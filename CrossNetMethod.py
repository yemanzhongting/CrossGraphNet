import os
import pandas as pd
import geopandas as gpd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import MultiheadAttention, Linear, L1Loss, MSELoss
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool

# Constants
DENSITY_SPEED_COLS = [
    ('VN211129', 'F2021_11_3'),
    ('VN220504', 'F2022_05_0'),
    ('VN221026', 'F2022_10_2'),
    ('VN231101', 'F2023_11_0')
]

PATH = './HKproject'

# Load and process edge data
with open(os.path.join(PATH, 'Road_Buffer.gwt'), 'r+') as f:
    edges = f.readlines()
edge_index = [[int(edge.split()[0]) - 1, int(edge.split()[1]) - 1] for edge in edges[1:]]
edge_index = torch.tensor(edge_index, dtype=torch.long).t()

# Load spatial data
gdf = gpd.read_file(os.path.join(PATH, './roadstatus/road_stats.shp'))
smooth_df = pd.read_csv(os.path.join(PATH, './Spatial.csv'))
coords = smooth_df.values[:, 1:]
print(coords.shape)


class SpatialSmoothingModule(nn.Module):
    """Spatial smoothing module for incorporating geographical dependencies"""

    def __init__(self, coords, sigma=1.0):
        super().__init__()
        self.coords = torch.tensor(coords, dtype=torch.float32)
        self.sigma = nn.Parameter(torch.tensor([sigma]), requires_grad=True)

    def calculate_distance_matrix(self):
        return torch.cdist(self.coords, self.coords)

    def calculate_weight_matrix(self):
        distances = self.calculate_distance_matrix()
        weights = torch.exp(-distances ** 2 / (2 * self.sigma ** 2))
        return weights / weights.sum(dim=1, keepdim=True)

    def forward(self, x):
        weights = self.calculate_weight_matrix()
        smoothed = torch.matmul(x, weights)
        alpha = torch.sigmoid(self.sigma)
        return alpha * x + (1 - alpha) * smoothed


class CrossGraphNet(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_heads, num_nodes, coords, smooth_sigma=1.0):
        super().__init__()
        self.gcn1 = GCNConv(num_features, hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, num_nodes)
        self.attention = MultiheadAttention(num_nodes, num_heads)
        self.linear = Linear(num_nodes, num_nodes)
        self.spatial_smooth = SpatialSmoothingModule(coords, smooth_sigma)
        self.gcn_final = GCNConv(hidden_channels, hidden_channels)

    def forward(self, data):
        embeddings = []
        for sub_data in data:
            x, edge_index, batch = sub_data.x, sub_data.edge_index, sub_data.batch
            x = F.relu(self.gcn1(x, edge_index))
            x = F.relu(self.gcn2(x, edge_index))
            x = global_mean_pool(x, batch)
            embeddings.append(x)

        embeddings = torch.stack(embeddings, dim=0)
        attended_embeddings, _ = self.attention(embeddings, embeddings, embeddings)
        out = self.linear(attended_embeddings.mean(dim=1))
        out = self.spatial_smooth(out)
        return out * (110 - 30) + 30


def prepare_data(density_col, speed_col):
    train_data = gdf.loc[(gdf[density_col].notnull()) & (gdf[speed_col].notnull()) &
                         (gdf[density_col] != 0) & (gdf[speed_col] != 0)]

    mask = torch.zeros(gdf.shape[0], 1, dtype=torch.bool)
    for i in range(gdf.shape[0]):
        mask[i][0] = i in train_data.index.values

    x = torch.tensor(gdf[density_col].values, dtype=torch.float).view(-1, 1)
    y = torch.tensor(gdf[speed_col].values, dtype=torch.float).view(-1, 1)

    x[torch.isnan(x)] = 0
    y[torch.isnan(y)] = 0

    train_mask = torch.tensor(mask, dtype=torch.bool).view(-1, 1)
    test_mask = train_mask.clone()

    return Data(x=x, edge_index=edge_index, y=y,
                batch=torch.zeros(x.size(0), dtype=torch.long),
                train_mask=train_mask, test_mask=test_mask)


def custom_loss(predictions, targets):
    lower_bound, upper_bound = 30, 110
    mse_loss = F.mse_loss(predictions, targets)
    penalty_lower = torch.sum(F.relu(lower_bound - predictions))
    penalty_upper = torch.sum(F.relu(predictions - upper_bound))
    return mse_loss + penalty_lower + penalty_upper


# Model and training parameters
model_params = {
    'num_features': 1,
    'hidden_channels': 64,
    'num_heads': 1,
    'num_nodes': gdf.shape[0],
    'smooth_sigma': 0.01
}

training_params = {
    'num_epochs': 200,
    'learning_rate': 0.001
}

# Prepare data
all_data = [prepare_data(*cols) for cols in DENSITY_SPEED_COLS]

# Training loop
metrics = {metric: [] for metric in ['MSE', 'MAE', 'RMSE', 'RSE', 'R2', 'Loss']}

for mask in [torch.tensor([False, True, True, True]),
             torch.tensor([True, False, True, True]),
             torch.tensor([True, True, False, True]),
             torch.tensor([True, True, True, False])]:

    # Initialize model and optimizer
    model = CrossGraphNet(**model_params, coords=coords)
    optimizer = optim.Adam(model.parameters(), lr=training_params['learning_rate'])

    y_list = [data.y for data in all_data]
    mask_list = [data.train_mask for data in all_data]

    # Training
    for epoch in range(training_params['num_epochs']):
        model.train()
        optimizer.zero_grad()

        outputs = model(all_data)
        outputs = outputs[mask]
        targets = torch.stack([y_list[i] for i in range(len(y_list)) if mask[i]])
        targets_squeezed = targets.squeeze(-1)

        train_mask = torch.stack([mask_list[i] for i in range(len(mask_list)) if mask[i]])
        train_mask = train_mask.squeeze(-1)

        loss = custom_loss(outputs[train_mask], targets_squeezed[train_mask])
        print(loss)

        loss.backward()
        optimizer.step()

    # Testing
    model.eval()
    with torch.no_grad():
        outputs = model(all_data)
        outputs = outputs[~mask]
        print(outputs[0].tolist())

        targets = torch.stack([y_list[i] for i in range(len(y_list)) if ~mask[i]])
        targets_squeezed = targets.squeeze(-1)

        train_mask = torch.stack([mask_list[i] for i in range(len(mask_list)) if ~mask[i]])
        train_mask = train_mask.squeeze(-1)

        true_values = outputs[train_mask]
        predicted_values = targets_squeezed[train_mask]

        # Calculate metrics
        test_loss = custom_loss(outputs[train_mask], targets_squeezed[train_mask])
        mse = F.mse_loss(predicted_values, true_values)
        mae = F.l1_loss(predicted_values, true_values)
        rmse = torch.sqrt(mse)

        variance = torch.var(true_values)
        rse = mse / variance

        total_variance = torch.sum((true_values - torch.mean(true_values)) ** 2)
        residual_variance = torch.sum((true_values - predicted_values) ** 2)
        r_squared = 1 - residual_variance / total_variance

        # Print and store metrics
        print(f'Test MSE: {mse.item()}')
        print(f'Test MAE: {mae.item()}')
        print(f'Test RMSE: {rmse.item()}')
        print(f'Test RSE: {rse.item()}')
        print(f'Test R²: {r_squared.item()}')

        for metric, value in zip(['MSE', 'MAE', 'RMSE', 'RSE', 'R2', 'Loss'],
                                 [mse, mae, rmse, rse, r_squared, test_loss]):
            metrics[metric].append(value.item())

    print(f"Final Loss: {loss.item()}")
