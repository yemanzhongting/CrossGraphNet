import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import os
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.utils import add_self_loops, degree

# 数据处理部分保持不变
path = '.\HKproject'
with open(os.path.join(path, 'Road_Buffer.gwt'), 'r+') as f:
    edges = f.readlines()
edge_index = []
for edge in edges[1:]:
    tmp = list(map(int, edge.split(' ')[0:2]))
    edge_index.append([tmp[0] - 1, tmp[1] - 1])
edge_index = torch.tensor(edge_index, dtype=torch.long)

gdf = gpd.read_file(os.path.join(path, './roadstatus/road_stats.shp'))


# 模型定义需要修改以接受三个时间步长的输入
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, edge_index=None):
        # x shape: (593, 3, 1)
        # edge_index shape: (2372, 2), 但在这个模型中不使用

        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 取最后一个时间步的输出，shape: (593, hidden_size)
        return self.fc(out)


class STRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1):
        super(STRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.gcn = GCNConv(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, edge_index):
        # x shape: (593, 3, 1)
        # edge_index shape: (2372, 2)

        out, _ = self.rnn(x)
        out = out[:, -1, :]  # 取最后一个时间步的输出，shape: (593, hidden_size)

        edge_index = edge_index.t().contiguous().long()  # 转置为 (2, 2372)

        out = self.gcn(out, edge_index)
        return self.fc(out)


class STLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1):
        super(STLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.gcn = GCNConv(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, edge_index):
        # x shape: (593, 3, 1)
        # edge_index shape: (2372, 2)

        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 取最后一个时间步的输出，shape: (593, hidden_size)

        edge_index = edge_index.t().contiguous().long()  # 转置为 (2, 2372)

        out = self.gcn(out, edge_index)
        return self.fc(out)


class CTLE(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1):
        super(CTLE, self).__init__()
        self.conv1d = nn.Conv1d(1, hidden_size, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.gcn = GCNConv(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, edge_index):
        # x shape: (593, 3, 1)
        # edge_index shape: (2372, 2)

        x = x.transpose(1, 2)  # (593, 1, 3)
        x = F.relu(self.conv1d(x))  # (593, hidden_size, 3)
        x = x.transpose(1, 2)  # (593, 3, hidden_size)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 取最后一个时间步的输出，shape: (593, hidden_size)

        edge_index = edge_index.t().contiguous().long()  # 转置为 (2, 2372)

        out = self.gcn(out, edge_index)
        return self.fc(out)


class BiLSTMCNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1):
        super(BiLSTMCNN, self).__init__()
        self.bilstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.conv1d = nn.Conv1d(hidden_size * 2, hidden_size, kernel_size=3, padding=1)
        self.gcn = GCNConv(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, edge_index):
        # x shape: (593, 3, 1)
        # edge_index shape: (2372, 2)

        out, _ = self.bilstm(x)
        out = out.transpose(1, 2)  # (593, hidden_size*2, 3)
        out = F.relu(self.conv1d(out))  # (593, hidden_size, 3)
        out = out[:, :, -1]  # (593, hidden_size)

        edge_index = edge_index.t().contiguous().long()  # 转置为 (2, 2372)

        out = self.gcn(out, edge_index)
        return self.fc(out)

class AttentionLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1):
        super(AttentionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, _):
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        out = self.fc(attn_out[:, -1, :])
        return out

class TransformerModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1, nhead=4, num_layers=2):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, batch_first=True),
            num_layers=num_layers
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, _):
        x = self.embedding(x)
        out = self.transformer(x)
        return self.fc(out[:, -1, :])

class LSTMGCN(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1):
        super(LSTMGCN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.gcn = GCNConv(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, edge_index):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 取最后一个时间步
        edge_index = edge_index.t().contiguous().long()
        out = self.gcn(out, edge_index)
        return self.fc(out)

def custom_loss(predictions, targets, mask):
    lower_bound, upper_bound = 50, 110
    # 应用 mask
    masked_predictions = predictions[mask]
    masked_targets = targets[mask]
    # 计算 MSE 损失
    mse_loss = F.mse_loss(masked_predictions, masked_targets)
    # 惩罚超出上下限的预测值
    penalty_lower = torch.sum(F.relu(lower_bound - masked_predictions))
    penalty_upper = torch.sum(F.relu(masked_predictions - upper_bound))
    # 计算总损失
    total_loss = mse_loss + penalty_lower + penalty_upper
    return total_loss

# 在训练循环中使用
criterion = custom_loss

def masked_mse_loss(pred, target, mask):
    loss = F.mse_loss(pred, target, reduction='none')
    mask = mask.float()
    loss = loss * mask
    return loss.sum() / mask.sum()


def masked_mae_loss(pred, target, mask):
    loss = F.l1_loss(pred, target, reduction='none')
    mask = mask.float()
    loss = loss * mask
    return loss.sum() / mask.sum()

# 主实验循环
density_cols = ['VN211129', 'VN220504', 'VN221026', 'VN231101']
speed_cols = ['F2021_11_3', 'F2022_05_0', 'F2022_10_2', 'F2023_11_0']

models = [LSTM(), STRNN(), STLSTM(), CTLE(), BiLSTMCNN(),LSTMGCN(),TransformerModel(),AttentionLSTM()]
model_names = ['LSTM', 'ST-RNN', 'ST-LSTM', 'CTLE', 'BiLSTM-CNN','LSTMGCN', 'TransformerModel', 'AttentionLSTM']

results = {}

for model, model_name in zip(models, model_names):
    print(f"Training {model_name}")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    criterion = custom_loss

    # 准备训练数据
    train_x = []
    train_y = []
    train_mask = []

    for i in range(3):  # 使用前三个时间段的数据
        density_col, speed_col = density_cols[i], speed_cols[i]

        valid_data = gdf.loc[(gdf[density_col].notnull()) & (gdf[speed_col].notnull()) &
                             (gdf[density_col] != 0) & (gdf[speed_col] != 0)]

        x = torch.tensor(gdf[density_col].values, dtype=torch.float).view(-1, 1)
        y = torch.tensor(gdf[speed_col].values, dtype=torch.float).view(-1, 1)

        mask = torch.zeros(gdf.shape[0], 1, dtype=torch.bool)
        mask[valid_data.index.values] = True

        x[torch.isnan(x)] = 0
        y[torch.isnan(y)] = 0

        train_x.append(x)
        train_y.append(y)
        train_mask.append(mask)

    train_x = torch.stack(train_x, dim=1)  # shape: (num_nodes, 3, 1)
    train_y = train_y[-1]  # 使用最后一个时间步的y作为目标
    train_mask = train_mask[-1]  # 使用最后一个时间步的mask

    # 训练过程
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        output = model(train_x, edge_index)#.t().contiguous()
        loss = criterion(output[train_mask], train_y[train_mask], train_mask[train_mask])
        loss.backward()
        optimizer.step()

    # 评估过程
    model.eval()
    density_col, speed_col = density_cols[-1], speed_cols[-1]

    test_data = gdf.loc[(gdf[density_col].notnull()) & (gdf[speed_col].notnull()) &
                        (gdf[density_col] != 0) & (gdf[speed_col] != 0)]

    test_x = train_x[:, 1:, :]  # 使用后两个时间步和最新的数据
    x = torch.tensor(gdf[density_col].values, dtype=torch.float).view(-1, 1)
    test_x = torch.cat([test_x, x.unsqueeze(1)], dim=1)  # shape: (num_nodes, 3, 1)

    y = torch.tensor(gdf[speed_col].values, dtype=torch.float).view(-1, 1)

    mask = torch.zeros(gdf.shape[0], 1, dtype=torch.bool)
    mask[test_data.index.values] = True

    x[torch.isnan(x)] = 0
    y[torch.isnan(y)] = 0

    with torch.no_grad():
        output_test = model(test_x, edge_index)
        test_loss = criterion(output_test[mask], y[mask], mask[mask])

        # 计算评估指标
        mse = masked_mse_loss(output_test[mask], y[mask], mask[mask])
        mae = masked_mae_loss(output_test[mask], y[mask], mask[mask])
        rmse = torch.sqrt(mse)

        # 计算R2分数和RSE (只考虑有效的y值)
        y_valid = y[mask].squeeze()
        output_valid = output_test[mask].squeeze()
        y_mean = torch.mean(y_valid)
        ss_tot = torch.sum((y_valid - y_mean) ** 2)
        ss_res = torch.sum((y_valid - output_valid) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        rse = ss_res / ss_tot  # 新增RSE计算

        results[model_name] = {
            'MSE': mse.item(),
            'MAE': mae.item(),
            'RMSE': rmse.item(),
            'R2': r2.item(),
            'RSE': rse.item(),  # 新增RSE结果
            'test_loss': test_loss.item()
        }

    print(f"{model_name} Test Loss: {test_loss.item():.4f}")

# 打印所有模型的结果
# 打印所有模型的结果
print("\nModel Comparison Results:")
for model_name, metrics in results.items():
    print(f"\n{model_name}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

# 找出表现最好的模型
best_model = min(results, key=lambda x: results[x]['MSE'])
print(f"\nBest performing model based on MSE: {best_model}")

# 可视化结果
metrics = ['MSE', 'MAE', 'RMSE', 'R2']
for metric in metrics:
    plt.figure(figsize=(10, 6))
    plt.bar(results.keys(), [results[model][metric] for model in results])
    plt.title(f'Comparison of {metric} across models')
    plt.xlabel('Models')
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 保存结果到CSV文件
results_df = pd.DataFrame(results).T
results_df.to_csv('model_comparison_results.csv')
print("Results saved to 'model_comparison_results.csv'")
