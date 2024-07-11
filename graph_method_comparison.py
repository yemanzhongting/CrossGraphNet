import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GCNConv
import torch
from torch.nn import Linear
import torch.nn.functional as F
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import ChebConv

class GCN1(nn.Module):
    def __init__(self, hidden_channel1):
        super(GCN1, self).__init__()
        self.conv1 = GCNConv(1, hidden_channel1)
        self.conv2 = GCNConv(hidden_channel1, 1)

    def forward(self,x, edge_index):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class GCN2(torch.nn.Module):
    def __init__(self, hidden_channel1,hidden_channel2=8):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(1, hidden_channel1)#768
        self.conv2 = GCNConv(hidden_channel1, hidden_channel2)
        self.conv3 = GCNConv(hidden_channel2,1)#dataset.num_classes

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv3(x, edge_index)
        return x

class ChebNet(nn.Module):
    def __init__(self, hidden_channel1=16, K=2):
        super(ChebNet, self).__init__()
        # K defines the order of the Chebyshev polynomial
        self.conv1 = ChebConv(1, hidden_channel1, K)
        self.conv2 = ChebConv(hidden_channel1, 1, K)

    def forward(self, x, edge_index):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class GraphSAGE(nn.Module):
    def __init__(self, hidden_channel1=16):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(1, hidden_channel1)
        self.conv2 = SAGEConv(hidden_channel1, 1)

    def forward(self,x, edge_index):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class GAT(nn.Module):
    def __init__(self,hidden_channel1=16):
        super(GAT, self).__init__()
        self.conv1 = GATConv(1,hidden_channel1, heads=8, dropout=0.6)
        self.conv2 = GATConv(hidden_channel1 * 8, 1, heads=1, concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x

import torch
import torch.nn.functional as F
from torch.nn import Module
from torch_geometric.nn import AGNNConv

class AGNNNet(Module):
    def __init__(self,hidden_channel1=16, num_layers=2):
        super(AGNNNet, self).__init__()
        self.lin = torch.nn.Linear(1, 16)  # 初始的全连接层，将特征维度从1映射到16
        self.convs = torch.nn.ModuleList([AGNNConv(requires_grad=True) for _ in range(num_layers)])
        self.out_lin = torch.nn.Linear(16, 1)  # 输出层，将特征映射回单个特征

    def forward(self, x, edge_index):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.lin(x))
        for conv in self.convs:
            x = conv(x, edge_index)
        x = self.out_lin(x)
        return x


import pandas as pd
import torch
from torch_geometric.data import Data
import numpy as np
from torch.nn import L1Loss, MSELoss
import numpy as np

MSEs = []
MAEs = []
RMSEs = []
RSEs = []
R2s = []
Losses = []

density_col1 = 'VN211129'
speed_col1 = 'F2021_11_3'

density_col2 = 'VN220504'
speed_col2 = 'F2022_05_0'

density_col3 = 'VN221026'
speed_col3 = 'F2022_10_2'

density_col4 = 'VN231101'
speed_col4 = 'F2023_11_0'

models = [AGNNNet, ChebNet, GCN1, GCN2, GAT, GraphSAGE]

import torch,os

path='.\HKproject'
with open(os.path.join(path,'Road_Buffer.gwt'), 'r+') as f:
  edges=f.readlines()
edge_index=[]
for edge in edges[1:]:
  tmp=list(map(int,edge.split(' ')[0:2]))
  edge_index.append([tmp[0]-1,tmp[1]-1])
edge_index = torch.tensor(edge_index, dtype=torch.long)
# torch.Size([2372, 2])
import geopandas as gpd
gdf=gpd.read_file(os.path.join(path,'./roadstatus/road_stats.shp'))
gdf.head()

results = {}
for graph in [AGNNNet, ChebNet, GCN1, GCN2, GAT, GraphSAGE]:
    model = graph(hidden_channel1=16)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    # 使用MSE criterion
    criterion = torch.nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    results[str(graph)] = []

    Meanes = {}
    Meanes['MSE'] = []
    Meanes['MAE'] = []
    Meanes['RMSE'] = []
    Meanes['RSE'] = []
    Meanes['R2'] = []
    Meanes['Loss'] = []

    for j in [(density_col1, speed_col1), (density_col2, speed_col2),(density_col3, speed_col3)]:
        density_col = j[0]
        speed_col = j[1]
        # 导入所需的库
        # 假设您希望筛选出'column1'和'column2'两列同时不为0或者不为空的行
        train_data = gdf.loc[(gdf[density_col].notnull()) & (gdf[speed_col].notnull()) &
                             (gdf[density_col] != 0) & (gdf[speed_col] != 0)]
        mask = torch.zeros(gdf.shape[0], 1, dtype=torch.bool)
        for i in range(gdf.shape[0]):
            # 构造一个embeddings，49*1大小，值为False
            if i in train_data.index.values:
                mask[i][0] = True
            else:
                mask[i][0] = False

        x = torch.tensor(gdf[density_col].values, dtype=torch.float)
        x = x.view(gdf.shape[0], 1)
        y = torch.tensor(gdf[speed_col].values, dtype=torch.float)
        y = y.view(gdf.shape[0], 1)

        x[torch.isnan(x)] = 0
        y[torch.isnan(y)] = 0

        data = Data(x=x, edge_index=edge_index.t().contiguous(),
                    y=y, train_mask=mask, test_mask=mask)

        model.train()
        for epoch in range(100):
            optimizer.zero_grad()
            output = model(data.x, data.edge_index)
            loss = criterion(output[data.train_mask], data.y[data.train_mask].view(-1, 1))

            # 这是loss记录
            results[str(graph)].append(loss.item())

            loss.backward()
            optimizer.step()

            # 在测试集上评估模型

    model.eval()
    #eval再走一遍

    density_col = density_col4
    speed_col = speed_col4
    # 导入所需的库
    train_data = gdf.loc[(gdf[density_col].notnull()) & (gdf[speed_col].notnull()) &
                         (gdf[density_col] != 0) & (gdf[speed_col] != 0)]

    mask = torch.zeros(gdf.shape[0], 1, dtype=torch.bool)
    for i in range(gdf.shape[0]):
        # 构造一个embeddings，49*1大小，值为False
        if i in train_data.index.values:
            mask[i][0] = True
        else:
            mask[i][0] = False
    # edge_index = edge_index
    x = torch.tensor(gdf[density_col].values, dtype=torch.float)
    x = x.view(gdf.shape[0], 1)
    y = torch.tensor(gdf[speed_col].values, dtype=torch.float)
    y = y.view(gdf.shape[0], 1)
    x[torch.isnan(x)] = 0
    y[torch.isnan(y)] = 0

    data = Data(x=x, edge_index=edge_index.t().contiguous(),
                y=y, train_mask=mask, test_mask=mask)

    with torch.no_grad():  # 在评估阶段不需要计算梯度
        # output_test = model(data.x, data.edge_index)
        output_test= model(data.x, data.edge_index)
        test_loss = criterion(output_test[data.test_mask], data.y[data.test_mask].view(-1, 1))
        print(f'Test Loss: {test_loss.item()}')

        true_values = data.y[data.test_mask].view(-1, 1)
        predicted_values = output_test[data.test_mask]

        # 计算MSE
        mse_loss = MSELoss()
        mse = mse_loss(predicted_values, true_values)

        # 计算MAE
        mae_loss = L1Loss()
        mae = mae_loss(predicted_values, true_values)

        # 计算RMSE
        rmse = torch.sqrt(mse)

        # 计算RSE
        variance = torch.var(true_values)
        rse = mse / variance

        # 计算R²
        total_variance = torch.sum((true_values - torch.mean(true_values)) ** 2)
        residual_variance = torch.sum((true_values - predicted_values) ** 2)
        r_squared = 1 - residual_variance / total_variance

        print(f'Test MSE: {mse.item()}')
        print(f'Test MAE: {mae.item()}')
        print(f'Test RMSE: {rmse.item()}')
        print(f'Test RSE: {rse.item()}')
        print(f'Test R²: {r_squared.item()}')