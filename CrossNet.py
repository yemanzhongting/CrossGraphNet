import torch
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.nn import MultiheadAttention, Linear
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
from torch.nn import L1Loss, MSELoss
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.nn import MultiheadAttention, Linear
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

density_col1='VN211129'
speed_col1='F2021_11_3'

density_col2='VN220504'
speed_col2='F2022_05_0'

density_col3='VN221026'
speed_col3='F2022_10_2'

density_col4='VN231101'
speed_col4='F2023_11_0'

path='./HKproject'

import torch,os
with open(os.path.join(path,'Road_Buffer.gwt'), 'r+') as f:
  edges=f.readlines()
edge_index=[]
for edge in edges[1:]:
  tmp=list(map(int,edge.split(' ')[0:2]))
  edge_index.append([tmp[0]-1,tmp[1]-1])
edge_index = torch.tensor(edge_index, dtype=torch.long)
# torch.Size([2372, 2]) ->转
# edge_index = torch.tensor(edge_index, dtype=torch.long)
edge_index = edge_index.t()

# Out[20]: torch.Size([2, 20])

import geopandas as gpd
gdf=gpd.read_file(os.path.join(path,'./roadstatus/road_stats.shp'))
gdf.head()

class CrossAttentionGCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_heads,num_nodes):
        super().__init__()
        self.gcn1 = GCNConv(num_features, hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, hidden_channels)
        self.attention = MultiheadAttention(hidden_channels, num_heads)
        self.linear = Linear(hidden_channels, num_nodes)
        self.gcn_final = GCNConv(hidden_channels, hidden_channels)  # 新添加的GCN层
      
    def forward(self, data):
        embeddings = []
        for sub_data in data:
            # in zip(x_list, edge_index_list, batch_ids_list):
            x, edge_index, batch =sub_data.x,sub_data.edge_index,sub_data.batch
            x = F.relu(self.gcn1(x, edge_index))
            x = F.relu(self.gcn2(x, edge_index))
            x = global_mean_pool(x, batch)
            embeddings.append(x)
        # torch.Size([1, 1])
        # torch.Size([1, 16])
        embeddings = torch.stack(embeddings, dim=0)  # Changed dim=1 to dim=0
        # torch.Size([4, 1, 1])
        # torch.Size([4, 1, 16])
        attended_embeddings, _ = self.attention(embeddings, embeddings, embeddings)
        # torch.Size([4, 1, 16])
        # torch.Size([1, 4, 4])
        attended_embeddings = attended_embeddings.mean(dim=1)
        #([4, 16])
      
        # 经过Attention后再进行GCN
        # batch_size, num_nodes = attended_embeddings.size(0), attended_embeddings.size(1)
        # edge_index = data[0].edge_index.repeat(1, batch_size)
        # edge_index[0] += torch.arange(0, batch_size * num_nodes, num_nodes).repeat_interleave(edge_index.size(1))
        # edge_index[1] += torch.arange(0, batch_size * num_nodes, num_nodes).repeat_interleave(edge_index.size(1))
        # 应用最后的GCN层
        # out = F.relu(self.gcn_final(attended_embeddings.view(-1, hidden_channels), edge_index))
        # out = self.linear(out).view(batch_size, num_nodes)

        #简单方法，直接linear输出
        # torch.Size([4, 16])
        out = self.linear(attended_embeddings)
        # torch.Size([4, 10]) num_nodes = 10
        # out = torch.relu(out)  # 限制输出为非负值

        lower_bound, upper_bound = 50, 110
        out  = out  * (upper_bound - lower_bound) + lower_bound
        return out

# density_col='VN231101', speed_col='F2023_11_0')
def prepare_data(density_col, speed_col):
    # Create a mask for rows where both density and speed are not null and not equal to zero
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

    # Convert the entire columns to tensors, disregarding whether they are valid according to the mask
    x = torch.tensor(gdf[density_col].values, dtype=torch.float).view(-1, 1)
    y = torch.tensor(gdf[speed_col].values, dtype=torch.float).view(-1, 1)

    # Replace NaN values with 0 in tensors
    x[torch.isnan(x)] = 0
    y[torch.isnan(y)] = 0

    # Create a boolean tensor from the valid_mask for use as train_mask and test_mask
    # train_mask = torch.tensor(mask.values, dtype=torch.bool).view(-1, 1)
    train_mask = torch.tensor(mask,dtype=torch.bool).view(-1, 1)
    test_mask = train_mask.clone()

    # Create a Data object for use in PyTorch Geometric
    return Data(x=x, edge_index=edge_index, y=y, batch=torch.zeros(x.size(0), dtype=torch.long),
                                train_mask=train_mask, test_mask=test_mask)

# Example usage
all_data = [prepare_data(*cols) for cols in [
    ('VN211129', 'F2021_11_3'),
    ('VN220504', 'F2022_05_0'),
    ('VN221026', 'F2022_10_2'),
    ('VN231101', 'F2023_11_0')
]]


def custom_loss(predictions, targets):
    lower_bound, upper_bound=50,110
    mse_loss = F.mse_loss(predictions, targets)
    # 惩罚超出上下限的预测值
    penalty_lower = torch.sum(F.relu(lower_bound - predictions))
    penalty_upper = torch.sum(F.relu(predictions - upper_bound))
    total_loss = mse_loss + penalty_lower + penalty_upper
    return total_loss

# Test MSE: 23.707172393798828
# Test MAE: 3.7119181156158447
# Test RMSE: 4.869001388549805
# Test RSE: 0.1372707337141037
# Test R²: 0.858916163444519
# Final Loss: 22.792165756225586

# def custom_loss(predictions, targets):
#     lower_bound, upper_bound = 50, 110
#     mse_loss = F.mse_loss(predictions, targets)
#     # 惩罚超出上下限的预测值
#     penalty_lower = torch.sum(F.relu(lower_bound - predictions) ** 2)
#     penalty_upper = torch.sum(F.relu(predictions - upper_bound) ** 2)
#     total_loss = mse_loss + penalty_lower + penalty_upper
#     return total_loss
# Test MSE: 24.785675048828125
# Test MAE: 3.6857142448425293
# Test RMSE: 4.978521347045898
# Test RSE: 0.13722822070121765
# Test R²: 0.8589599132537842
# Final Loss: 20.86109733581543


# 在训练循环中使用这个自定义损失函数
# optimizer.zero_grad()
# predictions = model(data)
# loss = custom_loss(predictions, targets, lower_bound, upper_bound)
# loss.backward()
# optimizer.step()
####loss


#可以进行增量学习
# out_mask = torch.tensor([True,True,True,False])
out_mask = torch.tensor([False,True,True,True])

for mask in [torch.tensor([False,True,True,True]),torch.tensor([True,False,True,True]),
             torch.tensor([True,True,False,True]),torch.tensor([True,True,True,False])]:
    out_mask=mask
# out_mask = torch.tensor([True,False,True,False])
#
# out_mask = torch.tensor([False,True,True,True])

    # Initialize model, optimizer, and loss function
    model = CrossAttentionGCN(num_features=1, hidden_channels=64,
                              num_heads=2, num_nodes=gdf.shape[0])
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    loss_fn = custom_loss

    y_list=[]
    for i in all_data:
        y_list.append(i.y)
    mask_list=[]
    for i in all_data:
        mask_list.append(i.train_mask)

    # 生成取反的布尔张量
    test_mask = ~out_mask

    # Training loop
    num_epochs = 200

    Meanes={}
    Meanes['MSE']=[]
    Meanes['MAE']=[]
    Meanes['RMSE']=[]
    Meanes['RSE']=[]
    Meanes['R2']=[]
    Meanes['Loss']=[]

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        optimizer.zero_grad()
        outputs = model(all_data)
        # torch.Size([3, 10])
        outputs = outputs[out_mask]  # Apply mask to select specific graphs
        targets = torch.stack([y_list[i] for i in range(len(y_list)) if out_mask[i]])
        # Select targets based on mask
        # 使用 squeeze() 方法移除最后一个维度
        targets_squeezed = targets.squeeze(-1)  # 指定-1移除最后一个维度，这里维度大小为1
        #别人的是torch.Size([2, 10]) 训练数+node_number
        #处理双重mask
        train_mask = torch.stack([mask_list[i] for i in range(len(mask_list)) if out_mask[i]])
        train_mask= train_mask.squeeze(-1)  # 指定-1移除最后一个维度，这里维度大小为1
        loss = loss_fn(outputs[train_mask],targets_squeezed[train_mask])

        print(loss)

        # print("输出的size大小为:",outputs.shape)
        loss.backward()
        optimizer.step()
        # print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    # Testing
    model.eval()
    with torch.no_grad():  # 在评估阶段不需要计算梯度

        outputs = model(all_data)
        outputs = outputs[~out_mask]  # Apply mask to select specific graphs
        print(outputs[0].tolist())

        #

        targets = torch.stack([y_list[i] for i in range(len(y_list)) if ~out_mask[i]])
        # Select targets based on mask
        # 使用 squeeze() 方法移除最后一个维度
        targets_squeezed = targets.squeeze(-1)  # 指定-1移除最后一个维度，这里维度大小为1
        # 别人的是torch.Size([2, 10]) 训练数+node_number
        # 处理双重mask
        train_mask = torch.stack([mask_list[i] for i in range(len(mask_list)) if ~out_mask[i]])
        train_mask = train_mask.squeeze(-1)  # 指定-1移除最后一个维度，这里维度大小为1

        true_values = outputs[train_mask]
        predicted_values = targets_squeezed[train_mask]

        test_loss = loss_fn(outputs[train_mask], targets_squeezed[train_mask])

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

        Meanes['MSE'].append(mse.item())
        Meanes['MAE'].append(mae.item())
        Meanes['RMSE'].append(rmse.item())
        Meanes['RSE'].append(rse.item())
        Meanes['R2'].append(r_squared.item())
        Meanes['Loss'].append(test_loss.item())

    print(f"Final Loss: {loss.item()}")
