# CrossGraphNet: A Fast Cross-Spatiotemporal Graph-based Method for Traffic Flow Reconstruction Using Remote Sensing Vehicle Detection

## English

### Introduction
CrossGraphNet is an innovative approach to traffic flow reconstruction using remote sensing vehicle detection data. It employs a cross-spatiotemporal graph-based method to effectively analyze and predict traffic patterns.

### Key Features
- Utilizes Graph Convolutional Networks (GCN) for spatial analysis
- Implements Multi-head Attention mechanism for temporal correlations
- Combines GCN and Attention for cross-spatiotemporal learning
- Supports incremental learning for continuous model improvement

### Requirements
- Python 3.x
- PyTorch
- PyTorch Geometric
- GeoPandas
- Pandas
- Other dependencies (see `requirements.txt`)

### Model Architecture
The core of CrossGraphNet is the `CrossAttentionGCN` class, which includes:
- Two GCN layers for spatial feature extraction
- Multi-head Attention layer for temporal analysis
- Linear layer for output prediction

### Data Preparation
The model uses road network data and traffic flow data, including:
- Road network topology (edge information)
- Traffic density and speed data for multiple time periods

### Training and Evaluation
- Customizable training process with flexible epoch settings
- Comprehensive evaluation metrics: MSE, MAE, RMSE, RSE, and R²
- Support for incremental learning with masked data selection

### Usage
1. Prepare your data in the required format (see `prepare_data` function)
2. Initialize the model with appropriate parameters
3. Train the model using the provided training loop
4. Evaluate the model performance using the test data

### Future Work
- Enhance model scalability for larger road networks
- Implement real-time prediction capabilities
- Explore integration with other traffic data sources

## 中文

### 简介
CrossGraphNet 是一种创新的交通流量重建方法，利用遥感车辆检测数据。它采用跨时空图形基础的方法来有效分析和预测交通模式。

### 主要特点
- 使用图卷积网络（GCN）进行空间分析
- 实现多头注意力机制以捕捉时间相关性
- 结合 GCN 和注意力机制进行跨时空学习
- 支持增量学习，持续改进模型

### 环境要求
- Python 3.x
- PyTorch
- PyTorch Geometric
- GeoPandas
- Pandas
- 其他依赖项（详见 `requirements.txt`）

### 模型架构
CrossGraphNet 的核心是 `CrossAttentionGCN` 类，包括：
- 两个 GCN 层用于空间特征提取
- 多头注意力层用于时间分析
- 线性层用于输出预测

### 数据准备
模型使用道路网络数据和交通流量数据，包括：
- 道路网络拓扑（边信息）
- 多个时间段的交通密度和速度数据

### 训练与评估
- 可定制的训练过程，灵活的 epoch 设置
- 全面的评估指标：MSE、MAE、RMSE、RSE 和 R²
- 支持通过数据掩码选择进行增量学习

### 使用方法
1. 按要求格式准备数据（参见 `prepare_data` 函数）
2. 使用适当的参数初始化模型
3. 使用提供的训练循环训练模型
4. 使用测试数据评估模型性能

### 未来工作
- 提高模型对大型道路网络的可扩展性
- 实现实时预测功能
- 探索与其他交通数据源的整合
