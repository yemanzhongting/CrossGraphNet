<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CrossGraphNet README</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            max-width: 800px;
            margin: 0 auto;
        }
        h1 {
            color: #333;
        }
        button {
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            margin-right: 10px;
        }
        button:hover {
            background-color: #45a049;
        }
        #content {
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <h1>CrossGraphNet: A Fast Cross-Spatiotemporal Graph-based Method for Traffic Flow Reconstruction Using Remote Sensing Vehicle Detection</h1>
    
    <button onclick="showLanguage('en')">English</button>
    <button onclick="showLanguage('zh')">中文</button>

    <div id="content"></div>

    <script>
        const content = {
            en: `
                <h2>Introduction</h2>
                <p>CrossGraphNet is an innovative approach to traffic flow reconstruction using remote sensing vehicle detection data. It employs a cross-spatiotemporal graph-based method to effectively analyze and predict traffic patterns.</p>

                <h2>Key Features</h2>
                <ul>
                    <li>Utilizes Graph Convolutional Networks (GCN) for spatial analysis</li>
                    <li>Implements Multi-head Attention mechanism for temporal correlations</li>
                    <li>Combines GCN and Attention for cross-spatiotemporal learning</li>
                    <li>Supports incremental learning for continuous model improvement</li>
                </ul>

                <h2>Requirements</h2>
                <ul>
                    <li>Python 3.x</li>
                    <li>PyTorch</li>
                    <li>PyTorch Geometric</li>
                    <li>GeoPandas</li>
                    <li>Pandas</li>
                    <li>Other dependencies (see \`requirements.txt\`)</li>
                </ul>

                <h2>Model Architecture</h2>
                <p>The core of CrossGraphNet is the \`CrossAttentionGCN\` class, which includes:</p>
                <ul>
                    <li>Two GCN layers for spatial feature extraction</li>
                    <li>Multi-head Attention layer for temporal analysis</li>
                    <li>Linear layer for output prediction</li>
                </ul>

                <h2>Data Preparation</h2>
                <p>The model uses road network data and traffic flow data, including:</p>
                <ul>
                    <li>Road network topology (edge information)</li>
                    <li>Traffic density and speed data for multiple time periods</li>
                </ul>

                <h2>Training and Evaluation</h2>
                <ul>
                    <li>Customizable training process with flexible epoch settings</li>
                    <li>Comprehensive evaluation metrics: MSE, MAE, RMSE, RSE, and R²</li>
                    <li>Support for incremental learning with masked data selection</li>
                </ul>

                <h2>Usage</h2>
                <ol>
                    <li>Prepare your data in the required format (see \`prepare_data\` function)</li>
                    <li>Initialize the model with appropriate parameters</li>
                    <li>Train the model using the provided training loop</li>
                    <li>Evaluate the model performance using the test data</li>
                </ol>

                <h2>Future Work</h2>
                <ul>
                    <li>Enhance model scalability for larger road networks</li>
                    <li>Implement real-time prediction capabilities</li>
                    <li>Explore integration with other traffic data sources</li>
                </ul>
            `,
            zh: `
                <h2>简介</h2>
                <p>CrossGraphNet 是一种创新的交通流量重建方法，利用遥感车辆检测数据。它采用跨时空图形基础的方法来有效分析和预测交通模式。</p>

                <h2>主要特点</h2>
                <ul>
                    <li>使用图卷积网络（GCN）进行空间分析</li>
                    <li>实现多头注意力机制以捕捉时间相关性</li>
                    <li>结合 GCN 和注意力机制进行跨时空学习</li>
                    <li>支持增量学习，持续改进模型</li>
                </ul>

                <h2>环境要求</h2>
                <ul>
                    <li>Python 3.x</li>
                    <li>PyTorch</li>
                    <li>PyTorch Geometric</li>
                    <li>GeoPandas</li>
                    <li>Pandas</li>
                    <li>其他依赖项（详见 \`requirements.txt\`）</li>
                </ul>

                <h2>模型架构</h2>
                <p>CrossGraphNet 的核心是 \`CrossAttentionGCN\` 类，包括：</p>
                <ul>
                    <li>两个 GCN 层用于空间特征提取</li>
                    <li>多头注意力层用于时间分析</li>
                    <li>线性层用于输出预测</li>
                </ul>

                <h2>数据准备</h2>
                <p>模型使用道路网络数据和交通流量数据，包括：</p>
                <ul>
                    <li>道路网络拓扑（边信息）</li>
                    <li>多个时间段的交通密度和速度数据</li>
                </ul>

                <h2>训练与评估</h2>
                <ul>
                    <li>可定制的训练过程，灵活的 epoch 设置</li>
                    <li>全面的评估指标：MSE、MAE、RMSE、RSE 和 R²</li>
                    <li>支持通过数据掩码选择进行增量学习</li>
                </ul>

                <h2>使用方法</h2>
                <ol>
                    <li>按要求格式准备数据（参见 \`prepare_data\` 函数）</li>
                    <li>使用适当的参数初始化模型</li>
                    <li>使用提供的训练循环训练模型</li>
                    <li>使用测试数据评估模型性能</li>
                </ol>

                <h2>未来工作</h2>
                <ul>
                    <li>提高模型对大型道路网络的可扩展性</li>
                    <li>实现实时预测功能</li>
                    <li>探索与其他交通数据源的整合</li>
                </ul>
            `
        };

        function showLanguage(lang) {
            document.getElementById('content').innerHTML = content[lang];
        }

        // 默认显示英文
        showLanguage('en');
    </script>
</body>
</html>
