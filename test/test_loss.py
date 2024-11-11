import torch
from src.utils.lossFunction_PWMR import LossFunctionsPWMR


# 生成随机数据
N = 64  # 节点数
D = 2048  # 特征维度

# 随机生成特征矩阵
features = torch.randn(N, D)

# 随机生成邻接矩阵
adj_matrix = torch.rand(N, N)

# 随机生成局部密度向量
local_densities = torch.rand(N)

# 创建 LossFunctionsPWMR 实例
loss_func = LossFunctionsPWMR()

# 计算平滑性损失
smoothness_loss_value = loss_func.smoothness_loss(features, adj_matrix, local_densities)

print(smoothness_loss_value)