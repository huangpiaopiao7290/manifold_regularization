from typing import Optional

import torch
from torch import Tensor, nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def knn_manual(x: Optional[Tensor], k: int):
    """
    knn
    Params:
        x (torch.Tensor): 数据点 shape [n_samples, n_features]
        k (int): 需要找到的最近邻居数量
    Returns:
        indices (torch.Tensor): 每个样本的k个最近邻居索引 shape [n_samples, k]
        distances (torch.Tensor): 每个样本到其k个最近邻居的距离 shape [n_samples, k]
    """
    if len(x.shape) > 2:
        x = x.view(x.size(0), -1)
    x = x.to(device)
    # 计算所有点之间的距离矩阵
    distances = torch.cdist(x, x)
    # 选择k+1个最小的距离及其对应的索引（包括自身）
    distances, indices = torch.topk(distances, k + 1, largest=False, sorted=True)
    # 排除自身点
    return indices[:, 1:], distances[:, 1:]

def build_adjacency_matrix(x: Optional[Tensor], ind: Optional[Tensor], dist: Optional[Tensor], k: int, sigma: float, mode: str = 'distance') -> Optional[Tensor]:
    """
    创建邻接矩阵w
    Params:
        x (torch.Tensor): 数据点 shape [n_samples, n_features]
        k (int): 每个数据点的邻居数
        sigma (float): 高斯核函数带宽参数
        ind (torch.Tensor): 每个样本的k个最近邻居索引 shape [n_samples, k]
        dist (torch.Tensor): 每个样本到其k个最近邻居的距离 shape [n_samples, k]
        mode (str): 'connectivity' 或 'distance'
    Returns:
        w (torch.Tensor): 邻接矩阵 shape [n_samples, n_samples]
    """
    # 初始化邻接矩阵
    n_samples = x.size(0)
    w = torch.zeros((n_samples, n_samples), dtype=torch.float32, device=device)
    # 填充邻接矩阵
    for i in range(n_samples):
        for j in range(k):
            neighbor_idx = ind[i, j].item()
            if mode == 'distance':
                weight = torch.exp(-dist[i, j] ** 2 / (2 * sigma ** 2))
            else:
                weight = 1.0
            w[i, neighbor_idx] = weight
            w[neighbor_idx, i] = weight

    return w

def calculate_local_density(distances: Optional[Tensor], u: tuple[Tensor, Tensor]) -> Optional[Tensor]:
    """
    计算数据点的局部密度p(xi)。

    Params:
        distances (torch.Tensor): 每个样本到其k个最近邻居的距离 shape [n_samples, k]
        u (tuple[Tensor, Tensor]): 隶属度(概率)

    Returns:
        p (torch.Tensor): 形状为(n_samples,)的局部密度向量p。
    """
    local_sum = torch.sum(distances, dim=1)     # 计算每个样本与K近邻的距离总和
    global_sum = torch.sum(distances)           # 计算全局距离总和

    part1 = 1 - (local_sum / global_sum)
    part2 = (torch.max(u[0], u[1]) if u is not None else 1.0)
    p = part1 * part2

    return p


def cross_entropy_loss(y_pre: Optional[Tensor], y_true: Optional[Tensor]) -> Optional[Tensor]:
    """
    对于有标记样本，比较模型的预测结果与样本的真实标记，计算交叉熵损失
    Params:
        y_pre (torch.Tensor): 模型预测结果，形状为 [batch_size, num_classes]
        y_true (torch.Tensor): 真实标签，形状为 [batch_size]
    Returns:
        如果reduction为 none , 形状 ()()(N)(N) 或 (N,d1,d2,...,dK)(N,d1,d2,...,dK),
        其中在K维损失的情况下, K≥1K≥1,取决于输入的形状。否则,标量
    """
    cross_entropy = torch.nn.CrossEntropyLoss()
    return cross_entropy(y_pre, y_true)

def consistency_loss(model: nn.Module, x: Optional[Tensor], alpha: float=1.) -> Optional[Tensor]:
    """
    正则化项: 对于无标记样本，采用数据增广计算一致性损失
    Params:
        model (Module): 模型
        x (torch.Tensor): 输入数据，形状为 [batch_size, features]
        alpha (float): Beta 分布的超参数。默认为1.0表示标准的 Mix_up。

    Returns:
        泛化后数据的一致损失
    """

    # ---------------------
    # 数据增广mix_up(a, b) = lam * a + (1-lam) * b
    # 1. 对数据集中任意两个样本点x_i, x_j以及对应的模型预测结果y_i, y_j
    # 2. 计算得到x = mix_up(x_i, x_j)以及模型对该插值的预测结果y = model(x)
    # 3. 要求y与mix_up(y_i, y_j)的一致性
    # ---------------------
    batch_size = x.size(0)
    indices = torch.randperm(batch_size).to(device)
    # lam遵循beta分布
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    mixed_x = lam * x + (1 - lam) * x[indices]
    # 计算原始输入和混合输入的预测
    with torch.no_grad():  # 不需要梯度计算，因为我们只使用预测结果作为目标
        output_x_i = model(x)
        output_x_j = model(x[indices])
    # targets_mixed = lam * output_x_i + (1 - lam) * output_x_j
    targets_mixed = lam * nn.functional.softmax(output_x_i, dim=1) + (1 - lam) * nn.functional.softmax(output_x_j, dim=1)
    output_mixed = model(mixed_x)

    # loss = cross_entropy_loss(output_mixed, targets_mixed)
    # loss = nn.functional.mse_loss(nn.functional.softmax(output_mixed, dim=1), targets_mixed.detach())
    loss = nn.functional.kl_div(output_mixed, targets_mixed, reduction="batchmean")
    

    return loss


def point_smoothness_loss(model, x, k, sigma, u: tuple[Tensor, Tensor]=None) -> Optional[Tensor]:
    """
    计算逐点平滑性损失的核心部分。

    Params:
        model (Module): 模型
        x (torch.Tensor): 所有数据点，包括有标签和无标签样本
        k (int): 邻居数
        sigma (float): 高斯核函数带宽参数
        u (tuple[Tensor, Tensor]): 隶属度(概率)

    Returns:
        逐点平滑性损失值（不包含系数）
    """
    indices, distances = knn_manual(x, k)
    mtr = build_adjacency_matrix(x, indices, distances, k, sigma,  'connectivity')   # 创建邻接矩阵
    p = calculate_local_density(distances, u)                                        # 计算局部密度

    with torch.no_grad():
        predictions = nn.functional.softmax(model(x), dim=1)

    n = x.size(0)
    smoothness_loss = torch.zeros(1, device=device)
    for i in range(n):
        neighbor_indices = indices[i]
        neighbor_weights = mtr[i, neighbor_indices]
        neighbors_sum = torch.sum(neighbor_weights.unsqueeze(1) * predictions[neighbor_indices], dim=0)

        diff = predictions[i] - neighbors_sum
        smoothness_loss += p[i] * torch.dot(diff, diff)

    return smoothness_loss


