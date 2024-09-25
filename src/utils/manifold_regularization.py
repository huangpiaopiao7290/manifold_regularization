from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_adjacency_matrix(features, k=8, sigma=1.0):
    """
    计算邻接矩阵 W。
    
    :param features: 样本特征 [N, D]
    :param k: K-NN 的邻居数量
    :param sigma: 高斯核带宽参数
    :return: 邻接矩阵 W [N, N]
    """
    # 将tensor转换为numpy
    features_np = features.detach().cpu().numpy()
    # 使用K-NN找到每个样本的k个最近邻居
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(features_np)
    _, indices = nbrs.kneighbors(features_np)

    # 初始化邻接矩阵W
    N = features.shape[0]
    W = torch.zeros((N, N), device=features.device)
    
    # 样本i邻居j
    for i in range(N):
        for j in indices[i, 1:]:  # 跳过自身
            # 计算样本i和邻居j之间的欧式距离
            dist = torch.norm(features[i] - features[j])
            # 计算两个样本之间的相似度
            W[i, j] = torch.exp(-dist**2 / (2 * sigma**2))
    
    # 归一化
    row_sums = W.sum(dim=1).view(-1, 1)
    W = W / (row_sums + 1e-8)  # 防止除零错误

    return W


# Mixup 插值函数
def mixup(x1, x2, alpha=0.75):
    lam = torch.distributions.Beta(alpha, alpha).sample()
    mixed_x = lam * x1 + (1 - lam) * x2
    return mixed_x, lam

# 定义交叉熵损失
criterion_supervised = nn.CrossEntropyLoss(reduction='mean')

def consistency_loss(predictions, perturbed_predictions, mask):
    """一致性损失"""
    mse = nn.MSELoss(reduction='none')
    loss = mse(predictions, perturbed_predictions)
    loss = (loss * mask.unsqueeze(1)).sum() / (mask.sum() + 1e-8)
    return loss

def smoothness_loss(features, W):
    """平滑性损失"""
    diff = features.unsqueeze(0) - features.unsqueeze(1)
    squared_diff = (diff ** 2).sum(dim=-1)
    loss = (W * squared_diff).sum() / 2
    return loss

def total_loss(model, images, labels, unlabeled_mask, lambda_c, lambda_s):
    """总损失"""
    # 获取模型预测
    outputs = model(images)
    
    # 分割有标签和无标签的数据
    labeled_outputs = outputs[~unlabeled_mask]
    labeled_labels = labels[~unlabeled_mask]
    
    # 交叉熵损失
    loss_supervised = criterion_supervised(labeled_outputs, labeled_labels)
    
    # 一致性损失
    # 对于无标签数据，进行Mixup增强
    unlabeled_images = images[unlabeled_mask]
    if unlabeled_images.size(0) > 0:
        mixed_images, lam = mixup(unlabeled_images, unlabeled_images)
        mixed_outputs = model(mixed_images)
        with torch.no_grad():
            teacher_outputs = model(unlabeled_images)
        interpolated_teacher_outputs = lam * teacher_outputs + (1 - lam) * teacher_outputs
        loss_consistency = consistency_loss(mixed_outputs, interpolated_teacher_outputs, unlabeled_mask)
    else:
        loss_consistency = 0.0
    
    # 平滑性损失
    feature_extractor = nn.Sequential(*list(model.children())[:-1])  # 假设倒数第二层是特征提取器
    with torch.no_grad():
        features = feature_extractor(images)
    W = compute_adjacency_matrix(features)
    loss_smoothness = smoothness_loss(features, W)
    
    # 总损失
    total_loss = loss_supervised + lambda_c * loss_consistency + lambda_s * loss_smoothness
    return total_loss, W