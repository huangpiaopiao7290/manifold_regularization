## @auther: pp
## @date: 2024/10/5
## @description: loss function

from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn


__all__ = ['compute_adjacency_matrix', 'consistency_loss', 'criterion_supervised', 'smoothness_loss', 'total_loss']


def compute_adjacency_matrix(features, k=8, sigma=1.0):
    """
    calculate adjacency matrix

    the input: a batch feature which size is [N, D], N is the number of samples, D is the feature dimension
    the `target` that an adjacency matrix which size is [N, N]
    K-NN algorithm is used to calculate the adjacency matrix, and the Gaussian kernel is used to smooth the adjacency matrix
    Gaussian kernel:
        -- math:
            W[i, j] = exp(-||x_i - x_j||^2 / (2 * sigma^2)),
    inside the exp function, the ||x_i - x_j|| is the Euclidean distance between x_i and x_j.
    in order to avoid the calculation of square root, we use the square of the Euclidean distance.
    the `params`:
        features: the input feature, a tensor with size [N, D]
        k: the number of nearest neighbors, an integer
        sigma: the parameter of the Gaussian kernel, a float
    """
    # 将tensor转换为numpy
    features_np = features.detach().cpu().numpy()
    # 使用K-NN找到每个样本的k个最近邻居
    nearest_neighbors = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(features_np)
    distances, indices = nearest_neighbors.kneighbors(features_np)
    # 初始化邻接矩阵
    _n = features.shape[0]
    _w = torch.zeros((_n, _n))
    # 对于每个样本 i, 找到它的 k 个最近邻居 j，计算高斯核函数值并赋值给邻接矩阵
    for i in range(_n):
        # 跳过自身
        for j in indices[i, 1:]:
            # 计算样本 (i, j) 的欧式距离
            dist = torch.norm(features[i] - features[j])
            # 计算两个样本之间的相似度
            _w[i, j] = torch.exp(-dist ** 2 / (2 * sigma ** 2))

    # 归一化
    try:
        row_sum = _w.sum(dim=1, keepdim=True).view(-1, 1)
        _w = _w / (row_sum + 1e-8)
    except ValueError as e:
        # 除数为零错误
        raise ValueError("calculation error: ｛｝".format(e))

    return _w

def mix_up(x1, x2, alpha=0.75):
    """
    By performing a simple linear transformation of the input data
    -- math:
        x = alpha * x1 + (1 - alpha) * x2

    the `params`:
        x1: the first batch, a tensor of shape (batch_size, ...)
        x2: the second batch, a tensor of shape (batch_size, ...)
        alpha: the interpolation coefficient, a scalar
    return: the mixed batch, a tensor of shape (batch_size, ...), and the interpolation coefficient
    """
    # beta分布中随机采样一个值作为混合系数
    lam = torch.distributions.Beta(alpha, alpha).sample().to(x1.device)
    # 按照混合系数进行线性插值
    mixed_x = lam * x1 + (1 - lam) * x2

    return mixed_x, lam

# 定义交叉熵损失
criterion_supervised = nn.CrossEntropyLoss(reduction='mean')

def consistency_loss(predictions, perturbed_predictions, mask):
    r"""
    calculate the consistency loss

    -- math:
        MSE:
            loss<i, c> = (Y<i, c> - \widehat{Y}<i, c>) ^ 2
            total_loss = \sigma_i=0^{N-1} m_i * \sigma_c=0^{C-1} loss<i, c> / (\sigma_i=0^{N-1} m_i +＼EPSILON)

            inside:
                Y: prediction of original inputs, the size is [N, C]
                \widehat{Y}: prediction of inputs after perturbation, the size is [N, C]


    :param predictions: the predictions of original data, a tensor of shape (batch_size, num_classes)
    :param perturbed_predictions: the predictions of the model with perturbation, a tensor of shape (batch_size, num_classes)
    :param mask: the mask of the data without label, a tensor of shape (batch_size, 1)
    :return: the consistency loss, a scalar

    """
    # 计算每个样本的均方误差损失 返回每个样本的损失值
    mse = nn.MSELoss(reduction='none')
    # 计算原始数据预测与扰动后输入预测之间的MSE损失  返回一个与predictions形状相同的tensor  其中每个元素对应于样本的MSE损失
    loss = mse(predictions, perturbed_predictions)
    # 计算加权平均损失
    loss = (loss * mask.unsqueeze(1)).sum() / (mask.sum() + 1e-8)
    
    return loss

def smoothness_loss(features, w):
    r"""
    smoothness loss
    If two sample points are close in the input space, then their outputs such as classification labels should  be similar.
    --math:
        l_s (\theta, \iota, \mu, w) = \sum_{}

    :param features: adjacency matrix, represents the feature vector for each sample
    :param w: weight matrix, which indicates the strength or similarity of connections between different samples
    """
    # 所有样本特征两两之间的差异矩阵
    diff = features.unsqueeze(0) - features.unsqueeze(1)
    # 计算 diff 的平方和 沿着最后一个维度求和  (每对样本之间的欧氏距离平方)
    squared_diff = (diff ** 2).sum(dim=-1)
    loss = (w * squared_diff).sum() / 2
    return loss

def total_loss(model, images, labels, unlabeled_mask, lambda_c, lambda_s):
    """
    total loss

    :param model:
    :param images:
    :param labels:
    :param unlabeled_mask:
    :param lambda_c:
    :param lambda_s:
    """
    # 获取模型预测
    outputs = model(images)
    
    # 分割有标签和无标签的数据
    labeled_outputs = outputs[~unlabeled_mask]
    labeled_labels = labels[~unlabeled_mask]
    
    # 带标签样本的交叉熵损失
    loss_supervised = criterion_supervised(labeled_outputs, labeled_labels)
    
    # 一致性损失
    # 对于无标签数据，进行Mixup增强
    unlabeled_images = images[unlabeled_mask]
    if unlabeled_images.size(0) > 0:
        mixed_images, lam = mix_up(unlabeled_images, unlabeled_images)
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
    w = compute_adjacency_matrix(features)
    loss_smoothness = smoothness_loss(features, w)
    
    # 总损失
    loss_all = loss_supervised + lambda_c * loss_consistency + lambda_s * loss_smoothness
    return loss_all, w