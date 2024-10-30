## @auther: pp
## @date: 2024/10/30
## @description:　

import torch
import torch.nn as nn


class PWMRLossFunctions:
    def __init__(self, device='cpu'):
        assert device, "Device must be specified or set to 'cpu' by default."
        self.device = device
        self.criterion_supervised = nn.CrossEntropyLoss(reduction='mean')

    def _to_device(self, tensor):
        return tensor.to(self.device)

    def compute_adjacency_matrix(self, features, k=8, sigma=1.0):
        """
        calculate adjacency matrix using K-NN and Gaussian kernel smoothing.
        """
        # 将多维特征展平成二维张量
        if len(features.shape) > 2:
            features = features.view(features.size(0), -1)

        # 确保所有张量都在同一个设备上
        features = features.to(self.device)

        # 计算每对样本之间的欧式距离
        pairwise_distances = torch.cdist(features, features, p=2.0)

        # 找到每个样本的 k 个最近邻居
        _, indices = torch.topk(pairwise_distances, k, largest=False, dim=1)

        # 初始化邻接矩阵
        _n = features.shape[0]
        _w = torch.zeros((_n, _n), device=self.device)

        # 对于每个样本 i, 找到它的 k 个最近邻居 j，计算高斯核函数值并赋值给邻接矩阵
        for i in range(_n):
            for j in indices[i]:
                if i != j:  # 跳过自身
                    dist = pairwise_distances[i, j]
                    _w[i, j] = torch.exp(-dist ** 2 / (2 * sigma ** 2))

        # 归一化
        try:
            row_sum = _w.sum(dim=1, keepdim=True).view(-1, 1)
            _w = _w / (row_sum + 1e-8)
        except ValueError as e:
            raise ValueError("calculation error: {}".format(e))

        return _w

    def calculate_local_density(self, adj_matrix, k=8):
        """
        Calculate the local density for each sample based on the adjacency matrix.
        """
        # 计算每个节点的度
        degrees = adj_matrix.sum(dim=1)
        # 使用K近邻的数量来归一化度，得到局部密度
        local_densities = degrees / k
        return local_densities

    def mix_up(self, x1, x2, alpha=0.75):
        """
        By performing a simple linear transformation of the input data.
        """
        # beta分布中随机采样一个值作为混合系数
        lam = torch.distributions.Beta(alpha, alpha).sample().to(x1.device)
        # 按照混合系数进行线性插值
        mixed_x = lam * x1 + (1 - lam) * x2

        return mixed_x, lam

    def consistency_loss(self, predictions, perturbed_predictions, mask):
        """
        calculate the consistency loss.
        """
        # 计算每个样本的均方误差损失 返回每个样本的损失值
        mse = nn.MSELoss(reduction='none')
        # 计算原始数据预测与扰动后输入预测之间的MSE损失  返回一个与predictions形状相同的tensor  其中每个元素对应于样本的MSE损失
        loss = mse(predictions, perturbed_predictions)
        # 计算加权平均损失
        loss = (loss * mask.unsqueeze(1)).sum() / (mask.sum() + 1e-8)

        return loss

    def smoothness_loss(self, features, w, local_densities):
        """
        smoothness loss with local densities.
        """
        # 将多维特征展平为二位张量
        if len(features.shape) > 2:
            features = features.view(features.size(0), -1)

        # 计算 diff (样本特征两两之间的差异矩阵) 的平方和 沿着最后一个维度求和  (每对样本之间的欧氏距离平方)
        squared_diff = ((features.unsqueeze(0) - features.unsqueeze(1)) ** 2).sum(dim=-1)
        # 使用局部密度作为权重
        loss = (local_densities.unsqueeze(1) * w * squared_diff).sum() / 2
        return loss

    def total_loss(self, model, images, labels, unlabeled_mask, lambda_c, lambda_s):
        """
        total loss with PWMR algorithm.
        """
        # 获取模型预测
        outputs = model(images)

        # 分割有标签和无标签的数据
        labeled_outputs = outputs[~unlabeled_mask]
        labeled_labels = labels[~unlabeled_mask]

        # 带标签样本的交叉熵损失
        loss_supervised = self.criterion_supervised(labeled_outputs, labeled_labels)

        # 一致性损失
        # 对于无标签数据，进行Mixup增强
        unlabeled_images = images[unlabeled_mask]
        if unlabeled_images.size(0) > 0:
            mixed_images, lam = self.mix_up(unlabeled_images, unlabeled_images)
            mixed_outputs = model(mixed_images)

            with torch.no_grad():
                teacher_outputs = model(unlabeled_images)
            interpolated_teacher_outputs = lam * teacher_outputs + (1 - lam) * teacher_outputs
            # 使用无标签数据的掩码
            loss_consistency = self.consistency_loss(mixed_outputs, interpolated_teacher_outputs,
                                                     torch.ones_like(mixed_outputs[:, 0], dtype=torch.bool))
        else:
            loss_consistency = 0.0

        # 平滑性损失
        feature_extractor = nn.Sequential(*list(model.children())[:-1])  # 假设倒数第二层是特征提取器
        with torch.no_grad():
            features = feature_extractor(images)
        w = self.compute_adjacency_matrix(features)
        local_densities = self.calculate_local_density(w)
        loss_smoothness = self.smoothness_loss(features, w, local_densities)

        # 总损失
        loss_all = loss_supervised + lambda_c * loss_consistency + lambda_s * loss_smoothness
        return loss_all, w, local_densities