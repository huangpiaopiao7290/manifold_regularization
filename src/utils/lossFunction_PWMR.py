## @auther: pp
## @date: 2024/10/30
## @description:　

import torch
import torch.nn as nn


class LossFunctionsPWMR:
    def __init__(self, device='cpu'):
        assert device, "Device must be specified or set to 'cpu' by default."
        self.device = device
        self.criterion_supervised = nn.CrossEntropyLoss(reduction='mean')

    def _to_device(self, tensor):
        return tensor.to(self.device)

    def compute_adjacency_matrix(self, features, k=8, sigma=1.0):
        r"""
        Calculate adjacency matrix using K-NN and Gaussian kernel smoothing.
        """
        if len(features.shape) > 2:
            features = features.view(features.size(0), -1)

        features = features.to(self.device)
        pairwise_distances = torch.cdist(features, features, p=2.0)
        _, indices = torch.topk(pairwise_distances, k, largest=False, dim=1)
        _n = features.shape[0]
        _w = torch.zeros((_n, _n), device=self.device)

        # 遍历结点
        # for i in range(_n):
        #     for j in indices[i]:
        #         if i != j:
        #             dist = pairwise_distances[i, j]
        #             _w[i, j] = torch.exp(-dist ** 2 / (2 * sigma ** 2))

        # 为前k个邻接结点创建mask
        mask = torch.zeros_like(_w, dtype=torch.bool, device=self.device)
        for i in range(_n):
            mask[i, indices[i, 1:]] = True      # 不包括自己

        # 添加高斯核得到边权重
        _w[mask] = torch.exp(-pairwise_distances[mask] ** 2 / (2 * sigma ** 2))

        row_sum = _w.sum(dim=1, keepdim=True).view(-1, 1)
        _w = _w / (row_sum + 1e-8)

        return _w

    def build_laplacian_matrix(self, adj_matrix):
        r"""
        Build the graph Laplacian matrix from the adjacency matrix.
        """
        degree_matrix = torch.diag(adj_matrix.sum(dim=1))
        laplacian_matrix = degree_matrix - adj_matrix
        return laplacian_matrix

    def calculate_local_density(self, adj_matrix, k=8):
        """
        Calculate the local density for each sample based on the adjacency matrix.
        """
        degrees = adj_matrix.sum(dim=1)
        local_densities = degrees / k
        #
        return local_densities

    def smoothness_loss(self, features, adj_matrix, local_densities):
        r"""
        Calculate the smoothness loss using the graph Laplacian matrix.
        """
        # 确保 features 是 2D 张量
        if len(features.shape) > 2:
            features = features.view(features.size(0), -1)

        # 计算逐点平滑度损失
        # features: (N, D), laplacian_matrix: (N, N), local_densities: (N)
        # 首先计算图拉普拉斯矩阵 特征矩阵的乘积：(N, D) x (N, N) = (N, D)
        laplacian_matrix = self.build_laplacian_matrix(adj_matrix)
        diff = torch.mm(laplacian_matrix, features)

        # 然后计算差值的平方和
        diff_squared = diff ** 2

        # 将局部密度与差值的平方和相乘
        # local_densities: (N), diff_squared: (N, D)
        weighted_diff_squared = local_densities.unsqueeze(1) * diff_squared

        # 计算总和并除以样本数
        smoothness_loss = torch.sum(weighted_diff_squared) / features.size(0)

        return smoothness_loss

    def mix_up(self, x1, x2, alpha=0.75):
        r"""
        Perform MixUp data augmentation.
        """
        lam = torch.distributions.Beta(alpha, alpha).sample().to(x1.device)
        mixed_x = lam * x1 + (1 - lam) * x2
        return mixed_x, lam

    def mix_up2(self, x1, x2, alpha=0.75):
        """
        接受两个不同epoch的数据, 对无标签样本使用随机噪声
        """
        if x2 is None:
            noise = torch.rand_like(x1) * 0.1
            x2 = x1 + noise
        
        lam = torch.distributions.Beta(alpha, alpha).sample().to(x1.device)
        mixed_x = lam * x1 + (1 - lam) * x2
        return mixed_x, lam

    def consistency_loss(self, predictions, perturbed_predictions, mask):
        r"""
        Calculate the consistency loss.
        """
        mse = nn.MSELoss(reduction='none')
        loss = mse(predictions, perturbed_predictions)
        loss = (loss * mask.unsqueeze(1)).sum() / (mask.sum() + 1e-8)
        return loss

    def total_loss(self, model, outputs, images, labels, unlabeled_mask, lambda_c, lambda_s):
        r"""
        Calculate the total loss with the PW_MR algorithm.
        """
        # 获取模型预测
        # outputs = model(images)

        # 分割有标签和无标签的数据
        labeled_mask = (labels != -1)
        labeled_outputs = outputs[labeled_mask]
        labeled_labels = labels[labeled_mask]

        # 带标签样本的交叉熵损失
        loss_supervised = self.criterion_supervised(labeled_outputs, labeled_labels)

        # 一致性损失
        unlabeled_images = images[unlabeled_mask]
        if unlabeled_images.size(0) > 0:
            # 使用不同的无标签样本进行MixUp，如果只有一个批次则使用随机噪声
            shuffled_indices = torch.randperm(unlabeled_images.size(0))
            mixed_images, lam = self.mix_up(unlabeled_images, unlabeled_images[shuffled_indices], alpha=0.75)
            mixed_outputs = model(mixed_images)
            with torch.no_grad():
                teacher_outputs = model(unlabeled_images)
                perturbed_teacher_outputs = model(mixed_images)
            interpolated_teacher_outputs = lam.unsqueeze(1) * teacher_outputs + (1 - lam.unsqueeze(1)) * perturbed_teacher_outputs
            loss_consistency = self.consistency_loss(mixed_outputs, interpolated_teacher_outputs,
                                                     torch.ones_like(mixed_outputs[:, 0], dtype=torch.bool))
        else:
            loss_consistency = torch.tensor(0.0, device=self.device)

        # 计算局部密度
        with torch.no_grad():
            # 提取特征
            features = model.extract_features(images)

        adj_matrix = self.compute_adjacency_matrix(features)
        local_densities = self.calculate_local_density(adj_matrix)

        # 平滑性损失
        loss_smoothness = self.smoothness_loss(features=features, adj_matrix=adj_matrix, local_densities=local_densities)

        # 总损失
        loss_all = loss_supervised + lambda_c * loss_consistency + lambda_s * loss_smoothness
        return loss_all, adj_matrix, local_densities