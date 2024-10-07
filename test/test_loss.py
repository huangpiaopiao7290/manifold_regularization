from src.utils.loss import compute_adjacency_matrix, mix_up, consistency_loss
import torch

# 示例数据
features = torch.tensor([
    [1.0, 2.0],
    [2.0, 3.0],
    [3.0, 4.0],
    [4.0, 5.0],
    [5.0, 6.0]
], dtype=torch.float32)

w = compute_adjacency_matrix(features, k=3, sigma=1.0)
print(w)


# 假设的样本数据
x1 = torch.tensor([1.0, 2.0, 3.0])
x2 = torch.tensor([4.0, 5.0, 6.0])

# 使用 mixup 函数
mixed_x, lam = mix_up(x1, x2)

print("mixing coefficient lam:", lam)
print("mixing sample mixed_x:", mixed_x)


# 假设的预测数据
prediction = torch.tensor([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]], requires_grad=True)
perturbed_prediction = torch.tensor([[0.15, 0.85], [0.25, 0.75], [0.35, 0.65]])
# 假设的掩码
mask = torch.tensor([True, False, False])

cons_loss = consistency_loss(prediction, perturbed_prediction, mask)
print("consistency_loss: ", cons_loss)


