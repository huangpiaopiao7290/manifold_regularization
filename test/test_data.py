import torch

# 检查 CUDA 是否可用
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# 打印更多调试信息
print(f"Device type: {type(device)}")
print(f"is_cuda: {device.type == 'cuda'}")