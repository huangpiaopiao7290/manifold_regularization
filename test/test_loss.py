import torch

from src.dataset.cifar10Dataset import get_data10_loaders


# 设定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模拟数据
batch_size = 4
image_channels = 3
image_size = 32
num_classes = 10

# 创建一批模拟图像数据 (batch_size, channels, height, width)
images = torch.randn(batch_size, image_channels, image_size, image_size, device=device)
ciFar10_labelNames = {}
ciFar10_label_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
for idx, name in enumerate(ciFar10_label_names):
    ciFar10_labelNames[name] = idx

img_train, img_test = get_data10_loaders("data/test/cifar10", ciFar10_labelNames, 64)


