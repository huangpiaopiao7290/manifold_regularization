## @author: pp
## @date: 2024/9/17
## @description: ciFar100数据集加载

import os
from torch.utils.data import DataLoader
from torchvision import transforms

from .cifarDataset import CiFarDataset

class CiFar100Dataset(CiFarDataset):
    def __init__(self, root, label_names_dict, transform=None):
        super(CiFar100Dataset, self).__init__(root, label_names_dict, transform)

def get_data100_loaders(root, label_names_dict,  batch_size=64, num_workers=5):
    """
    :@param root: 数据集根目录
    """
    train_transform = transforms.Compose([
        # transforms.RandomResizedCrop((28, 28)),  # CIFAR-10 的标准尺寸是 32x32
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomGrayscale(0.1),
        transforms.RandomRotation(90),
        transforms.ToTensor(),  # 将 PIL 图像转换为 Tensor
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])  # 归一化
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])

    dataset_train_path = os.path.join(root, "train")
    dataset_test_path = os.path.join(root, "test")

    dataset_train = CiFar100Dataset(root=dataset_train_path,label_names_dict=label_names_dict, transform=train_transform)
    dataset_test = CiFar100Dataset(root=dataset_test_path, label_names_dict=label_names_dict, transform=test_transform)

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return dataloader_train, dataloader_test

