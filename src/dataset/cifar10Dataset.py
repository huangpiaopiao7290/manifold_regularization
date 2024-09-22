## @author: piaopiao
## @date: 2024/9/17
## @description: 数据预处理

import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# cifar-10路径
cifar10 = os.path.join(os.getcwd(), "data", "processed", "cifar-10")

label_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
labels_dict = {}
for idx, name in enumerate(label_names):
    labels_dict[name] = idx

class Cifar10Dataset(Dataset):
    def __init__(self, root, transform=None) -> None:
        """
        读取数据信息
        """
        super(Cifar10Dataset, self).__init__()
        self.root = root
        self.transform = transform
        self.image_paths = []
        self.labels = []

        if os.path.split(root)[-1] == "train":
            # 加载训练集
            unlabeled_dir = os.path.join(root, "unlabel")
            for image_name in os.listdir(unlabeled_dir):
                self.image_paths.append(os.path.join(unlabeled_dir, image_name))
                self.labels.append(-1)                                              # -1 表示无标签

        # 加载验证集
        labeled_dir = os.path.join(root, "label")
        for label in os.listdir(labeled_dir):
            label_dir = os.path.join(labeled_dir, label)
            label = os.path.split(label_dir)[-1]
            for image_name in os.listdir(label_dir):
                self.image_paths.append(os.path.join(label_dir, image_name))
                self.labels.append(labels_dict[label])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_label = self.labels[idx]
        image_data = Image.open(image_path).convert('RGB')

        # 图像增强
        if self.transform:
            image = self.transform(image)

        return image_data, image_label

def get_data_loaders(root, batch_size=64, num_workers=4):
    """
    :@param root: 数据集根目录
    """
    transform = transforms.Compose([
        transforms.RandomResizedCrop((28, 28)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomGrayscale(0.1),
        transforms.RandomRotation(90),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.ToTensor()
    ])

    dataset_train_path = os.path.join(root, "train")
    dataset_test_path = os.path.join(root, "test")

    dataset_train = Cifar10Dataset(root=dataset_train_path, transform=transform)
    dataset_test = Cifar10Dataset(root=dataset_test_path, transform=transforms.ToTensor())

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)


    return dataloader_train, dataloader_test

if '__name__' == '__main__':
    pass
