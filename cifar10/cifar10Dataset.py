from torch.utils.data import Dataset

class LabeledCIFARDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        """
        有标签数据集
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = self.images[idx]
        y = self.labels[idx]
        if self.transform:
            x = self.transform(x)
        return x, y


class UnlabeledCIFARDataset(Dataset):
    def __init__(self, images, transform=None):
        """
        无标签数据集
        """
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = self.images[idx]
        if self.transform:
            x = self.transform(x)
        return x


class CIFARValDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        """
        验证/测试数据集
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = self.images[idx]
        y = self.labels[idx]
        if self.transform:
            x = self.transform(x)
        return x, y
