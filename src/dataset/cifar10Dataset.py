## @author: pp
## @date: 2024/9/17
## @description: 数据预处理
import logging
import os
from PIL import Image
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

        match os.path.split(root)[-1]:
            case "train":
                labeled_dir = os.path.join(root, "label")
            case "test":
                labeled_dir = root
            case _:
                labeled_dir = root

        for label in os.listdir(labeled_dir):
            label_dir = os.path.join(labeled_dir, label)
            label = os.path.split(label_dir)[-1]
            for image_name in os.listdir(label_dir):
                self.image_paths.append(os.path.join(label_dir, image_name))
                self.labels.append(labels_dict[label])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path: str= self.image_paths[index]
        image_label: int = self.labels[index]
        # image_data = Image.open(image_path).convert('RGB')

        try:
            image_data = Image.open(image_path).convert('RGB')
        except Exception as e:
            logging.error(f"Failed to open image at path: {image_path}, error: {e}")
            raise

        # 图像增强
        if self.transform:
            image_data = self.transform(image_data)

        return image_data, image_label

def get_data_loaders(root, batch_size=64, num_workers=4):
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

    dataset_train = Cifar10Dataset(root=dataset_train_path, transform=train_transform)
    dataset_test = Cifar10Dataset(root=dataset_test_path, transform=test_transform)

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)


    return dataloader_train, dataloader_test
