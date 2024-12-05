## @author: pp
## @date: 2024/9/17
## @description: ciFar10数据集加载
import logging
import os

from PIL import Image
from torch.utils.data import Dataset


class CiFarDataset(Dataset):
    def __init__(self, root, label_name_dict, transform) -> None:
        """
        读取数据信息
        """
        super(CiFarDataset, self).__init__()
        self.root = root                        # eg: data/processed/ciFar000/train
        self.transform = transform
        self.label_names_dict = label_name_dict
        self.samples = []   # 存储 (image_path, label) 元组

        labeled_dir = self.root

        if os.path.split(self.root)[-1] == "train":
            # 加载训练集
            unlabeled_dir = os.path.join(root, "unlabeled")
            for image_name in os.listdir(unlabeled_dir):
                image_path_unlabeled = os.path.join(unlabeled_dir, image_name)
                self.samples.append((image_path_unlabeled, -1))       # -1 表示无标签

            # 验证集
            labeled_dir = os.path.join(self.root, "label")                # xxx/train/label

        for label in os.listdir(labeled_dir):
            label_dir = os.path.join(labeled_dir, label)                        # xxx/train/label/xxx
            for image_name in os.listdir(label_dir):
                image_path_labeled = os.path.join(label_dir, image_name)        # xxx/train/label/xxx/xxxx.png
                self.samples.append((image_path_labeled, self.label_names_dict[label]))                   # -1 表示无标签

    def __len__(self):
        return len(self.samples )

    def __getitem__(self, index: int):

        image_path, image_label = self.samples[index][0], self.samples[index][1]

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

