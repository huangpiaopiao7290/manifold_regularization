## @auther: pp
## @date: 2024/12/06
## @description: svhn数据集加载

import os
import scipy.io as sio
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class SVHNDataset(Dataset):
    def __init__(self, root_dir, transform_=None, mode='train', labeled=True, num_labeled=500):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform_ (callable, optional): Optional transform to be applied on a sample.
            mode (string): 'train' or 'test'.
            labeled (bool): If True, returns only labeled data; otherwise, returns both labeled and unlabeled data.
            num_labeled (int): Number of labeled samples to use in semi-supervised learning.
        """
        self.root_dir = root_dir
        self.transform = transform_
        self.mode = mode
        self.labeled = labeled

        # Load digitStruct.mat
        mat_file = os.path.join(root_dir, f'{mode}.mat')
        self.digit_struct = sio.loadmat(mat_file)['digitStruct'][0]

        # Prepare indices for labeled and unlabeled data
        if labeled:
            self.indices = list(range(min(num_labeled, len(self.digit_struct))))
        else:
            self.indices = list(range(len(self.digit_struct)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        index = self.indices[idx]
        item = self.digit_struct[index]

        img_name = ''.join([chr(c[0]) for c in item['name'][0]])
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        bboxes = item['bbox'][0]
        labels = []
        cropped_images = []

        for bbox in bboxes:
            label = int(bbox['label'][0])
            top = int(max(bbox['top'][0], 1))
            left = int(max(bbox['left'][0], 1))
            height = int(bbox['height'][0])
            width = int(bbox['width'][0])

            bottom = min(top + height, image.height)
            right = min(left + width, image.width)

            cropped_image = image.crop((left, top, right, bottom))
            cropped_images.append(cropped_image)
            labels.append(label)

        # Apply transformations
        if self.transform:
            cropped_images = [self.transform(img) for img in cropped_images]

        if self.labeled:
            # Return only the first bounding box for simplicity
            return cropped_images[0], labels[0]
        else:
            # For unlabeled data, we can return the entire image or multiple crops
            return image if not cropped_images else cropped_images[0], -1  # Use -1 as a placeholder for no label


# Example usage
if __name__ == "__main__":
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Initialize dataset for training
    train_dataset = SVHNDataset(root_dir='./data/SVHN/train', transform_=transform, mode='train', labeled=True,
                                num_labeled=500)

    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Iterate through the DataLoader
    for images, labels in train_loader:
        # Training code here
        pass
