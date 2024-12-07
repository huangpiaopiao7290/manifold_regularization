## @auther: pp
## @date: 2024/12/06
## @description: svhn数据集加载

import os
import h5py
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class SVHNDataset(Dataset):
    def __init__(self, root_dir, transform_=None, labeled=True, num_labeled=500):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform_ (callable, optional): Optional transform to be applied on a sample.
            labeled (bool): If True, returns only labeled data; otherwise, returns both labeled and unlabeled data.
            num_labeled (int): Number of labeled samples to use in semi-supervised learning.
        """
        self.root_dir = root_dir
        self.transform = transform_
        self.labeled = labeled

        # Load digitStruct.mat
        mat_file = os.path.join(root_dir, 'digitStruct.mat')
        self.digit_struct = load_h5(mat_file)

        # Prepare indices for labeled and unlabeled data
        if labeled:
            self.indices = list(range(min(num_labeled, len(self.digit_struct))))
        else:
            self.indices = list(range(len(self.digit_struct)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        index = self.indices[idx]
        item = self.digit_struct['digitStruct'][index]

        img_name = ''.join([chr(c[0]) for c in item['name'][0]])
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        bboxes = item['bbox'][0]
        labels_num = []
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
            labels_num.append(label)

        # Apply transformations
        if self.transform:
            cropped_images = [self.transform(img) for img in cropped_images]

        if self.labeled:
            # Return only the first bounding box for simplicity
            return cropped_images[0], labels_num[0]
        else:
            # For unlabeled data, we can return the entire image or multiple crops
            return image if not cropped_images else cropped_images[0], -1  # Use -1 as a placeholder for no label

def load_h5(file_path):
    """Load .mat file using h5py and return a dictionary with the data."""
    def read_dataset(dataset):
        """Recursively read dataset elements."""
        if isinstance(dataset, h5py.Dataset):
            # If the dataset is of type string, decode it to Python string.
            if dataset.dtype == 'uint8' or dataset.dtype.kind == 'S':
                return ''.join(chr(i) for i in dataset[:])
            elif len(dataset.shape) == 0:
                return float(dataset[()])
            else:
                return [read_dataset(item) for item in dataset]
        elif isinstance(dataset, h5py.Group):
            # Recursively read group members.
            return {key: read_dataset(dataset[k]) for k in dataset.keys()}
        else:
            return dataset

    with h5py.File(file_path, 'r') as f:
        digit_struct = {}
        # Assuming the top-level group is named 'digitStruct'
        for key in f['digitStruct'].keys():
            digit_struct[key] = read_dataset(f['digitStruct'][key])

    return digit_struct


def get_loader(root_dir, batch_size=64, num_labeled=500):
    """
        Args:
            root_dir(string): Directory with svhn folder
            batch_size(int): the size of batch
            num_labeled(int): Number of labeled samples in the training set
    """
    # Define transformations for training and testing
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),  # 数据增强：随机水平翻转
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Initialize dataset for training and testing
    train_dataset_labeled = SVHNDataset(root_dir=os.path.join(root_dir, 'train'), transform_=train_transform,
                                        labeled=True, num_labeled=num_labeled)
    train_dataset_unlabeled = SVHNDataset(root_dir=os.path.join(root_dir, 'train'), transform_=train_transform,
                                          labeled=False)

    test_dataset = SVHNDataset(root_dir=os.path.join(root_dir, 'test'), transform_=test_transform, labeled=True)

    # Create DataLoader
    train_loader_labeled = DataLoader(train_dataset_labeled, batch_size=batch_size, shuffle=True)
    train_loader_unlabeled = DataLoader(train_dataset_unlabeled, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader_labeled, train_loader_unlabeled, test_loader