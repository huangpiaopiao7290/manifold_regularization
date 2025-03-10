import logging
import os
import pickle
import tarfile

import numpy as np
import torch


def un_tar(file_path, extract_to=".") -> None:
    """解压tar.gz到目标目录"""
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    with tarfile.open(file_path, 'r:gz') as tar:
        for member in tar.getmembers():
            tar.extract(member, extract_to)
            logging.info(f"extracted {member.name} to {extract_to}")

def parse_pickle(file) -> dict:
    """
    解析pickle文件
    """
    with open(file, 'rb') as fo:
        dct = pickle.load(fo, encoding='bytes')
    return dct

def load_c10_batch(file_path) -> tuple:
    """
    加载单个 batch 文件，返回图像和标签。
    """
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        images = batch[b'data']  # shape: (10000, 3072)
        labels = batch[b'labels'] if b'labels' in batch else batch[b'fine_labels']

        # 重塑图像为 (N, 3, 32, 32)
        images = images.reshape(-1, 3, 32, 32)
        images = images.astype(np.float32) / 255.0  # 归一化到 [0,1]

        # 转换为 Tensor
        images = torch.tensor(images)
        labels = torch.tensor(labels, dtype=torch.long)

    return images, labels

def load_c10_data(data_dir) -> tuple:
    """
    加载整个 ciFar-10 数据集，返回训练集和测试集的图像与标签。
    """
    train_images = []
    train_labels = []

    # 加载训练批次
    for i in range(1, 6):
        batch_file = os.path.join(data_dir, f'data_batch_{i}')
        images, labels = load_c10_batch(batch_file)
        train_images.append(images)
        train_labels.append(labels)

    # 拼接所有训练批次
    train_images = torch.cat(train_images, dim=0)  # shape: (50000, 3, 32, 32)
    train_labels = torch.cat(train_labels, dim=0)  # shape: (50000,)

    # 加载测试批次
    test_file = os.path.join(data_dir, 'test_batch')
    test_images, test_labels = load_c10_batch(test_file)  # shape: (10000, 3, 32, 32), (10000,)

    return train_images, train_labels, test_images, test_labels


# if __name__ == '__main__':
#     # ---------------------------
#     #     设置随机种子与设备
#     # ---------------------------
#     random.seed(42)
#     np.random.seed(42)
#     torch.manual_seed(42)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备类型
#     print(f"使用设备: {device}")

#     # ---------------------------
#     #     ciFar10 路径与相关配置
#     # ---------------------------
#     workspace = os.getcwd()                                                 # 项目根目录
#     ciFar10_tar_gz = "data\\cifar10\\cifar-10-python.tar.gz"                # 源数据相对位置
#     ciFar10_un_tar = "data\\cifar10"                                        # 解压相对位置
#     c10_dir_name = "cifar-10-batches-py"                                    # 解压目录名
#     path_tar = os.path.join(workspace, ciFar10_tar_gz)                      # 源数据绝对位置
#     path_un_tar = os.path.join(workspace, ciFar10_un_tar)                   # 解压绝对位置
#     c10_dir_absolute =  os.path.join(path_un_tar, c10_dir_name)             # 解压后绝对路径

#     # # 标签
#     # c10_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

#     # ---------------------------
#     #     ciFar10 数据处理
#     # ---------------------------
#     # 1. 解压数据集
#     # un_tar(path_tar, path_un_tar)
#     # 2. 加载batch
#     train_images_all, train_labels_all, test_images_all, test_labels_all = load_c10_data(c10_dir_absolute)

#     print("")
