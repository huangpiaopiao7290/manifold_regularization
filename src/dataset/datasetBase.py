## @auther: pp
## @date: 2024/10/5
## @description:

import os
import pickle
import numpy as np
import logging
import shutil
import random
from PIL import Image
from torchvision.transforms import ToPILImage
import asyncio

logging.basicConfig(level=logging.INFO)     # 定义日志级别

class DatasetBase:
    def __init__(self, dataset_name, labels=None, work_space=os.getcwd()):
        self.dataset_name = dataset_name
        self.labels = labels
        self.work_space = work_space
        self.processed_dir = os.path.join(work_space, "data", "processed", dataset_name)
        self.create_dataset_dir()

    def create_dataset_dir(self):
        """创建数据集目录结构"""
        if not os.path.exists(os.path.join(self.processed_dir, "train", "unlabel")):
            os.makedirs(os.path.join(self.processed_dir, "train", "unlabel"))
            logging.info("train/unlabel created...")

        if not os.path.exists(os.path.join(self.processed_dir, "test")):
            os.makedirs(os.path.join(self.processed_dir, "test"))
            logging.info("test created...")

        if self.labels:
            for label in self.labels:
                os.makedirs(os.path.join(self.processed_dir, "train", "label", label), exist_ok=True)
                os.makedirs(os.path.join(self.processed_dir, "test", label), exist_ok=True)
                logging.info(f"{self.dataset_name}/train|test/{label} created...")

    def unpickle(self, file):
        """解析pickle文件"""
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def decode_and_generate(self, batch_file, dataset_type):
        """根据数据集类型调用相应的方法"""
        raise NotImplementedError("This method should be implemented by subclasses.")

    async def split_ssl_dataset(self, num_label, num_unlabel):
        """划分训练集、验证集（异步版本）"""
        assert num_label + num_unlabel == 50000, "总数与数据集数量不一致"
        proportion = num_unlabel / (num_unlabel + num_label)
        total_num2move = num_unlabel
        unlabel_dir = os.path.join(self.processed_dir, "train", "unlabel")
        if not os.path.exists(unlabel_dir):
            raise FileNotFoundError(f"{unlabel_dir} not found")

        tasks = []
        for label in self.labels:
            label_dir_path = os.path.join(self.processed_dir, "train", "label", label)
            images = os.listdir(label_dir_path)
            images_num = len(images)
            num2move = int(images_num * proportion)
            if total_num2move - num2move < 0:
                num2move = total_num2move
            tasks.append((label_dir_path, unlabel_dir, num2move / images_num))
            total_num2move -= num2move
            if total_num2move <= 0:
                break

        await asyncio.gather(*[Utility.move_files_async(*task) for task in tasks])