import os
import tarfile
import pickle
import logging
from typing import List

import numpy as np
import random
import shutil
from concurrent.futures import ThreadPoolExecutor
from torchvision.transforms import ToPILImage

class DatasetProcessor:
    def __init__(self, dataset_name, raw_data_path, processed_data_path, labels, batch_keyword="data_batch", test_keyword="test_batch"):
        self.dataset_name = dataset_name
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        self.labels = labels
        self.batch_keyword = batch_keyword
        self.test_keyword = test_keyword
        self.transform = ToPILImage()
        self.logger = logging.getLogger(self.dataset_name)
        logging.basicConfig(level=logging.INFO)

    def untar(self):
        # 解压tar.gz文件
        if not os.path.exists(self.processed_data_path):
            os.makedirs(self.processed_data_path)
            with tarfile.open(self.raw_data_path, 'r:gz') as tar:
                tar.extractall(path=self.processed_data_path)
                self.logger.info(f"Extracted {self.raw_data_path} to {self.processed_data_path}")

    def create_directories(self):
        # 创建必要的目录结构
        for label in self.labels:
            os.makedirs(os.path.join(self.processed_data_path, "train", "label", label), exist_ok=True)
            os.makedirs(os.path.join(self.processed_data_path, "test", label), exist_ok=True)
            self.logger.info(f"Created directories for {label}")

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            data_dict = pickle.load(fo, encoding='bytes')
        return data_dict

    def save_image(self, img_data, img_extract_path):
        # 保存图像
        img_data = np.reshape(img_data, (3, 32, 32))
        img_data = np.transpose(img_data, (1, 2, 0))
        img = self.transform(img_data)
        img.save(img_extract_path)
        self.logger.info(f"Saved image to {img_extract_path}")

    def decode_and_generate(self):
        # 解码数据并生成图像
        train_list = self.find_filenames_with_keyword(self.processed_data_path, self.batch_keyword)
        test_list = self.find_filenames_with_keyword(self.processed_data_path, self.test_keyword)
        dataset = [(file, 'train') for file in train_list] + [(file, 'test') for file in test_list]

        for file, set_type in dataset:
            batch_dict = self.unpickle(file)
            target_dir = os.path.join(self.processed_data_path, set_type, "label" if set_type == "train" else "")
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(self.save_image, im_data, os.path.join(target_dir, self.labels[im_label], im_name.decode('utf-8')))
                           for im_idx, im_data in enumerate(batch_dict[b'data'])
                           for im_label, im_name in zip(batch_dict[b'labels'], batch_dict[b'filenames'])]
                for future in futures:
                    future.result()

    def find_filenames_with_keyword(self, root_dir, keyword):
        # 查找包含关键字的文件
        match_files = []
        for dirpath, _, files in os.walk(root_dir):
            match_files.extend([os.path.join(dirpath, file) for file in files if keyword in file])
        return match_files

    def split_ssl_dataset(self, num_label=5000, num_unlabel=45000):
        # 划分训练集、验证集
        assert num_label + num_unlabel == 50000, "总数与数据集数量不一致"
        proportion = num_unlabel / 50000
        unlabel_dir = os.path.join(self.processed_data_path, "train", "unlabel")
        os.makedirs(unlabel_dir, exist_ok=True)
        with ThreadPoolExecutor(max_workers=len(self.labels)) as executor:
            futures = [executor.submit(self.move_images, os.path.join(self.processed_data_path, "train", "label", label), unlabel_dir, proportion)
                       for label in self.labels]
            for future in futures:
                future.result()

    def move_images(self, source, dest, prop):
        # 获取源目录中的所有文件名
        files: List[str] = [f for f in os.listdir(source) if os.path.isfile(os.path.join(source, f))]
        num2move = int(len(files) * prop)
        selected_files = random.sample(files, num2move)
        for file in selected_files:
            src: str = str(os.path.join(source, file))
            dst: str= str(os.path.join(dest, file))
            shutil.move(src, dst)
            self.logger.info(f"Moved {file} from {source} to {dest}")

# 使用示例
if __name__ == '__main__':
    cifar10_processor = DatasetProcessor(
        "CIFAR-10",
        "data/raw/cifar/cifar-10-python.tar.gz",
        "data/processed/cifar-10",
        ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    )
    cifar10_processor.untar()
    cifar10_processor.create_directories()
    cifar10_processor.decode_and_generate()
    cifar10_processor.split_ssl_dataset()