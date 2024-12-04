import logging
import os
from concurrent.futures import ThreadPoolExecutor
import torchvision.transforms as transforms

import numpy as np

from src.utils.utility import Utility

# TODO 这里本来是要继承ciFar100Processor的，但是ciFar100Processor有问题，索性删除了，
#  而且这个预处理数据也只用的上一次，没必要封装， 后面有时间再重写


ciFar100_labels = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle", "bowl", "boy",
    "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur", "dolphin", "elephant", "flatfish",
    "forest", "fox", "girl", "hamster", "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse", "mushroom", "oak_tree", "orange",
    "orchid", "otter", "palm_tree", "pear", "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine",
    "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea", "seal", "shark", "shrew", "skunk",
    "skyscraper", "snail", "snake", "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper", "table",
    "tank", "telephone", "television", "tiger", "tractor", "train", "trout", "tulip", "turtle", "wardrobe", "whale",
    "willow_tree", "wolf", "woman", "worm"
]

train_list = Utility.find_filenames_with_keyword("data/raw/cifar/cifar-100", "train")
test_list = Utility.find_filenames_with_keyword("data/raw/cifar/cifar-100", "test")
dataset = [(file, 'train') for file in train_list] + [(file, 'test') for file in test_list]


# 用于保存有标签数据的函数
def save_labeled_images(batch_dict, target_dir, unlabeled_dir, labeled_ratio=0.1):
    # 按类别存储图片
    images_by_label = {label: [] for label in range(100)}

    for im_idx, im_data in enumerate(batch_dict[b'data']):
        im_label: int = batch_dict[b'fine_labels'][im_idx]
        im_name: str = batch_dict[b'filenames'][im_idx]
        images_by_label[im_label].append((im_data, im_name))

    for label, images in images_by_label.items():
        num_labeled = int(len(images) * labeled_ratio)
        labeled_images = images[:num_labeled]
        unlabeled_images = images[num_labeled:]

        for im_data, im_name in labeled_images:
            im_path_label = os.path.join(target_dir, ciFar100_labels[label], im_name.decode('utf-8'))
            yield (im_data, im_path_label)

        # 处理"无标签数据"
        for im_data, im_name in unlabeled_images:
            im_path_label = os.path.join(unlabeled_dir, im_name.decode('utf-8'))
            yield (im_data, im_path_label)


def save_image(img_data, img_extract_path):
    """
    保存图像数据
    :@param img_data: 图像数据
    :@param img_extract_path: 图像路径
    """
    # 确保目标目录存在
    os.makedirs(os.path.dirname(img_extract_path), exist_ok=True)

    # 将一维数组重塑为 (3, 32, 32) 的形状
    img_data = np.reshape(img_data, (3, 32, 32))
    img_data = np.transpose(img_data, (1,2,0))
    img = transforms.ToPILImage()(img_data)

    # 图像保存路径
    if not os.path.exists(img_extract_path):
        logging.log(logging.INFO, f"{img_extract_path}")
        img.save(img_extract_path)
    else:
        return

# dict_keys([b'filenames', b'batch_label', b'fine_labels', b'coarse_labels', b'data'])
for file, set_type in dataset:
    # unpickle
    dct: dict = Utility.parse_pickle(file)
    # 根据set_type决定是训练集还是测试集
    target_file = target_file_unlabeled = ""
    if set_type == "train":
        target_file = os.path.join("data/processed/cifar-100", "train", "label")
        target_file_unlabeled = os.path.join("data/processed/cifar-100", "train", "unlabeled")
    elif set_type == "test":
        target_file = os.path.join("data/processed/cifar-100", "test")
        target_file_unlabeled = os.path.join("data/processed/cifar-100", "test", "unlabeled")   # NULL

    # 拿到对应name和data根据label->存放到train对应文件夹下
    with ThreadPoolExecutor(max_workers=10) as executor:  # 线程数<=10
        futures = []
        for image_data, image_path_label in save_labeled_images(batch_dict=dct,
                                                                target_dir=target_file, 
                                                                unlabeled_dir=target_file_unlabeled,
                                                                labeled_ratio=0.1):

            # 提交任务给线程池
            future = executor.submit(save_image, image_data, image_path_label)
            futures.append(future)

        for future in futures:
            future.result()

