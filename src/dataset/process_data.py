## @auther: pp
## @date: 2024/9/14 2:40
## @description: 解压数据集文件，将数据分类

import tarfile
import pickle
import os
import logging
import numpy as np
import random
import shutil   
from concurrent.futures import ThreadPoolExecutor
import torchvision.transforms as transforms

# 定义日志级别
logging.basicConfig(level=logging.INFO)

## 获取工作目录
work_space = os.getcwd()

# 读取tar.gz文件
cifar10_tar_gz = os.path.join(work_space, "data", "raw", "cifar", "cifar-10-python.tar.gz")
extract_path = os.path.join(work_space, "data", "raw", "cifar", "cifar-10")

# Step2: 解压tar.gz文件
def untar(file_path, extract_to="."):
    """解压tar.gz到目标目录"""
    if os.path.exists(extract_path):
        return
    else:
        os.makedirs(extract_path)

    with tarfile.open(file_path, 'r:gz') as tar:
        for member in tar.getmembers():
            # 对每个batch提取
            tar.extract(member, extract_to)
            logging.log(logging.INFO, f"extracted {member.name} to {extract_to}")


# step1: 数据集解码

# 标签
labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# 在目标位置创建train test
processed_dir = os.path.join(work_space, "data", "processed")

def create_cifar10_dir(file_path):
    """
    划分数据集: 训练集(无标签), 验证集(带标签), 测试集
    """

    # 训练集(无标签)
    if not os.path.exists(os.path.join(processed_dir, file_path, " train", "unlabel")):
        os.makedirs(os.path.join(processed_dir, file_path, "train", "unlabel"))
        logging.log(logging.INFO, "train/unlabel created...")

    # 测试集
    if not os.path.exists(os.path.join(processed_dir, file_path, "test")):
        os.makedirs(os.path.join(processed_dir, file_path, "test"))
        logging.log(logging.INFO, "test created...")

    # 遍历标签列表, 创建每个标签子目录
    for label in labels:
        # 训练集带标签(验证集)
        os.makedirs(os.path.join(processed_dir, file_path, "train", "label", label), exist_ok=True)
        os.makedirs(os.path.join(processed_dir, file_path, "test", label), exist_ok=True)
        logging.log(logging.INFO, f"cifar-10/train|test/{label} created...")

# 官网提供pickle文件解析方法
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def find_filenames_with_keyword(root_dir, keyword):
    """
    查找指定文件路径
    :@param root_dir: 文件所在根目录
    :@param keyword: 筛选值
    """
    match_files = []
    # 遍历目录树
    for dirpath, dirnames, files in os.walk(root_dir):
        # 筛选出包含关键字的文件名
        match_files.extend([os.path.join(dirpath, file) for file in files if keyword in file])
    return match_files

def save_image(img_data, img_extract_path):
    """
    保存图像数据
    :@param img_data: 图像数据
    :@param img_extract_path: 图像路径
    """

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

def decode_and_generate(batch_file, target_base_file):
    """
    解码 生成数据集
    """
    # 获取训练集路径
    train_list: list = find_filenames_with_keyword(batch_file, "data_batch")
    # 获取测试集路径
    test_list: list = find_filenames_with_keyword(batch_file, "test_batch")

    logging.log(logging.INFO, f"{train_list},\n{test_list}")
    
    if len(train_list) == 0 or len(test_list) == 0:
        return

    # 将train_list, test_list合并
    dataset: list = [(trl, 'train') for trl in train_list] + [(tsl, 'test') for tsl in test_list]

    # 分组[name, lable, data]存储
    for tl_path, set_type in dataset:
        # unpickle
        batch_dict: dict = unpickle(tl_path)
        # 根据set_type决定是训练集还是测试集
        if set_type == "train":
            target_file = os.path.join(target_base_file, "train", "label")
        elif set_type == "test":
            target_file = os.path.join(target_base_file, "test")
        
        # 拿到对应name和data根据label->存放到train对应文件夹下
        with ThreadPoolExecutor(max_workers=10) as excutor:         # 线程数<=10
            futures = []
            for im_idx, im_data in enumerate(batch_dict[b'data']):
                im_label: int = batch_dict[b'labels'][im_idx]               # 图片标签
                im_name: str = batch_dict[b'filenames'][im_idx]             # 图片名字

                # logging.log(logging.INFO, f"{labels[im_label]}, {im_name.decode('utf-8')}, {im_data}")

                # 存储到训练集相同标签下
                im_path_label = os.path.join(target_file, labels[im_label], im_name.decode('utf-8'))
                # 提交任务给线程池
                future = excutor.submit(save_image, im_data, im_path_label)
                futures.append(future)

            for future in futures:
                future.result()


def split_ssl_dataset(base_dir, num_label=5000, num_unlabel=45000):
    """
    划分训练集、验证集
    :@param base_dir: 带标签数据集的基础目录
    :@param num_labeled: 新验证集中的样本数量
    :@param num_unlabeled: 无标签训练集中的样本数量
    """

    assert num_label + num_unlabel == 50000, "总数与数据集数量不一致"

    # proportion
    proportion = num_unlabel / (num_unlabel + num_label)
    total_num2move = num_unlabel

    # 目标路径
    labels_dir = os.path.join(base_dir, "train", "label")
    unlabel_dir = os.path.join(base_dir, "train", "unlabel")
    if not os.path.exists(unlabel_dir):
        raise FileNotFoundError(f"{unlabel_dir} not found")
    
    tasks = []
    # 遍历每个标签目录
    for label in labels:
        label_dir_path = os.path.join(labels_dir, label)                # 待处理标签路径
        images = os.listdir(label_dir_path)                             # 获取该标签下所有图片
        images_num = len(images)
        num2move = int(images_num * proportion)                         # 需要移动的图片数量

        # 如果当前是最后一个标签, 调整移动图像数量确保总数正确
        if total_num2move - num2move < 0:
            num2move = total_num2move

        # 移动图像数据
        # move_images(label_dir, unlabel_dir, num2move / images_num)
        tasks.append((label_dir_path, unlabel_dir, num2move / images_num))

        # 更新剩余需要移动图像总数
        total_num2move -= num2move
        if total_num2move <= 0:
            break
    
    with ThreadPoolExecutor(max_workers=len(labels)) as excutor:
        futures = [excutor.submit(move_images, *task) for task in tasks]
        for future in futures:
            future.result()


def move_images(source, dest, prop):
    """
    随机选择一定比例的文件
    :param source: 源目录
    :param dest: 目标目录
    :param prop: 移动比例
    """
    files: list = os.listdir(source)                                       # 图像数据列表
    num_files = len(files)
    num2move = int(num_files * prop)

    selected_files = random.sample(files, num2move)

    for file in selected_files:
        src = os.path.join(source, file)
        dst = os.path.join(dest, file)
        logging.log(logging.INFO, f"moved {src} to {dst}")
        shutil.move(src, dst)



if __name__ == '__main__':
    # 1.解压
    # untar(cifar10_tar_gz, extract_path)
    # 2.解码数据并保存
    create_cifar10_dir("cifar-10")
    # cifar10路径
    processed_cifar10 = os.path.join(processed_dir, "cifar-10")
    decode_and_generate(extract_path, processed_cifar10)
    # 划分数据集
    split_ssl_dataset(processed_cifar10)


    
