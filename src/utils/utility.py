## @auther: pp
## @date: 2024/10/5
## @description:
import os
import pickle
import tarfile
import logging
import shutil
import random
from shutil import copy2
import asyncio

logging.basicConfig(level=logging.INFO)     # 定义日志级别
work_space = os.getcwd()

class Utility:
    @staticmethod
    def un_tar(file_path, extract_to="."):
        """解压tar.gz到目标目录"""
        if not os.path.exists(extract_to):
            os.makedirs(extract_to)
        with tarfile.open(file_path, 'r:gz') as tar:
            for member in tar.getmembers():
                tar.extract(member, extract_to)
                logging.info(f"extracted {member.name} to {extract_to}")

    @staticmethod
    def parse_pickle(file):
        """
        解析pickle文件
        """
        with open(file, 'rb') as fo:
            dct = pickle.load(fo, encoding='bytes')
        return dct

    @staticmethod
    def find_filenames_with_keyword(root_dir, keyword):
        """查找指定文件路径"""
        match_files = []
        for dir_path, _, files in os.walk(root_dir):
            match_files.extend([os.path.join(dir_path, file) for file in files if keyword in file])
        return match_files

    @staticmethod
    async def move_files(source, dest, prop):
        """随机选择一定比例的文件并移动"""
        files = os.listdir(source)
        num_files = len(files)
        num2move = int(num_files * prop)
        selected_files = random.sample(files, num2move)
        for file in selected_files:
            src = os.path.join(source, file)
            dst = os.path.join(dest, file)
            logging.info(f"moving {src} to {dst}")
            await Utility._move_file(src, dst)

    @staticmethod
    async def _move_file(src, dst):
        """异步移动单个文件"""
        try:
            if hasattr(asyncio, 'to_thread'):
                await asyncio.to_thread(shutil.move, src, dst, copy2)
            else:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, shutil.move, src, dst, copy2)
        except Exception as e:
            logging.error(f"Failed to move {src} to {dst}: {e}")

    @staticmethod
    def move_file_sync(src, dst):
        """同步移动单个文件"""
        try:
            shutil.move(src, dst)
            logging.info(f"moved {src} to {dst}")
        except Exception as e:
            logging.error(f"Failed to move {src} to {dst}: {e}")


