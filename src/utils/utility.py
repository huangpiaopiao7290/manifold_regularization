## @auther: pp
## @date: 2024/10/5
## @description:
import os
import tarfile
import logging
from concurrent.futures import ThreadPoolExecutor
import shutil
import random
import aiofiles

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
    def find_filenames_with_keyword(root_dir, keyword):
        """查找指定文件路径"""
        match_files = []
        for dir_path, _, files in os.walk(root_dir):
            match_files.extend([os.path.join(dir_path, file) for file in files if keyword in file])
        return match_files

    @staticmethod
    async def move_files_async(source, dest, prop):
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
            await aiofiles.os.rename(src, dst)
        except Exception as e:
            logging.error(f"Failed to move {src} to {dst}: {e}")


