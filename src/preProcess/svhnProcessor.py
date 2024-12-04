import os

from src.utils.utility import Utility


work_space = os.getcwd()

# 解压svhn  train.tar.gz  test.tar.gz
svhn_train = os.path.join(work_space, "data/raw/svhn/train.tar.gz")
svhn_test = os.path.join(work_space, "data/raw/svhn/test.tar.gz")
svhn_dir = os.path.join(work_space, "data/processed/svhn")


Utility.un_tar(svhn_train, svhn_dir)
Utility.un_tar(svhn_test, svhn_dir)
