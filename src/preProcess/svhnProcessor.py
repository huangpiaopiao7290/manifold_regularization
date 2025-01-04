import os

from src.utils.utility import Utility


work_space = os.getcwd()

# 解压svhn  train.tar.gz  test.tar.gz
svhn_train = os.path.join(work_space, "data/raw/svhn/train.tar.gz")
svhn_test = os.path.join(work_space, "data/raw/svhn/test.tar.gz")
origin_svhn_dir = os.path.join(work_space, "data/raw/svhn")

Utility.un_tar(svhn_train, origin_svhn_dir)
Utility.un_tar(svhn_test, origin_svhn_dir)


import glob
import os
import pickle

import cv2
import h5py
import numpy as np
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm


class DataPreprocess():
    def __init__(self, set_type):
        self.set_type = set_type
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize((32, 32)),
             transforms.ToTensor()])
        self.digitStructName = None
        self.digitStructBbox = None
        self.f = None
        self.preprocess()

    def preprocess(self):
        print('Preprocess set:', self.set_type)
        data_dir = f"./{self.set_type}/"    # 数据集所在文件夹名称
        cache_dir = './cache'				# 用于存放处理好的数据

        print('Data directory is ', data_dir)
        self.f = h5py.File(os.path.join(data_dir, 'digitStruct.mat'), 'r')
        self.digitStructName = self.f['digitStruct']['name']
        self.digitStructBbox = self.f['digitStruct']['bbox']

        print('\nGetting name and bbox ...')
        os.makedirs(cache_dir, exist_ok=True)
        cache_image_dir = os.path.join(cache_dir, f'image_dict_{self.set_type}.pkl')
        if os.path.exists(cache_image_dir):
            # 读取已经处理好的
            image_dict = pickle.load(open(cache_image_dir, 'rb'))
        else:
            # 处理
            image_dict = {}
            for i in tqdm(range(len(self.digitStructName))):
                image_dict[self.getName(i)] = self.getBbox(i)

            with open(cache_image_dir, 'wb') as w:
                pickle.dump(image_dict, w)

        print('Done.\n\nPreprocessing ...')
        fnames = glob.glob(os.path.join(data_dir, '*.png'))
        save_dir = f"./{self.set_type}_new/"
        os.makedirs(save_dir, exist_ok=True)
        print('Save to ', save_dir)

        n_drop = 0
        for i in tqdm(range(len(fnames))):
            image = cv2.imread(fnames[i])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # because "torchvision.utils.save_image" use RGB

            _, fname = os.path.split(fnames[i])

            digit_dict = image_dict[fname]

            fname = fname.split('.')[0]
            for j in range(len(digit_dict['label'])):
                label = int(digit_dict['label'][j])
                if label == 10:
                    label = 0

                left = int(digit_dict['left'][j])
                upper = int(digit_dict['top'][j])
                right = int(left + digit_dict['width'][j])
                lower = int(upper + digit_dict['height'][j])
                if left < 0 or upper < 0:
                    n_drop += 1
                    continue
                img = image[upper:lower, left:right, :]
                img = self.transform(img)
                # 把图片以 1_0.jpg的格式存储，1是原始的图像名，0是它的标签
                save_file = os.path.join(save_dir, f'{fname}_{label}.jpg')
                torchvision.utils.save_image(img, save_file)
        print(f'Done.(Drop {n_drop} digits for the negative coordinate.)\n')

    def getName(self, n):
        # 从 mat 中获取图像name  e.g. 1.png
        return ''.join([chr(v[0]) for v in self.f[(self.digitStructName[n][0])]])

    def bboxHelper(self, attr):
        # 根据attr从bbox中取值，attr 可能是 height/left/top/width/label
        if len(attr) > 1:
            attr = [self.f[attr[j].item()][0][0] for j in range(len(attr))]
        else:
            attr = [attr[0][0]]
        return attr

    def getBbox(self, n):
        # 从 mat 中获取 digit 的bbox
        bbox = {}
        bb = self.digitStructBbox[n].item()
        bbox['height'] = self.bboxHelper(self.f[bb]["height"])
        bbox['left'] = self.bboxHelper(self.f[bb]["left"])
        bbox['top'] = self.bboxHelper(self.f[bb]["top"])
        bbox['width'] = self.bboxHelper(self.f[bb]["width"])
        bbox['label'] = self.bboxHelper(self.f[bb]["label"])
        return bbox


def loadData(set_type, reload=False):
    print(f'Loading {set_type} data ...')
    cache_dir = './cache'
    cache_dataset_dir = os.path.join(cache_dir, f'{set_type}.pkl')
    if os.path.exists(cache_dataset_dir) and not reload:
        # 读取已经处理好的
        print('Existed.')
        dataset = pickle.load(open(cache_dataset_dir, 'rb'))
    else:
        new_data_dir = f"./{set_type}_new/"
        fnames = glob.glob(os.path.join(new_data_dir, '*'))

        dataset = {}
        x = []
        y = []
        for i in tqdm(range(len(fnames))):
            fname = fnames[i]

            _, label = os.path.split(fname)
            label = int(label.split('.')[0].split('_')[1])
            y.append(label)

            image = cv2.imread(fname)
            x.append(image)

        dataset['data'] = np.array(x)
        dataset['label'] = np.array(y)

        with open(cache_dataset_dir, 'wb') as w:
            pickle.dump(dataset, w)

    print('Shape of x:', dataset['data'].shape, '\t|\tShape of y:', dataset['label'].shape)
    print('Done.')
    return dataset


if __name__ == '__main__':
    preprocess_train = DataPreprocess(set_type='train')
    preprocess_test = DataPreprocess(set_type='test')

    # train = loadData(set_type='train', reload=True)
    # test = loadData(set_type='test', reload=True)

    # print(train.shape)
    # print(test.shape)

