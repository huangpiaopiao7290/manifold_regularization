## @author: piaopiao
## @date: 2024/9/17
## @description: 定义模型

import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import torch.nn.functional as F
import os
import logging

# conv, pooling
cfg = {
    'VGG11': [64, 'P', 128, 'P', 256, 256, 'P', 512, 512, 'P', 512, 512, 'P'],
    'VGG13': [64, 64, 'P', 128, 128, 'P', 256, 256, 'P', 512, 512, 'P', 512, 512, 'P'],
    'VGG16': [64, 64, 'P', 128, 128, 'P', 256, 256, 256, 'P', 512, 512, 512, 'P', 512, 512, 512, 'P'],
    'VGG19': [64, 64, 'P', 128, 128, 'P', 256, 256, 256, 256, 'P', 512, 512, 512, 512, 'P', 512, 512, 512, 512, 'P'],
}

class VggNet(nn.Module):
    def __init__(self, vgg_name, num_class=10) -> None:
        super(VggNet, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        # 线性层分类器
        self.classifier = nn.Linear(512, num_class)

    def forward(self, x):
        # 提取特征
        out = self.features(x)
        # 展平特征
        out = self.view(out.size(0), -1)
        out = self.classifier(out)

        # out = F.log_softmax(out, dim=1)
        
        return out

    def _make_layer(self, cfg):
        layers = []
        in_channel = 3

        for x in cfg:
            if x == 'P':
                # 池化层
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                # 卷积层
                layers += [nn.Conv2d(in_channel, x, kernel_size=3, padding=1),
                            nn.BatchNorm2d(x),
                            nn.ReLU(inplace=True)]
                in_channel = x
        
        # 最后一层平均池化 
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


