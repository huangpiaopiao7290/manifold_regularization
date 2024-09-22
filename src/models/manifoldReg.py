## @auther## @author: piaopiao
## @date: 2024/9/17
## @description: 定义模型

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import numpy as np
import os
import logging


# class BasicNet(nn.module):
#     def __init__(self):
#         super(BasicNet, self).__int__()
    
#     def forward(self, x):



# class ManifoldRegBase(nn.Module):
#     def __init__(self, in_planes, planes, stride=1, num_class=10) -> None:
#         super(ManifoldRegBase, self).__init__()
#         # 28*28*3
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1),
#             nn.BatchNorm2d(planes),
#             nn.ReLU()
#         )

#         self.max_pooling1 = nn.MaxPool2d(kernel_size=2, stride=2)

#         # 14*14*3
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1),
#             nn.BatchNorm2d(planes),
#             nn.ReLU()
#         )
        
        
        
#         # 
#         # self.bn1 = nn.BatchNorm2d(in_planes)
#         # self.conv2 = nn.Conv2d(planes, 128, kernel_size=3, stride=stride, padding=1)
#         # self.bn2 = nn.BatchNorm2d(planes)


# 使用预训练的ResNet18模型提取特征

class FeatureExtractor(nn.Module):

    def __init__(self):

        super(FeatureExtractor, self).__init__()

        self.resnet = resnet18(pretrained=True)

        self.resnet.fc = nn.Identity()  # 移除全连接层，只保留特征提取部分


    def forward(self, x):

        return self.resnet(x)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
feature_extractor = FeatureExtractor().to(device)
feature_extractor.eval()


# 提取训练集和测试集的特征
def extract_features(dataloader, model):
    features = []
    labels = []
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            feature = model(images)
            features.append(feature.cpu())
            labels.append(targets)

    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)

    return features, labels


train_features, train_labels = extract_features(trainloader, feature_extractor)
test_features, test_labels = extract_features(testloader, feature_extractor)





