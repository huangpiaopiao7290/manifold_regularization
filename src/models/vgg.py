## @author: pp
## @date: 2024/9/17
## @description: 定义模型

import torch.nn as nn
import torch.nn.functional as F


# conv, pooling
# cfg = {
#     'VGG11': [64, 'P', 128, 128, 'P', 256, 256, 'P', 512, 512, 'P'],
#     'VGG13': [64, 64, 'P', 128, 128, 'P', 256, 256, 'P', 512, 512, 'P', 512, 512],
#     'VGG16': [64, 64, 'P', 128, 128, 'P', 256, 256, 256, 'P', 512, 512, 512, 'P', 512, 512, 512],
#     'VGG19': [64, 64, 'P', 128, 128, 'P', 256, 256, 256, 256, 'P', 512, 512, 512, 512, 'P', 512, 512, 512, 512],
# }

class VggNet(nn.Module):
    
    def __init__(self, num_class=10) -> None:
        super(VggNet, self).__init__()
        # self.features = self._make_layer(cfg[vgg_name])

        # 3 * 28 * 28
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.max_pooling_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 64 * 14 * 14
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.max_pooling_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 128 * 7 * 7
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )       
        self.max_pooling_3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        # 256 * 4 * 4
        self.conv4_1 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.conv4_2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.max_pooling_4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # batch_size * 512 * 2 * 2 ---->  reshape batch_size * 512 * 4
        self.classifier = nn.Linear(512 * 4, num_class)

    def forward(self, x):
        batch_size = x.size(0)
        # 提取特征
        # print(f"Input shape: {x.shape}")
        # out = self.features(x)
        out = self.conv1(x)
        out = self.max_pooling_1(out)
        out = self.conv2_1(out)
        out = self.conv2_2(out)
        out = self.max_pooling_2(out)
        out = self.conv3_1(out)
        out = self.conv3_2(out)
        out = self.max_pooling_3(out)
        out = self.conv4_1(out)
        out = self.conv4_2(out)
        out = self.max_pooling_4(out)
        # print(f"Features output shape: {out.shape}")
        # 展平特征
        out = out.view(batch_size, -1)
        out = self.classifier(out)
        out = F.log_softmax(out, dim=1)
        # print(f"Flattened shape: {out.shape}")

        return out

    # def _make_layer(self, cfg):
    #     layers = []
    #     in_channel = 3

    #     for x in cfg:
    #         if x == 'P':
    #             # 池化层
    #             layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
    #         else:
    #             # 卷积层
    #             layers += [nn.Conv2d(in_channel, x, kernel_size=3, padding=1), nn.BatchNorm2d(x), nn.ReLU(inplace=True)]
    #             in_channel = x
        
    #     # 最后一层平均池化 
    #     layers += [nn.AvgPool2d((1, 1))]
    #     return nn.Sequential(*layers)


