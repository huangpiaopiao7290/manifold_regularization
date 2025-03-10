import torch
from torch import nn

class BasicBlock(nn.Module):
    """
    浅层resnet18/34
    """
    expansion = 1       # 扩张因子
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        """
        Args:
            in_channel: 输入通道数
            out_channel: 输出通道数
            stride: 卷积层的步长
            downsample: 下采样
        """
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3,
                               padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3,
                               padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        """
        """
        identify = x                        # 保存输入信息以便于shortcut connection
        if self.downsample is not None:     # 进行下采样得到shortcut connection的输入
            identify = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identify                     # h(x) = f(x) + x
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    """
    深层resnet50/101/152
    """
    expansion = 4       # 主分支卷积核个数最后一层会变成第一层的4倍
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        """
        Args:
            in_channel: 输入通道数
            out_channel: 输出通道数
            stride: 卷积层的步长
            downsample: 下采样
        """
        super(Bottleneck, self).__init__()
        # 第一个1*1卷积层->压缩通道数
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1,
                               stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        # 第2个3*3卷积层->
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3,
                               padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        # 第3个1*1卷积层->恢复通道数为256
        self.conv3 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1,
                               stride=stride, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)

        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x):
        """
        """
        identify = x                        # 保存输入信息以便于shortcut connection
        if self.downsample is not None:     # 进行下采样得到shortcut connection的输入
            identify = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += identify                     # h(x) = f(x) + x
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    """
    resnet framework
    """
    def __init__(self, block, block_nums, num_classes=10, include_top=True):
        """
        Args:
            block: 网络选取 -example: basicBlock对应resnet18/34; bottleBlock对应resnet50/101/152
            block_nums: 残差结构的数目 residual num: as 34-layers [3,4,6,3]
            num_classes: 分类数
            include_top: 分类头 linear-layer
        """
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64
        # rgb: 3/gray: 1;
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU()
        self.max_pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # residual
        self.layer1 = self._residual(block=block, channel=64, block_num=block_nums[0], stride=1)
        self.layer2 = self._residual(block=block, channel=128, block_num=block_nums[1], stride=2)
        self.layer3 = self._residual(block=block, channel=256, block_num=block_nums[2], stride=2)
        self.layer4 = self._residual(block=block, channel=512, block_num=block_nums[3], stride=2)

        if self.include_top:
            self.avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        # 初始化卷积层权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def _residual(self, block, channel, block_num, stride=1):
        """
        创建残差层
        Args:
            block: 网络深度选择 BasicBlock/BottleNeck对象
            channel: 残差块中第一个卷积层对应通道数
            block_num: 残差块个数
            stride: 卷积步长
        """
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            # 对于resnet50以上的residual
            # -example：50-layers input=64, out=256
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion)
            )
        
        layers = [block(in_channel=self.in_channel, out_channel=channel, stride=stride, downsample=downsample)]
        # 更新通道数
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(in_channel=self.in_channel, out_channel=channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        """
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.max_pooling(out)     # 3×3 max pool, stride 2
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        if self.include_top:
            out = self.avg_pooling(out)
            out = torch.flatten(out, 1)
            out = self.fc(out)

        return out

def resnet18(num_classes=10, include_top=True, pretrained=False):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, include_top)

def resnet34(num_classes=10, include_top=True, pretrained=False):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, include_top)

def resnet50(num_classes=10, include_top=True, pretrained=False):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, include_top)

def resnet101(num_classes=10, include_top=True, pretrained=False):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, include_top)
