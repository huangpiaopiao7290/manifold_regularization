import math
import os
import random

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms

from cifar10Dataset import CIFARValDataset, UnlabeledCIFARDataset, LabeledCIFARDataset
from process import load_c10_data

from models.Resnet import resnet18, resnet34
from models.loss import consistency_loss, cross_entropy_loss, point_smoothness_loss

def evaluate_on_val(model: nn.Module, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x_val, y_val in test_loader:
            x_val = x_val.to(device)
            y_val = y_val.to(device)

            output = model.forward(x_val)

            pres = torch.argmax(output, dim=1)
            correct += (pres == y_val).sum().item()
            total += y_val.size(0)
    acc = correct / total
    return acc

def train_semi_supervised(model, labeled_loader, unlabeled_loader, val_loader, lam_c, lam_s, epochs=16, lr=1e-3):
    """

    Params:
        lam_c:
        lam_s:

    """
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()

        n_l = len(labeled_loader)
        n_u = len(unlabeled_loader)
        norm = math.pow(n_u + n_l, 2)

        total_loss = 0.0
        total_cross_loss = 0.0
        total_con_loss = 0.0
        total_smooth_loss = 0.0

        k = 7              # knn最近邻居个数
        sigma = 70.0        # gauss核带宽参数
        alpha = 0.75        # beta分布参数

        # 1.带标签数据训练
        for x_labeled, y_labeled in labeled_loader:
            x_labeled = x_labeled.to(device)
            y_labeled = y_labeled.to(device)

            optimizer.zero_grad()

            pre_labeled = model(x_labeled)
            cross_loss = cross_entropy_loss(pre_labeled, y_labeled)

            cross_loss.backward()
            optimizer.step()

            total_cross_loss +=  cross_loss.item()
            total_loss += cross_loss.item()

        # 2.无标签数据训练
        for x_unlabeled in unlabeled_loader:
            x_unlabeled = x_unlabeled.to(device)

            optimizer.zero_grad()

            # 计算mini-batch内一致性损失
            con_loss =  consistency_loss(model, x_unlabeled, alpha)
            # # 计算逐点平滑性损失
            # smooth_loss = point_smoothness_loss(model, x_unlabeled, k, sigma)

            # mini-batch总损失
            # loss = lam_c * con_loss + (lam_s / norm) * smooth_loss
            loss = lam_c * con_loss
            loss.backward()
            optimizer.step()

            total_con_loss += con_loss.item()
            # total_smooth_loss += smooth_loss.item()
            total_loss += loss.item()

        # 平均损失
        avg_total_loss = total_loss / (n_l + n_u)
        avg_cross_loss = total_cross_loss / n_l
        avg_con_loss = total_con_loss / n_u
        avg_smooth_loss = total_smooth_loss / n_u

        # 测试集评估
        val_acc = evaluate_on_val(model, val_loader)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_total_loss: .4f}, cross:{avg_cross_loss: .4f}, consistency: {avg_con_loss: .4f}, smoothness: {avg_smooth_loss: .4f} Acc: {val_acc: .4f}")

if __name__ == '__main__':
    # ---------------------------
    # 0. 设置随机种子与设备
    # ---------------------------
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备类型
    print(f"使用设备: {device}")

    # ---------------------------
    # 1. ciFar10 路径与相关配置
    # ---------------------------
    workspace = os.getcwd()                                                 # 项目根目录
    ciFar10_tar_gz = "data\\cifar10\\cifar-10-python.tar.gz"                # 源数据相对位置
    ciFar10_un_tar = "data\\cifar10"                                        # 解压相对位置
    c10_dir_name = "cifar-10-batches-py"                                    # 解压目录名
    path_tar = os.path.join(workspace, ciFar10_tar_gz)                      # 源数据绝对位置
    path_un_tar = os.path.join(workspace, ciFar10_un_tar)                   # 解压绝对位置
    c10_dir_absolute =  os.path.join(path_un_tar, c10_dir_name)             # 解压后绝对路径

    # 标签
    c10_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    # ---------------------------
    # 2. ciFar10 数据处理
    # ---------------------------
    # 解压数据集
    # un_tar(path_tar, path_un_tar)
    # 加载batch
    train_images_all, train_labels_all, test_images_all, test_labels_all = load_c10_data(c10_dir_absolute)

    print(f"训练集图像数量: {train_images_all.shape[0]}")
    print(f"测试集图像数量: {test_images_all.shape[0]}")

    # ---------------------------
    # 3. 拆分“有标签”和“无标签”数据
    # ---------------------------

    NUM_LABELED_PER_CLASS = 50  # 每类有标签样本数
    NUM_CLASSES = 10

    # 初始化计数器
    count_per_class = [0] * NUM_CLASSES

    labeled_images = []
    labeled_labels = []
    unlabeled_images = []
    unlabeled_labels = []

    # 打乱索引以确保随机性
    indices = list(range(len(train_images_all)))
    random.shuffle(indices)

    for idx in indices:
        img = train_images_all[idx]
        lbl = train_labels_all[idx].item()

        if count_per_class[lbl] < NUM_LABELED_PER_CLASS:
            labeled_images.append(img)
            labeled_labels.append(lbl)
            count_per_class[lbl] += 1
        else:
            unlabeled_images.append(img)
            unlabeled_labels.append(lbl)  # 标签仍然存在，但后续不使用

    print(f"有标签数据数量: {len(labeled_images)}")
    print(f"无标签数据数量: {len(unlabeled_images)}")

    # ---------------------------
    # 5. 创建 DataLoader
    # ---------------------------

    BATCH_SIZE = 32

    transform = transforms.Compose([
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # CIFAR-10的均值和标准差
    ])

    # 有标签 DataLoader
    labeled_dataset = LabeledCIFARDataset(labeled_images, labeled_labels, transform)
    labeled_loader = DataLoader(
        labeled_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # 无标签 DataLoader
    unlabeled_dataset = UnlabeledCIFARDataset(unlabeled_images, transform)
    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # 验证/测试 DataLoader
    val_dataset = CIFARValDataset(test_images_all, test_labels_all)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print(f"有标签数据批次数: {len(labeled_loader)}")
    print(f"无标签数据批次数: {len(unlabeled_loader)}")
    print(f"验证/测试数据批次数: {len(val_loader)}")
    # ---------------------------
    # 6. 训练
    # ---------------------------
    model34 =resnet18(num_classes=10, include_top=True).to(device)
    train_semi_supervised(model34, labeled_loader, unlabeled_loader, val_loader, 20., 3.)