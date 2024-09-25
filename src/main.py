## @author: piaopiao
## @date: 2024/9/17
## @description: 训练模型

from models.vgg import VggNet
from dataset.cifar10Dataset import get_data_loaders
from utils.manifold_regularization import compute_adjacency_matrix, consistency_loss, criterion_supervised, smoothness_loss, total_loss

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import os
import logging


epochs: int = 200                   # 训练批次
batch_size = 64                     # 批大小
learning_rate: float = 0.01         # 学习率
lambda_c: float = 1.0               # 一致性损失权重
lamda_s: float = 1.0                # 平滑性损失权重
alpha: float = 0.99                 # EMA动量项
device = torch.device("cuda" if torch.cuda.is_available else "cpu")

cifar10_dataset_root = os.path.join(os.getcwd(), "data", "processed", "cifar-10")

model = VggNet('VGG16', num_class=10).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
schedualer = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=5, gamma=0.9)

loss_func = nn.CrossEntropyLoss()


# 加载数据
train_loader, test_loader = get_data_loaders(root=cifar10_dataset_root, batch_size=batch_size)

# train
for epoch in range(epochs):
    logging.log(logging.INFO, "epoch is {}".format(epoch))
    model.train()

    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # 前向传播
        outputs = model(inputs)
        # loss
        
        # 一致性损失
        # 平滑性损失
        # 总损失

        loss = loss_func(outputs, labels)
        
        # 反向传播优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 
        _, pred = torch.max(outputs.data, dim=1)
        correct = pred.eq(labels.data).cpu().sum()
        logging.log(logging.INFO, f"step {i}, loss={loss.item()}, mini-batch correct is {100.0 * correct / batch_size}, 
                        learning rate is {optimizer.state_dict()["param_groups"][0]["lr"]}")

    # 更新学习率
    schedualer.step()

    # 保存模型
    if not os.path.exists(os.getcwd(), "models_t"):
        os.mkdir("models_t")
    torch.save(model.state_dict(), "models_t/{}".format(epoch + 1))

    # 测试
    # 测试
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for j, (test_inputs, test_labels) in enumerate(test_loader):
            test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
            test_outputs = model(test_inputs)
            _, test_predicted = torch.max(test_outputs.data, dim=1)
            test_total += test_labels.size(0)
            test_correct += (test_predicted == test_labels).sum().item()

    # 打印测试准确率
    logging.info(f"Test Accuracy: {100 * test_correct / test_total:.2f}%")

logging.info("Training completed.")
        


    

