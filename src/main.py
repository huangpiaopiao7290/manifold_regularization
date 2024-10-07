## @author: pp
## @date: 2024/9/17
## @description: 训练模型

from models.vgg import VggNet
from dataset.cifar10Dataset import get_data_loaders
from src.utils.loss import total_loss

import torch
import torch.nn as nn
import torch.optim as optim
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

device = torch.device("cuda" if torch.cuda.is_available else "cpu")
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

epochs: int = 200                   # 训练批次
batch_size = 64                     # 批大小
learning_rate: float = 0.01         # 学习率
lambda_c: float = 1.0               # 一致性损失权重
lambda_s: float = 1.0                # 平滑性损失权重
alpha: float = 0.99                 # EMA动量项

data_dir = os.path.join(os.getcwd(), "data", "processed")

model = VggNet(num_class=10).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=5, gamma=0.9)

loss_func = nn.CrossEntropyLoss()


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()

    logging.info("start...")


    # 加载数据
    cifar10_dir = os.path.join(data_dir, "cifar-10")
    if not os.path.exists(cifar10_dir):
        logging.error("CIFAR-10 dataset directory does not exist: %s", cifar10_dir)
        exit(1)
    try:
        train_loader, test_loader = get_data_loaders(root=cifar10_dir, batch_size=batch_size)
        logging.info(f"Data loaders created, train samples: {len(train_loader.dataset)}, test samples: {len(test_loader.dataset)}")
    except Exception as e:
        logging.error(f"Failed to create data loaders: {e}")
        exit(1)

    # 训练
    for epoch in range(epochs):
        logging.log(logging.INFO, "epoch is {}".format(epoch))
        model.train()
        running_loss = 0.0
        correct = 0.0
        total = 0.0

        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            outputs = model(inputs)

            # 过滤无标签数据
            labeled_mask = (labels != -1)
            no_labels_mask = (labels == -1)

            # 带标签数据 对应标签索引
            inputs_labels, has_labels = inputs[labeled_mask], labels[labeled_mask]
            outputs_labels = outputs[labeled_mask]
            # 1.计算带标签数据的交叉熵损失
            # loss = loss_func(outputs_labels, has_labels)

            # 计算总损失
            loss, w = total_loss(model=model, images=inputs, labels=labels,
                                 unlabeled_mask=no_labels_mask, lambda_c=lambda_c, lambda_s=lambda_s)

            # 反向传播优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计正确预测的数量
            with torch.no_grad():
                _, predicted = torch.max(outputs.data, dim=1)
                total += labels.size(0)
                correct = (predicted.eq(labels.data).cpu().sum())
                logging.log(logging.INFO,
                            f"step {i}, loss={loss.item()}, mini-batch correct is {100.0 * correct / batch_size},"
                            f" learning rate is {optimizer.state_dict()['param_groups'][0]['lr']}")

        # 更新学习率
        scheduler.step()

        # 保存模型
        if not os.path.exists(os.path.join(os.getcwd(), "models_t")):
            os.mkdir("models_t")
        torch.save(model.state_dict(), "models_t/{}".format(epoch + 1))

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
            


    

