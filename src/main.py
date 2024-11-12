## @author: pp
## @date: 2024/9/17
## @description: 训练模型

from models.vgg import VggNet
from dataset.cifar10Dataset import get_data_loaders
# from utils.lossFunction import LossFunctions
from utils.lossFunction_PWMR import LossFunctionsPWMR
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.optim as optim
import os
import yaml
import logging.config


current_dir = os.path.join(os.getcwd(), 'src')
application_yml_path = os.path.join(current_dir, 'application.yml')

# 日志文件保存位置
log_dir = os.path.join(os.getcwd(), 'log')
# tensorboard记录位置
lod_dir_tensorboard= os.path.join(log_dir, 'logdir')
os.makedirs(log_dir, exist_ok=True)
os.environ['LOG_DIR'] = log_dir

# 加载日志配置文件
with open(application_yml_path, 'r') as f:
    log_config = yaml.safe_load(f)
log_config['handlers']['fileHandler']['filename'] = os.path.join(log_dir, 'training.log')
logging.config.dictConfig(log_config)

# # 获取日志记录器
logger = logging.getLogger('exampleLogger')

# TensorBoard 记录器
writer = SummaryWriter(log_dir=lod_dir_tensorboard)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

epochs: int = 200                   # 训练批次
batch_size = 64                     # 批大小
learning_rate: float = 0.01         # 学习率
lambda_c: float = 0.1               # 一致性损失权重
lambda_s: float = 0.1               # 平滑性损失权重
alpha: float = 0.99                 # EMA动量项
best_test_accuracy = 90            # 准确率


data_dir = os.path.join(os.getcwd(), "data", "processed")

model = VggNet(num_class=10).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.9)

# 损失函数
# lossFunc = LossFunctions(device=device)
lossFunc = LossFunctionsPWMR(device=device)

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()

    logging.info("start...")

    # 加载数据cifar10
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
            no_labels_mask = (labels == -1)

            # # 带标签数据 对应标签索引
            # inputs_labels, has_labels = inputs[labeled_mask], labels[labeled_mask]
            # outputs_labels = outputs[labeled_mask]

            # 计算总损失

            features = model.extract_features(inputs)

            # loss, w = lossFunc.total_loss(model=model, images=inputs, labels=labels,
            #                      unlabeled_mask=no_labels_mask, lambda_c=lambda_c, lambda_s=lambda_s)

            loss, w, local_identities = lossFunc.total_loss(model=model, images=inputs, labels=labels,
                                          unlabeled_mask=no_labels_mask, lambda_c=lambda_c, lambda_s=lambda_s)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计正确预测的数量
            with torch.no_grad():
                _, predicted = torch.max(outputs.data, dim=1)
                total += labels.size(0)
                correct += (predicted.eq(labels.data)).sum().item()
                accuracy = 100.0 * correct / total
                # logging.debug(f"step {i}, loss={loss.item()}, mini-batch correct is {accuracy: .2f}"
                #               f" learning rate is {optimizer.state_dict()['param_groups'][0]['lr']}")

                # 记录到 TensorBoard
                writer.add_scalar('Train/Loss', loss.item(), epoch * len(train_loader) + i)
                writer.add_scalar('Train/Mini-Batch Accuracy', accuracy, epoch * len(train_loader) + i)
                writer.add_scalar('Train/Learning Rate', optimizer.state_dict()['param_groups'][0]['lr'],
                                  epoch * len(train_loader) + i)
        # 更新学习率
        scheduler.step()
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
        test_accuracy = 100 * test_correct / test_total
        logging.info(f"Test Accuracy: {test_accuracy:.2f}%")

        # 根据测试集的表现调整超参数 ----------- TODO 修改
        if test_accuracy < 80.0:
            lambda_c *= 0.9
            lambda_s *= 1.1
        elif test_accuracy > 90.0:
            lambda_c *= 1.1
            lambda_s *= 0.9

        # 记录到 TensorBoard
        writer.add_scalar('Validation/Accuracy', test_accuracy, epoch)

        # 保存最佳模型
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            if not os.path.exists(os.path.join(os.getcwd(), "models_t")):
                os.mkdir("models_t")
            torch.save(model.state_dict(), "models_t/{}.pt".format(epoch))
            logging.info(f"Saved new best model with accuracy: {best_test_accuracy:.2f}%")

    logging.info("Training completed.")
            


    

