## @author: pp
## @date: 2024/9/17
## @description: 训练模型
import shutil

from models.vgg import VggNet
from dataset.cifar10Dataset import get_data10_loaders
from dataset.cifar100Dataset import get_data100_loaders
from utils.lossFunction_PWMR import LossFunctionsPWMR
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.optim as optim
import os
import yaml
import logging.config

work_space = os.getcwd()
current_dir = os.path.join(work_space, 'src')
application_yml_path = os.path.join(current_dir, 'application.yml')

# 日志文件保存位置
log_dir = os.path.join(os.getcwd(), 'log')
# tensorboard记录位置
log_dir_tensorboard= os.path.join(log_dir, 'logdir')
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
_writer = SummaryWriter(log_dir=log_dir_tensorboard)

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# -----------------------------------------------
#                   训练参数
# -----------------------------------------------

_epochs: int = 200                   # 训练批次
_batch_size = 64                     # 批大小
_learning_rate: float = 0.01         # 学习率
_lambda_c: float = 0.1               # 一致性损失权重
_lambda_s: float = 0.1               # 平滑性损失权重
_alpha: float = 0.99                 # EMA动量项
_best_test_accuracy = 90             # 准确率

# -----------------------------------------------
#                 数据集相关信息
# -----------------------------------------------

CiFAR10 = os.path.join(work_space, 'data/processed/cifar-10')
CiFAR100 = os.path.join(work_space, 'data/processed/cifar-100')

ciFar10_labelNames = {}
ciFar10_label_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
for idx, name in enumerate(ciFar10_label_names):
    ciFar10_labelNames[name] = idx

ciFar100_labelNames = {}
ciFar100_label_names = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle", "bowl", "boy",
    "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur", "dolphin", "elephant", "flatfish",
    "forest", "fox", "girl", "hamster", "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse", "mushroom", "oak_tree", "orange",
    "orchid", "otter", "palm_tree", "pear", "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine",
    "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea", "seal", "shark", "shrew", "skunk",
    "skyscraper", "snail", "snake", "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper", "table",
    "tank", "telephone", "television", "tiger", "tractor", "train", "trout", "tulip", "turtle", "wardrobe", "whale",
    "willow_tree", "wolf", "woman", "worm"
]
for idx, name in enumerate(ciFar100_label_names):
    ciFar100_labelNames[name] = idx
# -----------------------------------------------

data_dir = os.path.join(os.getcwd(), "data", "processed")

_model = VggNet(num_class=10).to(_device)
_optimizer = optim.Adam(_model.parameters(), lr=_learning_rate)
_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=_optimizer, step_size=10, gamma=0.9)

# 损失函数
lossFunc = LossFunctionsPWMR(device=_device)

class Trainer:
    def __init__(self, model, optimizer, scheduler, loss_func, _device, lambda_c, lambda_s, best_test_accuracy):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_func = loss_func
        self._device = _device
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.best_test_accuracy = best_test_accuracy
        self.writer = SummaryWriter(log_dir=log_dir_tensorboard)

    def train(self, tr_dataloader, tr_epoch, tr_dataset_name):
        self.model.train()
        running_loss = 0.0
        correct = 0.0
        total = 0.0

        for i, data in enumerate(tr_dataloader):
            inputs, labels = data
            inputs, labels = inputs.to(self._device), labels.to(self._device)
            outputs = self.model(inputs)
            # 无标签标记
            unlabeled_mask = (labels == -1)
            self.optimizer.zero_grad()
            loss, w, local_identities = self.loss_func(model=self.model,
                                                            outputs=outputs,
                                                            images=inputs,
                                                            labels=labels,
                                                            unlabeled_mask=unlabeled_mask,
                                                            lambda_c=self.lambda_c,
                                                            lambda_s=self.lambda_s)
            loss.backward()
            self.optimizer.step()

            # 统计正确预测的数量
            with torch.no_grad():
                _, predicted = torch.max(outputs.data, dim=1)
                total += labels.size(0)
                correct += (predicted.eq(labels.data)).sum().item()
                accuracy = 100.0 * correct / total
                running_loss += loss.item()
                # 记录到 TensorBoard
                self.writer.add_scalar(f'{tr_dataset_name} Train/Loss', running_loss, tr_epoch * len(tr_dataloader) + i)
                self.writer.add_scalar(f'{tr_dataset_name} Train/Mini-Batch Accuracy', accuracy,
                                  tr_epoch * len(tr_dataloader) + i)
                self.writer.add_scalar(f'{tr_dataset_name} Train/Learning Rate',
                                  self.optimizer.state_dict()['param_groups'][0]['lr'],
                                  tr_epoch * len(tr_dataloader) + i)
            # 更新学习率
            self.scheduler.step()

    def test(self, ts_dataloader, ts_epoch, ts_dataset_name):
        # 测试
        self.model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for j, (test_inputs, test_labels) in enumerate(ts_dataloader):
                test_inputs, test_labels = test_inputs.to(_device), test_labels.to(_device)
                test_outputs = self.model(test_inputs)
                _, test_predicted = torch.max(test_outputs.data, dim=1)
                test_total += test_labels.size(0)
                test_correct += (test_predicted == test_labels).to(torch.float).sum().item()

        test_accuracy = 100 * test_correct / test_total
        logging.info(f"{ts_dataset_name} Test Accuracy: {test_accuracy:.2f}%")
        self.writer.add_scalar('Validation/Accuracy', test_accuracy, ts_epoch)

        return test_accuracy

    def adjust_hyperparameters(self, test_accuracy):
        if test_accuracy < 80.0:
            self.lambda_c *= 0.9
            self.lambda_s *= 1.1
        elif test_accuracy > 90.0:
            self.lambda_c *= 1.1
            self.lambda_s *= 0.9

    def save_model(self, test_accuracy, epoch_):
        """
        保存模型
        """
        if test_accuracy > self.best_test_accuracy:
            os.makedirs("models_t", exist_ok=True)
            torch.save(self.model.state_dict(), "models_t/{}.pt".format(epoch_))
            logging.info(f"Saved new best model with accuracy: {self.best_test_accuracy:.2f}%")
            self.best_test_accuracy = test_accuracy

def clear_tensorboard_logs(path):
    if os.path.exists(path) and os.path.isdir(path):
        logger.info("Deleting existing TensorBoard logs.")
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    logger.info(f"TensorBoard log directory cleared: {path}")


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()

    logging.info("start...")

    # 清楚tensorboard记录
    clear_tensorboard_logs(log_dir_tensorboard)

    try:
        c10_train_loader, c10_test_loader = get_data10_loaders(root=CiFAR10,
                                                             label_names_dict=ciFar10_labelNames,
                                                             batch_size=_batch_size)
        c100_train_loader, c100_test_loader = get_data100_loaders(root=CiFAR100,
                                                               label_names_dict=ciFar100_labelNames,
                                                               batch_size=_batch_size)
        logging.info(f"Data loaders created, train samples: {len(c10_train_loader.dataset)},"
                     f" test samples: {len(c10_test_loader.dataset)}")
        logging.info(f"Data loaders created, train samples: {len(c100_train_loader.dataset)},"
                     f" test samples: {len(c100_test_loader.dataset)}")
    except Exception as e:
        logging.error(f"Failed to create data loaders: {e}")
        exit(1)

    # 初始化训练器
    trainer = Trainer(model=_model,
                      optimizer=_optimizer,
                      scheduler=_scheduler,
                      loss_func=lossFunc,
                      _device=_device,
                      lambda_c=_lambda_c,
                      lambda_s=_lambda_s,
                      best_test_accuracy=_best_test_accuracy)

    # 训练
    for epoch in range(_epochs):
        logging.log(logging.INFO, "epoch is {}".format(epoch))

        # 训练CiFAR10
        trainer.train(tr_dataloader=c10_train_loader,
                      tr_epoch=epoch,
                      tr_dataset_name='CiFAR10')
        # 训练CiFAR100
        trainer.train(tr_dataloader=c100_train_loader,
                      tr_epoch=epoch,
                      tr_dataset_name='CiFAR100')
        # 测试CiFAR10
        test_accuracy_c10 = trainer.test(ts_dataloader=c10_test_loader,
                                         ts_epoch=epoch,
                                         ts_dataset_name='CiFAR10')
        # 测试CiFAR100
        test_accuracy_c100 = trainer.test(ts_dataloader=c100_test_loader,
                                         ts_epoch=epoch,
                                         ts_dataset_name='CiFAR100')

        # 更新参数
        avg_test_accuracy = (test_accuracy_c10 + test_accuracy_c100) / 2
        trainer.adjust_hyperparameters(avg_test_accuracy)
        # 保存模型
        trainer.save_model(avg_test_accuracy, epoch)

    logging.info("Training completed.")
            


    

