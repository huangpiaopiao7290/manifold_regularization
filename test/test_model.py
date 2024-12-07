# test svhn

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from src.dataset.svhnDataset import SVHNDataset
from src.models.vgg import VggNet
from src.utils.lossFunction_PWMR import LossFunctionsPWMR

def evaluate_model(model, test_loader, loss_fn, device):
    """
    Evaluate the model on the test dataset.

    Args:
        model (nn.Module): The neural network model.
        test_loader (DataLoader): DataLoader for the test set.
        loss_fn (LossFunctionsPWMR): Instance of the loss functions class.
        device (torch.device): Device to run the evaluation on ('cpu' or 'cuda').

    Returns:
        float: Accuracy of the model on the test set.
        float: Average total loss on the test set.
    """
    model.eval()  # 设置模型为评估模式
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():  # 禁用梯度计算以节省内存和加速计算
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 计算总损失
            if -1 not in labels:  # 如果批次中包含标签
                unlabeled_mask = torch.zeros_like(labels, dtype=torch.bool)
                lambda_c = 0.5  # 一致性损失权重
                lambda_s = 0.5  # 平滑性损失权重
                batch_loss, _, _ = loss_fn.total_loss(model, outputs, images, labels, unlabeled_mask, lambda_c, lambda_s)
                total_loss += batch_loss.item()

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else 0
    print(f'Accuracy of the model on the test images: {accuracy:.2f}%')
    print(f'Average total loss on the test images: {avg_loss:.4f}')
    return accuracy, avg_loss


def main():
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 定义转换（测试集不使用数据增强）
    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # 加载测试数据集
    svhn = "C:\\piao_programs\\py_programs\\DeepLearningProject\\Manifold_SmiLearn\\data\\processed\\svhn\\test"
    test_dataset = SVHNDataset(root_dir=svhn, transform_=test_transform, labeled=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 初始化模型并加载预训练权重（如果有的话）
    model = VggNet(num_class=10).to(device)
    # 如果有保存的模型权重，请取消注释以下行并提供正确的路径
    # model.load_state_dict(torch.load('path_to_your_model.pth'))

    # 初始化损失函数
    loss_fn = LossFunctionsPWMR(device=device)

    # 评估模型
    evaluate_model(model, test_loader, loss_fn, device)


if __name__ == "__main__":
    main()