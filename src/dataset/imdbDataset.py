import os
from collections import Counter
from typing import List, Dict

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from src.models.textCNN import TextCNN
from src.utils.lossFunction_PWMR import LossFunctionsPWMR

# =============================================

MAX_WORDS = 10000   # 词汇表大小
MAX_LEN = 200       # max length
BATCH_SIZE = 128
EMB_SIZE = 128      # embedding size
HID_SIZE = 128      # lstm hidden size
DROPOUT = 0.2

# ==============================================

def read_data(filepath, num=None) -> tuple[list, list]:
    """
    读取数据
    @param filepath: 文件路径
    @param num: 读取数据量
    """
    data = pd.read_excel(filepath, engine='xlrd')
    df = pd.DataFrame()
    if num is not None:
        df = data.iloc[:num]

    texts = df['text'].tolist()
    labels = df['label'].tolist()

    return texts, labels

def build_corpus(texts, embedding_num, max_vocab_size=MAX_WORDS):
    """
    构建语料库
    """
    # 初始化特殊标记
    word2index = {"<PAD>": 0, "<UNK>": 1}

    # 统计词频
    word_counts = Counter()
    for text in texts:
        word_counts.update(text)

    # 根据词频构建词汇表，保留最常见的max_vocab_size - len(special_tokens)个词
    special_tokens = ["<PAD>", "<UNK>"]
    vocab_size = min(max_vocab_size, len(word_counts) + len(special_tokens))
    most_common_words = [word for word, _ in word_counts.most_common(vocab_size - len(special_tokens))]

    # 构建词汇表
    for idx, word in enumerate(most_common_words, start=len(special_tokens)):
        word2index[word] = idx

    # 创建词嵌入层
    embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_num,
                                   padding_idx=word2index["<PAD>"])

    return word2index, embedding_layer


class TextDataset(Dataset):
    def __init__(self, texts: List[List[str]], labels: List[int], word2index: Dict[str, int], max_len: int) -> None:
        """
        @param texts:
        @param labels:
        @param word2index:
        @param max_len: 文本长度限制
        """
        self.texts: list = texts
        self.labels: list = labels
        self.word2index: dict = word2index
        self.max_len: int = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        # 获取文本及其标签
        text = self.texts[index][:self.max_len]
        label = int(self.labels[index])
        # 转成对应数值数据
        text2number = [self.word2index.get(i, 1) for i in text]
        text2number = text2number + [0] * (self.max_len - len(text2number))
        text2number = torch.tensor(text2number).unsqueeze(dim=0)

        return text2number, label


# Training function
def train(model, train_loader, valid_loader, test_loader, epochs, loss_fn, device, writer):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx_, batch_label_ in train_loader:
            batch_idx_ = batch_idx_.to(device)
            batch_label_ = batch_label_.to(device)
            optimizer.zero_grad()
            outputs_ = model.forward(batch_idx_)
            loss, _, _ = loss_fn.total_loss(model, outputs_, batch_idx_, batch_label_,
                                               unlabeled_mask=(batch_label_ == -1),
                                               lambda_c=0.2, lambda_s=0.8)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        writer.add_scalar('Train/Loss', avg_train_loss, epoch)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_idx_, batch_label_ in valid_loader:
                batch_idx_ = batch_idx_.to(device)
                batch_label_ = batch_label_.to(device)
                outputs_ = model.forward(batch_idx_)
                loss, _, _ = loss_fn.total_loss(model, outputs_, batch_idx_, batch_label_,
                                                    unlabeled_mask=(batch_label_ == -1),
                                                    lambda_c=0.9, lambda_s=1.1)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(valid_loader)
        writer.add_scalar('Validation/Loss', avg_val_loss, epoch)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")


        # Test accuracy calculation and logging
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, batch_label in test_loader:
                batch_idx = batch_idx.to(device)
                batch_label = batch_label.to(device)
                outputs = model_(batch_idx)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_label.size(0)
                correct += (predicted == batch_label).sum().item()

        accuracy = 100 * correct / total
        writer.add_scalar('Test/Accuracy', accuracy, epoch)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Test Accuracy: {accuracy:.3f}%")

        # Optionally, you can also log other information like histograms of weights
        for name, param in model.named_parameters():
            if 'weight' in name:
                writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)


if __name__ == '__main__':

    # 设备配置
    device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    file_paths = {
        'train': 'C:\\piao_programs\\py_programs\\DeepLearningProject\\Manifold_SmiLearn\\data\\processed\\imdb\\Train.xls',
        'valid': 'C:\\piao_programs\\py_programs\\DeepLearningProject\\Manifold_SmiLearn\\data\\processed\\imdb\\Valid.xls',
        'test': 'C:\\piao_programs\\py_programs\\DeepLearningProject\\Manifold_SmiLearn\\data\\processed\\imdb\\Test.xls'
    }

    ## TODO 样本数量目前只有500用于测试
    train_texts, train_labels = read_data(file_paths['train'], 500)
    valid_texts, valid_labels = read_data(file_paths['valid'], 500)
    test_texts, test_labels = read_data(file_paths['test'], 500)

    words2index, wordsEmbedding = build_corpus(train_texts + valid_texts, EMB_SIZE)

    train_dataset = TextDataset(train_texts, train_labels, words2index, MAX_LEN)
    valid_dataset = TextDataset(valid_texts, valid_labels, words2index, MAX_LEN)
    test_dataset = TextDataset(test_texts, test_labels, words2index, MAX_LEN)

    trainLoader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
    validLoader = DataLoader(valid_dataset, BATCH_SIZE, shuffle=False)
    testLoader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False)

    # Tensorboard位置
    log_dir = os.path.join(os.getcwd(), 'log')
    log_dir_tensorboard = os.path.join(log_dir, 'logdir/imdb')
    writer_ = SummaryWriter(log_dir=log_dir_tensorboard)
    # 创建模型
    num_classes = len(set(train_labels))  # 类别
    model_ = TextCNN(wordsEmbedding, num_classes=num_classes, max_len=MAX_LEN).to(device_)

    # 损失函数
    lossFunc = LossFunctionsPWMR(device=device_)

    train(model=model_,
          train_loader=trainLoader,
          valid_loader=validLoader,
          test_loader=testLoader,
          epochs=100,
          loss_fn=lossFunc,
          device=device_,
          writer=writer_)


    writer_.close()

