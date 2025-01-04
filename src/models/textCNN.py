import torch
import torch.nn as nn


class Block(nn.Module):
    """
    convolution
    activation
    max-pooling
    """
    def __init__(self,kernel_dim1, embedding_num, max_len):
        super(Block, self).__init__()
        """
        Args:
            kernel_dim1:
            embedding_num:

        Example:
            [5, 5] --kernel(dim1=2,embedding=5)--> [5 - 2 + 1, 5 - 5 + 1] => [4, 1]

        """
        # batch * channel * max_len * embedding  (文本处理时将通道数默认为1)
        self.con = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(kernel_dim1, embedding_num))
        self.act = nn.ReLU()
        self.maxPooling = nn.MaxPool1d(kernel_size=(max_len - kernel_dim1 + 1))

    def forward(self, batch_emd):
        convolution = self.con(batch_emd)
        activation = self.act(convolution).squeeze(dim=-1)
        max_pooling = self.maxPooling(activation).squeeze(dim=-1)

        return max_pooling


class TextCNN(nn.Module):
    def __init__(self, embedding_matrix, num_classes, max_len):
        super(TextCNN, self).__init__()
        self.embedding_num = embedding_matrix.weight.shape[1]
        self.block1 = Block(2, self.embedding_num, max_len)
        self.block2 = Block(3, self.embedding_num, max_len)
        self.block3 = Block(4, self.embedding_num, max_len)

        self.embedding_matrix = embedding_matrix

        # TODO 分类器参数要修改
        self.classifier = nn.Linear(6, num_classes)

        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, batch_idx, batch_labels=None):

        feature = self.extract_features(batch_idx)
        pre = self.classifier(feature.squeeze(-1))

        if batch_labels is not None:
            loss = self.loss_func(pre, batch_labels)
            return loss
        else:
            return torch.argmax(pre, dim=-1)

    def extract_features(self, batch_idx):
        # batch_embedding = self.embedding_matrix(x).unsqueeze(1)
        batch_embedding = self.embedding_matrix(batch_idx)
        result_block1 = self.block1(batch_embedding)
        result_block2 = self.block2(batch_embedding)
        result_block3 = self.block3(batch_embedding)
        # 特征拼接
        feature = torch.cat([result_block1, result_block2, result_block3], dim=1)
        return feature

