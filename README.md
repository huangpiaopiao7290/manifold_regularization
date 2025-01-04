## 目录


## 环境准备

- 安装[miniconda3](wget "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe" -outfile "./Downloads/Miniconda3-latest-Windows-x86_64.exe")，将项目中的environment.yml移动到合适的位置（环境目录），本项目环境配置如下。
```shell
conda env create -f environment.yml
conda activate dl
conda list
```

## 结构
```  
data  
|-processed  
|—raw  
├── cifar  
│   ├── cifar-10  
│   │   └── cifar-10-python.tar.gz  
│   └── cifar-100  
│       └── cifar-100-python.tar.gz  
├── imdb  
│   └── aclImdb_v1.tar.gz  
├── svhn  
│   ├── test.tar.gz  
│   └── train.tar.gz  
└── yahooAnswers  
    └── yahoo_answers_csv.tar.gz
```



> IMDB

1. 将imdb的三个xls文件（训练集、测试集、验证集）重新保存（目前实际是TSV文件）
2. 读取文件数据，且构建语料库将文本文件映射成数值数据
- 语料库
	1. 词频统计： 通过 `Counter` 来统计词频，并只选择最常见的词来构建词汇表
	2. 词汇表大小：设置了最大词汇表大小 `max_vocab_size`，确保不会过大
	3. 特殊标记：保证特殊标记如 `<PAD>` 和 `<UNK>` 不会被其他词覆盖
	4. 词嵌入层：创建了一个带有适当参数的 `nn.Embedding` 层，包括 `padding_idx` 参数，以便正确处理填充标记
3. 创建TextDataset，定义数据加载器
4. 创建TextCNN: [[https://arxiv.org/abs/1408.5882v2]]
5. 训练数据
- 集成PWMRloss到IMDB
	1. 修改TextCNN，添加特征输出
	2. 在train中添加自定义损失函数
	3. 样本测试程序已跑通
	4. 查看测试结果
```python
# 查看imdb测试结果-tensorboard

```
- 配置环境
