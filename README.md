##  decode dataset

## init
在根目录下新增folder：data/raw (存放原始数据集)
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


smooth_loss

$$
    Loss = Tr(DFLF^{T})
$$
D: 对角阵, 局部密度d  
F：特征矩阵(N, D) 节点数N 特征维度D
L：图拉普拉斯矩阵(N, N)


