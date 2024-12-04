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


- conda环境配置:   
dl_environment.yml 
- 先安装miniconda3或anaconda
```c
// 创建conda环境
conda env create -f environment.yml

conda activate dl
conda list
```

---

- 创建cifar100训练集、验证集、测试集
```shell
python src/dataset/cifar100Peocessor.py
```
- 加载cifar100
```pythonregexp
ciFar100Dataset.py
```


- 加载imdb数据集


- 预处理雅虎评论数据集
- 加载YaHuDataset




---

1. 将所有数据单独放置，不在本项目中，可通过配置文件的方式访问
2. 目录管理需要重新配置



