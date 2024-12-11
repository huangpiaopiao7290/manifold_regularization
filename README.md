# 最新代码见branch-2.0


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

