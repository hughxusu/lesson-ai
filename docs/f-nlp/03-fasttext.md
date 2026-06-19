# fasttext工具

[fasttext](https://fasttext.cc/)是NLP工程领域常用的工具包，主要有两大功能：进行文本分类和训练词向量。

* 模型具有十分简单的网络结构。
* 训练和预测速度快。
* 可以使用n-gram特征提取以弥补模型缺陷提升精度。

fasttext安装

```python
pip install fasttext
```

## 文本分类

文本分类的是将文档（例如：电子邮件、新闻、评论等）分配给一个或多个类别。文本分类的种类：

* 二分类：文本被分类两个类别中，往往这两个类别是对立面，比如: 判断评论是好评还是差评。
* 单标签多分类：文本被分入到多个类别中，且每条文本只能属于某一个类别。比如：判断是哪种语言。
* 多标签多分类：文本被分人到多个类别中，但每条文本可以属于多个类别。比如: 某些新闻分类。

使用fasttext工具进行文本分类的过程：

1. 获取数据
2. 训练集与验证集的划分
3. 训练模型
4. 使用模型进行预测并评估
5. 模型调优
6. 模型保存与重加载

### 获取数据

使用[IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)验证fasttext的分类功能。下载kaggle数据集需要安装api工具包

```shell
pip install kagglehub
```

下载原始数据集

```python
import os
import kagglehub

# 指定下载路径
os.environ["KAGGLEHUB_CACHE"] = "./data/"  
path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
print("Path to dataset files:", path)  
```

fasttext格式要求：`__label__<tag> <text>`，将数据转换成fasttext格式

```python
import pandas as pd

def csv_to_fasttext(input_csv, output_file):
    df = pd.read_csv(input_csv)
    
    with open(output_file, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            label = f"__label__{row['sentiment']}"
            text = row['review'].replace("\n", " ").strip()
            f.write(f"{label} {text}\n")

output_file = "./data/imdb_fasttext.txt"
csv_to_fasttext(
    input_csv=f"{path}/IMDB Dataset.csv",  
    output_file=output_file 
)
```

### 划分数据集

```python
import random

with open(output_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

random.shuffle(lines)
split_idx = int(0.8 * len(lines))

train_file = "./data/imdb_train.txt"
valid_file = "./data/imdb_valid.txt"

with open(train_file, "w", encoding="utf-8") as f_train:
    f_train.writelines(lines[:split_idx])
    
with open(valid_file, "w", encoding="utf-8") as f_valid:
    f_valid.writelines(lines[split_idx:])
```

### 训练模型

[`fasttext.train_supervised`](https://fasttext.cc/docs/en/supervised-tutorial.html)用于文本分类的模型训练

```python
import fasttext

model = fasttext.train_supervised(
    input=train_file,
    epoch=25,          # 迭代次数
    lr=0.5,            # 学习率
    wordNgrams=2,      # 使用2-gram特征
    dim=100,           # 词向量维度
    loss='ova'         # 二分类推荐One-vs-All
)
```

### 验证模型性能

```python
print("验证集性能:", model.test(valid_file))
```

### 保存模型

```python
model_path = "./data/imdb_model.bin"
model.save_model(model_path) 
```

### 读取模型

```python
loaded_model = fasttext.load_model(model_path)
text = "This movie was a brilliant portrayal of human resilience"
labels, probs = loaded_model.predict(text, k=2)  # k=2返回前2个预测
print(f"预测: {labels[0]} (置信度: {probs[0]:.2f})")
```

### 模型调优

1. 对训练数据进行处理。
2. 选择不同的超参数进行实验。
3. `autotuneValidationFile='valid_file', autotuneDuration=600`自动超参数调优。

## 训练词向量

使用[`fasttext.train_unsupervised`](https://fasttext.cc/docs/en/unsupervised-tutorial.html)可以训练词向量，词向量的模式包括skipgram和cbow两种模式。

### 下载数据集

训练词向量的数据集选用[THUCNews](https://www.kaggle.com/datasets/trumanjagan/thucnews)数据集。THUCNews数据集包含多个类别，总体数量比较大，选用其中的娱乐类数据训练验证词向量的数据。由于该数据是原始新闻，针对词向量的训练需要预处理，包括：去掉标点符号和分词等。处理后的数据保存在[THUCNewsYuleTrainingWords](https://www.kaggle.com/datasets/hughxusu/thucnewsyuletrainingwords)中。

```python
os.environ["KAGGLEHUB_CACHE"] = "./data/"
path = kagglehub.dataset_download("hughxusu/thucnewsyuletrainingwords")

print("Path to dataset files:", path)
```

### 训练词向量

```python
import fasttext

input_file = f"{path}/THUCNews_yule.txt"
model = fasttext.train_unsupervised(
    input=input_file,
    model='skipgram',  # 可以是'skipgram'或'cbow'
    dim=300,           # 词向量维度
    ws=5,              # 上下文窗口大小
    epoch=5,           # 训练轮数
    minCount=5,        # 最小词频
)
```

### 模型的保存和加载

```python
output_model = "./data/THUCNews_yule.bin"
model.save_model(output_model)
loaded_word_model = fasttext.load_model("./data/THUCNews_yule.bin")
```

### 测试模型

```python
loaded_word_model.get_nearest_neighbors("演唱会")
loaded_word_model.get_nearest_neighbors("粉丝")
loaded_word_model.get_nearest_neighbors("周杰伦")
```

## 预训练词向量

在大型语料库上已经进行训练完成的词向量模型

* [157种语言的词向量](https://fasttext.cc/docs/en/crawl-vectors.html)
* [294种语言的词向量](https://fasttext.cc/docs/en/pretrained-vectors.html)

下载词向量

```python
from fasttext import util
util.download_model('zh', if_exists='ignore')
```

加载模型文件

```python
model = fasttext.load_model("cc.zh.300.bin")
model.words[:100]
loaded_word_model.get_nearest_neighbors("演唱会")
loaded_word_model.get_nearest_neighbors("粉丝")
loaded_word_model.get_nearest_neighbors("周杰伦")
```



