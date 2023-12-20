

<img src="https://datatechlog.com/wp-content/uploads/2020/06/Pandas_logo.png" style="zoom: 33%;" />

# Pandas

- 2008年WesMcKinney开发出的库
- 专门用于数据挖掘的开源python库
- 以Numpy为基础，借力Numpy模块在计算方面性能高的优势
- 基于matplotlib，能够简便的画图
- 独特的数据结构

为什么使用Pandas

* 增强图表可读性
* 便捷的数据处理能力
* 读取文件方便
* 封装了Matplotlib、Numpy的画图和计算

## Pandas数据结构

Pandas中一共有三种数据结构，分别为：Series、DataFrame。其中Series是一维数据结构，DataFrame是二维的表格型数据结构。

### Series

Series是一个类似于一维数组的数据结构，它能够保存任何类型的数据，比如整数、字符串、浮点数等，主要由一组数据和与之相关的索引两部分构成。

`pd.Series(data, index, dtype)`

* data：传入的数据，可以是ndarray、list等。
* index：索引，必须是唯一的，且与数据的长度相等。如果没有传入索引参数，则默认会自动创建一个从0-N的整数索引。
* dtype：数据的类型。

```python
import pandas as pd
import numpy as np

pd.Series(np.arange(10))
pd.Series([6.7,5.6,3,10,2], index=[1,2,3,4,5])

color_count = pd.Series({'red':100, 'blue':200, 'green': 500, 'yellow':1000})
color_count

color_count.index # 数据索引
color_count.values # 数据
color_count[2] # 获得数据
```

### DataFrame

DataFrame是一个类似于二维数组或表格的对象，既有行索引，又有列索引

* 行索引，表明不同行，横向索引，叫index，0轴，axis=0
* 列索引，表名不同列，纵向索引，叫columns，1轴，axis=1

![](https://media.geeksforgeeks.org/wp-content/cdn-uploads/creating_dataframe1.png)

`pd.DataFrame(data, index, columns)`

* index：行标签。如果没有传入索引参数，则默认会自动创建一个从0-N的整数索引。
* columns：列标签。如果没有传入索引参数，则默认会自动创建一个从0-N的整数索引。

```python
score = np.random.randint(40, 100, (10, 5))
score_df = pd.DataFrame(score)

subjects = ["语文", "数学", "英语", "政治", "体育"]
stu = ['同学' + str(i) for i in range(score_df.shape[0])]
data = pd.DataFrame(score, columns=subjects, index=stu)
```

DataFrame的属性

```python
data.shape 
data.index # DataFrame的行索引列表
data.columns # DataFrame的列索引列表
data.values # DataFrame数据
data.T
data.head(5) # 显示前5行内容
data.tail(5) # 显示后5行内容

stu = ["学生_" + str(i) for i in range(score_df.shape[0])]
data.index = stu # 修改索引

# 重置索引
data.reset_index()
data.reset_index(drop=True) # 去掉圆索引


# 使用字典创建数据
df = pd.DataFrame({'month': [1, 4, 7, 10],
                    'year': [2012, 2014, 2013, 2014],
                    'sale':[55, 40, 84, 31]})

df.set_index('month') # 将某一列修改为索引
df.set_index(['year', 'month'])
```

## 基本数据操作

读取示例数据

```python
data = pd.read_csv("./data/stock_day.csv")
data = data.drop(["ma5","ma10","ma20","v_ma5","v_ma10","v_ma20"], axis=1)
```

### 索引操作

Numpy当中我们已经讲过使用索引选取序列和切片选择，pandas也支持类似的操作，也可以直接使用列名、行名

称，甚至组合使用。

```python
data['open']['2018-02-27'] # 直接使用行列索引名字的方式（先列后行）
```

切片索引

```python
data.loc['2018-02-27':'2018-02-22', 'open'] # 使用loc:只能指定行列索引的名字
data.iloc[:3, :5]

data.loc[data.index[0:4], ['open', 'close', 'high', 'low']]
data.iloc[0:4, data.columns.get_indexer(['open', 'close', 'high', 'low'])]
```

赋值操作

```python
data['close'] = 1
data.close = 1
```

排序

```python
data.sort_values(by="open", ascending=True).head() # 指定排序方式
data.sort_values(by=['open', 'high']) # 按照多个键进行排序
data.sort_index() # 给索引进行排序
```

Series排序

```python
data['p_change'].sort_values(ascending=True).head() # 只有一列
data['p_change'].sort_index().head() # 对索引排序
```

## DataFrame运算

算数运算

```python
# 加减法
data['open'].add(1)
data['open'].sub(1) 
```

逻辑运算

```python
data['open'] > 23

# 使用逻辑运算筛选数据
data[data["open"] > 23].head()
data[(data["open"] > 23) & (data["open"] < 24)].head()
```

逻辑运算函数

```python
data.query("open<24 & open>23").head() # 查询结构
data[data["open"].isin([23.53, 23.85])] # 判断数据是否存在
```

## 统计运算

综合分析: 能够直接得出很多统计结果`count`, `mean`, `std`, `min`, `max` 等

```python
data.describe()
```

对于单个函数去进行统计的时候，坐标轴还是按照默认列“columns” (axis=0, default)，如果要对行“index” 需要指定(axis=1)

```python
data.max(0) # 最大值
data.var(0) # 方差
data.std(0) # 标准差
data.median(0) # 中位数

data.idxmax(0) # 求出最大值的位置
data.idxmin(0) # 求出最小值的位置
```

自定义运算

```python
data[['open', 'close']].apply(lambda x: x.max() - x.min(), axis=0)
```

## 文件读取

Pandas 支持多种文档的存储和读取

<img src="https://pic2.zhimg.com/80/v2-da9199587b07a792a5af6e09aafd6899_1440w.webp" style="zoom:67%;" />

### CSV 文件

```python
data = pd.read_csv("./data/stock_day.csv", usecols=['open', 'close']) # 读取指定列数据
data[:10].to_csv("./data/test.csv", columns=['open']) # 保存数据到csv文件
data[:10].to_csv("./data/test.csv", columns=['open'], index=False) # 保存数据删除索引
```

### HDF5文件

HDF5文件的读取和存储需要指定一个键，值为要存储的DataFrame

```python
data.to_hdf('./test.h5', key='data') # 保存 HDF5格式 key存储标识符
data.read_hdf('./test.h5', key='data') # 读取文件
```

### JSON

JSON是一种前后端的交互经常用到数据格式，是一种键值对形式。

```python
json_read.to_json("./data/test.json", orient='records', lines=True) # 保存文件 orient josn 的存储形式 line 是否为一个对象一行
json_data = pd.read_json('./test.json') # 读取json文件
```

## 缺失值处理

```python
movie = pd.read_csv("./data/IMDB-Movie-Data.csv")

pd.notnull(movie) # 判断是否存在确实值
pd.isnull(movie) 
```

缺失值处理

```python
data = movie.dropna() # 移除缺失值
movie['Revenue (Millions)'].fillna(movie['Revenue (Millions)'].mean(), inplace=True) # 用平均值替换缺失值

data.replace(to_replace='?', value=np.nan) # 替换部分值
```

## 数据离散化

连续属性离散化的目的是为了简化数据结构，数据离散化技术可以用来减少给定连续属性值的个数。离散化方法经常作为数据挖掘的工具。

```python
qcut = pd.qcut(p_change, 10) # 自行分组
qcut.value_counts() # 计算到每个分组中的数据数目

bins = [-100, -7, -5, -3, 0, 3, 5, 7, 100]
p_counts = pd.cut(p_change, bins) # 自己指定分组区间

dummies = pd.get_dummies(p_counts, prefix="rise") # 生成 one-hot 编码
```

## 数据合并

将多张表中的数据合并后一起分析

```python
pd.concat([data, dummies], axis=1) # 按行索引
```

数据合并

```python
left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
                        'key2': ['K0', 'K1', 'K0', 'K1'],
                        'A': ['A0', 'A1', 'A2', 'A3'],
                        'B': ['B0', 'B1', 'B2', 'B3']})

right = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
                        'key2': ['K0', 'K0', 'K0', 'K0'],
                        'C': ['C0', 'C1', 'C2', 'C3'],
                        'D': ['D0', 'D1', 'D2', 'D3']})

result = pd.merge(left, right, on=['key1', 'key2']) # 内链接
result = pd.merge(left, right, how='left', on=['key1', 'key2']) # 左链接
result = pd.merge(left, right, how='right', on=['key1', 'key2']) # 右链接
result = pd.merge(left, right, how='outer', on=['key1', 'key2']) # 外链接
```

## 案例

数据来源：https://www.kaggle.com/damianpanek/sunday-eda/data

> [!tip]
>
> 1. 我们想知道这些电影数据中评分的平均分，导演的人数等信息，我们应该怎么获取？
> 2. 对于这一组电影数据，如果我们想rating，runtime的分布情况，应该如何呈现数据？
> 3. 对于这一组电影数据，如果我们希望统计电影分类(genre)的情况，应该如何处理数据？

读取数据

```python
import pandas  as pd
import numpy as np
from matplotlib import pyplot as plt

path = "./IMDB-Movie-Data.csv"
df = pd.read_csv(path)
```

案例 1

```python
df["Rating"].mean() # 得出评分的平均分
np.unique(df["Director"]).shape[0] # 得出导演人数信息
```

案例 2

```python
# 绘制得分直方图
plt.figure(figsize=(20,8),dpi=80)

max_ = df["Rating"].max()
min_ = df["Rating"].min()

t1 = np.linspace(min_,max_,num=21)
plt.xticks(t1)

plt.grid()
plt.hist(df["Rating"].values,bins=20)

plt.show()

# 绘制时间分布图
plt.figure(figsize=(20,8),dpi=80)
plt.hist(df["Runtime (Minutes)"].values,bins=20)

max_ = df["Runtime (Minutes)"].max()
min_ = df["Runtime (Minutes)"].min()

t1 = np.linspace(min_,max_,num=21)
plt.xticks(t1)
plt.grid()

plt.show()
```

案例 3

```python
temp_list = [i.split(",") for i in df["Genre"]]
genre_list = np.unique([i for j in temp_list for i in j])

for i in range(1000):
    temp_df.loc[i, temp_list[i]]=1
    
temp_df.sum().sort_values(ascending=False).plot(kind="bar",figsize=(20,8),fontsize=20,colormap="cool")
```

