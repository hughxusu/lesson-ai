# K近邻 （KNN）

K近邻算法K-Nearest Neighbors（KNN）

1. 存在一定量的数据，包括特征和类别
2. 计算未知类别的数据与所有已知数据的距离。
3. 选择K个距离最小的样本，以最近的K个样本进行投票。

![](../_images/base/popular-knn-metrics-0.png)

KNN的基本思想是样本距离只够接近，样本的类型可以划分为一类。

已知样本数据集如下。

> [!tip]
>
> 根据下列样本数据实践KNN算法

```python
raw_data_x = [[1.4, 0.2],
              [1.7, 0.4],
              [1.5, 0.1],
              [1.9, 0.2],
              [1.6, 0.4],
              [4.7, 1.4],
              [4.9, 1.5],
              [4.0, 1.3],
              [4.4, 1.4],
              [3.9, 1.1]]
raw_data_y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
x = [4.2, 1.5]
```

绘制上述样本数据的二维图像

```python
import numpy as np
import matplotlib.pyplot as plt

x_train = np.array(raw_data_x)
y_train = np.array(raw_data_y)
t = np.array(row_text)

plt.scatter(x_train[y_train == 0, 0], x_train[y_train == 0, 1], color='g')
plt.scatter(x_train[y_train == 1, 0], x_train[y_train == 1, 1], color='r')
plt.scatter(t[0], t[1], color='b')
plt.show()
```

使用欧拉距离来表示两个样本点之间的差异，对于 $n$ 维向量 $x$​ 其距离公式为，欧拉距离为：
$$
\sqrt{\sum_{i=1}^n\left(x_i^{(a)}-x_i^{(b)} \right)^2}
$$

1. 计算样本间所有距离

```python
from math import sqrt

distances = []
for x_i in x_train:
    d = sqrt(np.sum((x_i - t) ** 2))
    distances.append(d)
    
# 使用列表生成式来计算全部距离
distances = [sqrt(np.sum((x_i - t) ** 2)) for x_i in x_train]
```

2. 对全部距离进行排序

```python
nearest = np.argsort(distances)
```

3. 选择最近的K个样本，并获取相应的标签

```python
k = 7
top_k_y = [y_train[i] for i in nearest[:k]]
```

4. 统计标签结果

```python
from collections import Counter
votes = Counter(top_k_y)
most = votes.most_common(1)
predict = most[0][0]
```

[`Counter`类是一个，用于数据统计](https://docs.python.org/zh-cn/3.9/library/collections.html?highlight=collections#counter-objects)

将上述算法过程封装成为一个函数

```python
import numpy as np
from math import sqrt
from collections import Counter

def knn_classify(k, x_train, y_train, x):
    assert 1 <= k <= x_train.shape[0], 'k must be valid'
    assert x_train.shape[0] == y_train.shape[0], 'the size of x_train must be equal to the size of y_train'
    assert x_train.shape[1] == x.shape[0], 'the feature number of x must be equal to x_train'
    
    distances = [sqrt(np.sum((x_i - x) ** 2)) for x_i in x_train]
    nearest = np.argsort(distances)
    
    top_k_y = [y_train[i] for i in nearest[:k]]
    votes = Counter(top_k_y)
    
    return votes.most_common(1)[0][0]
```







> [!warning]
>
> K 值的取值一般基于经验。




$$
\sqrt{\sum_{i=1}^n\left(x_i^{(a)}-x_i^{(b)} \right)^2}
$$

> [!attention]
>
> KNN算法是一个不需要训练过程的算法，可以认为KNN算法发的模型就是全部训练数据本身。

```python
raw_data_x = [[3.393533211, 2.331273381],
              [3.110073483, 1.781539638],
              [1.343808831, 3.368360954],
              [3.582294042, 4.679179110],
              [2.280362439, 2.866990263],
              [7.423436942, 4.696522875],
              [5.745051997, 3.533989803],
              [9.172168622, 2.511101045],
              [7.792783481, 3.424088941],
              [7.939820817, 0.791637231]]
raw_data_y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

x_train = np.array(raw_data_x)
y_train = np.array(raw_data_y)

x = np.array([8.093607318, 3.365731514])

from sklearn.neighbors import KNeighborsClassifier
kNN_classifier = KNeighborsClassifier(n_neighbors=6)
kNN_classifier.fit(x_train, y_train)
x_predict = x.reshape(1, -1)
y_predict = kNN_classifier.predict(x_predict)
print(predict_y)
```

> [!attention]
>
> 1. KNN算法中没有模型参数。
> 2. KNN算法中的K是典型的超参数

手写数值识别

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn import datasets

digits = datasets.load_digits()
print(digits.keys())
print(digits.DESCR)

x = digits.data
print(x.shape)

y = digits.target
print(y.shape)

print(digits.target_names)
print(y[:100])

print(x[:10])

some_digit = x[666]
some_digit_image = some_digit.reshape(8, 8)
plt.imshow(some_digit_image, cmap=matplotlib.cm.binary)
plt.show()

print(y[666])
```

使用KNN算法测试手写数字识别

```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=666)

from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(x_train, y_train)
y_predict = knn_clf.predict(x_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_predict)

knn_clf.score(x_test, y_test)
```

[KNeighborsClassifier 说明文档](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier)

## 超参数

### 近邻数K

```python
best_score = 0.0
best_k = -1
for k in range(1, 11):
    knn_clf = KNeighborsClassifier(n_neighbors=k)
    knn_clf.fit(x_train, y_train)
    score = knn_clf.score(x_test, y_test)
    if score > best_score:
        best_k = k
        best_score = score
        
print('best_k =', best_k)
print('best_score =', best_score)
```

### 距离权重

<img src="https://raw.githubusercontent.com/hughxusu/lesson-ai/developing/_images/knn/1681090-20230523083454966-1157335194.png" style="zoom: 67%;" />

对于投票的KNN算法，预测的点属于蓝色类。

> [!warning]
>
> 投票类KNN算法忽略了，样本点之间的距离的影响。更进一步的KNN算法，增加了距离权重的参数，权重等于距离的倒数。

$$
\text{Red}=1\\
\text{Blue}=\frac{1}{3}+\frac{1}{4}=\frac{7}{12}
$$



```python
best_method = ''
best_score = 0.0
best_k = -1
for method in ['uniform', 'distance']:
    for k in range(1, 11):
        knn_clf = KNeighborsClassifier(n_neighbors=k, weights=method)
        knn_clf.fit(x_train, y_train)
        score = knn_clf.score(x_test, y_test)
        if score > best_score:
            best_k = k
            best_score = score
            best_method = method

print('best_k =', best_k)
print('best_score =', best_score)
print('best_method =', best_method)
```

#### 距离类型

1. 欧拉距离与曼哈顿距离

![](https://raw.githubusercontent.com/hughxusu/lesson-ai/developing/_images/knn/1541640266.png)
$$
d=\sqrt{\sum_{i=1}^n\left(x_i^{(a)}-x_i^{(b)} \right)^2} \\
d=\sum_{i=1}^N|x_i-y_i|
$$

2. 明可夫斯基距离

$$
d=\left(\sum_{i=1}^N|x_i-y_i|^p\right)^{\frac{1}{p}}
$$

> [!warning]
>
> $p$ 可以认为是一个超参数

```python
best_p = -1
best_score = 0.0
best_k = -1

for k in range(1, 11):
    for p in range(1, 6):
        knn_clf = KNeighborsClassifier(n_neighbors=k, weights='distance', p=p)
        knn_clf.fit(x_train, y_train)
        score = knn_clf.score(x_test, y_test)
        if score > best_score:
            best_k = k
            best_score = score
            best_p = p
            
print('best_k =', best_k)
print('best_score =', best_score)
print('best_p =', best_p)
```

3. 其他距离
   * 向量空间余弦相似度 Cosine Similarity
   * 调整余弦相似度 Adjust Cosine Similarity
   * 皮尔逊相关系数 Pearson Correlation Coefficient
   * Jaccard相似系数 Jaccard Coefficient

## 网格搜索

使用sklearn的网格搜索来选择最优参数

```python
param_grid = [
    {
        'weights': ['uniform'],
        'n_neighbors': [i for i in range(1, 11)]
    },
    {
        'weights': ['distance'],
        'n_neighbors': [i for i in range(1, 11)],
        'p': [i for i in range(1, 6)]
    }
]

knn_clf = KNeighborsClassifier()

from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(knn_clf, param_grid)
grid_search.fit(x_train, y_train)

print(grid_search.best_estimator_)
print(grid_search.best_score_)
print(grid_search.best_params_)

knn_clf = grid_search.best_estimator_
knn_clf.score(x_test, y_test)

grid_search = GridSearchCV(knn_clf, param_grid, n_jobs=-1, verbose=2) # n_jobs多核运行 verbose打印搜索信息
grid_search.fit(x_train, y_train)
```

## KNN算法特点

* 可以解决分类问题（包括多分类问题）

* 使用k近邻算法可以解决回归问题，取K个近邻的平均值，或加权平均值。

* k近邻算法的计算效率低

* k近邻算法预测结果不具有可解释性

* k近邻算法容易陷入位数灾难

  
