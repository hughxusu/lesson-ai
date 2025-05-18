### 手写数字识别

scikit-learn测试数据集中包含一个图像数据集`load_digits`

1. 有1797个数据，每个数据是64维特征值，表示一个 $8\times8$ 大大小的图像。
2. 共有10个类别的数据，分别是数字0~9。

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn import datasets

digits = datasets.load_digits()
print(digits.DESCR)

x = digits.data
print(x.shape)
y = digits.target
print(y.shape)
print(digits.target_names)

print(y[:70])
print(x[:1])
```

绘制其中一个元素的图像，使用[`imshow`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html)可以绘制图像

```python
import matplotlib
import matplotlib.pyplot as plt

some_digit = x[0]
print(y[0])
some_digit_image = some_digit.reshape(8, 8)
plt.imshow(some_digit_image, cmap=matplotlib.cm.binary)
plt.show()
```







## 特征归一化

判断肿瘤是良性还是恶性

|       | 肿瘤大小（厘米） | 发现时间（天） | 发现时间（年） |
| ----- | ---------------- | -------------- | -------------- |
| 样本1 | 1                | 200            | 0.55年         |
| 样本2 | 5                | 100            | 0.27年         |

数据不同维度的量纲不同，会直接影响数据距离的计算。

1. 当发现时间的单位为天时，样本间的距离被发现时间所主导。
2. 当发现时间的单位为年时，样本间的距离被肿瘤大小所主导。

数据归一化是将所有不同量纲的数据，映射在一个尺度下。

> [!warning]
>
> 理论上可以证明，归一化数据不影响分类结果，但可以加快学习速率。

### 最值归一化

把所有的数据映射到0~1之间
$$
x_{\text{sacle}}=\frac{x-x_{\min}}{x_{\max}-x_{\min}}
$$
适用于分布有明显边界特征，受异常值影响比较大，如：数据集 $1,2,3, 1000, …$

* 学生考试成绩 $[0, 100]$
* 图像像素点 $[0, 255]$​

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.random.randint(0, 100, size=20)
print((x - np.min(x)) / (np.max(x) - np.min(x)))
```

### 均值方差归一化

把所有的数据归一到平均值为0方差为1的分布中，适用于数据分布没有明显边界
$$
x_{\text{sacle}}=\frac{x-\mu}{\sigma}
$$

> [!warning]
>
> 如果数据没有明显的边界，一般都采用均值方差归一化方法。

```python
x = np.random.randint(0, 100, size=20)
print((x - np.mean(x)) / np.std(x))
```

### `Scaler`归一化工具

<img src="https://raw.githubusercontent.com/hughxusu/lesson-ai/developing/_images/base/原始数据.png" style="zoom: 50%;" />

1. 真实数据无法获得均值和方差。
2. 采用均值方差归一化是要保留测试数据的均值和方差，用于处理预测数据。预测时，预测数据同样需要归一化。对数据的归一化也可以理解为算法的一部分。

在scikit-learn中可以借助`Scaler`工具完成特征值归一化和均值方差保存的工作。

1. `StandardScaler`均值方差归一化

```python
# 导入鸢尾花数据集
import numpy as np
from sklearn import datasets
iris = datasets.load_iris()
x = iris.data
y = iris.target
print(x[:5])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=666)

# 对数据进行归一化处理
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=666)
standardScaler = StandardScaler()
standardScaler.fit(x_train)
print(standardScaler.mean_)
print(standardScaler.scale_)

x_train = standardScaler.transform(x_train)
x_test = standardScaler.transform(x_test)
print(x_train[:5])
```

1. `standardScaler.fit`函数可以对数据进行规划化，计算出均值和方差。
2. `standardScaler.mean_`和`standardScaler.scale_`计算出的均值和方差。
3. `standardScaler.transform`对训练数据和测试数据进行规一化处理。

绘制归一化后的鸢尾花数据图像

```python
import matplotlib.pyplot as plt

name = ['Setosa', 'Versicolour', 'Virginica']

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
for target in set(y_train):
    subset = x_train[y_train == target]
    plt.scatter(subset[:, 2], subset[:, 3], label=f'{name[target]}')
    
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.legend()

plt.subplot(1, 2, 2)
for target in set(y_train):
    subset = train_scale[y_train == target]
    plt.scatter(subset[:, 2], subset[:, 3], label=f'{name[target]}')
    
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.legend()
plt.show()
```

使用归一化后的特征进行模型训练和预测

```python
from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(train_scale, y_train)
print(knn_clf.score(test_scale, y_test))
```

> [!tip]
>
> scikit-learn的最值归一化封装在[`MinMaxScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)中，使用最值归一化处理手写数字识别数据，并测试KNN模型的性能。







[!warning]

理论上可以证明，归一化数据不影响分类结果，但可以加快学习速率。






