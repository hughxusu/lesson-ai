# 主成分分析

主成分分析（Principal Component Analysis）是一种多变量统计分析技术。它的主要目的是通过线性变换，将原始数据的多个变量（特征）转换为一组新的、数量较少的变量，这些新变量被称为主成分。

* 非监督的机器学习算法。
* 主要用于数据降维。
* 通过降维，可以发现更便于人类理解的特征。
* 可视化、去噪。

主成分分析并不只应用在机器学习领域，也是**统计分析领域**的重要方法。

<img src="https://raw.githubusercontent.com/hughxusu/lesson-ai/develop/images/base/Xnip2025-06-03_18-12-33.jpg" style="zoom:85%;" />

> [!warning]
>
> PCA降维的标准是，在降维过程中信息损失最小，即方差最大。

PCA的操作步骤：

1. 去中心化，把坐标原点放在数据中心。
1. 找到方差最大方向，这个方向是第一主成分。
1. 方向与第一主成分正交（垂直），且在剩余方差中最大，是第二主成分。
1. 以此类推可以得到剩余主成分。

PCA的本质就是将原空间坐标系，变换到新的坐标系中，取出前$k$个重要的主成分，就可以在$k$个轴上获得一个低维的数据信息。降维后的数据丢失了部分信息。

> [!note]
>
> [怎样找到方差的最大方向？]( https://www.bilibili.com/video/BV1E5411E71z/?share_source=copy_web&vd_source=aa661569ff3138d0b604d53a96184bf2)

## sklearn的主成分分析

导入鸢尾花数据集

```python
from sklearn import datasets

iris = datasets.load_iris()
x = iris.data
y = iris.target
```

在`sklearn.decomposition`包中，有PCA降维的方法，使用该方法对测试数据降维有

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(x)
print(pca.components_)
x_reduction = pca.transform(x)
print(x_reduction.shape)
print(x_reduction[0:5])
```

绘制降维后的数据分布

```python
import matplotlib.pyplot as plt

def plot_pca(x_std, y):
    plt.figure(figsize=(10, 8))
    plt.scatter(x_std[:, 0], x_std[:, 1], c=y, s=100)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()
    
plot_pca(x_reduction, y)
```

比较原始数据的特征组合与降维数据的对比

<img src="https://raw.githubusercontent.com/hughxusu/lesson-ai/develop/images/base/iris-data-show.png" style="zoom: 75%;" />

### 使用降维数据分类

导入癌症数据，使用pca对数据进行降维

```python
from sklearn.preprocessing import StandardScaler

cancer = datasets.load_breast_cancer()
x = cancer.data
y = cancer.target
x_std = StandardScaler().fit_transform(x)
pca = PCA(n_components=2)
pca.fit(x_std)
x_reduction = pca.transform(x_std)
plot_pca(x_reduction, y)
```

划分训练集和测试集

```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_reduction, y, random_state=42)
print(x_train.shape)
print(x_test.shape)
```

使用逻辑回归对数据进行分类

```python
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)
print(log_reg.score(x_train, y_train))
print(log_reg.score(x_test, y_test))
```

> [!note]
>
> PCA降维中降维后的维度$k$应该如何选择？

sklearn的PCA提供了一个方法就是可以计算出每个主成分代表的方差比率

```python
print(pca.explained_variance_ratio_)
```

使用PCA计算出全部主成分，然后看看每个主成分的方差比率是如何变化。

```python
pca = PCA(n_components=x_std.shape[1])
pca.fit(x_std)
print(pca.explained_variance_ratio_)
```

绘制上述方差的累计曲线

```python
import numpy as np

plt.figure(figsize=(10, 8))
plt.plot([i for i in range(x_std.shape[1])], 
         [np.sum(pca.explained_variance_ratio_[:i+1]) for i in range(x_std.shape[1])], 
         linewidth=3)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()
```

sklearn的PCA初始化时提供了一个参数，表示期望达到的总方差率为多少，然后会帮我们自动计算出主成分个数

```python
pca = PCA(0.95)
pca.fit(x_std)
print(pca.n_components_)
```

重新划分训练集和测试集

```python
x_reduction = pca.transform(x_std)
x_train, x_test, y_train, y_test = train_test_split(x_reduction, y, random_state=42)
print(x_train.shape)
print(x_test.shape)
```

使用逻辑回归对数据进行分类

```python
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)
print(log_reg.score(x_train, y_train))
print(log_reg.score(x_test, y_test))
```

> [!warning]
>
> 某些情况下，PCA降维后的数据，分类性能有所提升，这是在降维的过程中对数据进行了降噪。

## 决策边界

将癌症分类数据压缩为2维，打印模型参数

```python
pca = PCA(n_components=2)
pca.fit(x_std)
x_reduction = pca.transform(x_std)
log_reg = LogisticRegression()
log_reg.fit(x_reduction, y)

print(log_reg.coef_)
print(log_reg.intercept_)
```

对于逻辑回归有分类函数表示为
$$
\hat{p}=
\sigma \left( \theta^{T}\cdot x_b \right)=\frac{1}{1+e^{\theta^{T}\cdot x_b}} \qquad
\hat{y}=
\begin{cases}
 1, & \hat{p}\ge 0.5 \Rightarrow \theta^{T}\cdot x_b \ge 0\\
 0, & \hat{p}< 0.5 \Rightarrow \theta^{T}\cdot x_b < 0 \\
\end{cases}
$$
所以
$$
\theta^{T}\cdot x_b = 0
$$
称为决策边界。当特征维度为2时，决策边界可以表示为
$$
\theta_0+\theta_1x_1+\theta_2x_2=0
$$
绘制上面模型的决策边界如下

```python
def x2(x1):
    return (-log_reg.coef_[0][0] * x1 - log_reg.intercept_) / log_reg.coef_[0][1]

x1_plot = np.linspace(-5, 15, 1000)
x2_plot = x2(x1_plot)

plt.figure(figsize=(10, 8))
plt.scatter(x_reduction[y==0, 0], x_reduction[y==0, 1], color='red')
plt.scatter(x_reduction[y==1, 0], x_reduction[y==1, 1], color='blue')
plt.plot(x1_plot, x2_plot)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()
```

> [!warning]
>
> 在二维平面上，逻辑回归可以看做预测一个点相对于一条直线的位置。

[DecisionBoundaryDisplay.from_estimator](https://scikit-learn.org/stable/modules/generated/sklearn.inspection.DecisionBoundaryDisplay.html)给定一个估计量，绘制决策边界。

```python
from sklearn.inspection import DecisionBoundaryDisplay

plt.figure(figsize=(10, 8))
DecisionBoundaryDisplay.from_estimator(
    log_reg, x_reduction, cmap=plt.cm.Paired, response_method="predict", ax=plt.gca()
)
plt.scatter(x_reduction[:, 0], x_reduction[:, 1], c=y, s=100)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()
```









