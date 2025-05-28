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





## 交叉验证

```mermaid
flowchart LR
		z(数据集)-->a(训练数据)
		z-->b(测试数据)
		a-->c(模型)
		b-->c
```

> [!note]
>
> 使用训练数据集和测试数据集划分的方式验证模型，可能造成准对测试数据的过拟合现象。

```mermaid
flowchart LR
		z(数据集)-->a(训练数据)
		z(数据集)-->d(验证数据)
		z-->b(测试数据)
		a-->c(模型)
		d-->c
```

1. 训练数据集训练模型。
2. 验证数据集调整模型，主要用于调整超参数。
3. 测试数据集验证模型，测试数据不参与模型的创建，用于评价模型的最终性能。

交叉验证（Cross Validation）

```mermaid
flowchart LR
		z(训练数据)-->a(A)
		z-->b(B)
		z-->c(C)
```

将数据分割为A、B、C三部分。

* 使用B、C训练；使用A验证。
* 使用A、C训练；使用B验证。
* 使用A、B训练；使用C验证。

上面的三种训练可以得到3个模型，使用3个模型的均值作为结果进行调参。训练数据可以分割为$k$份，进行$k$份的交叉验证。

### 使用训练集和测试集

```python
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

digits = datasets.load_digits()
X = digits.data
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=666)

best_score, best_p, best_k = 0, 0, 0
for k in range(2, 11):
    for p in range(1, 6):
        knn_clf = KNeighborsClassifier(weights='distance', n_neighbors=k, p=p)
        knn_clf.fit(X_train, y_train)
        score = knn_clf.score(X_test, y_test)
        if score > best_score:
            best_score, best_p, best_k = score, p, k

print('best k =', best_k)
print('best p =', best_p)
print('best score =', best_score)
```

### 使用交叉验证

`cross_val_score`交叉验证的测试分类器性能。

```python
from sklearn.model_selection import cross_val_score

knn_clf = KNeighborsClassifier()
score = cross_val_score(knn_clf, X_train, y_train)
print(score)
```

使用交叉验证选择参数

```python
best_score, best_p, best_k = 0, 0, 0
for k in range(2, 11):
    for p in range(1, 6):
        knn_clf = KNeighborsClassifier(weights='distance', n_neighbors=k, p=p)
        scores = cross_val_score(knn_clf, X_train, y_train)
        score = np.mean(scores)
        if score > best_score:
            best_score, best_p, best_k = score, p, k
            
print('best k =', best_k)
print('best p =', best_p)
print('best score =', best_score)
```

使用测试集测试分类器性能

```python
best_knn_clf = KNeighborsClassifier(weights='distance', n_neighbors=2, p=2)
best_knn_clf.fit(X_train, y_train)
print(best_knn_clf.score(X_test, y_test))
```

> [!warning]
>
> 在网格搜索类中`GridSearchCV`，包含了交叉验证。

把训练数据分割为$k$份，使用交叉验证的方式训练模型，称为k-folds cross validation。

留一法（Leave-One-Out，LOO）是一种特殊的交叉验证方法，每次训练只留下一个作为预测值，其它数据全部用来训练。留一法的优点是几乎利用了所有数据进行训练，评估结果相对准确，且不受随机分组的影响，结果具有较高的稳定性和可靠性。但缺点是计算成本高，当数据集较大时，训练和验证的次数会非常多，计算量巨大。



## 决策边界

打印模型参数

```python
print(log_reg.coef_)
print(log_reg.interception_)
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
    return (-log_reg.coef_[0] * x1 - log_reg.interception_) / log_reg.coef_[1]

x1_plot = np.linspace(4, 8, 1000)
x2_plot = x2(x1_plot)

plt.scatter(X[y==0, 0], X[y==0, 1], color='red')
plt.scatter(X[y==1, 0], X[y==1, 1], color='blue')
plt.plot(x1_plot, x2_plot)
plt.show()
```

> [!warning]
>
> 逻辑回归可以看做预测一个点相对于一条直线的位置。

根据模型绘制决策边界（该函数仅了解即可）

```python
def plot_decision_boundary(model, axis):
    x0, x1 = np.meshgrid(
        np.linspace(axis[0], axis[1], int((axis[1]-axis[0])*100)).reshape(-1, 1),
        np.linspace(axis[2], axis[3], int((axis[3]-axis[2])*100)).reshape(-1, 1),
    )
    X_new = np.c_[x0.ravel(), x1.ravel()]
    y_predict = model.predict(X_new)
    zz = y_predict.reshape(x0.shape)
    
    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#EF9A9A', '#FFF59D', '#90CAF9'])
    
    plt.contourf(x0, x1, zz, linewidth=5, cmap=custom_cmap)
```

调用上述函数绘制线性回归的决策边界

```python
plot_decision_boundary(log_reg, axis=[4, 7.5, 1.5, 4.5])
plt.scatter(X[y==0, 0], X[y==0, 1], color='red')
plt.scatter(X[y==1, 0], X[y==1, 1], color='blue')
plt.show()
```

### 验证KNN算法的决策边界

使用KNN分类器分类上面的数据并绘制决策边界

```python
from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)
print(knn_clf.score(X_test, y_test))

plot_decision_boundary(knn_clf, axis=[4, 7.5, 1.5, 4.5])
plt.scatter(X[y==0, 0], X[y==0, 1], color='red')
plt.scatter(X[y==1, 0], X[y==1, 1], color='blue')
plt.show()
```

绘制3类的分类边界

```python
knn_clf_all = KNeighborsClassifier()
knn_clf_all.fit(iris.data[:, :2], iris.target)
plot_decision_boundary(knn_clf_all, axis=[4, 8, 1.5, 4.5])
plt.scatter(iris.data[iris.target==0, 0], iris.data[iris.target==0, 1], color='red')
plt.scatter(iris.data[iris.target==1, 0], iris.data[iris.target==1, 1], color='blue')
plt.scatter(iris.data[iris.target==2, 0], iris.data[iris.target==2, 1], color='green')
plt.show()
```

当`n_neighbors=50`绘制决策边界

```python
knn_clf_all = KNeighborsClassifier(n_neighbors=50)
knn_clf_all.fit(iris.data[:, :2], iris.target)
plot_decision_boundary(knn_clf_all, axis=[4, 8, 1.5, 4.5])
plt.scatter(iris.data[iris.target==0, 0], iris.data[iris.target==0, 1], color='red')
plt.scatter(iris.data[iris.target==1, 0], iris.data[iris.target==1, 1], color='blue')
plt.scatter(iris.data[iris.target==2, 0], iris.data[iris.target==2, 1], color='green')
plt.show()
```

> [!warning]
>
> 在KNN算法中，当面模型的`n_neighbors`参数越大模型越简单，在分类边界中表现为边界越规整。



上面的逻辑回归本质上是找到一条直线，用直线来分割样本的类别。通过对数据添加多项式向，可以使得逻辑回归对非线性的数据同时起作用。
