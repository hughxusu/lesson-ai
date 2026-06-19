# 支持向量机

支持向量机（supported vector machine，简称：SVM）的算法的本质是找到一个在两类样本中间位置的分界线。等价于两个类别距离分界线最近的点，到分界线的距离相等。

* 两个类别距离分界线最近的点，构成一个区域，理想条件下，这个区域内没有样本点。
* 两个类别距离分界线最近的点，被称为支撑向量。
* SVM特别适用于中小型复杂数据集的分类。

<img src="https://raw.githubusercontent.com/hughxusu/lesson-ai/develop/images/base/svm-compare.png" style="zoom:65%;" />

> [!warning]
>
> 当两类数据间可以选择多条分类边界时，称为不适定问题。

支撑向量机算法：

1. 找到这些支撑向量。
2. 最大化margin。

<img src="https://raw.githubusercontent.com/hughxusu/lesson-ai/develop/images/base/pF70P6x.png" style="zoom:90%;" />

## Margin的数学表达

在$n$维空间中直线方程可以表示为$w^Tx+b=0$，也可以表示为$\theta^Tx_b=0$。设正样本$1$表示，负样本用$-1$表示。上述式子可以化简为
$$
\left\{\begin{matrix}
w^Tx^{(i)}+b \ge 1 & \forall y^{(i)}=1 \\
w^Tx^{(i)}+b \le -1  & \forall y^{(i)}=-1
\end{matrix}\right.
$$
中间分界线的方程为
$$
w^Tx^{(i)}+b = 0
$$
重新定义直线的参数这有
$$
w^Tx+b = 1 \\
w^Tx+b = 0 \\
w^Tx+b = -1
$$
直线的示意图如下

<img src="https://raw.githubusercontent.com/hughxusu/lesson-ai/develop/images/base/image-20190814141836897.png" style="zoom:80%;" />

支持向量机公式表示为
$$
\left\{\begin{matrix}
w^Tx^{(i)}+b \ge 1 & \forall y^{(i)}=1 \\
w^Tx^{(i)}+b \le -1  & \forall y^{(i)}=-1
\end{matrix}\right.
$$
所以上面的分类器可以统一为
$$
y^{(i)}(w^Tx^{(i)}+b) \ge 1
$$
支持向量机的算法目标是最大化间隔$d$。根据点到超平面距离公式
$$
d=\frac{|w^Tx+b|}{||w||}
$$
等价于
$$
\max \frac{2|w^Tx+b|}{||w||}
$$
由于所有的$x$都是支撑向量，所以$|w^Tx+b|=1$，所以上述公式可以表示为
$$
\max \frac{2}{||w||}
$$
最大化上面的值可以表示为，最小化公式
$$
\max \frac{2}{||w||} \Rightarrow  \frac{1}{\min\frac{1}{2}||w||}
$$
所以SVM的优化目标为
$$
\begin{cases}
y^{(i)}(w^Tx^{(i)}+b) \ge 1\\
\min \frac{1}{2}||w||^2 \\
\end{cases}
$$

在这个优化目标函数之间没有任何样本点，称为Hard Margin SVM。

> [!warning]
>
> 支持向量机的最优化是有条件的最优化问题，可以使用[拉格朗日乘子法]( https://www.bilibili.com/video/BV1NH4y1q7Ef/?share_source=copy_web&vd_source=aa661569ff3138d0b604d53a96184bf2)求解。

## Soft Margin SVM

一般的情况下，大部分数据是线性不可分的。软边缘：目标是尽可能在保持最大间隔，和限制间隔违例之间找到平衡。

<img src="https://raw.githubusercontent.com/hughxusu/lesson-ai/develop/images/base/pF7kub4.png" style="zoom: 40%;" />

SVM分类器无法使得所有的$i\in M$满足下列公式
$$
y^{(i)}(w^Tx^{(i)}+b) \ge 1
$$

为了能够正确分类，可以放松分类器的限制。在Hard Margin SVM目标函数中增加一个宽松量，表示如下
$$
y^{(i)}(w^Tx^{(i)}+b) \ge 1-\zeta_i, \quad \zeta_i>0
$$

上面的目标函数表示为

<img src="https://raw.githubusercontent.com/hughxusu/lesson-ai/develop/images/base/maxresdefault.jpg" style="zoom:65%;" />

其中，对于每个样本数据存在不同$\zeta_i$。如果当$\zeta$无穷大时，意味着容错性无穷大，故而分不出类别。为控制$\zeta$的范围，增加正则项
$$
\min \left(\frac{1}{2}||w||^2+C\sum_i^m\zeta_i\right)
$$
其中$C$是超参数，用于平衡超参数的比例。Soft Margin SVM的分类器目标函数表示如下：
$$
\begin{cases}
y^{(i)}(w^Tx^{(i)}+b) \ge 1-\zeta_i, \quad \zeta_i>0\\
\min \left(\frac{1}{2}||w||^2+C\sum_i^m\zeta_i\right) \\
\end{cases}
$$
上面的目标函数相当于增加了L1正则。L2正则的目标函数表示如下
$$
\begin{cases}
y^{(i)}(w^Tx^{(i)}+b) \ge 1-\zeta_i, \quad \zeta_i>0\\
\min \left(\frac{1}{2}||w||^2+C\sum_i^m\zeta_i^2\right) \\
\end{cases}
$$

* C值低，分类的错误样本较多，更容易出现欠拟合现象。
* C值高，分类的错误样本较少，更容易出现过拟合现象。

> [!warning]
>
> 对于线性不可分的情况，支持向量是由边界点和错误点共同组成。

## sklearn中的svm

> [!attention]
>
> 使用SVM前需要对数据进行标准化处理。

使用癌症数据集并使用PCA降维得到

```python
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

cancer = datasets.load_breast_cancer()
x = cancer.data
y = cancer.target
x_std = StandardScaler().fit_transform(x)
pca = PCA(n_components=2)
pca.fit(x_std)
x_reduction = pca.transform(x_std)
```

绘制数据图像

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

对数据进行标准化处理，并划分训练集与测试集

```python
from sklearn.preprocessing import StandardScaler

standardScaler = StandardScaler()
standardScaler.fit(x_reduction)
x_standard = standardScaler.transform(x_reduction)
```

导入SVM类，C表示正则的强弱：小C值强正则化；大C值弱正则化。`C=1e9`取一个非常大的值，SVM分类器为Hard SVM，训练模型

```python
from sklearn.svm import LinearSVC

svc = LinearSVC(C=1e9)
svc.fit(x_standard, y)
print(svc.score(x_standard, y))
```

绘制分类边界

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import LinearSVC

def plot_svm_boundary(svc, X, y):
    plt.figure(figsize=(10, 8))
    DecisionBoundaryDisplay.from_estimator(
        svc, X, colors="k", alpha=0.5, plot_method="contour",
        levels=[-1, 0, 1], linestyles=["--", "-", "--"], ax=plt.gca()
    )
    
    plt.scatter(X[:, 0], X[:, 1], c=y, s=100, cmap=plt.cm.Paired)

    if hasattr(svc, 'coef_'):
        decision_function = svc.decision_function(X)
        support_vector_indices = np.where(np.abs(decision_function) <= 1 + 1e-15)[0]
        support_vectors = X[support_vector_indices]
        plt.scatter(support_vectors[:, 0], support_vectors[:, 1],
                    s=100, facecolors='none', edgecolors='k', linewidths=1.5,
                    label='Support Vectors')
    
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()

plot_svm_boundary(svc, x_standard, y)
```

当`C=0.01`时，SVM分类器为Soft SVM，训练模型，绘制分类边界

```python
svc2 = LinearSVC(C=0.01)
svc2.fit(x_standard, y)
print(svc2.score(x_standard, y))
plot_svm_boundary(svc2, x_standard, y)
```

## 非线性数据分类

使用sklearn的`datasets.make_moons`函数生成测试数据

```python
x, y = datasets.make_moons()
print(x.shape)
print(y.shape)
plt.figure(figsize=(10, 8))
plt.scatter(x[y==0,0],x[y==0,1], color='red', s=100)
plt.scatter(x[y==1,0],x[y==1,1], color='blue', s=100)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()
```

给生成数据集添加扰动

```python
x, y = datasets.make_moons(noise=0.15, random_state=666)
plt.figure(figsize=(10, 8))
plt.scatter(x[y==0,0],x[y==0,1],color='red', s=100)
plt.scatter(x[y==1,0],x[y==1,1],color='blue', s=100)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()
```

对于非线性数据，可以使用多项式特征对非线性数据分类

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

def PolynomialSVC(degree, C=1.0):
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('std_scaler', StandardScaler()),
        ('linearSVC', LinearSVC(C=C))
    ])

poly_svc = PolynomialSVC(degree=3)
poly_svc.fit(x,y)
print(poly_svc.score(x,y))
```

绘制分界面

```python
plot_svm_boundary(poly_svc, x, y)
```

### 核函数

核函数的作用就是一个从低维空间到高维空间的映射，而这个映射可以把低维空间中线性不可分的两类点变成线性可分的。

<img src="https://raw.githubusercontent.com/hughxusu/lesson-ai/develop/images/base/up-dim.jpg" style="zoom:55%;" />

> [!warning]
>
> 核函数这种转换方式，不止限于SVM分类器中。

$$
w^Tx+b = 0  \Rightarrow w^TK(x)+b = 0
$$

常用的核函数

| 核函数                          | 公式                                       | 对应升维空间 |
| ------------------------------- | ------------------------------------------ | ------------ |
| 线性核<br />Linear Kernel       | $K( x_i, x)=\langle x_i, x \rangle$        | 原始特征空间 |
| 多项式核<br />Polynomial Kernel | $K( x_i, x)=(a\langle x_i, x \rangle+b)^d$ | 多项式空间   |
| 高斯核<br />Gaussian Kernel     | $K(x_i, x)=\exp{(-\gamma||x_i-x||^2)}$     | 无穷维空间   |

### 多项式核

SVM分类器中有多项式核函数可以直接对非线性数据进行分类，分类器为[`SVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)，训练分类器

```python
from sklearn.svm import SVC

def PolynomialKernelSVC(degree, C=1.0):
    return Pipeline([
        ('std_scaler', StandardScaler()),
        ('kernelSVC', SVC(kernel='poly', degree=degree, C=C))
    ])

poly_kernel_svc = PolynomialKernelSVC(degree=3)
poly_kernel_svc.fit(x,y)
print(poly_kernel_svc.score(x,y))
plot_svm_boundary(poly_kernel_svc, x, y)
```

`degree`表示多项式的阶数，`C`正则参数。其中
$$
K( x_i, x)=\left(\gamma \cdot \langle x_i, x \rangle+\text{coef0} \right )^{\text{degree}}
$$
`gamma`控制单个样本的影响范围：

* 样本影响范围大，更平滑，更线性。
* 样本影响范围小，更复杂，更非线性。

```python
def PolynomialKernelSVC2(degree, C=1.0):
    return Pipeline([
        ('std_scaler', StandardScaler()),
        ('kernelSVC', SVC(kernel='poly', degree=degree, C=C, coef0=1.0, gamma=1.0))
    ])

poly_kernel_svc = PolynomialKernelSVC2(degree=3)
poly_kernel_svc.fit(x,y)
print(poly_kernel_svc.score(x,y))
plot_svm_boundary(poly_kernel_svc, x, y)
```

### 高斯核函数

高斯核函数也称为RBF核（Radial Basis Function Kernel）。特征升维可以使线性不可分的数据线性可分。训练SVM模型有，其中使用高斯核函数。其中`gamma=1.0`

```python
def RBFKernelSVC(gamma=1.0):
    return Pipeline([
        ('std_scaler', StandardScaler()),
        ('svc', SVC(kernel='rbf', gamma=gamma))
    ])

rbf_svc = RBFKernelSVC(gamma=1.0)
rbf_svc.fit(x, y)
print(rbf_svc.score(x, y))
plot_svm_boundary(rbf_svc, x, y)
```

当`gamma=10`时绘制决策边界

```python
rbf_svc = RBFKernelSVC(gamma=10)
rbf_svc.fit(x, y)
print(rbf_svc.score(x, y))
plot_svm_boundary(rbf_svc, x, y)
```

当`gamma=0.1`时绘制决策边界

```python
rbf_svc = RBFKernelSVC(gamma=0.1)
rbf_svc.fit(x, y)
print(rbf_svc.score(x, y))
plot_svm_boundary(rbf_svc, x, y)
```

对于每个样本点都有围绕它的一个高斯分布图，所以连起来就形成了一片区域，然后形成了决策区域和决策边界。

<img src="https://raw.githubusercontent.com/hughxusu/lesson-ai/develop/images/base/6d96b78336d261269b68a73e06b24350.jpg" style="zoom:50%;" />

> [!warning]
>
> `gamma`过大造成过拟合，`gamma`过小造成欠拟合。`gamma`实际上再调整模型复杂度。

## SVM解决回归问题

SVM解决回归问题的思路和解决分类问题的思路正好是相反的。找到一条拟合直线，使得这条直线的Margin区域中的样本点越多，说明拟合的越好，反之依然。Margin边界到拟合直线的距离称为$\epsilon$是SVM解决回归问题的一个超参数。

<img src="https://raw.githubusercontent.com/hughxusu/lesson-ai/develop/images/base/svm-ress.png" style="zoom:75%;" />

导入糖尿病数据集

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split

diabetes = datasets.load_diabetes()
x = diabetes.data
y = diabetes.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(x_train.shape)
print(x_test.shape)
```

使用SVM进行回归预测

```python
from sklearn.svm import LinearSVR

def StandardLinearSVR(epsilon=0.1):
    return Pipeline([
        ('std_scaler', StandardScaler()),
        ('linearSVR', LinearSVR(epsilon=epsilon))
    ])

svr = StandardLinearSVR()
svr.fit(x_train, y_train)
print(svr.score(x_test, y_test))
```

