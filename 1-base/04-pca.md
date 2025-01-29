# 主成分分析

主成分分析（Principal Component Analysis）是一种多变量统计分析技术。它的主要目的是通过线性变换，将原始数据的多个变量（特征）转换为一组新的、数量较少的变量，这些新变量被称为主成分。

* 非监督的机器学习算法。
* 主要用于数据降维。
* 通过降维，可以发现更便于人类理解的特征。
* 可视化、去噪。

> [!warning]
>
> 主成分分析并不只应用在机器学习领域，也是统计分析领域的重要方法。

二维特征可以绘制在如下平面

![](https://raw.githubusercontent.com/hughxusu/lesson-ai/developing/_images/base/96db7d87e7af97e05b630366cc02bcf1.jpg)

如果只保留特征1

![](https://raw.githubusercontent.com/hughxusu/lesson-ai/developing/_images/base/b51d7ec017736d4f47fb7e836c33f074.jpg)

如果只保留特征2

![](https://raw.githubusercontent.com/hughxusu/lesson-ai/developing/_images/base/aa709dfbb796ada2b6b6dae9d0d43d64.jpg)

从上面的结果来看，特征1的区分度较高。如果能找到一条直线，可以拟合样本的投影

![](https://raw.githubusercontent.com/hughxusu/lesson-ai/developing/_images/base/0525efddb7f229a87f9b6443cd26dd8a.jpg)

降维后点和点之间的区分度更接近，原来点的分布。

> [!warning]
>
> 这一直线的标准就是，样本映射到该直线后，方差最大。

1. 将样本的均值归0，即样本值减去均值，根据方差的公式。

$$
Var(x)=\frac{1}{m}\sum_{i=1}^m(x_i-\bar{x})^2，\bar{x}=0 \Rightarrow Var(x)=\frac{1}{m}\sum_i^mx_i^2
$$

<img src="https://raw.githubusercontent.com/hughxusu/lesson-ai/developing/_images/base/format.jpg" style="zoom:65%;" />

2. 找到投影轴的方向$w=\left( w_1, w_2  \right)$，使得所有样本映射到新的坐标轴有

$$
Var(X_{project})=\frac{1}{m}\sum_{i}^{m}\left \|X_{project}^{(i)}-\bar{X}_{project}\right\|^2
$$

最大。由于向量进行了归0处理，则有
$$
Var(X_{project})=\frac{1}{m}\sum_{i}^{m}\left \|X_{project}^{(i)}\right\|^2 \tag{1}
$$
最大。投影计算如下

<img src="https://raw.githubusercontent.com/hughxusu/lesson-ai/developing/_images/base/pca-project.jpg" style="zoom:85%;" />

上面的映射计算，就是
$$
X^{(i)}\cdot w=\left\| X^{(i)} \right\|\cdot\left\| w \right\| \cdot \cos \theta
$$
式子中寻找的$w$，是一个方向，可以用方向向量表示，所以有$\left\| w \right\|=1 $，上式可以化简为
$$
X^{(i)}\cdot w=\left\| X^{(i)} \right\| \cdot \cos \theta
$$
所以有
$$
X^{(i)}\cdot w=\left \|X_{project}^{(i)}\right\|
$$
所以公式 $(1)$ 可以变为
$$
Var(X_{project})=\frac{1}{m}\sum_{i}^{m}\left ( X^{(i)}\cdot w  \right )^2
$$
所以主成分分析就是求$w$使得上式最大。推广到$n$维样本点，则有
$$
Var(X_{project})=\frac{1}{m}\sum_{i}^{m}\left ( X^{(i)}_1 w_1+X^{(i)}_2 w_2+…+ X^{(i)}_n w_n \right )^2 \tag{2}
$$
所以主成分分析法，就是求目标函数 $(2)$ 的最优化问题。该问题可以使用梯度上升法来解决（该问题也有解析解）。

> [!attention]
>
> 主成分分析法与线性回归的区别：在主成分分析中横纵坐标都是特征；而在线性回归中，横坐标是特征，纵坐标是预测值。

目标函数可以表示为
$$
f\left(X \right)=\frac{1}{m}\sum_{i}^{m}\left ( X^{(i)}_1 w_1+X^{(i)}_2 w_2+…+ X^{(i)}_n w_n \right )^2 \tag{3}
$$
主成分分析法就是求目标函数 $(3)$ 最大。所以目标函数的梯度可以表示为
$$
\nabla f = \begin{pmatrix}
\frac{\partial f}{\partial w_1 } \\
\frac{\partial f}{\partial w_2 } \\
… \\
\frac{\partial f}{\partial w_n } 
\end{pmatrix} =\frac{2}{m}\begin{pmatrix}
\sum_{i=1}^m \left ( X_1^{(i)}w_1 + X_2^{(i)}w_2 + … + X_n^{(i)}w_n \right ) X_1^{(i)} \\
\sum_{i=1}^m \left ( X_1^{(i)}w_1 + X_2^{(i)}w_2 + … + X_n^{(i)}w_n \right ) X_2^{(i)} \\
… \\
\sum_{i=1}^m \left ( X_1^{(i)}w_1 + X_2^{(i)}w_2 + … + X_n^{(i)}w_n \right ) X_n^{(i)} \\
\end{pmatrix} \\  =  \frac{2}{m}\begin{pmatrix}
\sum_{i}^{m}\left ( X^{(i)} w  \right )X_1^{(i)} \\
\sum_{i}^{m}\left ( X^{(i)} w  \right )X_2^{(i)} \\
… \\
\sum_{i}^{m}\left ( X^{(i)} w  \right )X_n^{(i)}
\end{pmatrix} \tag{4}
$$
考虑下面的式子
$$
\frac{2}{m}\cdot\left(X^{(1)}w, X^{(2)}w, … X^{(m)}\right) \cdot \begin{pmatrix}
X_1^{(1)} & X_2^{(1)} & … & X_n^{(1)} \\
X_1^{(2)} & X_2^{(2)} & … & X_n^{(2)} \\
… \\
X_1^{(m)} & X_2^{(m)} & … & X_n^{(m)} \\
\end{pmatrix} = \frac{2}{m}\cdot \left( Xw \right)^T\cdot X \tag{5}
$$
结合公式 $(5)$ ，可以将公式 $(4)$ 化简为
$$
\nabla f =\frac{2}{m}\begin{pmatrix}
\sum_{i}^{m}\left ( X^{(i)} w  \right )X_1^{(i)} \\
\sum_{i}^{m}\left ( X^{(i)} w  \right )X_2^{(i)} \\
… \\
\sum_{i}^{m}\left ( X^{(i)} w  \right )X_n^{(i)}
\end{pmatrix} = \frac{2}{m}\cdot X^T \cdot \left( Xw \right)
$$

## PCA的过程模拟

生成测试数据

```python
import numpy as np
import matplotlib.pyplot as plt

X = np.empty([100, 2])
X[:,0] = np.random.uniform(0., 100., size=100)
X[:,1] = 0.75 * X[:,0] + 3. + np.random.normal(0, 5, size=100)
plt.scatter(X[:,0], X[:,1])
plt.show()
```

对数据进行归0处理

```python
def demean(X):
    return X - np.mean(X, axis=0)

X_demean = demean(X)
plt.scatter(X_demean[:,0], X_demean[:,1])
print(np.mean(X_demean[:,0]))
print(np.mean(X_demean[:,1]))
```

目标函数$Var(X_{project})$和梯度的计算为

```python
def f(w, X):
    return np.sum((X.dot(w)**2)) / len(X)

def df(w, X):
    return X.T.dot(X.dot(w)) * 2. / len(X)

def df_debug(w, X, epsilon=0.0001):
    res = np.empty(len(w))
    for i in range(len(w)):
        w_1 = w.copy()
        w_1[i] += epsilon
        w_2 = w.copy()
        w_2[i] -= epsilon
        res[i] = (f(w_1, X) - f(w_2, X)) / (2 * epsilon)
    return res
```

梯度上升法的求解过程为

```python
def direction(w):
    return w / np.linalg.norm(w)

def gradient_ascent(df, X, initial_w, eta, n_iters=1e4, epsilon=1e-8):
    w = direction(initial_w)
    cur_iter = 0

    while cur_iter < n_iters:
        gradient = df(w, X)
        last_w = w
        w = w + eta * gradient
        w = direction(w)
        if (abs(f(w, X) - f(last_w, X)) < epsilon):
            break

        cur_iter += 1

    return w
```

其中`direction`函数是转换$w$向量，使其值为$\left\| w \right\|=1$。测试梯度上升法

```python
initial_w = np.random.random(X.shape[1])
print(initial_w)

eta = 0.001
w = gradient_ascent(df_debug, X_demean, initial_w, eta)
print(w)
w = gradient_ascent(df, X_demean, initial_w, eta)
print(w)

plt.scatter(X_demean[:,0], X_demean[:,1])
plt.plot([w[0]*-50, w[0]*50], [w[1]*-50, w[1]*50], color='r')
plt.show()
```

假设没有扰动

```python
X2 = np.empty(X.shape)
X2[:,0] = np.random.uniform(0., 100., size=100)
X2[:,1] = 0.75 * X2[:,0] + 3.
plt.scatter(X2[:,0], X2[:,1])
plt.show()
```

使用pca方法计算方向向量

```python
X2_demean = demean(X2)
w2 = gradient_ascent(df, X2_demean, initial_w, eta)
print(w2)
plt.scatter(X2_demean[:,0], X2_demean[:,1])
plt.plot([w2[0]*-50, w2[0]*50], [w2[1]*-50, w2[1]*50], color='r')
plt.show()
```

## 数据前n个主成分

PCA的本质是对特征空间的坐标轴重新排列，在样本的原始特征空间内寻找一套新的坐标轴替换掉样本的原始坐标轴；在这套新的坐标轴里，第一主成分轴捕获了样本最大的方差，第二主成分轴轴次之，第三主成分轴再次之，以此类推；

<img src="https://raw.githubusercontent.com/hughxusu/lesson-ai/developing/_images/base/project-abc.png" style="zoom:50%;" />

上面例子中样本数据，在二维空间表示为$\vec{c}$，主成分分析看作是将横坐标轴旋转到$w=\left( w_1, w_2  \right)$，而向量$\vec{b}$可以看做第一主成分，是在$w$上的投影。另外一个分量$\vec{a}$垂直于$\vec{b}$向量。
$$
\vec{b}=\left \|X_{project}^{(i)}\right\|w \\ 
X^{(i)}\cdot w=\left \|X_{project}^{(i)}\right\|\\
\vec{a} = \vec{c} - \vec{b}
$$
可以计算出
$$
\vec{a} =X^{(i)}-\left(X^{(i)}\cdot w\right)w
$$
在此基础上对向量$\vec{a}$再次进行主成分分析，得到第二组成。以此类推可以得到前n个主成分。

虚拟数据如下

```python
X = np.empty((100, 2))
X[:,0] = np.random.uniform(0., 100., size=100)
X[:,1] = 0.75 * X[:,0] + 3. + np.random.normal(0, 10., size=100)

def demean(X):
    return X - np.mean(X, axis=0)

X_demean = demean(X)
plt.scatter(X_demean[:,0], X_demean[:,1])
plt.show()
```

计算数据的第一个主成分

```python
def f(w, X):
    return np.sum((X.dot(w)**2)) / len(X)

def df(w, X):
    return X.T.dot(X.dot(w)) * 2. / len(X)

def direction(w):
    return w / np.linalg.norm(w)

def first_component(X, initial_w, eta, n_iters=1e4, epsilon=1e-8):
    w = direction(initial_w)
    cur_iter = 0

    while cur_iter < n_iters:
        gradient = df(w, X)
        last_w = w
        w = w + eta * gradient
        w = direction(w)
        if (abs(f(w, X) - f(last_w, X)) < epsilon):
            break

        cur_iter += 1

    return w

initial_w = np.random.random(X.shape[1])
eta = 0.01
w = first_component(X_demean, initial_w, eta)
print(w)
```

计算出减去第一主成分的向量

```python
X2 = np.empty(X.shape)
for i in range(len(X)):
    X2[i] = X[i] - X[i].dot(w) * w
    
plt.scatter(X2[:,0], X2[:,1])
```

计算第二主成分，其中第一主成分和第二主成分是垂直的。

```python
w2 = first_component(X2, initial_w, eta)
print(w2)
print(w.dot(w2))
```

其中减去第一主成分的过程，可以使用向量化计算

```python
X2 = X - X.dot(w).reshape(-1, 1) * w
```

对上述过程重新进行封装得到求前n个主成分的函数

```python
def first_n_components(n, X, eta=0.01, n_iters=1e4, epsilon=1e-8):
    X_pca = X.copy()
    X_pca = demean(X_pca)
    res = []

    for i in range(n):
        initial_w = np.random.random(X_pca.shape[1])
        w = first_component(X_pca, initial_w, eta)
        res.append(w)
        X_pca = X_pca - X_pca.dot(w).reshape(-1, 1) * w

    return res

result = first_n_components(2, X)
print(result)
```

## 高维数据映射为低维数据

在前面的计算过程中，矩阵$X$是$n$​维特征向量构成的特征矩阵，
$$
X=\begin{pmatrix}
  X_1^{(1)}&  X_2^{(1)}&  …& X_n^{(1)}\\
  X_1^{(2)}&  X_2^{(2)}&  …& X_n^{(2)}\\
  …&  …&  …& … \\
  X_1^{(m)}&  X_2^{(m)}&  …& X_n^{(m)}
\end{pmatrix}
$$
通过主成分分析的过程可以求出空间转换矩阵$W_k$​
$$
W_k=\begin{pmatrix}
  W_1^{(1)}&  W_2^{(1)}&  …& W_n^{(1)}\\
  W_1^{(2)}&  W_2^{(2)}&  …& W_n^{(2)}\\
  …&  …&  …& … \\
  W_1^{(k)}&  W_2^{(k)}&  …& W_n^{(k)}
\end{pmatrix}
$$
其中$k$表示主成分的数量，表示特征空间在这$k$个方向上更为重要。特征空间仍然的维度仍是$n$维，其中维度并没有减少。将原有特征矩阵与空间转换矩阵相乘
$$
X\cdot W_k^T=X_k 
$$
得到的矩阵就是降维后的特征矩阵。
$$
X_k=\begin{pmatrix}
  X_1^{(1)}&  X_2^{(1)}&  …& X_k^{(1)}\\
  X_1^{(2)}&  X_2^{(2)}&  …& X_k^{(2)}\\
  …&  …&  …& … \\
  X_1^{(m)}&  X_2^{(m)}&  …& X_k^{(m)}
\end{pmatrix}
$$
对于上述矩阵$X_k$来说还存在
$$
X_k\cdot W_k=X_m
$$
同样高维矩阵也可以映射到低维。

封装PCA的类如下

```python
class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components_ = None
        
    def fit(self, X, eta=0.01, n_iters=1e4):
        assert self.n_components <= X.shape[1], "n_components must not be greater than the number of features"
        
        def demean(X):
            return X - np.mean(X, axis=0)
        
        def f(w, X):
            return np.sum((X.dot(w)**2)) / len(X)
        
        def df(w, X):
            return X.T.dot(X.dot(w)) * 2. / len(X)
        
        def direction(w):
            return w / np.linalg.norm(w)
        
        def first_component(X, initial_w, eta=0.01, n_iters=1e4, epsilon=1e-8):
            w = direction(initial_w)
            cur_iter = 0

            while cur_iter < n_iters:
                gradient = df(w, X)
                last_w = w
                w = w + eta * gradient
                w = direction(w)
                if (abs(f(w, X) - f(last_w, X)) < epsilon):
                    break

                cur_iter += 1

            return w
        
        X_pca = demean(X)
        self.components_ = np.empty(shape=(self.n_components, X.shape[1]))
        for i in range(self.n_components):
            initial_w = np.random.random(X_pca.shape[1])
            w = first_component(X_pca, initial_w, eta)
            self.components_[i] = w
            X_pca = X_pca - X_pca.dot(w).reshape(-1, 1) * w
            
        return self
    
    def transform(self, X):
        assert X.shape[1] == self.components_.shape[1], "the number of features must be equal to the number of features of X"
        return X.dot(self.components_.T)
    
    def inverse_transform(self, X):
        assert X.shape[1] == self.components_.shape[0], "the number of features must be equal to the number of components"
        return X.dot(self.components_)
```

使用上述类来计算主成分

```python
pac = PCA(2)
pac.fit(X)
print(pac.components_)
```

使用主成分对数据进行降维

```python
pac = PCA(1)
pac.fit(X)
X_reduction = pac.transform(X)
print(X_reduction.shape)
```

对降维的数据进行还原

```python
X_restore = pac.inverse_transform(X_reduction)
print(X_restore.shape)
```

比较绘制原始坐标系和降维后的点

```python
plt.scatter(X[:,0], X[:,1])
plt.scatter(X_restore[:,0], X_restore[:,1], color='r')
plt.show()
```

> [!warning]
>
> 1. PCA的本质就是将原空间坐标系，变换到新的坐标系中，取出前$k$个重要的主成分，就可以在$k$个轴上获得一个低维的数据信息。
>
> 2. 降维后的数据丢失了部分信息。

## 使用scikit-learn的主成分分析法

在`sklearn.decomposition`包中，有PCA降维的方法，使用该方法对测试数据降维有

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=1)
pca.fit(X)
print(pca.components_)
X_reduction = pca.transform(X)
print(X_reduction.shape)
```

比较绘制原始坐标系和降维后的点

```python
X_restore = pca.inverse_transform(X_reduction)
plt.scatter(X[:,0], X[:,1])
plt.scatter(X_restore[:,0], X_restore[:,1], color='r')
plt.show()
```

### 使用PCA降维手写数字识别

导入手写数字识别数据，并划分为训练集和测试集

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()
X = digits.data
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

print(X_train.shape)
```

使用KNN算法对手写数字识别进行预测

```python
from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)

print(knn_clf.score(X_test, y_test))
```

对手写数字识别数据进行降维，降维到2维

```python
pca = PCA(n_components=2)
pca.fit(X_train)
X_train_reduction = pca.transform(X_train)
X_test_reduction = pca.transform(X_test)
```

在此使用KNN算法对降维后的数据进行测试

```python
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_reduction, y_train)
print(knn_clf.score(X_test_reduction, y_test))
```

> [!note]
>
> PCA降维中降维后的维度$k$​应该如何选择？

scikit-learn的PCA提供了一个方法就是可以计算出每个主成分代表的方差比率

```python
print(pca.explained_variance_ratio_)
```

使用PCA将64维数据计算出64个主成分，然后看看每个主成分的方差比率是如何变化。

```python
pca = PCA(n_components=X_train.shape[1])
pca.fit(X_train)
print(pca.explained_variance_ratio_)
```

绘制上述方差的累计曲线

```python
plt.plot([i for i in range(X_train.shape[1])], [np.sum(pca.explained_variance_ratio_[:i+1]) for i in range(X_train.shape[1])])
plt.show()
```

scikit-learn的PCA初始化时提供了一个参数，表示期望达到的总方差率为多少，然后会帮我们自动计算出主成分个数

```python
pca = PCA(0.95)
pca.fit(X_train)
print(pca.n_components_)
```

根据上述主成分分析再次对数据降维，使用KNN算法进行预测

```python
X_train_reduction = pca.transform(X_train)
X_test_reduction = pca.transform(X_test)
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_reduction, y_train)
print(knn_clf.score(X_test_reduction, y_test))
```

PCA降维还可以用于数据的可视化，将数据降维的2维，绘制平面图像

```python
pca = PCA(n_components=2)
pca.fit(X)
X_reduction = pca.transform(X)
print(X_reduction.shape)
for i in range(10):
    plt.scatter(X_reduction[y==i, 0], X_reduction[y==i, 1], alpha=0.8)
plt.show()
```

### 使用PCA处理MINST数据集

使用`fetch_openml`可以下载完整的minst数据

```python
from sklearn.datasets  import fetch_openml 

mnist = fetch_openml('mnist_784', version=1) 

X, y = mnist['data'], mnist['target']
print(X.shape)
```

针对minst数据可划分训练数据集合测试数据集

```python
X_train = np.array(X[:60000], dtype=float)
print(X_train.shape)
y_train = np.array(y[:60000], dtype=float)
print(y_train.shape)
X_test = np.array(X[60000:], dtype=float)
print(X_test.shape)
y_test = np.array(y[60000:], dtype=float)
print(y_test.shape)
```

使用PCA对minst数据集进行降维

```python
from sklearn.decomposition import PCA

pca = PCA(0.9)
pca.fit(X_train)
X_train_reduction = pca.transform(X_train)
print(X_train_reduction.shape)
X_test_reduction = pca.transform(X_test)
```

对降维后的数据进行预测

```python
%%time
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_reduction, y_train)
print(knn_clf.score(X_test_reduction, y_test))
```

可以发现降维后的数据，预测性能反而提升了。

> [!warning]
>
> PCA数据在降维的过程中对数据进行了降噪。

## 使用PCA对数据进行降噪

线性模型模拟数据如下

```python
X = np.empty((100, 2))
X[:,0] = np.random.uniform(0., 100., size=100)
X[:,1] = 0.75 * X[:,0] + 3. + np.random.normal(0, 10., size=100)

plt.scatter(X[:,0], X[:,1])
plt.show()
```

使用PCA进行降维后，在将数据变换为原有维度

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=1)
pca.fit(X)
X_reduction = pca.transform(X)
X_restore = pca.inverse_transform(X_reduction)

plt.scatter(X_restore[:,0], X_restore[:,1], color='r')
plt.show()
```

从上面的图像可以看出，降维的数据虽然丢失了一些信息，但是这些信息本身很有可能就是噪音。

使用scikit-learn中的手写数字识别，并对图片增加一定数量的噪音

```python
from sklearn import datasets

digits = datasets.load_digits()
X = digits.data
y = digits.target

noisy_digits = X + np.random.normal(0, 4, size=X.shape)

example_digits = noisy_digits[y==0,:][:10]
for num in range(1, 10):
    X_num = noisy_digits[y==num,:][:10]
    example_digits = np.vstack([example_digits, X_num])
    
print(example_digits.shape)
```

绘制增加噪音后的数组图片得到

```python
def plot_digits(data):
    fig, axes = plt.subplots(10, 10, figsize=(10, 10), subplot_kw={'xticks':[], 'yticks':[]}, gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(data[i].reshape(8, 8), cmap='binary', interpolation='nearest', clim=(0, 16))
    plt.show()
    
plot_digits(example_digits)
```

使用PCA对上述数据的进行主成分分析，得到主成分后回复原有数据绘图可以得到

```python
pca = PCA(0.5)
pca.fit(noisy_digits)
print(pca.n_components_)
components = pca.transform(example_digits)
filtered_digits = pca.inverse_transform(components)
plot_digits(filtered_digits)
```

## PCA在人脸识别中的应用

对于空间转换矩阵$W_k$
$$
W_k=\begin{pmatrix}
  W_1^{(1)}&  W_2^{(1)}&  …& W_n^{(1)}\\
  W_1^{(2)}&  W_2^{(2)}&  …& W_n^{(2)}\\
  …&  …&  …& … \\
  W_1^{(k)}&  W_2^{(k)}&  …& W_n^{(k)}
\end{pmatrix}
$$
从样本的特征角度可以考虑，可以看做是选择了$k$个最重要的样本，从人脸识别的角度考虑，这$k$个样本中每一个都反应了原始人脸样本的主成分，称为特征脸。

使用scikit-learn中的人脸数据来验证特征脸

```python
from sklearn.datasets import fetch_lfw_people

faces = fetch_lfw_people()
print(faces.keys())
print(faces.data.shape)
print(faces.images.shape)
```

随机绘制部分人脸

```python
random_indexes = np.random.permutation(len(faces.data))
X = faces.data[random_indexes]
example_faces = X[:36, :]

def plot_faces(faces):
    fig, axes = plt.subplots(6, 6, figsize=(10, 10), subplot_kw={'xticks':[], 'yticks':[]}, gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(faces[i].reshape(62, 47), cmap='bone')
    plt.show()
    
plot_faces(example_faces)
```

### 特征脸

对上述人脸数据集进行主成分分析

```python
%%time
from sklearn.decomposition import PCA
pca = PCA(svd_solver='randomized')
pca.fit(X)

print(pca.components_.shape)
```

绘制特征脸

```python
plot_faces(pca.components_[:36, :])
```

