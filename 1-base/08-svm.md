# 支持向量机

对于线性可分的两类数据有如下图的分布

![](https://raw.githubusercontent.com/hughxusu/lesson-ai/developing/_images/base/pF70P6x.png)

> [!warning]
>
> 当两类数据间可以选择多条分类边界时，称为不适定问题。

支持向量机的算法的本质是找到一个在两类样本中间位置的分界线。

* 等价于两个类别距离分界线最近的点，到分界线的距离相等。
* 两个类别距离分界线最近的点，构成一个区域，理想条件下，这个区域内没有样本点。
* 两个类别距离分界线最近的点，被称为支撑向量。

支撑向量机算法：

1. 找到这些支撑向量。
2. 最大化margin。

如果数据是线性可分，或者在高维度线性可分，称为Hard Margin SVM。如数据线性不可分，找到分线或分界面称为Soft Margin SVM。

## Margin的数学表达

如果支撑向量，到分界线的距离定义为$d$，则两类支持向量间的距离$\text{margin}=2d$。

在$n$维空间中直线方程可以表示为$w^Tx+b=0$，也可以表示为$\theta^Tx_b=0$。点到直线的距离公式
$$
d=\frac{|w^Tx+b|}{||w||}
$$
其中$||w||=\sqrt{w_1^2+w_2^2+…+w_n^2}$。假设存在上述决策边界，则有
$$
\left\{\begin{matrix}
\frac{w^Tx^{(i)}+b}{||w||} \ge d & \forall y^{(i)}=1 \\
\frac{w^Tx^{(i)}+b}{||w||} \le -d  & \forall y^{(i)}=-1
\end{matrix}\right.
\Rightarrow
\left\{\begin{matrix}
\frac{w^Tx^{(i)}+b}{||w||d} \ge 1 & \forall y^{(i)}=1 \\
\frac{w^Tx^{(i)}+b}{||w||d} \le -1  & \forall y^{(i)}=-1
\end{matrix}\right.
$$
其中正样本是$1$，负样本是$-1$。上述式子可以化简为
$$
\left\{\begin{matrix}
w_d^Tx^{(i)}+b_d \ge 1 & \forall y^{(i)}=1 \\
w_d^Tx^{(i)}+b_d \le -1  & \forall y^{(i)}=-1
\end{matrix}\right.
$$
对于$w^Tx+b=0$两侧同时除$||w||d$，所以中间分界线的方程为
$$
w_d^Tx^{(i)}+b_d = 0
$$
重新定义直线的参数这有
$$
w^Tx+b = 1 \\
w^Tx+b = 0 \\
w^Tx+b = -1
$$
直线的示意图如下

<img src="https://raw.githubusercontent.com/hughxusu/lesson-ai/developing/_images/base/pFh239A.jpg" style="zoom:50%;" />

支持向量机公式表示为
$$
\left\{\begin{matrix}
w^Tx^{(i)}+b \ge 1 & \forall y^{(i)}=1 \\
w^Tx^{(i)}+b_\le -1  & \forall y^{(i)}=-1
\end{matrix}\right.
$$
所以上面的分类器可以统一为
$$
y^{(i)}(w^Tx^{(i)}+b) \ge 1
$$
支持向量机的算法目标是最大化$d$。等价于
$$
\max \frac{|w^Tx+b|}{||w||}
$$
由于所有的$x$都是支撑向量，所以$|w^Tx+b|=1$，所以等价于
$$
\max \frac{1}{||w||}\Rightarrow \min||w|| \Rightarrow \min \frac{1}{2}||w||^2
$$
所以SVM的优化目标为
$$
\begin{cases}
y^{(i)}(w^Tx^{(i)}+b) \ge 1\\
\min \frac{1}{2}||w||^2 \\
\end{cases}
$$

> [!warning]
>
> 支持向量机的最优化是有条件的最优化问题。







> [!warning]
>
> 使用拉格朗日求极值，可以推导出 $w$​ 的计算方法。通过全部的训练样本寻找边界点。

$w$ 求解公式如下
$$
w=\sum_{i\in \text{SN}} \alpha_i y_i x_i
$$
其中 $i \in \text{SN}$ 表示所有边界向量组成的集合。$\alpha_i$ 表示低 $i$ 条数据的权重，表示该数据点在分类器参数计算的重要性，且 $\alpha_i>0$。$w_0$ 计算法方法如下
$$
w_0=y_i-wx_i
$$
$i$ 为任意的边界点，当 $y_i=1$ 时表示正样本，否则表示负样本。

> [!warning]
>
> SVM 算法两部分：第一是在训练集中求所有的边界向量，第二值利用向量求参数。
>
> 边界点的向量称为支持向量。

支持向量的特点：

1. 抗噪声：分界线仅有支持向量决定，异常点不会影响分界线。
2. SVM没有概率意义，只有几何意义，仅仅表示点到分界面的距离。
3. SVM训练过程不能使用部分数据。

对于支持向量机
$$
wx=\sum_{i\in \text{SN}} \alpha_i y_i (x_i \cdot x)
$$
其中 $(x_i, x)$ 表数据点与支持向量的内积。

## 线性不可分

<img src="https://raw.githubusercontent.com/hughxusu/lesson-ai/developing/_images/base/pF7kub4.png" style="zoom: 40%;" />

对于线性不可分情况SVM分类器无法使得所有的 $i\in N$ 满足下列公式
$$
(wx_i+w_0)\cdot y_i>1
$$

现实问题基本是线性不可分的。

> [!warning]
>
> 为了能够正确分类，可以放松分类器的限制。

所以对每一个特征增加一个 $\epsilon_i\geq 0 $​ 满足如下公式
$$
(wx_i+w_0)\cdot y_i>1-\epsilon_i
$$

<img src="https://raw.githubusercontent.com/hughxusu/lesson-ai/developing/_images/base/1_LPHJnXPMSzAstBnHaSKmhA.png" style="zoom:80%;" />

对于上述算式中的 $\epsilon_i$ 不是一个固定的值，对于每一个支持向量都有一个 $\epsilon_i$，对其求和得
$$
\sum_{i=1}^{n}\epsilon_i
$$
越小分类器分类效果越好，所以SVM的优化目标可以变化为
$$
\begin{cases}
(wx_i+w_0)\cdot y_i>1-\epsilon_i\\
\min{\left(||w||+C\sum_{i=1}^{n}\epsilon_i\right)} \\
\end{cases}
$$
其中 $C$ 错误权重，也是一个超参数。

> [!attention]
>
> 对于线性不可分的情况，支持向量是由边界点和错误点共同组成。

所以 $w$ 求解公式
$$
w=\sum_{i\in \text{SN}} \alpha_i y_i x_i
$$
其中 $i \in \text{SN}$ 包括边界点和错误点。

> [!attention]
>
> 使用SVM前需要对数据进行标准化处理。

对于训练数据解 `train_data` 和 `test_data` 

```python
from sklearn import svm
import numpy as np

def read_data(path):
    with open(path) as f:
        lines = f.readlines()
    lines = [eval(line.strip()) for line in lines]
    x, y = zip(*lines)
    x = np.array(x)
    y = np.array(y)
    return x, [i[0] for i in y]

x_train, y_train = read_data("train_data")
model = svm.SVC()
model.fit(x_train, y_train)
print(model.support_vectors_)
print(model.support_)

x_test, y_test = read_data("test_data")
score = model.score(x_test, y_test)
print(score)
```

> [!attention]
>
> 实际过程中对于大量数据，SVM计算量比较大为 $n^3$

对于SVM分类器公式
$$
\min{||w||+C\sum_{i=1}^{n}\epsilon_i}
$$
中参数 $C$ 的验证可以使用网格搜索来完成。

```python
from sklearn import svm
import numpy as np
from sklearn.model_selection import GridSearchCV

def read_data(path):
    with open(path) as f:
        lines = f.readlines()
    lines = [eval(line.strip()) for line in lines]
    x, y = zip(*lines)
    x = np.array(x)
    y = np.array(y)
    return x, [i[0] for i in y]

x_train, y_train = read_data("train_data")
model = svm.SVC()
search_space = {'C': np.logspace(-3, 3, 7)}
print(search_space['C'])
gridsearch = GridSearchCV(model, param_grid=search_space)
gridsearch.fit(x_train, y_train)
cv_performance = gridsearch.best_score_
print("C", gridsearch.best_params_['C'])
```

> [!attention]
>
> 对于大量数据可以在子数据集上进行网格搜索来确定 $C$ 的参数，然后再进行训练。

对于线性不可分情况 $\epsilon_i$ 可以表示为
$$
\epsilon_i=\max{(0,1-y_i(wx_i+w_0))}
$$
带入公式
$$
\min{||w||+C\sum_{i=1}^{n}\epsilon_i}=\min{||w||+C\sum_{i=1}^{n}\max{(0,1-y_i(wx_i+w_0))}}
$$
上面公式可以理解为 `正则项+损失函数`

<img src="https://raw.githubusercontent.com/hughxusu/lesson-ai/developing/_images/base/Loss-functions-for-commonly-used-classifier-hinge-loss-SVM-cross-entropy-loss.png" style="zoom: 67%;" />

## 核函数

> [!warning]
>
> 对于线性不可分文可以使用升维的方式实现数据的线性可分。

![](https://raw.githubusercontent.com/hughxusu/lesson-ai/developing/_images/base/pFreb0e.png)
$$
(x_1, x_2)\rightarrow(x_1, x_2, x_1\cdot x_2)
$$
$w$ 求值过程可以表示为
$$
wx=\sum_{i\in \text{SN}} \alpha_i y_i \langle x_i, x \rangle
$$
对于特征升维则存在：$x \rightarrow \phi(x) $ 和 $x_i \rightarrow \phi(x_i)$，所以上面公式可以表示为
$$
wx=\sum_{i\in \text{SN}} \alpha_i y_i \langle \phi(x_i), \phi(x) \rangle
$$

> [!attention]
>
> 由于高维空间的内积可以由低维空间表示

所以存在一些快捷算法使得 $\langle \phi(x_i), \phi(x) \rangle$ 表示为一些特殊的函数
$$
\langle \phi(x_i), \phi(x) \rangle=K(x_i, x)
$$

| 核函数                          | 公式                                       | 对应升维空间 |
| ------------------------------- | ------------------------------------------ | ------------ |
| 线性核<br />Linear Kernel       | $K( x_i, x)=\langle x_i, x \rangle$        | 原始特征空间 |
| 多项式核<br />Polynomial Kernel | $K( x_i, x)=(a\langle x_i, x \rangle+b)^d$ | 多项式空间   |
| 高斯核<br />Gaussian Kernel     | $K(x_i, x)=\exp{(-\gamma||x_i-x||^2)}$     | 无穷维空间   |

使用不同的核函数进行预测

```python
from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split as ts

iris = datasets.load_iris()
x = iris.data
y = iris.target
x_train, x_test, y_train, y_test = ts(x, y, test_size=0.3)

clf_rbf = svm.SVC(kernel='rbf')  # 高斯核函数
clf_rbf.fit(x_train, y_train)
score_rbf = clf_rbf.score(x_test, y_test)
print("The score of rbf is : %f" % score_rbf)

clf_linear = svm.SVC(kernel='linear')  # 线性和函数
clf_linear.fit(x_train, y_train)
score_linear = clf_linear.score(x_test, y_test)
print("The score of linear is : %f" % score_linear)

clf_poly = svm.SVC(kernel='poly')  # 多项式核函数
clf_poly.fit(x_train, y_train)
score_poly = clf_poly.score(x_test, y_test)
print("The score of poly is : %f" % score_poly)
```

