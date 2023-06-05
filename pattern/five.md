# K-近邻算法

![](http://www.itcast.cn/files/image/202104/20210428112138157.jpg)

根据你的“邻居”来推断出你的类别

K Nearest Neighbor算法又叫KNN算法，这个算法是机器学习里面一个比较经典的算法， 总体来说KNN算法是相对比较容易理解的算法。

如果一个样本在特征空间中的k个最相似(即特征空间中最邻近)的样本中的大多数属于某一个类别，则该样本也属于这个类别。

距离公式：两个样本的距离可以通过如下公式计算，又叫欧式距离 ，关于距离公式会在后面进行讨论。

![](file:///Users/xusu/Downloads/lesson/%E9%98%B6%E6%AE%B55-%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD%E7%BB%8F%E5%85%B8%E7%AE%97%E6%B3%95%E7%BC%96%E7%A8%8B/3.others/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0(%E7%AE%97%E6%B3%95%E7%AF%87)/images/1.3%20%E6%AC%A7%E5%BC%8F%E8%B7%9D%E7%A6%BB1.png)

**电影类型分析**

假设我们现在有几部电影

![](file:///Users/xusu/Downloads/lesson/%E9%98%B6%E6%AE%B55-%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD%E7%BB%8F%E5%85%B8%E7%AE%97%E6%B3%95%E7%BC%96%E7%A8%8B/3.others/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0(%E7%AE%97%E6%B3%95%E7%AF%87)/images/knn%E7%94%B5%E5%BD%B1%E4%B8%BE%E4%BE%8B1.png)

其中？ 号电影不知道类别，如何去预测？我们可以利用K近邻算法的思想

![](file:///Users/xusu/Downloads/lesson/%E9%98%B6%E6%AE%B55-%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD%E7%BB%8F%E5%85%B8%E7%AE%97%E6%B3%95%E7%BC%96%E7%A8%8B/3.others/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0(%E7%AE%97%E6%B3%95%E7%AF%87)/images/knn%E7%94%B5%E5%BD%B1%E4%B8%BE%E4%BE%8B2.png)

分别计算每个电影和被预测电影的距离，然后求解

![](file:///Users/xusu/Downloads/lesson/%E9%98%B6%E6%AE%B55-%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD%E7%BB%8F%E5%85%B8%E7%AE%97%E6%B3%95%E7%BC%96%E7%A8%8B/3.others/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0(%E7%AE%97%E6%B3%95%E7%AF%87)/images/knn%E7%94%B5%E5%BD%B1%E4%B8%BE%E4%BE%8B3.png)

最近邻 (k-Nearest Neighbors，KNN) 算法是一种分类算法，1968年由 Cover 和 Hart 提出，应用场景有字符识别、文本分类、图像识别等领域。

该算法的思想是：一个样本与数据集中的k个样本最相似，如果这k个样本中的大多数属于某一个类别。

实现流程：

1. 计算已知类别数据集中的点与当前点之间的距离
2. 按距离递增次序排序
3. 选取与当前点距离最小的k个点
4. 统计前k个点所在的类别出现的频率
5. 返回前k个点出现频率最高的类别作为当前点的预测分类

## Scikit-learn

- Python语言的机器学习工具
- Scikit-learn包括许多知名的机器学习算法的实现
- Scikit-learn文档完善，容易上手，丰富的API

[Scikit-learn 官方网站](https://scikit-learn.org/stable/#)

```python
from sklearn.neighbors import KNeighborsClassifier

x = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]

estimator = KNeighborsClassifier(n_neighbors=2)
estimator.fit(x, y)
estimator.predict([[2]])
```

## 距离度量

### 欧氏距离

![](https://s1.ax1x.com/2023/06/05/pCPZRgg.png)

### 曼哈顿距离

在曼哈顿街区要从一个十字路口开车到另一个十字路口，驾驶距离显然不是两点之前的直线距离。这个实际的驾驶距离就是"曼哈顿距离"。曼哈顿距离也称“城市街区距离”。

![](https://s1.ax1x.com/2023/06/05/pCPZ5bn.png)

### 切比雪夫距离

国际象棋中，国王可以直行、横行、斜行，所以国王走一步可以移动到相邻8个方格中的任意一个。国王从格子(x1,y1)走到格子(x2,y2)最少需要多少步？这个距离就叫切比雪夫距离。

![](https://s1.ax1x.com/2023/06/05/pCPZbCT.png)

### 闵可夫斯基距离

闵氏距离不是一种距离，而是一组距离的定义，是对多个距离度量公式的概括性的表述。

两个n维变量a(x11,x12,…,x1n)与b(x21,x22,…,x2n)间的闵可夫斯基距离定义为：

![](https://s1.ax1x.com/2023/06/05/pCPZxbR.png)

其中p是一个变参数：

当p=1时，就是曼哈顿距离；

当p=2时，就是欧氏距离；

当p→∞时，就是切比雪夫距离。

根据p的不同，闵氏距离可以表示某一类/种的距离。

### 标准化欧氏距离

标准化欧氏距离是针对欧氏距离的缺点而作的一种改进。

思路：既然数据各维分量的分布不一样，那先将各个分量都“标准化”到均值、方差相等。假设样本集X的均值(mean)为m，标准差(standard deviation)为s，X的“标准化变量”表示为：

![](https://s1.ax1x.com/2023/06/05/pCPei8O.png)

如果将方差的倒数看成一个权重，也可称之为加权欧氏距离(Weighted Euclidean distance)。

### 余弦距离

几何中，夹角余弦可用来衡量两个向量方向的差异；机器学习中，借用这一概念来衡量样本向量之间的差异。

- 二维空间中向量A(x1,y1)与向量B(x2,y2)的夹角余弦公式：

![](https://s1.ax1x.com/2023/06/05/pCPeVrd.png)

- 两个n维样本点a(x11,x12,…,x1n)和b(x21,x22,…,x2n)的夹角余弦为：

  ![](https://s1.ax1x.com/2023/06/05/pCPelRS.png)

即：

![](https://s1.ax1x.com/2023/06/05/pCPe8MQ.png)

夹角余弦取值范围为[-1,1]。余弦越大表示两个向量的夹角越小，余弦越小表示两向量的夹角越大。当两个向量的方向重合时余弦取最大值1，当两个向量的方向完全相反余弦取最小值-1。

## k 值的选择

K值过小：容易受到异常点的影响

k值过大：受到样本均衡的问题

1. 选择较小的K值，就相当于用较小的领域中的训练实例进行预测，“学习”近似误差会减小，只有与输入实例较近或相似的训练实例才会对预测结果起作用，与此同时带来的问题是“学习”的估计误差会增大，换句话说，K值的减小就意味着整体模型变得复杂，容易发生过拟合；
2. 选择较大的K值，就相当于用较大领域中的训练实例进行预测，其优点是可以减少学习的估计误差，但缺点是学习的近似误差会增大。这时候，与输入实例较远（不相似的）训练实例也会对预测器作用，使预测发生错误，且K值的增大就意味着整体的模型变得简单。
3. K=N（N为训练样本个数），则完全不足取，因为此时无论输入实例是什么，都只是简单的预测它属于在训练实例中最多的类，模型过于简单，忽略了训练实例中大量有用信息。

在实际应用中，K值一般取一个比较小的数值，例如采用交叉验证法（简单来说，就是把训练数据在分成两组:训练集和验证集）来选择最优的K值。对这个简单的分类器进行泛化，用核方法把这个线性模型扩展到非线性的情况，具体方法是把低维数据集映射到高维特征空间

## 鸢尾花种类预测--数据集介绍

Iris数据集是常用的分类实验数据集，由Fisher, 1936收集整理。Iris也称鸢尾花卉数据集，是一类多重变量分析的数据集。关于数据集的具体介绍：

![](https://s1.ax1x.com/2023/06/05/pCPmkF0.png)

获取数据集

```python
from sklearn.datasets import load_iris
iris = load_iris()

print("鸢尾花数据集的返回值：\n", iris)
print("鸢尾花的特征值:\n", iris["data"])
print("鸢尾花的目标值：\n", iris.target)
print("鸢尾花特征的名字：\n", iris.feature_names)
print("鸢尾花目标值的名字：\n", iris.target_names)
print("鸢尾花的描述：\n", iris.DESCR)
```

### 查看数据分布

通过创建一些图，以查看不同类别是如何通过特征来区分的。 在理想情况下，标签类将由一个或多个特征对完美分隔。 在现实世界中，这种理想情况很少会发生。

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

iris_d = pd.DataFrame(iris['data'], columns = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width'])
iris_d['Species'] = iris.target

def plot_iris(iris, col1, col2):
    sns.lmplot(x = col1, y = col2, data = iris, hue = "Species", fit_reg = False)
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.show()
plot_iris(iris_d, 'Petal_Width', 'Sepal_Length')
```

### 数据集的划分

机器学习一般的数据集会划分为两个部分：

- 训练数据：用于训练，**构建模型**
- 测试数据：在模型检验时使用，用于**评估模型是否有效**

划分比例：

- 训练集：70% 80% 75%
- 测试集：30% 20% 25%

**数据集划分api**

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()

x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=22)
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=2)
```

