# 深度学习DNN

Deep Neural Network (DNN)

```mermaid
flowchart LR
		z(野猪)-->a(肉猪)
```

只有感性认识，理论支持较弱。

深度学习只是机器学习的一种工具。

对于线性不可分数据集

<img src="../_images/dnn/8557a73d-684c-4d8e-8f4b-a35238ce763e.webp" style="zoom:67%;" />

> [!note]
>
> 线性变换是否可以将线性不可分变为线性可分？

$$
\begin{pmatrix}
a & b \\
c & d
\end{pmatrix}
\begin{pmatrix}
x \\
y
\end{pmatrix}
=
\begin{pmatrix}
x' \\
y'
\end{pmatrix}
$$

线性变换无法将线性不可分变为线性可分，如果想让数据可分需要通过非线性变换。
$$
y = f(Wx + b)
$$
其中$f$函数是Sigmoid函数

![](../_images/dnn/neural-network-2.png)

隐层的输出表示为，下式：
$$
H=f(W_1x+W_0)
$$
$f$为激活函数。每个神经元可以看做是一个逻辑回归。当隐层足够多的时候即为深度学习。

![](../_images/dnn/1*N8UXaiUKWurFLdmEhEHiWg.jpeg)

最后一层是标准的逻辑回归，输出有概率意义。

> [!warning]
>
> 层数越多模型越复杂，容易过拟合。在满足任务的条件下层数越少越好。

每层的神经元个数可以任意组合。

```mermaid
flowchart LR
		z(x输入变量)-->a(神经网络)-->b(概率值)
```

神经网络中的激活函数可以不是Sigmoid，只要是非线性函数即可。

对于二分类问题输出一类的概率为$y$，另一类的概率为$1-y$。

对于多分类的概率$y_1+y_2+y_3+y_4=1$​，对于Sigmoid输出的概率值无法保证概率和为1。

对于多分类的最后一层输出使用softmax函数。
$$
\text{softmax}(x) = \frac{e^{-W_jx}}{\sum_{j=1}^{N} e^{-W_jx}}
$$
对于二分类问题softmax函数为
$$
\left\{
\begin{array}{ccc}
y_1 = \frac{e^{-W_1x}}{e^{-W_1x} + e^{-W_2x}} \\
y_2 = \frac{e^{-W_2x}}{e^{-W_1x} + e^{-W_2x}} \\
\end{array}
\right.
$$
对于$y_1$的输出
$$
y_1=\frac{1}{1 + e^{-(W_1-W_2)x}}
$$
将$W_1-W_2=W$则输出结果可以看做是Sigmoid函数。对于二分类问题输出是LR比softmax更好，因为可以少计算一组参数。

softmax也是一个非线性函数，但由于其计算量较大只用在最后一层，中间层一般不用。

## 机器学习框架

| 框架                                               | 优点                                                         | 缺点                                                         | GitHub Stars | 公司     |
| -------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------ | -------- |
| [Caffe](https://caffe.berkeleyvision.org/)         | \- 简单易用的接口                                            | \- 功能相对较为有限                                          | 约 33k       | Facebook |
| [TensorFlow](https://www.tensorflow.org/?hl=zh-cn) | - 强大的生态系统和支持<br>- 良好的文档和社区支持<br>- 支持灵活的部署（包括移动端和嵌入式设备） | - 相对较复杂，学习曲线较陡峭<br>- 部分功能可能不够直观       | 约 160k      | Google   |
| [PyTorch](https://pytorch.org/)                    | - 动态图模式更直观，易于调试<br>- 灵活性高，易于定制<br>- 易于在GPU上进行加速计算 | - 相对TensorFlow较新，生态系统可能不及其成熟<br>- 文档相对不够完善 | 约 66k       | Facebook |
| [Keras](https://keras.io/)                         | - 高度模块化，易于使用<br>- 抽象层次较高，适合快速原型设计<br>- 与多个后端兼容（如TensorFlow、Theano等） | - 灵活性相对较差，不够适合定制<br>- 性能可能略逊于TensorFlow和PyTorch | 约 52k       | Google   |
| [PaddlePaddle](https://www.paddlepaddle.org.cn/)   | - 易用性高，提供了易于上手的高级API和简单直观的编程接口<br>- 灵活性高，支持静态图和动态图两种模式 | - 生态系统相对较小，社区资源和第三方工具相对较少<br>- 文档和教程相对不足<br>- 国际化程度有限 | 约 23k       | Baidu    |
