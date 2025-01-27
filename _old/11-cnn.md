# 人工智能与计算机视觉

## 卷积神经网络

卷积神经网络：以卷积层为主的深度神经网络结构。

卷积神经网络的结构

* 卷积层
* 激活层
* BN层
* 池化层
* FC层
* 损失层

### 卷积层

对图像和滤波矩阵做内积运算（逐个元素相乘再求和）的操作

#### 感受野（Receptive Field）

感受野是指神经网络中神经元看到的输入区域，在卷积神经网络中，特征图上某元素的计算受输入图像上某个区域的影响，这个区域即该元素的感受野。

<img src="https://raw.githubusercontent.com/hughxusu/lesson-ai/developing/_images/cnn/Q4wXPf.png" style="zoom:67%;" />

一般使用小的卷积核来代替大的卷积核

* 逐渐增加感受野
* 可以使网络层数加深

### 池化层

对输入图像进行压缩

* 减小特征图的变换，简化网络计算复杂度。
* 对特征进行压缩，提取主要特征

池化层最大池化、平均池化

![](https://raw.githubusercontent.com/hughxusu/lesson-ai/developing/_images/cnn/avgpooling_maxpooling.png)

### 激活层

激活层是增加了网络的非线性表达能力。激活函数一般是Relu激活函数

### 全连接层

将输出值给分类器

* 可以将数字特征图映射到不同长度的向量中
* 实现分类或回归分析

### 经典的神经网络结构

![](https://raw.githubusercontent.com/hughxusu/lesson-ai/developing/_images/cnn/20190716171704843.png)

#### Resnet

![](https://raw.githubusercontent.com/hughxusu/lesson-ai/developing/_images/cnn/v2-6fc1664743c9b73501d059a9010bc6e2_720w.jpg)

#### inception net

![](https://raw.githubusercontent.com/hughxusu/lesson-ai/developing/_images/cnn/1*iBkUmf7IEF7BeH1WX7_fTw.png)

## CIFAR-10数据集

CIFAR-10/100是从8000万个微小图像中提取的分类任务。

CIFAR-10数据集包含60000张32x32像素的彩色图像，共分为10个类。以下代码展示了如何加载数据、预处理数据、定义模型、编译模型以及训练模型。

```python
  from keras.datasets import cifar10

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
```

![](https://raw.githubusercontent.com/hughxusu/lesson-ai/developing/_images/cnn/4fdf2b82-2bc3-4f97-ba51-400322b228b1.png) +  
