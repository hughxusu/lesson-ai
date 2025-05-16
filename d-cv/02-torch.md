# PyTorch入门

[Pytorch](https://pytorch.org/)是由Facebook AI Research (FAIR)开发的开源深度学习框架，是一个基于Numpy的科学计算包，向它的使用者提供了两大功能。与Tensorflow对比，PyTorch在GitHub上的开源项目，数量和社区活跃度方面，略占优势，尤其在研究和学术领域。

* Hugging Face Transformers提供BERT、GPT-2、T5、RoBERTa等预训练模型，支持文本分类、翻译、生成等任务。
* 常用的开源大语言模型基于PyTorch
  * Deepseek官方推荐PyTorch。
  * Facebook开源模型LLaMA基于PyTorch。
* YOLO开源工具V5、V8、V11均基于PyTorch。
* Stable Diffusion基于 扩散模型（Diffusion Models）的文本到图像生成工具。
* FaceFusion高度真实感的换脸工具。
* LLM-Driver结合了大型语言模型（LLM）与自动驾驶任务。

> [!warning]
>
> 目前知名的开源项目，大多是基于PyTorch开发。

核心特点

* 动态计算图：计算图，在代码运行时动态构建，灵活性高，适合调试和研究。对比 TensorFlow 1.x 的静态图（Define-and-Run），PyTorch 更直观，适合快速实验。
* 张量计算：作为Numpy的替代者，向用户提供使用GPU强大功能的能力。
* 自动微分：自动计算梯度，简化反向传播。
* 模块化神经网络：做为一款深度学习的平台，向用户提供最大的灵活性和速度。
* GPU 加速（CUDA 支持）：只需简单命令即可实现GPU加速。
* 数据加载与预处理：使用`Dataset` 和 `DataLoader`方便数据批处理和多线程加载。

工作流

<img src="https://raw.githubusercontent.com/hughxusu/lesson-ai/developing/_images/cv/01_a_pytorch_workflow.png" style="zoom:75%;" />

可视化

* 支持Tensorboard可视化分析。

模型仓库

* PyTorch提供了[PyTorch Hub](https://pytorch.org/hub/)官方模型仓库。
* [HuggingFace](https://huggingface.co/)：开源的AI工具库中的预训练模型。

学习资料

* 学习网站
  * [官方教程](https://pytorch.org/tutorials/)
  * [20天吃掉那只Pytorch](https://jackiexiao.github.io/eat_pytorch_in_20_days/)
  * [深入浅出PyTorch](https://datawhalechina.github.io/thorough-pytorch/index.html#)
* 学习书籍
  * [《PyTorch深度学习实战》](https://book.douban.com/subject/35776474/)

PyTorch的安装

* [安装命令](https://pytorch.org/get-started/locally/)
  * `torch`为PyTorch的核心包。
  * `torchvision`专为计算机视觉任务设计的扩展库。
  * `torchaudio`音频处理库。


模型的转换

* ONNX可以对不同框架模型进行转换，可以将Tensorflow模型转换为PyTorch模型。

## PyTorch基本语法

### 张量及其操作

张量（Tensors）是一种多为数组，它可以看做是矩阵和向量的推广。

<img src="https://raw.githubusercontent.com/hughxusu/lesson-ai/developing/_images/cv/0ca2fd5a6590d22027e3058b497fdff1.jpeg" style="zoom:50%;" />

在PyTorch中，张量的概念类似于Numpy中的`ndarray`数据结构，最大的区别在于Tensor可以利用GPU的加速功能。张量的类型为`tensor`，具有数据类型和形状。

<img src="https://raw.githubusercontent.com/hughxusu/lesson-ai/developing/_images/cv/656a769280b04c.jpg" style="zoom:75%;" />

使用PyTorch前，需要引入相关包

```python
import torch
```

### 基本方法

创建一个没有初始化的矩阵

```python
x = torch.empty(5, 3)
print(x)
```

> [!warning]
>
> 有些版本的PyTorch，创建一个未初始化的矩阵时，分配给矩阵的内存，包含残留数据。

随机初始化矩阵（标准高斯分布初始化数据）

```python
x = torch.rand(5, 3)
print(x)
```

创建一个全零矩阵，指定数据类型为`long`

```python
x = torch.zeros(5, 3, dtype=torch.long)
print(x)
```

直接通过数据创建张量。`tensor`可以封装不同类型的数据

```python
x = torch.tensor([2.5, 3.5])
print(x)
```

通过已有的张量，创建相同尺寸的新张量。

```python
x = x.new_ones(5, 3, dtype=torch.double)
print(x)

y = torch.randn_like(x, dtype=torch.float)
print(y)

```

得到张量的尺寸。`size()`方法返回的是一个元组，它支持一切元组的操作，如：拆包等。

```python
print(x.size())
print(y.size())
a, b = x.size()
print(a, b)
```

创建数字、向量和矩阵

```python
rank_0_tensor = torch.tensor(4)
print(rank_0_tensor)

rank_1_tensor = torch.tensor([2.0, 3.0, 4.0])
print(rank_1_tensor)

rank_2_tensor = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float16)
print(rank_2_tensor)
```

<img src="https://raw.githubusercontent.com/hughxusu/lesson-ai/developing/_images/cv/98602c4fbd104a9a91a9eba0b7099fb7.png" style="zoom:55%;" />

创建3维张量

```python
rank_3_tensor = torch.tensor([
    [[0, 1, 2, 3, 4],
     [5, 6, 7, 8, 9]],
    [[10, 11, 12, 13, 14],
     [15, 16, 17, 18, 19]],
    [[20, 21, 22, 23, 24],
     [25, 26, 27, 28, 29]],])
print(rank_3_tensor)
```

<img src="https://raw.githubusercontent.com/hughxusu/lesson-ai/developing/_images/cv/6206db46441e4f10b33bd752d7892b41.png" style="zoom:55%;" />

### 张量的运算

[加法操作](https://pytorch.org/docs/stable/generated/torch.add.html)

```python
print(x + y)
print(torch.add(x, y))
```

指定输出变量的加法操作

```python
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)
```

加法操作，原地置换（in-place）

```python
print(y)
y.add_(x)
print(y)
```

> [!warning]
>
> 原地置换会修改变量，所有原地置换操作函数，都有一个下划线的后缀，如：`x.copy_(y)`。

用类似于Numpy的方式对张量进行操作

```python
print(y[:, 1])
```

改变张量的形状`torch.view()`

* 操作需要保证数据元素的总数量不变。
* `-1`代表自动匹配个数。

```python
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)
print(x.size(), y.size(), z.size())

```

如果张量中**只有一个元素**，可以用`item()`将值取出，作为一个python number。

```python
x = torch.randn(1)
print(x)
print(x.item())
```

> [!warning]
>
> `item()`常用在获取损失（loss）值时。获取Python列表[`tolist`](https://pytorch.org/docs/stable/generated/torch.Tensor.tolist.html)

### `tensor`和`ndarray`的转换

Torch的`tensor`和Numpy的`ndarray`共享底层的内存空间，因此改变其中一个的值，另一个也会随之被改变。

```python
a = torch.ones(5)
print(a)
```

将`tensor`转换为`ndarray`

```python
b = a.numpy()
print(b)
```

对其中一个进行加法操作，另一个也随之被改变。

```python
a.add_(1)
print(a)
print(b)
```

将`ndarray`转换为`tensor`

```python
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
print(a)
print(b)
print(20*'-')
np.add(a, 1, out=a)
print(a)
print(b)
```

> [!warning]
>
> 所有在CPU上的`tensor`，除了CharTensor，都可以转换为`ndarray`并可以反向转换。

## 模型搭建

鸢尾花分类

<img src="https://raw.githubusercontent.com/hughxusu/lesson-ai/developing/_images/base/Iris-Dataset-Classification.png" style="zoom:50%;" />

读取数据集

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
```

### sk-learn实现

使用逻辑回归模型分类鸢尾花

```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(x_train, y_train)
print(lr.score(x_test, y_test))
```

### PyTorch实现

构建神经网络的一般流程

1. 定义一个拥有可学习参数的神经网络。
2. 遍历训练数据集。
3. 输入数据计算前向传播结果。
4. 根据计算结果和监督数据，计算损失值。
5. 根据损失值，计算梯度进行反向传播。
6. 以一定的规则，更新网络的权重。
7. 重复上述过程，迭代一定的次所，或者损失函数不在减少时停止。

鸢尾花的分类网络如下

<img src="https://raw.githubusercontent.com/hughxusu/lesson-ai/developing/_images/cv/51c3298d5b3f85b7b51530b6fb30ef02.png" style="zoom:65%;" />

其中两隐层为10个神经元。

1. 准备数据，将数据转换为张量形式。

```python
x_train_tensor = torch.FloatTensor(x_train)
y_train_tensor = torch.LongTensor(y_train)
x_test_tensor = torch.FloatTensor(x_test)
y_test_tensor = torch.LongTensor(y_test)
```

2. PyTorch来构建神经网络，主要工具都在`torch.nn`包中

```python
from torch import nn
```

3. 使用`nn.Sequential`创建模型，模型是层的线性堆叠，该模型的构造函数会采用一系列层实例。

```python
model = nn.Sequential(
    nn.Linear(4, 10), 
    nn.ReLU(),           
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10, 3),
    nn.Softmax(dim=1)
)
```

4. 安装`pip install torchsummary`，打印网络信息。

```python
from torchsummary import summary
summary(model, input_size=(4,))
```

5. 创建损失函数和优化器

```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

6. 训练模型，PyTorch需要手动编写训练循环。

```python
num_epochs = 500

model.train()
for epoch in range(num_epochs):
    # 前向传播
    outputs = model(x_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    _, train_predicted = torch.max(outputs.data, 1)
    train_accuracy = (train_predicted == y_train_tensor).sum().item() / y_train_tensor.size(0)
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, 训练集准确率: {train_accuracy:.4f}')
```

> [!warning]
>
> PyTorch的设计哲学更倾向于灵活性和显式控制训练过程，而TensorFlow的`compile()`提供了更高层次的抽象。

`model.train()`设置为训练模式，不同层状态可能不同。修改优化器为`Adam`

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
num_epochs = 100
```

7. 评价模型性能。`model.eval()`设置为验证模式，`with torch.no_grad()`不需要反向传播运算。

```python
model.eval()
with torch.no_grad():
    outputs = model(x_test_tensor)
    _, predicted = torch.max(outputs.data, 1)
    accuracy = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
    print(f'测试集准确率: {accuracy:.4f}')
```

