# PyTorch常用API

## 自动求导（autograd）

在整个PyTorch框架中，所有的神经网络，本质上都是一个自动求导工具包（autograd package），它提供了一个，对Tensors上所有的操作，进行自动微分的功能。

### 关于`torch.tensor`

`torch.tensor`是整个package中的核心类：

*  将属性`requires_grad`设置为`True`，将追踪在这个类上定义的所有操作。
  * 当代码进行反向传播时，直接调用`backward()`，可以自动计算梯度。
  * `tensor`上的所有梯度，被累加进属性`grad`中。
* 如果终止一个`tensor`在计算图中的回溯，只需要执行`detach()`，就可以将该`tensor`从计算图中撤下。
* 如果想终止对整个计算图的回溯，也就是不再进行反向传播， 可以采用代码块的方式`with torch.no_grad():`，一般适用于模型推理阶段。

### 关于`torch.function`

`torch.function`是和`tensor`同等重要的一个核心类：

* 每一个`tensor`拥有一个`grad_fn`属性，代表引用了哪个函数创建了该`tensor`。
* 用户自定义的`tensor`是，`grad_fn`属性是`None`。

### 自动求导属性

```python
import torch

x1 = torch.ones(3, 3)
print(x1)

x = torch.ones(2, 2, requires_grad=True)
print(x)
```

对`requires_grad=True`的`tensor`执行加法操作

```python
y = x + 2
print(y)
```

打印`tensor`的`grad_fn`属性

```python
print(x.grad_fn)
print(y.grad_fn)
```

在`tensor`上执行更复杂的操作

```python
z = y * y * 3
out = z.mean()
print(z, out)
```

关于方法`requires_grad_()`方法，可以原地改变`tensor.requires_grad`的属性值，默认值为`False`

```python
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)
```

在PyTorch中，反向传播是依靠`backward()`实现

```python
out.backward()
print(x.grad)
```

关于自动求导的属性，可以通过`requires_grad=True`设置，也可以通过代码块来停止自动求导

```python
print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)
```

可以通过`detach()`获得一个新的`tensor`，拥有相同内容，不需要自动求导

```python
print(x.requires_grad)
y = x.detach()
print(y.requires_grad)
print(x.eq(y).all())
```

## 激活函数

### Sigmoid

```python
import numpy as np
import matplotlib.pyplot as plt

x = torch.tensor(np.linspace(-10, 10, 100)) 
y = nn.Sigmoid()(x)

plt.figure(figsize=(10, 5))
plt.plot(x, y) 
plt.grid()
plt.xlabel('x')
plt.ylabel('sigmoid(x)')
plt.show()
```

### Tanh

```python
x = torch.tensor(np.linspace(-10, 10, 100)) 
y = nn.Tanh()(x)

plt.figure(figsize=(10, 5))
plt.plot(x, y) 
plt.grid()
plt.xlabel('x')
plt.ylabel('tanh(x)')
plt.show()
```

### Relu

```python
x = torch.tensor(np.linspace(-10, 10, 100)) 
y = nn.ReLU()(x)

plt.figure(figsize=(10, 5))
plt.plot(x, y) 
plt.grid()
plt.xlabel('x')
plt.ylabel('relu(x)')
plt.show()
```

### Softmax

```python
x = torch.tensor([0.2, 0.02, 0.15, 1.3, 0.5, 0.06, 1.1, 0.05, 3.75])
y = nn.Softmax(dim=0)(x)
print(y)
print(torch.sum(y))
```

## 网络构建

PyTorch构建网络的常用方式有两种。

### `nn.Sequential`

> [!warning]
>
> 在构建网络的同时，框架会自动对权重进行初始化。

### Model的子类构建模型

### 模型方法

模型中所有的可训练参数, 可以通过`parameters()`来获得

## 损失函数

MSE损失函数`nn.MSELoss`

```python
import torch
from torch import nn

y_true = torch.tensor([[0.], [1.]])
y_pred = torch.tensor([[1.], [1.]], requires_grad=True)

mse = nn.MSELoss()
loss = mse(y_true, y_pred)
```

在PyTorch中执行反向传播，调用`loss.backward()`：

1. 整张计算图将对损失函数进行自动求导；
2. 有属性`requires_grad=True`的Tensor将参计算梯度；
3. 梯度计算的结果，累加到这些Tensors的`grad`属性中；

```python
loss.backward()
print(loss)
print(y_pred.grad)
print(y_true.grad)
```

交叉熵损失函数

```python
y_true = torch.tensor([1, 2])
y_pred = torch.tensor([[0.05, 0.95, 0], [0.1, 0.8, 0.1]])

loss = nn.CrossEntropyLoss()
loss_value = loss(y_pred, y_true)
print(loss_value)
```

## 网络优化

测试用例

1. 损失函数为$0.5x^2$。
2. 初始值为1。
3. 计算一次梯度下降后的值。

### 梯度下降算法

随机梯度下降的计算公式
$$
W \leftarrow W-\eta\frac{\partial L}{\partial W}
$$

在优化网络时，参数的更新不止一次，需要反复迭代，以达到最优值。


```python
from torch import optim

def loss(var):
    return (var ** 2) / 2.0

var = torch.tensor(1.0, requires_grad=True)
opt = optim.SGD([var], lr=0.1)

for i in range(10):
    opt.zero_grad()
    loss_value = loss(var)
    loss_value.backward()
    opt.step()
    print(f"第 {i+1} 次更新后的参数值:", var.item())
```

> [!warning]
>
> 在执行反向传播之前，要将梯度清零，调用`opt.zero_grad()`，否则梯度会在不同的批次数据之间被累加。

### Dropout

```python
data = np.arange(1, 11).reshape(5, 2).astype(np.float32)
data_tensor = torch.from_numpy(data)  
print("输入数据:\n", data_tensor)

dropout_layer = torch.nn.Dropout(p=0.5)
dropout_layer.train()  # 设置为训练模式
outputs = dropout_layer(data_tensor)
print("Dropout输出:\n", outputs)
```

