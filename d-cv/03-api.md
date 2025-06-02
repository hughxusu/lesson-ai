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

## 连接层

### 全连接层

`in_features`是输入数据维度；`out_features`是输出数据维度。

```python
nn.Linear(in_features, out_features)
nn.Linear(4, 10)
```

`nn.Flatten()`将特征展平

### 卷积层

```python
nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
nn.Conv2d(1, 6, 3, 1, 'same')
```

* `in_channels`输入图像的通道数量。
* `out_channels`输出图像的通道数量（卷积核的数量）。
* `kernel_size`卷积核的大小。
* `stride`卷积核的步长。
* `padding`边界填充值，`padding='same'`卷积后保持图像大小。

### 池化层

最大池化

```python
nn.MaxPool2d(kernel_size, stride)
nn.MaxPool2d(2, 1)
```

* `kernel_size`池化层大小。
* `stride`窗口移动的步长。

平均池化

```python
nn.AvgPool2d(2, 2)
```

## 网络构建

PyTorch构建网络的常用方式有两种。

### `Sequential`

按顺序创建堆叠模型

```python
model = nn.Sequential(
    nn.Linear(4, 10),    
    nn.ReLU(),           
    nn.Linear(10, 10),   
    nn.ReLU(),
    nn.Linear(10, 3),  
    nn.Softmax(dim=1)
)

from torchsummary import summary
summary(model, input_size=(4,))
```

> [!warning]
>
> 在构建网络的同时，框架会自动对权重进行初始化。

### 子类构建模型

继承`nn.Module`在`__init__`函数中定义网络层，在`forward`方法中定义网络的前向传播过程。

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 1, padding='same')
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 1, padding='same')
        self.pool2 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

model = Net()
summary(model, input_size=(1, 28, 28))
```

打印模型参数`model.named_parameters()`

```python
for name, param in model.named_parameters():
    print(name, param.size())
```

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

### 随机梯度下降

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

### Adam

```python
var = torch.tensor(1.0, requires_grad=True)
opt = optim.Adam([var], lr=0.1)

for i in range(10):
    opt.zero_grad()
    loss_value = loss(var)
    loss_value.backward()
    opt.step()
    print(f"第 {i + 1} 次更新后的参数值:", var.item())
```

优化器直接内置L2正则化（模型参数添加正则化）

```python
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5) 
```

> [!warning]
>
> PyTroch可以实现，部分参数正则化或不同参数正则化不同值。

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

