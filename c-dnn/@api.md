# PyTorch常用API

## 连接层

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

## 网络优化

优化器直接内置L2正则化（模型参数添加正则化） 

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

