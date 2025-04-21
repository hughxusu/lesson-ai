# 实践案例

## 神经网络

使用手写数字的MNIST数据集，该数据集包含60000个用于训练的样本和10000个用于测试的样本，图像是固定大小（28$\times$28像素），其值为0到255。

### 加载数据集

```python
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),          
    transforms.Lambda(lambda x: x.view(-1)) 
])

train = datasets.MNIST(root='./data', 
                       train=True, 
                       download=True,
                       transform=transform)
test = datasets.MNIST(root='./data', 
                      train=False, 
                      download=True, 
                      transform=transform)
```

* 分别读取训练数据了测试数据`train`和`test`。
* `root='./data'`指定下载路径。
* `transform`读取数据转换为`tensor`类型，并归一化到0~1之间，784维。

打印数据信息

```python
image, label = train[0]
print(image.shape) 
print(image.max().item())
print(label)
```

`image`时图片信息，`laebel`是标签信息。绘制数据图像

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 8))
for i in range(9):
    image, label = train[i]
    image = image.view(28, 28)  
    plt.subplot(3, 3, i+1)
    plt.imshow(image, cmap='gray', interpolation='none')
    plt.title("number {}".format(label))
    
plt.tight_layout()
plt.show()
```

将数据包装成`DataLoader`形式才可以进行训练

```python
from torch.utils.data import DataLoader

batch_size = 100 
# 训练集需要打乱
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True) 
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

data_iter = iter(train_loader)
image, label = next(data_iter)
print(image.shape)
print(label.shape)
```

### 构建网络

```python
from torch import nn
from torchsummary import summary

model = nn.Sequential(
    nn.Linear(784, 512), 
    nn.ReLU(),           
    nn.Dropout(0.5),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 10),
    nn.Softmax(dim=1)
)

print(summary(model, input_size=(784,)))
```

* 使用`Sequential`构建网络
* `Dropout`放在每层的激活函数之后

打印模型参数`model.named_parameters()`

```python
for name, param in model.named_parameters():
    print(name, param.size())
```

### 损失函数

定义交叉熵损失函数，

```python
from torch import optim

loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)
```

跟踪loss反向传播的方向，使用`grad_fn`属性打印，查看反向传播的路径。

```python
outputs = model(image)
loss = loss_fn(outputs, label)

node = loss.grad_fn
while node is not None:
    print(f"→ {str(node)}")  # 截断输出避免过长
    node = node.next_functions[0][0] if node.next_functions else None
```

### 模型训练







### 模型评估

### 模型保存与加载



## CNN网络

### 数据处理

卷积神经网络的输入要求是：

* N图片数量

* C图片的通道，因为是灰度图，通道为1。

* H图片高度

* W图片宽度
