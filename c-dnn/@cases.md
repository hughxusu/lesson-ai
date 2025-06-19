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

## Cloud Studio

[Cloud Studio](https://ide.cloud.tencent.com/)是腾讯云推出的一款云端集成开发环境（IDE），基于浏览器运行，为开发者提供无需本地配置的远程开发体验。

* 通过Web IDE实现远程编码、调试和部署，支持Java、Python、Node.js等多种语言及框架模板。
* 支持GPU运算，可以用于神经网络训练。
* 支持TensorFlow和PyTorch框架。
* 集成VSCode开发环境和AI代码助手。
* 支持主流代码仓库（GitHub、CODING）的云端克隆。

可以使用微信登录账号

<img src="https://raw.githubusercontent.com/hughxusu/lesson-ai/developing/_images/cv/Xnip2025-05-02_11-21-06.jpg" style="zoom:40%;" />

选择模板

<img src="https://raw.githubusercontent.com/hughxusu/lesson-ai/developing/_images/cv/Xnip2025-05-02_11-26-03.jpg" style="zoom:40%;" />

选择以创建过的应用

<img src="https://raw.githubusercontent.com/hughxusu/lesson-ai/developing/_images/cv/Xnip2025-05-02_11-29-58.jpg" style="zoom:40%;" />

进入开发

<img src="https://raw.githubusercontent.com/hughxusu/lesson-ai/developing/_images/cv/Xnip2025-05-02_11-33-06.jpg" style="zoom:40%;" />

代码编辑机器类似于VSCode

<img src="https://raw.githubusercontent.com/hughxusu/lesson-ai/developing/_images/cv/Xnip2025-05-02_11-38-02.jpg" style="zoom:40%;" />

查看GPU配置

```python
import torch

if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    print(f"可用的GPU数量: {gpu_count}")
    
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        print(f"GPU {i}: {gpu_name}")
else:
    print("没有可用的GPU")

```

## CNN网络

### 数据加载

```python
from torchvision import datasets, transforms

train = datasets.MNIST(root='./data', 
                       train=True, 
                       download=True,
                       transform=transforms.ToTensor())
test = datasets.MNIST(root='./data', 
                      train=False, 
                      download=True, 
                      transform=transforms.ToTensor())

image, label = train[0]
print(image.shape) 
print(label)
```

使用`DataLoader`加载数据

```python
from torch.utils.data import DataLoader

batch_size = 256  
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

data_iter = iter(train_loader)
image, label = next(data_iter)
print(image.shape)
print(label.shape)
```

卷积神经网络的输入要求是

* N图片数量

* C图片的通道，因为是灰度图，通道为1。

* H图片高度

* W图片宽度

### 构建模型

```python
import torch.nn as nn
from torchsummary import summary

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, padding='same')
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3, padding='same')
        self.pool2 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 7 * 7, 84)
        self.fc2 = nn.Linear(84, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.softmax(self.fc2(x))
        return x
```

自动匹配GPU

```python
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net()
model.to(device)
summary(model, input_size=(1, 28, 28))
```

### 定义损失函数

```python
from torch import optim

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4) 
```

### 模型训练

```python
epochs = 50

# 记录训练过程中的损失和准确率
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

for epoch in range(epochs):
    # 训练阶段
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    # 计算训练指标
    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    
    # 验证阶段
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # 计算验证指标
    test_loss = test_loss / len(test_loader)
    test_accuracy = 100 * correct / total
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)
    
    # 打印每个epoch的结果
    print(f"Epoch {epoch+1} / {epochs}: "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
          f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%")

# 打印最终结果
print("\nTraining complete!")
print(f"Best Test Accuracy: {max(test_accuracies):.2f}% at epoch {test_accuracies.index(max(test_accuracies))+1}")
```

绘制训练过程曲线

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_accuracies, label='Train Accuracy', color='blue', marker='o')
plt.plot(test_accuracies, label='Test Accuracy', color='green', marker='o')
plt.title('Accuracy vs. Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(train_losses, label='Train Loss', color='red', marker='o')
plt.plot(test_losses, label='Test Loss', color='orange', marker='o')
plt.title('Loss vs. Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

