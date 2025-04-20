# 实践案例

## 神经网络

使用手写数字的MNIST数据集，该数据集包含60000个用于训练的样本和10000个用于测试的样本，图像是固定大小（28$\times$28像素），其值为0到255。

### 加载数据集

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
```

* 分别读取训练数据了测试数据`train`和`test`。
* `root='./data'`指定下载路径。
* `transform=transforms.ToTensor()`下载的数据转换为`tensor`类型，并归一化到0~1之间。

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
    plt.subplot(3, 3, i+1)
    plt.imshow(image[0], cmap='gray', interpolation='none')
    plt.title("number {}".format(label))
    
plt.tight_layout()
plt.show()
```

将图像数据转换成一维向量

```python
print(train.data.size())
print(train.targets.size())
print(test.data.size())
print(test.targets.size())

print(50*'*')

def dataset_to_matrix(dataset):
    return dataset.view(len(dataset), -1)

x_train = dataset_to_matrix(train.data)
x_test = dataset_to_matrix(test.data)
y_train = train.targets
y_test = test.targets

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
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

```python
```



### 损失函数

关于方向传播的链条，如果我们跟踪loss反向传播的方向，使用`grad_fn`属性打印, 将可以看到一张完整的计算。



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
