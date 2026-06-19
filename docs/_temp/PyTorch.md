# PyTorch

## PyTorch的基本使用

### Tensor使用

#### 创建张量

```python
import torch
a = torch.Tensor([[1, 2],[3, 4]]) # 指定矩阵创建
d = torch.tensor(((1, 2), (3, 4))) # 使用元组创建

d.type() # 数据类型

b = torch.Tensor(2, 2) # 指定形状创建

d = torch.ones(2, 2) # 全1矩阵
d = torch.eye(2, 2) # 对角线为1
d = torch.zeros(2,3) # 全0矩阵

d = torch.ones_like(d) # 与某个矩阵形状相同的全1矩阵
d = torch.zeros_like(d) # 与某个矩阵形状相同的全0矩阵

d = torch.rand(2, 3) # 创建随机张量，值为0~1之间

# 满足正态分布的随机张量
d = torch.normal(mean=0, std=1, size=(2, 3), out=b)
d = torch.normal(mean=torch.rand(5), std=torch.rand(5))

d = torch.Tensor(2, 2).uniform_(-1, 1) # 均匀分布的随机张量

d = torch.arange(2, 10, 2) # 序列定义，从2到10不包含10，步长为2

d = torch.linspace(10, 2, 3) # 生成等间隔的张量

d = torch.randperm(10) # 生成随机序列
```

#### Tensor属性

* `.dtype` 数据类型
* `.device` 数据对象创建之后所存储的设备名称（GPU或CPU）
* `.layout` 数据在内存中的布局（稠密或稀疏）

```python
import torch

dev = torch.device("cpu") # 数据定义在cpu上, 默认设备是cpu
dev = torch.device("cuda") # 数据定义在gpu上
a = torch.tensor([2, 2], dtype=torch.float32, device=dev) # 指定数据类型和设备
print(a)

# 定义稀疏矩阵
i = torch.tensor([[0, 1, 2], [0, 1, 2]]) # 非零元素坐标x轴和y轴
v = torch.tensor([1, 2, 3]) # 非零元素值
a = torch.sparse_coo_tensor(i, v, (4, 4),
                            dtype=torch.float32,
                            device=dev)
print(a)
print(a.to_dense()) # 将稀疏矩阵转换为稠密矩阵
```

#### 运算

##### 算数运算

对应位置的元素进行运算，矩阵的行踪需要相同

```python
import torch

a = torch.rand(2, 3)
b = torch.rand(2, 3)

# 前三种表达式计算结构相同
c = a + b
c = a.add(b)
c = torch.add(a, b)
a.add_(b) # 将计算结果写入矩阵a，原位操作(in-place)

print(a - b)
print(a.sub(b))
print(torch.sub(a, b))
print(a.sub_(b))

print(a * b)
print(a.mul(b))
print(torch.mul(a, b))

print(a.mul_(b))

print(a / b)
print(a.div(b))
print(torch.div(a, b))
print(a.div_(b))

# 指数
a = torch.tensor([1, 2])
print(a**3)
print(a.pow(3))
print(torch.pow(a, 3))
print(a.pow_(3))

# 以e为底对应位置计算指数
a = torch.tensor([1, 2], dtype=torch.float32)
print(torch.exp(a))
print(a.exp())
print(torch.exp_(a)) # 将计算结果写入矩阵a
print(a.exp_())

# 计算对应位置以e为底的对数值
a = torch.tensor([10, 2], dtype=torch.float32)
print(torch.log(a))
print(a.log())
print(torch.log_(a))
print(a.log_())

# 计算对应位置开方值
a = torch.tensor([10, 2], dtype=torch.float32)
print(torch.sqrt(a))
print(a.sqrt())
print(torch.sqrt_(a))
print(a.sqrt_())
```

##### 矩阵运算

```python
# 矩阵乘法
a = torch.ones(2, 1)
b = torch.ones(1, 2)

# 下面五种写法等价
print(a @ b)
print(a.matmul(b))
print(torch.matmul(a, b))
print(torch.mm(a, b))
print(a.mm(b))

# 高维tensor计算，需要满足相应的计算规则
a = torch.ones(1, 2, 3, 4)
b = torch.ones(1, 2, 4, 3)
print(a.matmul(b))
```

##### 广播机制

广播机制：张量参数可以自动扩展为相同大小，广播机制需要满足两个条件

* 每个张量至少有一个维度
* 满足右对齐

```python
a = torch.rand(2, 2) 
b = torch.rand(1, 2)

c = a + b # a的每一行与b相加得到c

a = torch.rand(2, 1) 
b = torch.rand(1, 2)
c = a + b # 得到2x2的矩阵，a的每一列与b的每一行相加
```

##### 取正与取余

```python
import torch

a = torch.rand(2, 2)
a = a * 10

print(torch.floor(a)) # 包含in-place操作
print(torch.ceil(a))
print(torch.round(a)) # 四舍五入
print(torch.trunc(a)) # 取整数部分
print(torch.frac(a)) # 取小数部分
print(a % 2) # 取除2的余数
```

##### 比较运算

```python
import torch

a = torch.rand(2, 3) # 两个比较矩阵形状必须相同
b = torch.rand(2, 3)

# 比较每个位置上的值大小，生成一个bool类型的张量，表示比较结果
print(torch.eq(a, b))
print(torch.ge(a, b))
print(torch.gt(a, b))
print(torch.le(a, b))
print(torch.lt(a, b))
print(torch.ne(a, b))

# 所有位置的值都必须相等返回ture
print(torch.equal(a, b))
```

#### 数学函数

函数具体内容可以查阅手册

1. 三角函数
2. 常用函数 `torch.abs()`，`torch.sign()`，`torch.sigmoid()`
3. 统计学函数，包括：`torch.mean()`，`torch.sum()` ，`torch.histc`（直方图）等
4. `torch.distributions()` 中定义了分布函数
5. 随机抽样函数

```python
import torch
torch.manual_seed(1) # 定义随机种子后，下面两次随机会取得的抽样结果是一致的
mean = torch.rand(1, 2)
std  = torch.rand(1, 2)
print(torch.normal(mean, std))
```

6. 范数运算`torch.dist()`，`torch.norm()`等
6. 矩阵分解：LU分解；QR分解；EVD分解；SVD分解。

#### 操作

##### 排序

```python
a = torch.tensor([[1, 4, 4, 3, 5],
                  [2, 3, 1, 3, 5]])
print(a.shape)
print(torch.sort(a, dim=1, descending=False))

# 返回k个最大值
a = torch.tensor([[2, 4, 3, 1, 5],
                  [2, 3, 5, 1, 4]])
print(torch.topk(a, k=2, dim=1, largest=False))
```

##### 裁剪

```python
a = torch.rand(2, 2) * 10
a = a.clamp(2, 5) # 对张量进行裁剪，小于等于2的数全部设置为2，大于等于5的数全部设置为5，其它数保持不变。
```

##### 数据筛选

```python
import torch
#torch.where

a = torch.rand(4, 4)
b = torch.rand(4, 4)
out = torch.where(a > 0.5, a, b) # a大于0.5的位置选择a中元素，否则选择b中元素

a = torch.tensor([[0, 1, 2, 0], [2, 3, 0, 1]])
out = torch.nonzero(a) # 选择非0元素

# 下列均是选择函数，具体操作查阅文档
torch.index_select()
torch.gather()
torch.masked_select()
torch.take()
```

##### 拼接与切片

```python
import torch

# 拼接
a = torch.zeros((2, 4))
b = torch.ones((2, 4))

out = torch.cat((a,b), dim=0) # 根据行拼接
out = torch.cat((a,b), dim=1) # 根据列拼接

# 增加维度拼接
a = torch.linspace(1, 6, 6).view(2, 3)
b = torch.linspace(7, 12, 6).view(2, 3)
out = torch.stack((a, b), dim=2)

# 切片
a = torch.rand((3, 4))
out = torch.chunk(a, 2, dim=1) # 根据列平均切分为几块，最后一块是剩余值

a = torch.rand((10, 4))
out = torch.split(a, 3, dim=0) # 平均切分为3份
out = torch.split(a, [1, 3, 6], dim=0) # 按指定数据进行切分
```

##### 张量变形

```python
import torch

a = torch.rand(2, 3)
out = torch.reshape(a, (3, 2)) 

print(torch.t(out)) # 转置
print(torch.transpose(out, 0, 1)) # 维度交换

# 其他变形函数，具体操作查阅文档
torch.squeeze()
torch.unsqueeze(1)
torch.unbind()
torch.flip()
torch.rot90()
```

##### 其他操作

```python
import torch

# 填充操作
a = torch.full((2,3), 10) # 填充元素为10
```

### 搭建简单网络

深度学习工作流：

1. 准备数据，包括：读取数据集和标签、清洗数据等。data
2. 搭建网络：深度学习模型。net
3. 定义损失函数。loss
4. 定义优化器。optimiter
5. 训练网络。training
6. 测试网络。test
7. 保存模型数据 save
8. 使用模型推理 predict

#### 回归模型

```python
# 波士顿房价预测

# data
import numpy as np
import re

data = []
with open("./data/housing.data") as ff: # 将数据读取到数组内
    lines = ff.readlines()
    for item in lines:
        out = re.sub(r"\s{2,}", " ", item).strip()
        data.append(out.split(" "))

data = np.array(data).astype(np.float) # 将数组转换为np数据类型

Y = data[:, -1]  # 获得标签和特征
X = data[:, 0:-1]

X_train = X[0:496, ...] # 划分训练集合测试集
Y_train = Y[0:496, ...]
X_test = X[496:, ...]
Y_test = Y[496:, ...]

print('train x shape ${}, y shape ${}'.format(X_train.shape, Y_train.shape))
print('test x shape ${}, y shape ${}'.format(X_test.shape, Y_test.shape))

# net
class Net(torch.nn.Module): # 定义一个网络
    def __init__(self, n_feature, n_output):
        super(Net, self).__init__()
        self.predict = torch.nn.Linear(n_feature, n_output) # 线性层: 输入数和输出数

    def forward(self, x): # 定义前向运算形式
        out = self.predict(x)
        return out
      
class Net(torch.nn.Module): # 包含一个隐藏层的神经网络
    def __init__(self, n_feature, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, 100)
        self.predict = torch.nn.Linear(100, n_output)
    def forward(self, x):
        out = self.hidden(x)
        out = torch.relu(out)
        out = self.predict(out)
        return out

net = Net(13, 1) # n_feature:13,  n_output:1

# loss
loss_func = torch.nn.MSELoss() # 定义损失函数

# optimiter
optimizer = torch.optim.Adam(net.parameters(), lr=0.01) # 定义优化器: 传入网络参数和学习率lr

# training
for i in range(1000):
    x_data = torch.tensor(X_train, dtype=torch.float32) # 初始化训练集，np数据转换为tensor
    y_data = torch.tensor(Y_train, dtype=torch.float32)
    pred = net.forward(x_data) # 前向传播预测
    pred = torch.squeeze(pred)
    loss = loss_func(pred, y_data) * 0.001 # 缩小损失函数

    optimizer.zero_grad() # 每次训练用设置初始梯度为0
    loss.backward() # 反向传播计算
    optimizer.step() # 优化参数
    if (i + 1) % 100 == 0:
        print("ite:{}, loss_train:{}".format(i + 1, loss))
        print(pred[0:10])
        print(y_data[0:10])
        
torch.save(net, "model/model.pkl") # 保存模型
torch.load("") # 可以直接加载模型

torch.save(net.state_dict(), "params.pkl") # 只保存参数
net.load_state_dict("") # 需要通过网络加载参数

# test
net = torch.load("model/model.pkl") # 加载模型
x_data = torch.tensor(X_test, dtype=torch.float32)
y_data = torch.tensor(Y_test, dtype=torch.float32)
pred = net.forward(x_data) # 测试性能
```

#### 分类模型

```python
# 手写数组识别
import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torch.utils.data as data_utils

# data 使用torch库下载已有数据
train_data = dataset.MNIST(root="./data",
                           train=True,
                           transform=transforms.ToTensor(),
                           download=True)
test_data = dataset.MNIST(root="mnist",
                          train=False,
                          transform=transforms.ToTensor(),
                          download=False) # 设置为False不会下载，下载过的数据不会在下载

train_loader = data_utils.DataLoader(dataset=train_data,  # batchsize 使用DataLoader读取一部分数据
                                     batch_size=64, # 每个batch为64
                                     shuffle=True) # 打乱全部数据后抽样
test_loader = data_utils.DataLoader(dataset=test_data,
                                    batch_size=64,
                                    shuffle=True)


# net
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv =torch.nn.Sequential( # 定义卷积层
            torch.nn.Conv2d(1, 32, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.fc = torch.nn.Linear(14 * 14 * 32, 10) # 定义全连接层
        
    def forward(self, x):
        out = self.conv(x) # 卷积操作
        out = out.view(out.size()[0], -1) # 卷积结果变为线性结构
        out = self.fc(out) # 分类操作
        return out

cnn = CNN()
# cnn = cnn.cuda() 使用cuda训练

# loss
loss_func = torch.nn.CrossEntropyLoss()

# optimizer
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.01)

# training
for epoch in range(10): # 1个epoch表示训练完1次全部样本数据数据
    for i, (images, labels) in enumerate(train_loader): # 每次训练数据从样本集中抽取64条
        # images = images.cuda() 将数据传入cuda
        # labels = labels.cuda()

        outputs = cnn(images) # 对图像进行分类，images中是64条数据
        loss = loss_func(outputs, labels) # 计算损失函数，每次64条
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("epoch is {}, ite is {}/{}, loss is {}".format(epoch+1, i, len(train_data) // 64, loss.item()))

    # eval/test 每个epoch后测试性能
    loss_test = 0
    accuracy = 0
    for i, (images, labels) in enumerate(test_loader):
        outputs = cnn(images)
        loss_test += loss_func(outputs, labels)
        _, pred = outputs.max(1)
        accuracy += (pred == labels).sum().item()

    accuracy = accuracy / len(test_data)
    loss_test = loss_test / (len(test_data) // 64)

    print("epoch is {}, accuracy is {}, loss test is {}".format(epoch + 1, accuracy, loss_test.item()))
    
# save
torch.save(cnn, "./models/mnist_model.pkl") # 保存模型

# predict
cnnSave = torch.load("./models/mnist_model.pkl") # 使用模型预测
for i, (images, labels) in enumerate(test_loader):
    # images = images.cuda()
    # labels = labels.cuda()
    outputs = cnnSave(images)
    _, pred = outputs.max(1)
    accuracy += (pred == labels).sum().item()

    images = images.cpu().numpy()
    labels = labels.cpu().numpy()
    pred = pred.cpu().numpy()

    for idx in range(images.shape[0]):
        im_data = images[idx]
        im_label = labels[idx]
        im_pred = pred[idx]
        im_data = im_data.transpose(1, 2, 0)

accuracy = accuracy / len(test_data)
print(accuracy)
```

## PyTroch部署

* Torch Server

* ONNX 可以使模型在不同框架内进行转换

  
