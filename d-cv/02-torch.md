# PyTorch入门

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

