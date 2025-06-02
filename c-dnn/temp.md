## 手写数识别

机器学习问题可以分为学习和推理两部分。在神经网络中，推理过程也称为前向传播（forward propagation）。下面使用训练好的参数来实践，手写数字识别的推理过程。

导入手写数字识别的数据集函数如下

```python
from sklearn.datasets import fetch_openml
import numpy as np

def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    mnist = fetch_openml('mnist_784', version=1)

    X, y = mnist['data'], mnist['target']
    X_train = np.array(X[:60000], dtype=float)
    y_train = np.array(y[:60000], dtype=int)  # 转换为整数类型
    X_test = np.array(X[60000:], dtype=float)
    y_test = np.array(y[60000:], dtype=int)  # 转换为整数类型
    
    if normalize:
        X_train /= 255.0
        X_test /= 255.0
        
    if one_hot_label:
        y_train = np.eye(10)[y_train]
        y_test = np.eye(10)[y_test]
    
    if not flatten:
        X_train = X_train.reshape(-1, 1, 28, 28)
        X_test = X_test.reshape(-1, 1, 28, 28)
        
    return (X_train, y_train), (X_test, y_test)
```

绘制数据图像

```python
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)

def display_mnist_image(data, label):
    img = data.reshape(28, 28)
    
    plt.imshow(img, cmap='gray')
    plt.title(f'Label: {label}')
    plt.axis('off')
    plt.show()
    

data = x_train[0]
print(data.shape)
label = t_train[0]
display_mnist_image(data, label)


def display_mnist_image(data, label):
    img = data.reshape(28, 28)
    
    plt.imshow(img, cmap='gray')
    plt.title(f'Label: {label}')
    plt.axis('off')
    plt.show()

data = x_train[0]
print(data.shape)
label = t_train[0]
display_mnist_image(data, label)
```

读取[参数文件](https://github.com/hughxusu/lesson-ai/blob/developing/_examples/sample_weight.pkl)文件，并进行预测

```python
import pickle

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
        
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    
    return y

```

`x = np.clip(x, -500, 500)`防止sigmoid函数计算过程中产生溢出。使用测试数据对网络进行预测

```python
x, t = x_test, t_test
network = init_network()

accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
```

上述神经网络结构如下

```mermaid
flowchart LR
		a(输入层 X-784)-->b(隐层 50)-->c(隐层 100)-->d(输出层 Y-10)
```

使用矩阵运算来直接统计准确率

```python
x, t = x_test, t_test
network = init_network()

y_batch = predict(network, x)
p = np.argmax(y_batch, axis=1)
accuracy_cnt = np.sum(p == t)
print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
```

创建`utils.py`将`load_mnist`等函数写入其中，后面的常用函数，也包含在这个文件中。