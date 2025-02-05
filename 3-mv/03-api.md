# TensorFlow常用API

## 激活函数

### Sigmoid

```python
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 100)
y = tf.nn.sigmoid(x)
plt.plot(x, y)
plt.grid()
plt.show()
```

### Tanh

```python
x = np.linspace(-10, 10, 100)
y = tf.nn.tanh(x)
plt.plot(x, y)
plt.grid()
plt.show()
```

### Relu

```python
x = np.linspace(-10, 10, 100)
y = tf.nn.relu(x)
plt.plot(x, y)
plt.grid()
plt.show()
```

### Leaky Relu

对Relu激活函数的改进$f(x)=\max(0.1x, x)$

```python
x = np.linspace(-10, 10, 100)
y = tf.nn.leaky_relu(x)
plt.plot(x, y)
plt.grid()
plt.show()
```

### Softmax

```python
x = tf.constant([0.2, 0.02, 0.15, 1.3, 0.5, 0.06, 1.1, 0.05, 3.75])
y = tf.nn.softmax(x)
print(y)
print(tf.reduce_sum(y))
```

## 权重初始化

### Xavier初始化

正态初始化

```python
init = tf.keras.initializers.glorot_normal()
values = init(shape=(9, 1))
print(values)
```

均匀初始化

```python
init = tf.keras.initializers.glorot_uniform()
values = init(shape=(9, 1))
print(values)
```

### He初始化

正太初始化

```python
init = tf.keras.initializers.he_normal()
values = init(shape=(9, 1))
print(values)
```

标准初始化

```python
init = tf.keras.initializers.he_uniform()
values = init(shape=(9, 1))
print(values)
```

## 网络的构建

### `keras.Sequential`

按顺序创建堆叠模型。

```python
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(3, activation='relu', kernel_initializer='he_normal', name='layer1', input_shape=(3,)),
    keras.layers.Dense(2, activation='relu', kernel_initializer='he_normal', name='layer2'),
    keras.layers.Dense(2, activation='sigmoid', kernel_initializer='he_normal', name='layer3')
], name='my_Sequential')

print(model.summary())
```

`keras.Sequential`只能创建简单的序列模型。

### function API构建网络

使用方法是将层作为可调用对象并返回张量，并将输入向量和输出向量传给模型

```python
input = keras.Input(shape=(3,), name='input')
x = keras.layers.Dense(3, activation='relu', kernel_initializer= 'he_normal', name='layer1')(input)
x = keras.layers.Dense(2, activation='relu', kernel_initializer= 'he_normal', name='layer2')(x)
output = keras.layers.Dense(2, activation='sigmoid', kernel_initializer= 'he_normal', name='layer3')(x)

model = keras.Model(inputs=input, outputs=output, name='my_model')
print(model.summary())
```

Functional API可以建立更为复杂的模型。

### Model的子类构建模型

继承`keras.Model`在`__init__`函数中定义网络层，在`call`方法中定义网络的前向传播过程。

```python
class MyModel(keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = keras.layers.Dense(3, activation='relu', kernel_initializer='he_normal', name='layer1')
        self.dense2 = keras.layers.Dense(2, activation='relu', kernel_initializer='he_normal', name='layer2')
        self.dense3 = keras.layers.Dense(2, activation='sigmoid', kernel_initializer='he_normal', name='layer3')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.dense3(x)
    
model = MyModel()
x = tf.ones((1, 3))
y = model(x)
print(model.summary())
```

如果没有输入值无法打印网络参数。