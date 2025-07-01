# TensorFlow常用API

## 激活函数

### Sigmoid

```python

import tensorflow as tf
import numpy as np
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
from tensorflow import keras

init = keras.initializers.glorot_normal()
values = init(shape=(9, 1))
print(values)
```

均匀初始化

```python
init = keras.initializers.glorot_uniform()
values = init(shape=(9, 1))
print(values)
```

### He初始化

正太初始化

```python
init = keras.initializers.he_normal()
values = init(shape=(9, 1))
print(values)
```

标准初始化

```python
init = keras.initializers.he_uniform()
values = init(shape=(9, 1))
print(values)
```

## 网络的构建

### `keras.Sequential`

按顺序创建堆叠模型。

```python
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

> [!warning]
>
> 使用Functional API可以用循环来创建模型

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

## 损失函数

### 分类任务

1. 多分类任务。

```python
y_true = tf.constant([[0, 1, 0], [0, 0, 1]])
y_pred = tf.constant([[0.05, 0.95, 0], [0.1, 0.8, 0.1]])

cue = keras.losses.CategoricalCrossentropy()
print(cue(y_true, y_pred))
```

计算的交叉熵损失函数是mini-batch的交叉熵损失。

2. 二分类任务。

```python
y_true = tf.constant([[0], [1]])
y_pred = tf.constant([[0.4], [0.6]])

bce = keras.losses.BinaryCrossentropy()
print(bce(y_true, y_pred))
```

### 回归任务

1. MAE损失函数

```python
y_true = tf.constant([[0.], [0.]])
y_pred = tf.constant([[1.], [1.]])

mae = keras.losses.MeanAbsoluteError()
print(mae(y_true, y_pred))
```

2. MSE损失函数

```python
y_true = tf.constant([[0.], [1.]])
y_pred = tf.constant([[1.], [1.]])

mse = keras.losses.MeanSquaredError()
print(mse(y_true, y_pred))
```

3. smooth L1损失

$$
loss(x)= \begin{cases}
 0.5x^2 & \text{ if } \left | x \right | <1 \\
 \left | x \right | - 0.5 & \text{ otherwise }
\end{cases}
$$

函图像为

<img src="https://raw.githubusercontent.com/hughxusu/lesson-ai/develop/images/cv/a352281be9094b22e2ce8f2f101830b0.png" style="zoom:50%;" />

```python
y_true = tf.constant([[0.], [1.]])
y_pred = tf.constant([[.6], [.4]])

huber = keras.losses.Huber()
print(huber(y_true, y_pred))
```

上图中我们展示了一维和多维的损失函数，损失函数呈现碗状。

## 网络优化

测试用例

1. 损失函数为$0.5x^2$。
2. 初始值为1。
3. 计算一次梯度下降后的值。

### 梯度下降算法

```python
loss = lambda: (var ** 2) / 2.0

var = tf.Variable(1.0)
opt = keras.optimizers.SGD(learning_rate=0.1)
for i in range(10):
    with tf.GradientTape() as tape:
        loss_value = loss()
    grads = tape.gradient(loss_value, [var])
    opt.apply_gradients(zip(grads, [var]))
    print(f"第 {i+1} 次更新后的参数值:", var.numpy())
```

### 动量梯度下降算法

在`keras.optimizers.SGD`函数中增加，`momentum=0.9`即时动量梯度下降算法。

```python
var = tf.Variable(1.0)
opt = keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
for i in range(10):
    with tf.GradientTape() as tape:
        loss_value = loss()
    grads = tape.gradient(loss_value, [var])
    opt.apply_gradients(zip(grads, [var]))
    print(f"第 {i+1} 次更新后的参数值:", var.numpy())
```

### AdaGrad梯度下降算法

对学习率进行修正。

```python
var = tf.Variable(1.0)
opt = tf.keras.optimizers.Adagrad(learning_rate=0.1) 
for i in range(10):
    with tf.GradientTape() as tape:
        loss_value = loss()
    grads = tape.gradient(loss_value, [var])
    opt.apply_gradients(zip(grads, [var]))
    print(f"第 {i+1} 次更新后的参数值:", var.numpy())
```

### RMSprop

AdaGrad算法在迭代后期由于学习率过小，RMSProp算法对AdaGrad算法做了一点改进。

```python
var = tf.Variable(1.0)
opt = tf.keras.optimizers.RMSprop(learning_rate=0.1) 
for i in range(10):
    with tf.GradientTape() as tape:
        loss_value = loss()
    grads = tape.gradient(loss_value, [var])
    opt.apply_gradients(zip(grads, [var]))
    print(f"第 {i+1} 次更新后的参数值:", var.numpy())
```

### Adam

```python
var = tf.Variable(1.0)
opt = tf.keras.optimizers.Adam(learning_rate=0.1) 
for i in range(10):
    with tf.GradientTape() as tape:
        loss_value = loss()
    grads = tape.gradient(loss_value, [var])
    opt.apply_gradients(zip(grads, [var]))
    print(f"第 {i+1} 次更新后的参数值:", var.numpy())
```

### 学习率退火

在训练神经网络时，一般情况下学习率都会随着训练而变化。主要是由于，在神经网络训练的后期

* 学习率过高，会造成loss的振荡。
* 学习率减小的过快，又会造成收敛变慢的情况。

1. 分段常数衰减。

在事先定义好的训练次数区间上，设置不同的学习率常数。刚开始学习率大一些，之后越来越小，区间的设置需要根据样本量调整，一般样本量越大区间间隔应该越小。

```python
# 设置的分段的step值
boundaries = [100000, 110000]
# 不同的step对应的学习率
values = [1.0, 0.5, 0.1]
# 实例化进行学习的更新
learning_rate_fn = keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
```

返回了一个回调函，该函数可以设置在优化器中。

2. 指数衰减。

$$
\alpha=\alpha_0e^{-kt}
$$

`keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps,                                               decay_rate)`

* 返回回调函数
* `initial_learning_rate`初始学习率。
* `decay_steps`k值。
* `decay_rate`指数的底。

```python
def decayed_learning_rate(step):
		return initial_learning_rate * decay_rate ^ (step / decay_steps)
```

3. $\frac{1}{t}$衰减

$$
\alpha=\frac{\alpha_0}{1+kt} 
$$

`keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate, decay_steps,                                               decay_rate)`

* 返回回调函数
* `decay_steps`和`decay_rate`共同控制k值

```python
def decayed_learning_rate(step):
		return initial_learning_rate / (1 + decay_rate * step / decay_step)
```

### 正则化

1. L1正则化`keras.regularizers.L1(l1=0.01)`
2. L2正则化`keras.regularizers.L2(l2=0.01)`
3. L1-L2正则化`keras.regularizers.L1L2(l1=0.0, l2=0.0 )`

给每一层设置正则项

```python
from tensorflow.keras import regularizers

model = keras.models.Sequential()
model.add(keras.layers.Dense(16, kernel_regularizer=regularizers.l2(0.001), activation='relu', input_shape=(10,)))
model.add(tf.keras.layers.Dense(16, kernel_regularizer=regularizers.l1(0.001), activation='relu'))
model.add(tf.keras.layers.Dense(16, kernel_regularizer=regularizers.L1L2(0.001, 0.01), activation='relu'))
print(model.summary())
```

### Dropout

```python
data = np.arange(1, 11).reshape(5, 2).astype(np.float32)
print(data)

layer = keras.layers.Dropout(0.5, input_shape=(2,))
outputs = layer(data, training=True)
print(outputs)
```

`Dropout`层跟在`Dense`后面，如果想让哪一层`Dense`失活，在其后面添加`Dropout`。

> [!warning]
>
> 未被失活的输入将按$\frac{1}{1-\text{rate}}$放大

### 提前停止

提前停止（early stopping）是将一部分训练集作为验证集。 当验证集的性能越来越差时或者性能不再提升，则立即停止对该模型的训练。 这被称为提前停止。

`keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)`

* 返回回调函数，设置在训练过程中。
* `monitor`监测量，`val_loss`表示验证集损失。
* `patience`设置连续epochs数量。

```python
callback = keras.callbacks.EarlyStopping(monitor='loss', patience=3)
model = keras.models.Sequential([tf.keras.layers.Dense(10)])

model.compile(keras.optimizers.SGD(), loss='mse')
history = model.fit(np.arange(100).reshape(5, 20), 
                    np.array([0,1,2,1,2]),
                    epochs=10, 
                    batch_size=1, 
                    callbacks=[callback],
                    verbose=1)
print(history.history['loss'])
```

### 批标准化

BN层（Batch Normalization）属于网络中的一层。

`keras.layers.BatchNormalization(epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones')`

* `epsilon`防止分母为0
* `center=True`是否使用偏移量
* `scale=True`是否缩放
* `beta_initializer` $\beta$权重初始值
* `gamma_initializer`$\gamma$权重初始值

$$
f(x) = \begin{cases}
0.5x^2 \cdot \text{sign}(x), & |x| < 1 \\
(|x| - 0.5) \cdot \text{sign}(x), & |x| \geq 1
\end{cases}
$$

