# 实践案例

## 神经网络

使用手写数字的MNIST数据集，该数据集包含60000个用于训练的样本和10000个用于测试的样本，图像是固定大小（28$\times$28像素），其值为0到255。

### 加载数据集

```python
from tensorflow.keras.datasets import mnist

nb_classes = 10
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print("训练样本数据", X_train.shape)
print("训练样本标签", y_train.shape)
```

绘制数据图像

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 8))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(X_train[i], cmap='gray', interpolation='none')
    plt.title("number {}".format(y_train[i]))
    
plt.tight_layout()
plt.show()
```

### 数据处理

<img src="https://raw.githubusercontent.com/hughxusu/lesson-ai/developing/_images/mv/c41429c8bfa5f1aad08b3a18aaf41f01.png" style="zoom:55%;" />

keras中的mnist数据是28$\times$28的矩阵形式，在神经网络中需要转换成一个向量，同时进行归一化。

```python
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

print("训练集：", X_train.shape)
print("测试集：", X_test.shape)
```

需要将标签值转换为one-hot编码

```python
from tensorflow.keras import utils

Y_train = utils.to_categorical(y_train, nb_classes)
Y_test = utils.to_categorical(y_test, nb_classes)
print("训练集标签：", Y_train.shape)
print("测试集标签：", Y_test.shape)
```

### 构建网络

网络结构

```mermaid
flowchart LR
    a(输入数据_784)-->b(隐藏层1_512)-->c(隐藏层2_512)-->d(输出_10)
```

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras import regularizers

model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))                          
model.add(Dropout(0.2))  

model.add(Dense(512, kernel_regularizer=regularizers.l2(0.001)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(10))
model.add(Activation('softmax')) 

print(model.summary())
```

* `Activation`函数和`Dense`层可以分开添加。
* `Dropout`层放在最后每一组最后即可。
* `BatchNormalization`添加在`Dense`和`Activation`之间。
* 输入层的数量，在第一个隐层中指定。

### 模型编译

设置模型训练使用的损失函和优化方法等。

```python
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

* 使用交叉熵损失和adam优化。
* 评估指标使用准确率。

### 模型训练

```python
history = model.fit(X_train, Y_train, batch_size=128, epochs=4, verbose=1, validation_data=(X_test, Y_test))
print(history.history.keys())
print("Epochs: ", history.epoch)
print("Training Parameters: ", history.params)
```

* `verbose=1`打印训练过程信息。
* `history`保存训练过程信息。
* `validation_data`设置交叉验证数据。

绘制训练过程损失函数

```python
plt.figure()
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.legend()
plt.grid()
```

绘制训练过程的准确率曲线

```python
plt.figure()
plt.plot(history.history["accuracy"], label="train_acc")
plt.plot(history.history["val_accuracy"], label="val_acc")
plt.legend()
plt.grid()
```

使用tensoboard监控训练过程

```python
from tensorflow import keras

tensorboard = keras.callbacks.TensorBoard(log_dir='./graph', histogram_freq=1, write_graph=True,write_images=True)
history = model.fit(X_train, 
                    Y_train,
                    batch_size=128, 
                    epochs=4,
                    verbose=1,
                    callbacks=[tensorboard],
                    validation_data=(X_test, Y_test))
```

使用命令打开tensoboard

```shell
tensorboard --logdir="./"
```

### 模型评估

```python
score = model.evaluate(X_test, Y_test, verbose=1)
print('测试集准确率:', score)
```

### 模型保存与加载

保存模型。保存成`.keras`文件

```python
model.save('my_model.keras')
```

模型加载

```python
model.save('my_model.h5')
model = tf.keras.models.load_model('my_model.h5')
```

## CNN网络

构建LeNet-5

![](https://raw.githubusercontent.com/hughxusu/lesson-ai/developing/_images/mv/V0L3hpYW93ZWl0ZTE.png)

* f=5表示卷积核大小为$5\times5$。
* s=1表示步长为1。
* n=6表示卷积核为6个。
* 激活函数为sigmod的函数。

### 卷积API

1. 卷积层。

```python
keras.layers.Conv2D(filters, 
                    kernel_size, 
                    strides=(1, 1), 
                    padding='valid', 
                    activation=None)
```

* `filters`卷积核数量，对应输出特征图的通道数。
* `kernel_size`滤波器的大小。
* `strides`步长。
* `padding=valid`不进行padding；`padding=same`保证输入特征图和输出特征图大小相等。
* `activation`激活函数。

2. 最大池化。

```python
keras.layers.MaxPool2D(
    pool_size=(2, 2), strides=None, padding='valid'
)
```

* `pool_size`池化层大小。
* `strides`窗口移动的步长，默认为1。
* `padding`是否进行填充，默认是不进行填充的

3. 平均池化。

```python
keras.layers.AveragePooling2D(
    pool_size=(2, 2), strides=None, padding='valid'
)
```

### 数据加载

```python
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
```

### 数据处理

卷积神经网络的输入要求是：

* N图片数量
* H图片高度
* W图片宽度
* C图片的通道，因为是灰度图，通道为1。

```python
import tensorflow as tf

train_images = tf.reshape(train_images, 
                          (train_images.shape[0], train_images.shape[1], train_images.shape[2], 1))

test_images = tf.reshape(test_images, 
                         (test_images.shape[0], test_images.shape[1], test_images.shape[2], 1))

print(train_images.shape)
print(test_images.shape)
```

### 模型构建

```python
net = keras.models.Sequential([
    # 卷积层1，激活sigmoid
    keras.layers.Conv2D(filters=6, kernel_size=5, activation='sigmoid', input_shape=(28,28,1)),
    keras.layers.MaxPool2D(pool_size=2, strides=2),
    # 卷积层2，激活sigmoid
    keras.layers.Conv2D(filters=16, kernel_size=5, activation='sigmoid'),
    keras.layers.MaxPool2D(pool_size=2, strides=2),
    # 维度调整为1维数据
    keras.layers.Flatten(),
    # 全连接层1，激活sigmoid
    keras.layers.Dense(120, activation='sigmoid'),
    # 全连接层1，激活sigmoid
    keras.layers.Dense(84, activation='sigmoid'),
    # 输出层，激活softmax
    keras.layers.Dense(10, activation='softmax')
])
print(net.summary())
```

* `keras.layers.Flatten()`专门用于卷积层展开。

### 模型编译

```python
optimizer = keras.optimizers.SGD(learning_rate=0.9)
net.compile(optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
```

* `keras.optimizers.SGD`使用随机梯度下降优化器。
* `loss='sparse_categorical_crossentropy'`标签没有one-hot编码使用的损失函数。

### 模型训练

```python
history = net.fit(train_images, train_labels, epochs=10, validation_split=0.1)
```

* `validation_split=0.1`使用训练集的$10\%$，进行验证。

绘制训练过程曲线

```python
plt.figure()
plt.plot(history.history["accuracy"], label="train_acc")
plt.plot(history.history["val_accuracy"], label="val_acc")
plt.legend()
plt.grid()
plt.show()
```

### 模型评估

```python
score = net.evaluate(test_images, test_labels, verbose=1)
print('Test accuracy:', score[1])
```

