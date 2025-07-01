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
    image, label = train[i]
    plt.subplot(3, 3, i+1)
    plt.imshow(image[0], cmap='gray', interpolation='none')
    plt.title("number {}".format(label))
    
plt.tight_layout()
plt.show()
```

### 数据处理

<img src="https://raw.githubusercontent.com/hughxusu/lesson-ai/develop/images/cv/c41429c8bfa5f1aad08b3a18aaf41f01.png" style="zoom:55%;" />

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

![](https://raw.githubusercontent.com/hughxusu/lesson-ai/develop/images/cv/V0L3hpYW93ZWl0ZTE.png)

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

* `filters`卷积核数量，对应输出特征图的通道数。`Conv2D`会根据图像的输入通道数自动调整卷积核。当`filters=3`时：
  * 当输入是一通道图像时，`Conv2D` 会创建3个卷积核。每个卷积核的形状为 `(kernel_height, kernel_width, 1)`。
  * 当输入是三通道图像时，`Conv2D` 会创建3个卷积核，但每个卷积核的形状为 `(kernel_height, kernel_width, 3)`。这意味着每个卷积核都会考虑所有三个通道的信息。

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

> [!warning]
>
> 数据没有归一化，标签也没有转化成one-hot编码。

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

## Colab的使用

Google Colab（Colaboratory）是一个由Google提供的云端Jupyter笔记本服务，主要用于机器学习、数据分析和教育等领域。

* 用户可以在浏览器中直接使用，无需在本地安装任何软件。
* Colab提供免费的GPU和TPU支持，适合进行深度学习训练。
* 可以直接从Google Drive导入和导出文件，方便数据管理（免费15G数据存储）。
* 自带许多常用的Python库，如TensorFlow、Keras、Pandas等，方便进行数据科学和机器学习项目。
* 支持Matplotlib、Seaborn等可视化库，便于数据展示。

[Colab首页](https://colab.research.google.com/)

### Colab基本使用

登陆Colab需要使用Google账户

<img src="https://raw.githubusercontent.com/hughxusu/lesson-ai/develop/images/cv/Xnip2025-02-08_15-38-23.jpg" style="zoom:40%;" />

Google硬盘功能

<img src="https://raw.githubusercontent.com/hughxusu/lesson-ai/develop/images/cv/Xnip2025-02-08_15-45-25.jpg" style="zoom:40%;" />

新建文件

<img src="https://raw.githubusercontent.com/hughxusu/lesson-ai/develop/images/cv/Xnip2025-02-08_15-49-59.jpg" style="zoom:40%;" />

* 需要手动链接一下Google的云端硬盘。

测试程序执行

```python
import sys
import tensorflow as tf
print(sys.version)
print(tf.__version__)
```

打印当前路径

```python
pwd
```

创建文件夹

```python
!mkdir data
```

切换文件夹

```python
cd /content/data
cd ..
```

> [!warning]
>
> 在编辑器里使用命令行时，Windows命令直接使用，linux命令前面需要使用`!`。

可以使用git命令克隆github上的项目。

```python
!git clone https://github.com/jwyang/faster-rcnn.pytorch.git
```

设置GPU执行程序

<img src="https://raw.githubusercontent.com/hughxusu/lesson-ai/develop/images/cv/Xnip2025-02-08_15-57-20.jpg" style="zoom:40%;" />

> [!warning]
>
> GPU一般可以连续运行10小时左右，运行时保存模型。

检查GPU是否可用

```python
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
```

查看GPU信息

```shell
!nvidia-smi
```

> [!warning]
>
> 当训练Keras模型时，TensorFlow会默认检查可用的GPU。如果系统中有GPU，TensorFlow会优先使用它。

常见问题：

1. colab运行的文件夹叫`/content`。
   * `/content`文件夹是Colab分配给你的虚拟机实例的临时文件系统。
   * 可以将文件上传到`/content`文件夹，或者在 Colab 中创建新的文件和文件夹。
   * `/content`文件夹中的数据仅在当前会话期间有效。当会话结束或超时时，`/content`文件夹中的所有数据都将被删除。

2. `/content/drive`文件夹是 Google Drive挂载到Colab虚拟机实例上的目录。
   * 通过挂载Google Drive，可以像访问本地文件一样访问Google Drive中的文件和文件夹。
   * 使用`!cp`命令将`/content`文件夹中的文件复制到`/content/drive`文件夹中，从而将数据保存到你的Google Drive中。

3. 删除文件夹可以使用命令`!rm -r /content/your_folder_name`。

## 其它免费GPU

[kaggle](https://www.kaggle.com/) 每周30个小时

[阿里云天池](https://tianchi.aliyun.com/notebook-ai/) 一共60个小时
