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

