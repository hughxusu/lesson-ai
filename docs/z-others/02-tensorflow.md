# Tensorflow基础

2015年11月9日，Google发布深度学习框架[TensorFlow](https://www.tensorflow.org/?hl=zh-cn)并宣布开源，并迅速得到广泛关注，在图形分类、音频处理、推荐系统和自然语言处理等场景下都被大面积推广。支持Python和C++接口，目前到2.x版本。

![](https://raw.githubusercontent.com/hughxusu/lesson-ai/develop/images/cv/logo_icon.jpeg)

* TF托管在github平台，有google groups和contributors共同维护。
* TF提供了丰富的深度学习相关的API，支持Python和C/C++接口。
* TF提供了可视化分析工具Tensorboard，方便分析和调整模型。
* TF支持Linux平台，Windows平台，Mac平台，甚至手机移动设备等各种平台。

TensorFlow 2.x版本，将专注于简单性和易用性

<img src="https://raw.githubusercontent.com/hughxusu/lesson-ai/develop/images/cv/tf_arch_logic_architecture.png" style="zoom:85%;" />

自动微分

* 只要使用TensorFlow提供的运算操作，TensorFlow就够自动地计算出模型中每个操作的梯度。
* 在自定义的神经元和激活函数中，只需要根据设计使用TensorFlow运算完成前向传播即可。

训练过程

- `tf.data`用于加载训练用的原始数据，数据可以是公开数据集，也可以是私有数据集，数据格式为TF Datasets
- `tf. Keras`（官方推荐）或Premade Estimators用于构建、训练和验证模型。
- eager execution用于运行和调试。
- distribution strategy用于进行分布式训练，支持单机多卡以及多机多卡的训练场景。

可视化

* Tensorboard可视化分析。

模型仓库

* TensorFlow hub用于保存训练好的TensorFlow模型，供推理或重新训练使用。

部署

* TensorFlow Serving，即TensorFlow允许模型通过REST以及gPRC对外提供服务
* TensorFlow Lite，即TensorFlow针对移动和嵌入式设备提供了轻量级的解决方案
* TensorFlow.js，即TensorFlow支持在 JavaScript 环境中部署模型
* TensorFlow 还支持其他语言 包括 C, Java, Go, C#, Rust 等

TensorFlow的安装

1. 安装标准版本

```shell
pip install tensorflow
```

2. 安装GPU版本

```shell
pip install tensorflow-gpu
```

> [!attention]
>
> GPU版本只有windows和linux版本，从2.12版本后Tensorflow不再区分CPU版和GPU版。

## 张量及其操作

张量是一种多为数组，它可以看做是矩阵和向量的推广。

<img src="https://raw.githubusercontent.com/hughxusu/lesson-ai/develop/images/cv/0ca2fd5a6590d22027e3058b497fdff1.jpeg" style="zoom:55%;" />

在tensorflow中，用`tf.Tensor`对象表示，与Numpy ndarray对象类似，`tf.Tensor`对象也具有数据类型和形状。

<img src="https://raw.githubusercontent.com/hughxusu/lesson-ai/develop/images/cv/656a769280b04c.jpg" style="zoom:75%;" />

`tf.Tensor`可以驻留在加速器内存（例如 GPU）中。TensorFlow提供了丰富的运算库（`tf.math.add`、`tf.linalg.matmul` 和 `tf.linalg.inv` 等），这些运算使用和生成`tf.Tensor`。在进行张量操作之前需先导入相应的工具包。

```python
import tensorflow as tf
import numpy as np
```

### 基本方法

1. 张量的创建

```python
rank_0_tensor = tf.constant(4)
print(rank_0_tensor)

rank_1_tensor = tf.constant([2.0, 3.0, 4.0])
print(rank_1_tensor)

rank_2_tensor = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.float16)
print(rank_2_tensor)
```

<img src="https://raw.githubusercontent.com/hughxusu/lesson-ai/develop/images/cv/98602c4fbd104a9a91a9eba0b7099fb7.png" style="zoom:55%;" />

默认创建张量时，整形默认是`int32`，浮点型默认是`float32`。创建3维张量

```python
rank_3_tensor = tf.constant([
    [[0, 1, 2, 3, 4],
     [5, 6, 7, 8, 9]],
    [[10, 11, 12, 13, 14],
     [15, 16, 17, 18, 19]],
    [[20, 21, 22, 23, 24],
     [25, 26, 27, 28, 29]],])
print(rank_3_tensor)
```

<img src="https://raw.githubusercontent.com/hughxusu/lesson-ai/develop/images/cv/6206db46441e4f10b33bd752d7892b41.png" style="zoom:55%;" />

> [!attention]
>
> `tf.constant`创建为常量，不能修改里面的值，无法进行索引。

2. 张量转换为numpy中的`ndarray`的形式

```python
print(np.array(rank_2_tensor))
print(rank_2_tensor.numpy())
```

`ndarray`转换为张量

```python
np_array = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
tf_tensor = tf.convert_to_tensor(np_array)
print(tf_tensor)
```

### 张量的运算

对于张量，可以做⼀些基本的数学运算，包括加法、元素乘法和矩阵乘法等

```python
a = tf.constant([[1, 2], 
                 [3, 4]
                ]) 
b = tf.constant([[1, 1], 
                 [1, 1]
                ]) 
 
print(tf.add(a, b)) # 计算张量的和 
print(tf.multiply(a, b)) # 计算张量的元素乘法 
print(tf.matmul(a, b)) # 计算乘法（矩阵乘法）
```

张量也可以用于各种聚合运算

```python
c = tf.constant([[4.0, 5.0], [10.0, 1.0]])

print(tf.reduce_max(c)) # 计算张量的最大值
print(tf.argmax(c)) # 计算张量的最小值
print(tf.reduce_mean(c)) # 计算张量的均值
```

### 变量

变量是⼀种特殊的张量，形状是不可变，但可以更改其中的参数。

```python
my_variable = tf.Variable([[1.0, 2.0], [3.0, 4.0]]) 

print("Shape: ",my_variable.shape) 
print("DType: ",my_variable.dtype) 
print("As NumPy: ", my_variable.numpy)
```

修改变量的值

```python
my_variable.assign([[2.0, 3.0], [4.0, 5.0]])
print(my_variable)
```

修改变量的形状

```python
my_variable.assign([2.0, 3.0])
```

## Keras

`tf.keras`是TensorFlow 2.x高阶API接口，为TensorFlow的代码提供了新的风格和设计模式，提升了TF代码的简洁性和复用性，官方也推荐使用`tf.keras`来进行模型设计和开发。

<img src="https://raw.githubusercontent.com/hughxusu/lesson-ai/develop/images/cv/20220212105648110.png" style="zoom:50%;" />

### 常用模块

`tf.keras`中常用模块

| **模块**      | **概述**                                                     |
| ------------- | ------------------------------------------------------------ |
| activations   | 激活函数                                                     |
| applications  | 预训练网络模块                                               |
| Callbacks     | 在模型训练期间被调用                                         |
| datasets      | tf.keras数据集模块，包括boston_housing，cifar10，fashion_mnist，imdb ，mnist |
| layers        | Keras层API                                                   |
| losses        | 各种损失函数                                                 |
| metircs       | 各种评价指标                                                 |
| models        | 模型创建模块，以及与模型相关的API                            |
| optimizers    | 优化方法                                                     |
| preprocessing | Keras数据的预处理模块                                        |
| regularizers  | 正则化，L1,L2等                                              |
| utils         | 辅助功能实现                                                 |

### 常用方法

深度学习实现的主要流程

```mermaid
flowchart LR
    a(数据获取)-->b(数据处理)-->c(模型创建与训练)-->d(模型测试与评估)-->e(模型预测)
```

1. 导入`tf.keras`

```python
from tensorflow import keras
```

2. 数据输入。对于小的数据集，可以直接使用numpy格式的数据进行训练、评估模型，对于大型数据集或者要进行跨设备训练时使用`tf.data.datasets`来进行数据输入。
3. 模型构建
   * 简单模型使用Sequential进行构建
   * 复杂模型使用函数式编程来构建
   * 自定义layers
4. 训练与评估

配置训练过程

```python
# 配置优化方法，损失函数和评价指标
model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

模型训练

```python
model.fit(dataset, epochs=10, batch_size=3, validation_data=val_dataset)
```

模型评估

```python
model.evaluate(x, y, batch_size=32)
```

模型预测

```python
model.predict(x, batch_size=32)
```

5. 回调函数。回调函数用在模型训练过程中，来控制模型训练行为，可以自定义回调函数，也可使用`tf.keras.callbacks`内置回调函数：
   * `ModelCheckpoint`定期保存checkpoints。 
   * `LearningRateScheduler`动态改变学习速率。 
   * `EarlyStopping`当验证集上的性能不再提高时，终止训练。 
   * `TensorBoard`使用TensorBoard监测模型的状态。

6. 模型的保存和恢复。

只保存参数

```python
model.save_weights('./my_model')
model.load_weights('my_model')
```

保存整个模型，保存模型架构与权重在h5文件中。

```python
model.save('my_model.h5')
model = keras.models.load_model('my_model.h5')
```

## 模型搭建

鸢尾花分类

<img src="https://raw.githubusercontent.com/hughxusu/lesson-ai/develop/images/base/Iris-Dataset-Classification.png" style="zoom:50%;" />

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

### tensorflow实现

1. 将数据转换为张量数据类型

```python
x_train_tf = tf.convert_to_tensor(x_train, dtype=tf.float32)
x_test_tf = tf.convert_to_tensor(x_test, dtype=tf.float32)
y_train_tf = tf.convert_to_tensor(y_train, dtype=tf.int32)
y_test_tf = tf.convert_to_tensor(y_test, dtype=tf.int32)
print(x_train_tf.shape, x_test_tf.shape, y_train_tf.shape, y_test_tf.shape)
```

2. TensorFlow的标签需要为one-hot编码，标签数据转换

```python
num_classes = 3 
y_train_one_hot = tf.one_hot(y_train, depth=num_classes)
y_test_one_hot = tf.one_hot(y_test, depth=num_classes)
print(y_train_one_hot.shape, y_test_one_hot.shape)
```

3. 设计网络模型

<img src="https://raw.githubusercontent.com/hughxusu/lesson-ai/develop/images/cv/51c3298d5b3f85b7b51530b6fb30ef02.png" style="zoom:65%;" />

使用`tf.keras.Sequential`创建模型，模型是层的线性堆叠，该模型的构造函数会采用一系列层实例。

```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(10, activation='relu', input_shape=(4,)),
    Dense(10, activation='relu'),
    Dense(3, activation='softmax')
])

print(model.summary())
```

* `Dense`表示全连接网络，有10个神经元，激活函数是relu。
* 最后一层有三个神经元，激活函数是softmax。
* 通过`model.summary()`可以查看模型的架构

4. 训练模型

```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train_tf, y_train_one_hot, epochs=10, batch_size=1, verbose=1)
```

* `model.compile`可以设置模型的优化策略、损失函数和模型精度的计算方法。
* `model.fit`训练模型，`verbose=1`是打印训练过程信息。

5. 评价模型

```python
loss, accuracy = model.evaluate(x_test_tf, y_test_one_hot)
print("Accuracy", accuracy)
```

