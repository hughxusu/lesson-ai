# 深度学习DNN

## 机器学习框架

### Keras安装

```shell
pip install tensorflow
pip install --upgrade keras
```

查看是否安装成功

```python
import keras
print(keras.__version__)
```

### 创建网络模型

创建训练和测试数据集

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

iris = datasets.load_iris()
x = iris.data
y = iris.target
y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```

创建模型并训练数据

```python
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(units=5, input_dim=4, activation='sigmoid')) # 输入维度应该和特征维度相等
model.add(Dense(units=6, input_dim=5, activation='sigmoid'))
model.add(Dense(units=3, activation='softmax')) # 输出维度应该等于类别数
model.compile(loss='categorical_crossentropy', optimizer="adam") # 损失函数和学习方法
model.fit(x_train, y_train, batch_size=8, epochs=10, shuffle=True) # 训练的迭代控制
```

评估训练结果

```python
score = model.evaluate(x_test, y_test)
print(score)
```

增加中间层的数量

```python
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(units=5, input_dim=4, activation='sigmoid'))
model.add(Dense(units=10, input_dim=5, activation='sigmoid'))
model.add(Dense(units=3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=8, epochs=500, shuffle=True)
score = model.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```

`epochs` 每个周期训练的次数，`batch_size`每次训练输入的数据数量，一般都取$2^n$。`shuffle`将数据随机打乱。`metrics`保存训练模型的准确度。

保存和读取模型

```python
from keras.models import load_model

path="model.keras"
model.save(path)
model_load = load_model(path)
result = model_load.predict(x_test)
print(result)
```

```mermaid
flowchart LR
		z(x输入)-->c(5个神经元)-->a(10个神经元)-->b(输出3个类别)
```

可以给中间层设置正则`kernel_regularizer`，

```python
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers

model = Sequential()
model.add(Dense(units=5, input_dim=4, activation='sigmoid', name='layer1'))
model.add(Dense(units=10, input_dim=5, activation='sigmoid', kernel_regularizer=regularizers.l1(0.01), name='layer2'))
model.add(Dense(units=3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=8, epochs=500, shuffle=True)
```

`name`给网络的一层定义名字，打印参数。

```python
print(model.get_layer('layer2').get_weights()[0])
```

### 函数式模型

```python
from keras.models import Model
from keras.layers import Dense,Input

feature_input = Input(shape=(4,))

#first model
m11 = Dense(units=5,input_dim=4,activation='relu')(feature_input)
m12 = Dense(units=6, activation='relu')(m11)

#second model
m21 = Dense(units=5,input_dim=4,activation='relu')(feature_input)
m22 = Dense(units=6, activation='relu')(m21)

m = keras.layers.Concatenate(axis= 1)([m21,m22])
output = Dense(units=3, activation='softmax')(m)
model = Model(inputs=feature_input, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=8, epochs=100, shuffle=True)
score = model.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```

```mermaid

		
```





