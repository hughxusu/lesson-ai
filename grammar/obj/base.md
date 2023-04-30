# 基本语法

## 类和对象

### 类的定义

语法

```python
class ClzName():
    statement_block
```

实例

```python
class Horse():
    pass
```

### 创建对象

语法

```python
instance = ClzName()
```

实例

```python
horse = Horse()
print(horse)
```

## 类的组成

### 属性

属性对应于事物的特征，用来保存一个对象的数据，本质是对象的一个变量。

语法

```python
instance.property = value
```

实例

```python
horse.color = 'white'
print(horse.color)
```

### 方法

方法对应于事物的行为，本质是对象的一个函数。

语法

```python
class ClzName():
    def method(self):
        statement_block
```

实例

```python
class Horse():
    def run(self):
        print('马儿跑……')

horse = Horse()
horse.run()
```

### `self` 

`self` 指的是调用该函数的对象。

```python
class Horse():
    def run(self):
        print('马儿跑……')
        print(self)

horse = Horse()
horse.run()
print(horse)
```

在类内调用对象属性和方法

```python
class Horse():
    def run(self):
        print('马儿跑……')

    def desc(self):
        print('一匹 ', end='')
        print(self.color, end=' ')
        self.run()

horse = Horse()
horse.color = '白色'
horse.desc()
```

## 魔法方法

在Python中，一些具有特殊功能的函数，被称为魔法方法，魔法方法的函数名 `__name__`。

### `__init__()`

初始化对象，当创建对象时会自动调用该函数。

```python
class Horse():
    def __init__(self):
        self.color = '红色'

    def run(self):
        print('马儿跑……')

    def desc(self):
        print('一匹 ', end='')
        print(self.color, end=' ')
        self.run()

horse = Horse()
horse.desc()
```

#### 带参数的初始化

```python
class Horse():
    def __init__(self, color='白色'):
        self.color = color

    def run(self):
        print('马儿跑……')

    def desc(self):
        print('一匹 ', end='')
        print(self.color, end=' ')
        self.run()

horse = Horse('黑色')
horse.desc()
```

### `__str__()`

当使用 `print` 输出对象的时候，默认打印对象的内存地址。如果类定义了该方法，那么就会打印从在这个方法中 `return` 的数据。

```python
class Horse():
    def __str__(self):
        return '这是一个马儿类'

horse = Horse()
print(horse)
```

### `__del__()`

当删除对象时，python解释器也会默认调用该方法。

```python
class Horse():
    def __del__(self):
        print('放马儿跑了')

horse = Horse()
del horse
```

> [!tip]
>
> 使用 user 类优化用户登陆代码

## 访问限制

让内部属性或方法不被外部访问，可以将其变为私有，在其名称前加上两个下划线`__`，只能在类内部访问，外部不能访问。

```python
class Horse():
    def __init__(self):
        self.__color = '红色'

    def __run(self):
        print('马儿跑……')

    def desc(self):
        print('一匹 ', end='')
        print(self.__color, end=' ')
        self.__run()

horse = Horse()
horse.desc()
horse.__run()
```

> [!warning]
>
> 私有属性和方法与魔法方法定义的区别。

### 私有属性的访问与修改

在 Python 中，一般定义函数名 `get_xx` 用来获取私有属性，定义 `set_xx` 用来修改私有属性值。

```python
class Horse():
    def __init__(self):
        self.__color = 'red'

    def get_color(self):
        return self.__color

    def set_color(self, color):
        self.__color = color

horse = Horse()
horse.set_color('blue')
print(horse.get_color())
```

