# 装饰器

## 闭包

**闭包**：在函数嵌套的前提下，内部函数使用了外部函数的变量，并且外部函数返回了内部函数，这个使用外部函数变量的内部函数称为闭包。闭包也是高级函数的一种。

闭包的使用：

1. 定义外部函数
2. 定义内部函数（函数嵌套）
3. 外部函数返回了内部函数

```python
# 1. 定义外部函数
def func_out(num1):
    # 2. 定义内部函数
    def func_inner(num2):
        result = num1 + num2
        print("结果是:", result)
    # 3. 外部函数返回了内部函数
    return func_inner

# 创建闭包实例    
f = func_out(1)
# 执行闭包
f(2)
f(3)
```

上述案例中闭包保存了外部函数内的变量num1，每次执行闭包都是在num1 = 1 基础上进行计算。

> [!warning]
>
> 1. 闭包可以保存外部函数内的变量，不会随着外部函数调用完而销毁。
>
> 2. 由于闭包引用了外部函数的变量，则外部函数的变量没有及时释放，消耗内存。

### 闭包的使用

> [!tip]
>
> 需求: 根据配置信息使用闭包实现不同人的对话信息，
>
> 例如：
>
> Tom: How are you?
>
> Jack: Fine thank you, and you?
>
> Tom: I am fine, too.

```python
def config_name(name):
    def inner(msg):
        print(name + ": " + msg)
    return inner

tom = config_name("tom")
jack = config_name("jack")

tom("How are you?")
jack("Fine thank you, and you?")
tom("I am fine, too.")
```

> [!note]
>
> 闭包的本质是一个函数与其相关的引用环境组合的一个整体（实体），可以看做一个简略的类。

### 修改闭包内使用的外部变量

闭包中保存的外部函数内的变量可以被修改

```python
def func_out():
    num1 = 10

    def func_inner():
        nonlocal num1
        num1 = 20
        
        result = num1 + 10
        print(result)

    print("修改前的外部变量:", num1)
    func_inner()
    
    print("修改后的外部变量:", num1)
    return func_inner

new_func = func_out()
new_func()
```

## 装饰器

**装饰器**：就是给已有函数增加额外功能的函数，它本质上就是一个闭包函数。

装饰器的特点：

1. 不修改已有函数的源代码
2. 不修改已有函数的调用方式
3. 给已有函数增加额外的功能

```python
def check(fn):
    def inner():
        print("已登录....")
        fn()
    return inner


def comment():
    print("发表评论")

comment = check(comment)
comment()
```

> [!warning]
>
> 1. 闭包函数有且只有一个参数，必须是函数类型，这样定义的函数才是装饰器。
> 2. 不改变被装饰函数的名称和调用方式。

### 装饰器的标准写法

python 语言中有专门的写法来表示装饰器（语法糖）

```python
def check(fn):
    def inner():
        print("已登录....")
        fn()
    return inner


@check
def comment():
    print("发表评论")

comment()
```

> [!note]
>
> `@check` 等价于 `comment = check(comment)`

### 装饰器的使用

> [!tip]
>
> 实现一个统计函数运算时间的装饰器

```python
import time

def compute_time(func):
    def inner():
        begin = time.time()
        func()
        end = time.time()

        result = end - begin
        print("函数执行完成耗时:", result)

    return inner


@compute_time
def work():
    for i in range(10000):
        print(i)

work()
```

### 通用装饰器

1. 装饰带有参数的函数

```python
def logging(fn):
    def inner(num1, num2):
        print("--正在努力计算--")
        fn(num1, num2)
    return inner

@logging
def sum_num(a, b):
    result = a + b
    print(result)

sum_num(1, 2)
```

2. 装饰带有返回值的函数

```python
def logging(fn):
    def inner(num1, num2):
        print("--正在努力计算--")
        result = fn(num1, num2)
        return result
    return inner

@logging
def sum_num(a, b):
    result = a + b
    return result


result = sum_num(1, 2)
print(result)
```

3. 装饰带有不定长参数的函数

```python
def logging(fn):
    def inner(*args, **kwargs):
        print("--正在努力计算--")
        fn(*args, **kwargs)
    return inner


@logging
def sum_num(*args, **kwargs):
    result = 0
    for value in args:
        result += value

    for value in kwargs.values():
        result += value

    print(result)

sum_num(1, 2, a=10)
```

4. 通用装饰器

```python
def logging(fn):
    def inner(*args, **kwargs):
        print("--正在努力计算--")
        result = fn(*args, **kwargs)
        return result
    return inner

@logging
def subtraction(a, b):
    result = a - b
    print(result)

result = subtraction(4, 2)
print(result)
```

### 多个装饰器的使用

一个函数可以使用多个装饰器装饰

```python
def make_div(func):
    def inner(*args, **kwargs):
        return "<div>" + func() + "</div>"
    return inner

def make_p(func):
    def inner(*args, **kwargs):
        return "<p>" + func() + "</p>"
    return inner

@make_div
@make_p
def content():
    return "hello, world"

result = content()

print(result)
```

多个装饰器可以对函数进行多个功能的装饰，装饰顺序是由内到外的进行装饰。

### 带有参数的装饰器

使用带有参数的装饰器，其实是在装饰器外面又包裹了一个函数，使用该函数接收参数，返回是装饰器，因为 @ 符号需要配合装饰器实例使用

```python
def make_label(label):
    def decorator(func):
        def inner(*args, **kwargs):
            return f"<{label}>" + func() + f"</{label}>"
        return inner
    return decorator

@make_label('div')
def content():
    return "hello, world"
```

### 类装饰器的使用

可以通过定义一个类来装饰函数。

```python
class Check(object):
    def __init__(self, fn):
        self.__fn = fn

    def __call__(self, *args, **kwargs):
        print("请先登陆...")
        self.__fn()

@Check
def comment():
    print("发表评论")

comment()
```

1. `@Check` 等价于 `comment = Check(comment)`，所以需要提供一个 `init` 方法，并多增加一个 `fn` 参数。
2. 要想类的实例对象能够像函数一样调用，需要在类里面实现 `call` 方法，把类的实例变成可调用对象(callable)，也就是说可以像调用函数一样进行调用。
3. 在 `call`方法里进行对 `fn` 函数的装饰，可以添加额外的功能。

# property属性

property 属性就是负责把一个方法当做属性进行使用，用于简化代码

```python
class User(object):
    def __init__(self):
        self.__value = 0

    @property
    def percent(self):
        return self.__value * 100

    @percent.setter
    def percent(self, value):
        if value > 100:
            self.__value = 1
        elif value < 0:
            self.__value = 0
        else:
            self.__value = value / 100
               
user = User()
print(user.percent)
user.percent = 44
print(user.percent)
user.percent = 120
print(user.percent)
```

类属性方式

```python
class User(object):
    def __init__(self):
        self.__value = 0

    def get_percent(self):
        return self.__value * 100

    def set_percent(self, value):
        if value > 100:
            self.__value = 1
        elif value < 0:
            self.__value = 0
        else:
            self.__value = value / 100
    percent = property(get_percent, set_percent)
    
user = User()
user.percent = 44

user2 = User()
user2.percent = 66

print(user.percent)
print(user2.percent)

User.percent = 77

print(user.percent)
print(user2.percent)
```

property 的参数说明：

1. 第一个参数是获取属性时要执行的方法
2. 第二个参数是设置属性时要执行的方法
