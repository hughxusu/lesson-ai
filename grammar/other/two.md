# `with` 语句与上下文管理器

通常打开文件的方法

```python
 f = open("demo.txt", "w")
 f.write("hello, world!")
 f.close()
```

文件使用完后必须关闭，因为文件对象会占用操作系统的资源，并且操作系统同一时间能打开的文件数量也是有限的。

文件打开是可能存在异常，更完备的写法：

```python
try:
    f = open("demo.txt", "r")
    f.write("hello, python!")

except IOError as e:
    print("文件操作出错")
finally:
    f.close()
```

Python 语言中为上述代码提供更简洁的语法规范，使用 `with` 语句。

```python
with open("demo.txt", "w") as f:
    f.write("hello world")
```

使用 `with` 语句后执行完成以后自动调用关闭文件操作，即使出现异常也会自动调用关闭文件操作。

## 上下文管理器

实现了上下文管理器功能的函数或类可以使用 `with` 语句。

### 类上下文管理器

一个类只要实现了 `__enter__()` 和 `__exit__()` 这个两个方法，通过该类创建的对象我们就称之为上下文管理器。

```python
class File(object):

    # 初始化方法
    def __init__(self, file_name, file_model):
        # 定义变量保存文件名和打开模式
        self.file_name = file_name
        self.file_model = file_model

    # 上文方法
    def __enter__(self):
        print("进入上文方法")
        # 返回文件资源
        self.file = open(self.file_name,self.file_model)
        return self.file

    # 下文方法
    def __exit__(self, exc_type, exc_val, exc_tb):
        print("进入下文方法")
        self.file.close()


if __name__ == '__main__':

    # 使用with管理文件
    with File("1.txt", "r") as file:
        file_data = file.read()
        print(file_data)
```

1. `__enter__` 表示上文方法，需要返回一个操作文件对象。
2. `__exit__` 表示下文方法，with语句执行完成会自动执行，即使出现异常也会执行该方法。
3. `__exit__` 方法如果返回 `True` ，`with` 语句中出现异常将被拦截，不会向上层抛出。

### 函数上下文管理器

Python 提供 `@contextmanager` 装饰器，可以使一个函数成为上下文管理器。

```python
from contextlib import contextmanager

@contextmanager
def my_open(path, mode):
    try:
        file = open(file_name, file_mode)
        yield file
    except Exception as e:
        print(e)
    finally:
        print("over")
        file.close()
        
with my_open('out.txt', 'w') as f:
    f.write("hello , the simplest context manager")
```

1. 使用 `yield` 返回值为函数的返回值，`yield` 将函数分割为两部分。
2. `yield` 上面的语句类似于 `__enter__` 方法中的语句。
3. `yield` 下面的语句类似于 `__exit__` 方法中的语句。
4. 装饰函数本身不能处理异常。