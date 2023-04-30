# 异常、模块和包

## 异常

异常：当检测到一个错误时，解释器就无法继续执行。

打开文件异常

```python
open('test.txt', 'r')
```

### 基本写法

```python
try:
    error_code # 可能发生错误的代码
except:
    other_code # 替代代码
```

打开不存在的文件

```python
try:
    f = open('test.txt', 'r')
except:
    f = open('test.txt', 'w')
```

#### 捕获指定异常

```python
try:
    error_code
except error_type: # 异常类型
    other_code
```

捕获名称异常

```python
try:
    print(num)
except NameError:
    print('名称异常')
```

#### 捕获多个指定异常

当捕获多个异常时，可以把要捕获的异常类型的名字，放到except 后，并使用元组的方式进行书写。

```python
try:
    print(1/0)
except (NameError, ZeroDivisionError):
    print('有异常')
```

#### 捕获异常描述信息

```python
try:
    print(num)
except (NameError, ZeroDivisionError) as result:
    print(result)
```

#### 捕获所有异常

Exception是所有程序异常类的父类。

```python
try:
    print(num)
except Exception as result:
    print(result)
```

### 异常的 `else`

`else` 表示的是如果没有异常要执行的代码。

```python
try:
    print(1)
except Exception as result:
    print(result)
else:
    print('没有异常的时候执行的代码')
```

### 异常的 `finally`

`finally` 表示的是无论是否异常都要执行的代码。

```python
try:
    f = open('test.txt', 'r')
except Exception as result:
    f = open('test.txt', 'w')
else:
    print('没有异常')
finally:
    f.close()
```

### 自定义异常

在Python中，抛出自定义异常的语法为 `raise` 异常类对象。

```python
# 自定义异常类
class ShortInputError(Exception):
    def __init__(self, length, min_len):
        self.length = length
        self.min_len = min_len

    # 设置抛出异常的描述信息
    def __str__(self):
        return f'你输入的长度是{self.length}, 不能少于{self.min_len}个字符'


def main():
    try:
        con = input('请输入密码：')
        if len(con) < 6:
            raise ShortInputError(len(con), 6)
    except Exception as result:
        print(result)
    else:
        print('密码已经输入完成')


main()
```

## 

### 模块

一个 Python 文件可以看做一个模块，模块能定义函数，类和变量，模块里也能包含可执行的代码。

### 导入模块

1. `import`

```python
import math
print(math.sqrt(9))
```

2. ` from import`

```python
from math import sqrt
print(sqrt(9))
```

3. `from import *`

```python
from math import *
print(sqrt(9))
```

4. `as` 别名

```python
# 模块别名
import math as m
print(m.sqrt(9))


# 功能别名
from math import sqrt as sq
print(sq(9))
```

### 自定义模块

在Python中，每个Python文件都可以作为一个模块，模块的名字就是文件的名字。

> [!warning]
>
> 自定义模块名必须要符合标识符命名规则。

1. 创建模块：创建一个 Python 文件，并定义函数。

```python
def add(a, b):
    print(a + b)
```

2. 测试模块：可以通过函数调用测试模块中的函数。

```python
if __name__ == '__main__':
    add(1, 1)
```

> [!note]
>
>  只有当执行文件的入口为当前文件时才满足上述条件。

3. 模块的调用

```python
import demo_module
demo_module.add(2, 2)
```

#### 模块定位顺序

导入模块时，Python解析器对模块位置的搜索顺序是：

1. 当前目录。
2. 环境变量的配置目录。
3. Python安装默认路径（Linux 系统，默认路径一般为/usr/local/lib/python/）

> [!warning]
>
> 1. 自己的文件名不要和已有模块名重复，否则导致模块功能无法使用。
> 2. 如果使用`from import`或`from import *`导入多个模块的时候，且模块内有同名功能。当调用这个同名功能的时候，调用到的是后面导入的模块的功能。

#### `__all__` 变量

使用 `__all__` 变量可以控制模块中导出的函数。

1. 定义模块

```python
__all__ = ['testA']

def testA():
    print('testA')

def testB():
    print('testB')
```

2. 导入模块的文件代码

```python
from demo_module import *
testA()
```

## 包

Python 中的包就是包含多个模块文件的文件夹，并且在这个文件夹有一个 `__init__.py` 文件。

### 定义包文件

1. 创建文件夹。
2. 添加`__init__.py` 文件。
3. 添加模块。

### 导入包中的文件

1. 导入包中的模块

```python
import demo_package.first
demo_package.first.print_info()
```

2. 导入全部模块

```python
from demo_package import *
first.print_info()
```

3. 导入指定模块

```python
from demo_package import first
first.print_info()
```

4. 导入指定函数

```python
from demo_package.first import print_info
print_info()
```

### `__all__` 变量

在 `__init__.py` 文件中添加 `__all__ = []`，控制允许导入的模块列表。
