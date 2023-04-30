# 几种特殊函数

## 递归函数

递归是一种编程思想，函数内部自己调用自己。

斐波那契数列
$$
f(n) =
\begin{cases} 
0,  & n=0 \\
1, & n=1 \\
f(n-1) + f(n-2), & n \geqslant 2, n\in N^* \\
\end{cases}
$$

| 位置 | 0    | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    | 10   | 11   | 12   |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 值   | 0    | 1    | 1    | 2    | 3    | 5    | 8    | 13   | 21   | 34   | 55   | 89   | 144  |

使用递归来实现斐波那契数列

```python
def fib(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fib(n - 2) + fib(n - 1)

print(fib(6))
```

<img src="https://s1.ax1x.com/2023/03/28/pp6qZz6.png" style="zoom:90%;" />

> [!attention]
>
> 1. 递归函数必须有终止条件。
> 2. Python 中递归调用的深度是900层~1000层之间。

## 匿名函数

Python 使用 `lambda` 来创建匿名函数。

* `lambda` 表达式的参数形式和一般函数相同。
* `lambda` 的主体是一个表达式，而不是一个代码块，只能返回一个表达式的值。
* `lambda` 函数拥有自己的命名空间，且不能访问自己参数列表之外或全局命名空间里的参数。

### 基本语法

```python
lambda args: expression
```

使用实例

```python
square = lambda x: x ** 2
print(type(square))
print(square(2))

# 等价于
def square(x):
    return x ** 2
print(square(2))
```

### `lambda` 参数形式

#### 无参数

```python
fn = lambda: 100
print(fn())
```

#### 必要参数

```python
fn = lambda top, bottom, h: (top + bottom) * h / 2
print(fn(3, 5, 4))
```

#### 关键字参数

```python
fn = lambda top, bottom, h: (top + bottom) * h / 2
print(fn(3, h=4, bottom=5))
```

#### 默认参数

```python
fn = lambda top, bottom, h=1: (top + bottom) * h / 2
print(fn(3, 5))
```

#### 可变参数

```python
fn = lambda *args, **kwargs: [args, kwargs]
print(fn(1, 2, 3, name='tom', age=18, is_male=True))
```

### `lambda` 应用

1. 三目运算符与 `lambda` 相关结合

```python
max = lambda a, b: a if a > b else b
print(max(5, 3))
```

2. 数组排序

```python
students = [
    {'name': 'Tom', 'age': 20},
    {'name': 'Jack', 'age': 18},
    {'name': 'Mike', 'age': 22}
]

# 按age值升序排列
students.sort(key=lambda x: x['age'])
print(students)

# 按age值降序排列
students.sort(key=lambda x: x['age'], reverse=True)
print(students)
```

## 函数作为参数

在 Python 中函数名本身也是变量，所以可以将函数名作为另外一个函数的参数。函数作为参数是**高阶函数**的一种。

```python
print(abs(-5))

def add(x, y, f):
    return f(x) + f(y)

print(add(-5, 6, abs))
```

### 内置高阶函数

#### `map()`

`map()` 函数接收两个参数，一个是函数，一个是 `Iterable`，`map` 将传入的函数依次作用到序列的每个元素，并把结果作为新的 `Iterator` 返回。

```python
arr = [1, 2, 3, 4, 5, 6, 7, 8, 9]

def square(x):
    return x ** 2

result = map(square, arr)

print(type(result))
print(list(result))

# 使用 lambda 函数替代
result = map(lambda x: x ** 2, arr)

print(type(result))
print(list(result))
```

<img src="https://s1.ax1x.com/2023/03/28/pp6qUOS.png" style="zoom:80%;" />

#### `reduce()`

`reduce()` 将一个数据序列中的所有数据进行下列操作：用传给 `reduce` 中的函数有两个参数先对集合中的第 1、2 个元素进行操作，得到的结果再与第三个数据用函数运算，最后得到一个结果。`reduce` 只返回一个结果。

![](https://www.mybluelinux.com/img/post/posts/0149/python-reduce-3.png)

```python
import functools

arr = [1, 2, 3, 4]

def sum(a, b):
    return a + b

result = functools.reduce(sum, arr) 

print(result)
```

<img src="https://www.pythontut.com/static/pictures/reduce_flow_chart.png" style="zoom:60%;" />

#### `filter()`

`filter()` 函数用于过滤序列，过滤掉不符合条件的元素，返回一个迭代器对象，如果要转换为列表，可以使用 `list()` 来转换。该接收两个参数，第一个为函数，第二个为序列，序列的每个元素作为参数传递给函数进行判断，然后返回 True 或 False，最后将返回 True 的元素放到新列表中。

<img src="https://s1.ax1x.com/2023/02/26/pppgaV0.png" style="zoom: 50%;" />

```python
odds = filter(lambda x: x % 2 == 1, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(type(odds))
print(list(odds))
```

