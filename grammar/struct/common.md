# 相似操作

## 运算符

| 运算符 |      描述      |      支持的容器类型      |
| :----: | :------------: | :----------------------: |
|   +    |      合并      |    字符串、列表、元组    |
|   *    |      复制      |    字符串、列表、元组    |
|   in   |  元素是否存在  | 字符串、列表、元组、字典 |
| not in | 元素是否不存在 | 字符串、列表、元组、字典 |

### 加 `+`

```python
# str
str1 = 'hello'
str2 = 'world'
str3 = str1 + str2
print(type(str3))
print(str3)

# list
arr1 = ['red', 'blue']
arr2 = ['yellow', 'green']
arr3 = arr1 + arr2
print(type(arr3))
print(arr3)


# tuple
tup1 = ('red', 'blue')
tup2 = ('yellow', 'green')
tup3 = tup1 + tup2
print(type(tup3))
print(tup3)
```

### 乘 `*`

```python
# str
print('-' * 10)

# list
list1 = ['red']
print(list1 * 4)

# tuple
t1 = ('red',)
print(t1 * 4)
```

### `in` 或 `not in`

```python
# str
print('a' in 'abcd')  
print('a' not in 'abcd') 

# list
list1 = ['red', 'blue', 'yellow', 'green']
print('red' in list1)  
print('red' not in list1) 

# tuple
t1 = ('red', 'blue', 'yellow', 'green')
print('red' in t1)  
print('red' not in t1) 
```

## 公共方法

| 函数             | 描述                                                         |
| ---------------- | ------------------------------------------------------------ |
| `len()`          | 计算容器中元素个数                                           |
| `del` 或 `del()` | 删除                                                         |
| `max()`          | 返回容器中元素最大值                                         |
| `min()`          | 返回容器中元素最小值                                         |
| `enumerate()`    | 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。 |

### `len()`

```python
str = 'hello world'
colors = ['red', 'blue', 'yellow', 'green']
t1 = ('red', 'blue', 'yellow', 'green')
person = {'name': '张三', 'age': 20, 'is_male': True }
s1 = {'red', 'blue', 'yellow', 'green'}

# str
print(len(str))

# list
print(len(colors))

# tuple
print(len(t1))

# dict
print(len(person))

# set
print(len(s1))
```

### `del()`

```python
del str

del(colors[0])
print(colors)

# 元组无法删除

del(person['name'])
print(person)
```

### `min()` 或 `max()`

```python
chars = 'abcdefg'
nums = [10, 20, 30, 40]
d1 = { 'a': 1, 'b': 2, 'c': 3}

print(max(chars))
print(max(nums))
print(max(d1))
print(max(d1.values()))

print(min(chars))
print(min(nums))
print(min(d1))
print(min(d1.values()))
```

### `enumerate()`

```python
colors = ['red', 'blue', 'yellow', 'green']

for i in enumerate(colors):
    print(i)

for index, color in enumerate(colors, start=2):
    print(f'索引是{index}, 对应的颜色是{color}')
```

## 容器转换

* `tuple()` 转换为元组
* `list()` 转换为数组
* `set()` 转换为集合

```python
colors = ['red', 'blue', 'yellow', 'green']
chars = ('a', 'b', 'c', 'd')
student = {'tom', 'harry', 'kim'}

print(tuple(colors))
print(set(colors))
print(list(chars))
```

## 推导式

`range(start, end, step)` 生成从start到end的数字，步长为 step，供for循环使用。

```python
print(range(1, 10, 1))

for i in range(1, 10, 1):
    print(i, end='\t')
print()

for i in range(1, 10, 2):
    print(i, end='\t')
print()

for i in range(10):
    print(i, end='\t')
```

Python 推导式是 Python 特有的语法，用一个表达式创建一个有规律序列。包括：

* 列表与元组推导式
* 字典推导式
* 集合推导式

> [!tip]
>
> 1. 使用 `while` 循环创建 0~10 的列表。
> 2. 使用 `for` 循环创建 0~10 的偶数列表。

### 列表与元组推导式

```python
nums = [i for i in range(11)]
print(nums)

powers = [i**2 for i in range(11)]
print(nums)

evens = [i for i in range(0, 11, 2)]
print(evens)
```

#### 推导式中使用 `if` 

```python
evens = [i for i in range(11) if i % 2 == 0]
print(evens)
```

#### 列表推导式嵌套

```python
groups = [(i, j) for i in range(1, 3) for j in range(3)]
print(groups)
```

#### 元组

元组推导式返回的结果是一个生成器对象。

```python
evens = (i for i in range(0, 11, 2))
print(type(evens))
print(evens)
result = tuple(evens)
print(type(result))
print(result)
```

### 字典推导式

```python
# 生成字典
dict1 = {i: i**2 for i in range(1, 5)}

# 组合字典，注意数组对齐
keys = ['name', 'age', 'is_male']
values = ['Tom', 20, True]

persons = {keys[i]: values[i] for i in range(len(keys))}
print(persons)

# 字典中提取
stocks = {'apple': 268, 'google': 218, 'twitter': 122, 'facebook': 153, 'tesla': 230}
better = {key: value for key, value in stocks.items() if value >= 200}
print(better) 
```

### 集合推导式

集合推导式可以去重。

```python
nums = [1, 1, 2]
powers = {i ** 2 for i in nums}
print(powers) 
```

## 查阅参考手册

[Python 官方网站](https://www.python.org/)

[Python 中文手册](https://docs.python.org/zh-cn/3.9/)

1. Python 主页

<img src="https://s1.ax1x.com/2023/03/21/ppUq9Ug.jpg" style="zoom:50%;" />

2. 选择文档语言和版本

<img src="https://s1.ax1x.com/2023/03/21/ppUqC5Q.jpg" style="zoom:50%;" />

3. [菜鸟 Python 3 教程](https://www.runoob.com/python3/python3-tutorial.html)

## 综合练习

> [!tip]
>
> 统计文章中的一段使用了多少个汉字和每个汉字出现的次数。
>
> [文章链接](https://baijiahao.baidu.com/s?id=1720661522278169835&wfr=spider&for=pc)
