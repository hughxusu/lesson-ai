# 条件控制

> 有所为而有所不为。

## 关系运算符

| 运算符 | 描述                                                         |
| ------ | ------------------------------------------------------------ |
| ==     | 检查两个操作数的值是否相等，如果是，则条件成立，返回 True    |
| !=     | 检查两个操作数的值是否不相等，如果是，则条件成立，返回 True  |
| >      | 检查左操作数的值是否大于右操作数的值，如果是，则条件成立，返回 True |
| <      | 检查左操作数的值是否 小于右操作数的值，如果是，则条件成立，返回 True |
| >=     | 检查左操作数的值是否大于或等于右操作数的值，如果是，则条件成立，返回 True |
| <=     | 检查左操作数的值是否小于或等于右操作数的值，如果是，则条件成立，返回 True |

```python
a = 10
b = 10
c = 6

print(a == b)
print(a >= b)
print(a < c)

a = '张三' 
b = '张三'
c = '李四'

print(a == b) # 比较两个字符串值是否相等
print(c == c)
```

> [!warning]
>
> 由于字符串比较大小计算规则复杂，通常只使用 `==` 运算符。

## 条件语句

根据条件做出判断，并以一定的策略来应对，这是生活中极为常见的场景。

计算机中已条件语句来模拟上述的生活场景，**条件的定义**：

* 如果满足条件，才能做某件事情，
* 如果不满足条件，就做另外一件事情，或者什么也不做。

> 条件语句又被称为”分支语句“，正是条件语句，才让程序有了无穷的变化。

### `if` 语句

Python 中使用 `if` 语句，来实现条件的控制。

```python
if condition:
    statement_block

following_block
```

代码执行过程

<img src="https://static.runoob.com/images/mix/python-if.webp" style="zoom:40%;" />

```python
day = int(input("今天星期几（1~7）: "))

print("=" * 20)
if day == 3:
    print('下午要上课……')
    print('喝杯咖啡提提神！')

print("=" * 20)
print('不管星期几都要做核酸！')
```

条件语句流程图

<img src="https://s1.ax1x.com/2023/03/07/ppen1BD.jpg" style="zoom: 75%;" />

>[!warning]
>
>代码块：与其他语言中使用 `{}` 表示代码块不同，Python 中使用缩进来表示代码块

### `else` 语句

Python 中使用 `else` 语句，来实现不满足条件的操作。

```python
if condition:
  statement_block_1
else:
  statement_block_2
  
following_block
```

程序流程图

<img src="https://s1.ax1x.com/2023/03/07/ppenagP.jpg" style="zoom:75%;" />

```python
day = int(input("今天星期几（1~7）: "))

print("=" * 20)
if day == 3:
    print('下午要上课……')
    print('喝杯咖啡提提神！')
else:
    print('下午不上课，美美睡个午觉……')

print("=" * 20)
print('不管星期几都要做核酸！')
```

### `elif` 语句

Python 中使用 `elif` 语句，来实现多种条件的操作。

```python
if condition_1:
  	statement_block_1
elif condition_2:
  	statement_block_2
else:
  	statement_block_3
    
following_block
```

程序流程图

<img src="https://s1.ax1x.com/2023/03/07/ppen23q.jpg" style="zoom:60%;" />

```python
day = int(input("今天星期几（1~7）: "))
print("=" * 20)

if day == 3:
    print('下午要上课……')
    print('喝杯咖啡提提神！')
elif day == 5:
    print('上午要上课……')
    print('不能睡到自然醒！')
else:
    print('下午不上课，美美睡个午觉……')

print("=" * 20)
print('不管星期几都要做核酸！')
```

> [!warning]
>
> 1. `else` 语句和 `elif` 不能单独使用，必须与 `if` 语句一起使用。
> 2. `if`、`elif` 和 `else` 组合在一起使用时，通常被看做一个代码块。
> 3. Python 中没有 `switch/case` 语句用于分支判断。

### `pass` 语句

Python 中 `pass` 不做任何事情，一般用做占位语句。

```python
day = age = int(input("今天星期几（1~7）: "))
print("=" * 20)

if day == 3:
    print('下午要上课……')
    print('喝杯咖啡提提神！')
elif day == 5:
    pass # 代码为完成时占位使用
else:
    pass
```

### 三目运算符

Python 中的三目运算符。

```python
a = 10
b = 11
max = a if a > b else b # 为真时的结果 if 判定条件 else 为假时的结果
```

## 逻辑运算符

| 名称 | 运算符 | 逻辑表达式 | 描述                                                         |
| ---- | ------ | ---------- | ------------------------------------------------------------ |
| 与   | and    | x and y    | 只有 x 和 y 的值都为 True，才会返回 True<br />否则只要 x 或者 y 有一个值为 False，就返回 False |
| 或   | or     | x or y     | 只要 x 或者 y 有一个值为 True，就返回 True<br />只有 x 和 y 的值都为 False，才会返回 False |
| 非   | not    | not x      | 如果 x 为 True，返回 False<br />如果 x 为 False，返回 True   |

1. 与运算

```python
heat = float(input("今天体温是: "))

if heat >= 35 and heat <= 38:
    print('你的体温正常')
else:
    print('你的体温异常')
```

2. 或运算

```python
student_1 = 36.5
student_2 = 38.5

if student_1 > 38 or student_2 > 2:
    print('都要隔离！')
else:
    print('躲过一劫~~')
```

3. 非运算

```python
is_white = True

if not is_white:
    print('不是白名单，禁止入校园！')
```

## `if`嵌套

在局部代码块中可嵌套 `if` 代码块。

```python
day = int(input("今天星期几（1~7）: "))
week = int(input("本周是第几周（1~16）: "))
print("=" * 20)

if day == 3:
    print('下午要上课……')
    print('喝杯咖啡提提神！')
elif day == 5:
    if week % 2 == 0:
        print('上午要上课……')
        print('不能睡到自然醒！')
    else:
        print('下午不上课，美美睡个午觉……')
else:
    print('下午不上课，美美睡个午觉……')

print("=" * 20)
print('不管星期几都要做核酸！')
```

> [!tip]
>
> 可以用逻辑运算符对上述代码进行优化，应该如何优化？

## 综合练习

> [!tip]
>
> 需求
>
> 1. 从控制台输入要出的拳 —— 石头（1）／剪刀（2）／布（3）。
> 2. 电脑随机出拳，比较胜负。
>
> ```python
> player = int(input("请出拳 石头（1）／剪刀（2）／布（3）："))
> 
> # 编写后续代码
> ```

**随机数的处理**

在 Python 中，要使用随机数，首先需要导入随机数的模块 —— “工具包”

```python
import random # 首先需要导入随机数的模块工具包

c = random.randint(12, 20)  # 生成的随机数 12 <= n <= 20  
random.randint(20, 20)  # 结果永远是 20   
random.randint(20, 10)  # 该语句是错误的，下限必须小于上限
```

