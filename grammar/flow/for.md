# 循环控制

> 日复一日，圈复一圈

## 赋值运算符

在 Python 中，除 `=` 可以给变量赋值外，还提供了一系列的与算术运算符对应的赋值运算符，来简化代码的编写。

| 运算符 | 描述                       | 实例                                        |
| ------ | -------------------------- | ------------------------------------------- |
| `=`    | 简单的赋值运算符           | `c = a + b` 将 `a + b` 的运算结果赋值为 `c` |
| `+=`   | 加法赋值运算符             | `c += a` 等效于 `c = c + a`                 |
| `-=`   | 减法赋值运算符             | `c -= a` 等效于 `c = c - a`                 |
| `*=`   | 乘法赋值运算符             | `c *= a` 等效于 `c = c * a`                 |
| `/=`   | 除法赋值运算符             | `c /= a` 等效于 `c = c / a`                 |
| `//=`  | 取整除赋值运算符           | `c //= a` 等效于 `c = c // a`               |
| `%=`   | 取 **模** (余数)赋值运算符 | `c %= a` 等效于`c = c % a`                  |
| `**=`  | 幂赋值运算符               | `c \**= a` 等效于`c = c ** a`               |

```python
a = 3
b = 4

b *= a
b -= b
b += a
b **= a
```

## 循环语句

循环的作用就是让指定的代码块，重复执行一定的次数。

### `while` 循环

```python
while condition: # 条件为 Ture 执行循环，条件为 False 退出循环
    statement_block

following_block
```

`while` 循环流程图

<img src="https://s1.ax1x.com/2023/03/07/ppeumVg.jpg" style="zoom:75%;" />

```python
i = 1 # 计数器

while i <= 5:
    print("Hello Python")
    i += 1

print("循环结束后的 i = %d" % i)
```

> [!warning]
>
> 1. `while` 及循环语句通常被视为同一代码块。
> 2. 循环结束后，计数器依旧保留循环语句中最后一次执行的值。

#### 程序中的计数原则

* 自然计数法，从1开始计数。
* 程序计数法从0开始计数。在编写程序时，应该尽量养成习，循环计数从0开始。

#### 循环计算

利用循环进行统计计算，是循环在程序开发中的一个重要应用。

需求：计算0 ~ 100之间所有偶数求和。

```python
result = 0

i = 0
while i <= 100:
    if i % 2 == 0:
        result += i

    i += 1

print("0~100之间偶数求和结果 = %d" % result)

# 当增量为2时，即 i+=2 如何改写该例子。
```

> [!attention]
>
> **死循环**：忘记在循环内部修改循环的判断条件，导致循环持续执行，程序无法终止！

###  `break` 和 `continue` 

`break` 和 `continue` 是专门在循环中使用的关键字：

* `break` 某一条件满足时，退出循环，不再执行后续重复的代码。
* `continue` 某一条件满足时，不执行后续重复的代码。

<img src="https://static.runoob.com/images/mix/python-while.webp" style="zoom:55%;" />

#### `break`

在循环过程中，如果 某一个条件满足后，不再希望循环继续执行，可以使用 `break` 退出循环。

```python
i = 0

while i < 10:

    if i == 3: # i == 3 后退出循环。
        break

    print(i)
    i += 1

print("over")
```

#### `continue`

在循环过程中，如果某一个条件满足后，不希望执行循环代码，但是又不希望退出循环，可以使用 `continue`。

```python
i = 0

while i < 5:

    if i == 3:  # i == 3 时不希望打印数字
        i += 1  # 在使用 continue 之前，同样应该修改计数器否则会出现死循环
        continue

    print('当前计数 = %d' % i)
    i += 1
```

> [!warning]
>
> `break` 和 `continue` 只针对当前所在循环有效。

### `for` 循环

Python for 循环可以遍历任何可迭代对象，如一个字符串。

```python
for <variable> in <sequence>:
    statement_block
```

`for` 循环流程图

<img src="https://s1.ax1x.com/2023/03/07/ppeugde.jpg" style="zoom:75%;" />



```python
shcool = '北方工业大学-理学院'
for i in shcool:
	print(i)
    
# break
for i in shcool:
	if i == '-':
		break
	print(i)
  
# continue
for i in shcool:
	if i == '-':
		continue
	print(i)
    
```

### 循环嵌套

循环嵌套，就是一个循环中嵌套另一个循环，每个循环的语法相同。

<img src="https://pic1.zhimg.com/80/v2-b18b8b9092554dd4f9af9067db109528_1440w.webp" style="zoom:75%;" />

需求：用三行四列的方式打印1~12个数

```python
result = 1
row = 0

while row < 3:
	col = 0
	while col < 4:
		print('%d, ' % result, end='')
		result += 1
		col += 1
	print()
	row += 1
```

> [!tip]
>
> [打印九九乘法表](https://jennifercodingworld.files.wordpress.com/2016/06/e4b998e6b395e8a1a8.jpeg)

### 循环和`else`

循环可以和else配合使⽤用，else下⽅方缩进的代码指的是当循环正常结束之后要执⾏的代码。

#### `while` + `else`

```python
# case break
i = 1

while i <= 5:
    if i == 3:
        print(f"星期 {i} 忘了做核酸~~")
        i += 1
        break
    else:
        print(f"星期 {i} 做核酸~~")
    i += 1
else:
    print('核酸天天做')
    

# case continue
i = 1

while i <= 5:
    if i == 3:
        print(f"星期 {i} 忘了做核酸~~")
        i += 1
        continue
    else:
        print(f"星期 {i} 做核酸~~")
    i += 1
else:
    print('核酸天天做')
    

```

#### `for` + `else`

```python
shcool = '北方工业大学-理学院'

# break
for i in shcool:
    if i == '-':
        break
    print(i)
else:
    print('开了一门编程课~~~')

# continue
for i in shcool:
    if i == '-':
        continue
    print(i)
else:
    print('开了一门编程课~~~')
```

## 运算符优先级

以下表格的算数优先级由高到最低顺序排列。

| 运算符                     | 描述                   |
| -------------------------- | ---------------------- |
| `**`                       | 幂 (最高优先级)        |
| `* / % //`                 | 乘、除、取余数、取整除 |
| `+ -`                      | 加法、减法             |
| `<= < > >=`                | 比较运算符             |
| `== !=`                    | 等于运算符             |
| `= %= /= //= -= += *= **=` | 赋值运算符             |
| `not or and`               | 逻辑运算符             |

运算符的结合性：先执行左边的叫左结合性，先执行右边的叫右结合性。

* Python 中大部分运算符都具有左结合性，也就是从左到右执行。
* 乘方运算符、单目运算符、赋值运算符和三目运算符例外，它们具有右结合性，也就是从右向左执行。

$$
a =\displaystyle{\frac{\displaystyle{\frac{2^3-1}{5+6}}}{5-\displaystyle{\frac{3}{4}}}}
$$

```python
a = ((2 ** 3 - 1) / (5 + 6)) / (5 - 3 / 4)
```

