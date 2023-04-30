# 字符串

字符串是 Python 中最常用的数据类型，使用引号来创建字符串。

```python
name = '北方工业大学-理学院'
print(type(name))
```

## 创建字符串

* 单引号或双引号

```python
name1 = 'Tom'
name2 = "Andy"

sentence = '鲁迅写到："在我的后园，可以看见墙外有两株树，一株是枣树，还有一株也是枣树。"'
```

* 三个引号字符串，三引号形式的字符串支持换行。

```python
school1 = '''
北方工业大学
理学院
'''
school2 = """
北方工业大学
理学院
"""
print(school1)
print(school2)
```

## 输入与输出

### 字符串输入

```python
name = input('请输入您的名字：')
```

### 格式化输出

```python
print('您好，我是 %s'%name)
print(f'您好，我是 {name}')
```

### 转义字符

| 转义字符      | 描述       |
| ------------- | ---------- |
| `\`(在行尾时) | 续行符     |
| `\\`          | 反斜杠符号 |
| `\"`          | 双引号     |
| `\'`          | 单引号     |
| `\n`          | 换行       |
| `\t`          | 横向制表符 |

## 字符截取

字符串在内存中连续存储的

<img src="https://s1.ax1x.com/2023/02/26/pppgCE6.jpg" style="zoom: 50%;" />

###  索引

快速查找内存中连续数据的方法。

```python
name = '北方工业大学'
print(name[0])
print(name[3])
print(name[-3])
print(name[-6])
```

### 切片

切片是指对操作的对象截取其中一部分的操作。字符串、列表、元组等都支持切片操作。

```python
name[begin:end:step]
```

> [!warning]
>
> 1. `end` 不包含结束位置下标对应的数据，遵循计算机计数法。
> 2. `begin:end` 正负整数均可。
> 3. 步长是选取间隔，正负整数均可，默认步长为1。负数步长是从右向左计算。
> 4. 字符串切片后返回的数据类型是字符串。

```python
sub = name[2:5:1]
print(sub)
print(type(sub))
print(name[2:5])  
print(name[:5])  
print(name[1:]) 
print(name[:])  
print(name[::2])  
print(name[:-1])  
print(name[-4:-1])  
print(name[::-1]) # begin = -1 end = -1
```

## 常用方法

字符串的常用操作方法有查找、修改和判断三大类。

### 查找

查找子串在字符串中的位置或次数。

1. `find()`：检测某个子串是否包含在这个字符串或其切片中，存在返回子串开始位置的下标，否则返回-1。

```python
# str.find(sub, begin, end)
language = 'c and c++ and java and python and javascript and go'
language.find('and')

print(language.find('and', 15, 30))
print(language.find('c#'))
```

2. `index()`：用法与 `find()` 一样，不存在时报异常。

```python
print(language.index('and'))
print(language.index('and', 15, 30))
print(language.index('c#'))
```

3. `rfind()`：用法与 `find()` 一样，但查找方向从右侧开始。
4. `rIndex()`：用法与 `index()` 一样，但查找方向从右侧开始。
5. `count()`：返回某个子串在字符串中出现的次数。

```python
# str.count(sub, begin, end)
print(language.count('and'))
print(language.count('and', 15, 30))
print(language.count('c#'))
```

### 修改

通过函数修改字符串的数据，生成新字符串。

1. `replace()`： 替换

```python
# str.replace(old, new, count) count 替换几次，不写全部替换

# 字符串变量是不可变类型，修改后需要用新的变量来保存结果。
result = language.replace('and', '&')
print(language)
print(result)

print(language.replace('and', '&', 2))
```

2. `split()`： 分割

```python
# str.split(sub, count)

print(language.split('and'))
print(language.split(' '))
print(language.split('and', 2))
```

3. `join`：用一个字符或子串，将一个数组拼接为新字符串。

```python
# str.join(list)
langs = language.split('and')
'_'.join(langs)
```

4. `capitalize()`：字符串首字母换成大写。

```python
print(language.capitalize())
```

5. `title()`：每个单词首字母大写。
6. `lower()`：字符串全部转换为小写。
7. `upper()`：字符串全部转换为大写。

```python
title = language.title()
print(title)
print(title.lower())
print(title.upper())
```

8. `lstrip()`：删除字符串左侧空白字符。
9. `rstrip()`：删除字符串右侧空白字符。
10. `strip()`：删除字符串两侧空白字符。

```python
language = '  c and c++ and java and python and javascript and go   '
print(language)
print(language.lstrip())
print(language.rstrip())
print(language.strip())
```

11. `ljust()`：返回一个原字符串左对齐，并使用指定字符（默认空格）填充至对应长度 的新字符串。
12. `rjust()`：和 `ljust()` 类似，右对齐。
13. `center()`：和 `ljust()` 类似，居中对齐。

```python
# str.ljust(len, chars)

language = 'python'
print(language)
print(language.ljust(21, '#'))
print(language.rjust(21, '#'))
print(language.center(21, '#'))
```

### 判断

检测字符串或其子串是否满足条件，返回的结果是布尔型数据类型：True 或 False。

1. `startswith()`：检查字符串是否是以指定子串开头，是返回 True，否返回 False。
2. `endswith()`：和 `startswith()` 类似，检测指定结尾。

```python
language = 'c and c++ and java and python and javascript and go'

print(language.startswith('c'))
print(language.startswith('c', 5, 20))

print(language.endswith('go'))
print(language.endswith('go', -20, -11))
```

3. `isalpha()`：字符串全部是字母，返回 True, 否则返回 False。

```python
print('hello'.isalpha())
print('hello '.isalpha())
print('hello#'.isalpha())
print('hello12345'.isalpha())
```

4. `isdigit()`：字符串全部是数字，返回 True, 否则返回 False。

```python
print('12345'.isdigit())
print('12345 '.isdigit())
print('12345#'.isdigit())
print('hello12345'.isdigit())
```

5. `isalnum()`：字符串只包含字母或数字，返回 True，否则返回 False。

```python
print('hello12345'.isalnum())
print('hello12345 '.isalnum())
print('hello12345#'.isalnum())
```

6. isspace()：如果字符串中只包含空白，返回 True，否则返回 False。

```python
print('1 2 3 4 5'.isspace())
print('     '.isspace())
```





