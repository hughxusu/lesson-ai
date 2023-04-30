# 集合

集合是一个无序的不重复元素序列。

创建集合使用`{}`或`set()`。如果要创建空集合只能使用`set()`，因为`{}`用来创建空字典。

```python
s1 = {'tiktok', 40, 'google', 17.5}
print(s1)

colors = {'red', 'blue', 'yellow', 'purple'}
print(colors)

s3 = set('abcdefg')
print(s3)

s4 = set()
print(type(s4))

s5 = {}
print(type(s5))
```

> [!warning]
>
> 1. 集合可以去掉重复数据。
> 2. 集合数据是无序的，故不支持下标。

## 集合的操作

### 添加

1. `add()` 向集合内追加数据，如果集合中存在该数据，则不进行任何操作。

```python
colors.add('white')
print(colors)
colors.add('red')
print(colors)
```

2. update() 向集合中追加序列。

```python
colors.update(['gray', 'pink'])
colors.update('black')
colors.update(10)
```

### 删除

1. `remove()` 删除集合中的指定数据，如果数据不存在则报错。
2. `discard()` 删除集合中的指定数据，如果数据不存在<mark>不会报错</mark>。

```python
colors = {'red', 'blue', 'yellow', 'purple', 'gray', 'pink'}
colors.remove('gray')
print(colors)
colors.remove('gray')

colors.discard('yellow')
print(colors)
colors.discard('yellow')
```

3. `pop()` 随机删除集合中的某个数据，并返回这个数据。

```python
color = colors.pop()
print(colors)
print(color)
```

4. `clear()` 清空集合。

```python
colors.clear()
```

### 判断

```python
colors = {'red', 'blue', 'yellow', 'purple', 'gray', 'pink'}

print('red' in colors)
print('red' not in colors)
```

### 遍历

```python
for color in colors:
    print(color)
```

