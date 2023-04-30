# 字典

字典是一种可变容器模型，且可存储任意类型对象。字典里面的数据是以键值对形式出现。

* 字典中的数据是无顺的。
* 字典中的键可以使字符串，也可以是数字，通常以字符串作为键。

```python
person = {'name': '张三', 'age': 20, 'is_male': True }
numbers = {1: '李四', 2: 20}
box = {}
pack = dict()
```

<img src="https://s1.ax1x.com/2023/02/26/pppgEgH.jpg" style="zoom:45%;" />

## 字典的操作

### 添加

通过索引方式来增加数据项

```python
person['job'] = '码农'
person['id'] = 10001
print(person)
```

### 修改

通过已存在的索引修改数据

```python
person['name'] = '王五'
person['age'] = 28
print(person)
```

> [!warning]
>
> 字典本身为可变数据类型。

### 删除

1. `del` 删除键值对

```python
person = {'name': '张三', 'age': 20, 'is_male': True }
del person['age']
print(person)
```

2. `clear()` 清空字典

```python
person.clear()
```

### 读取

```python
person = {'name': '张三', 'age': 20, 'is_male': True }
print(person['name'])
print(person['id'])
```

`get()` 函数：`dict.get(key, default)`

```python
person = {'name': '张三', 'age': 20, 'is_male': True }
print(person.get('name'))
print(person.get('id', 110))
print(person.get('id'))
```

#### 字典的遍历

1. `keys()` 返回一个所有键组成的可迭代对象。

```python
print(person.keys()) # 可迭代对象类似于数组
for key in person.keys():
    print(person[key])
```

2. `value()` 返回一个所有值组成的可迭代对象。

```python
print(person.values())
for value in person.values():
    print(value)
```

3. `items()` 返回一个键值对组成的可迭代对象，每个键值对是一个元组。

```python
print(person.items())
for key, value in person.items(): # 元组的拆包
  	print(f'{key} = {value}')
```

> [!warning]
>
> 字典的拆包只能获得键，用处不大。
>
> ```python
> age, name, is_male = person
> print(age)
> ```





