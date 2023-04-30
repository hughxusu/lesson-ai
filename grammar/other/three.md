# 生成器与拷贝

## 生成器

根据规则循环生成数据，当条件不成立时则生成数据结束。数据不是一次性全部生成处理，而是使用一个个生成，可以节约大量的内存。

### 创建生成器的方式

1. 生成器推导式
2. `yield` 关键字

#### 生成器推导式

与列表推导式类似，只不过生成器推导式使用小括号。

```python
my_generator = (i * 2 for i in range(5))
print(my_generator)

next获取生成器下一个值
value = next(my_generator)
print(value)

for value in my_generator:
    print(value)
```

1. `next` 函数获取生成器中的下一个值。
2. `for` 循环遍历生成器中的每一个。

#### `yield` 关键字

只要在 `def` 函数里面看到有 `yield` 关键字那么就是生成器

```python
def mygenerater(n):
  for i in range(n):
    print('开始生成...')
    yield i
    print('完成一次...')
        
        
if __name__ == '__main__':
  g = mygenerater(3)
  
  value = next(result)
	print(value)
  
  while True:
    try:
      result = next(g)
      print(result)
    except StopIteration as e:
      break
  
 	for value in result:
    print(value)
```

1. 代码执行到 `yield` 会暂停，然后把结果返回出去，下次启动生成器会在暂停的位置继续往下执行。
2. 生成器如果把数据生成完成，再次获取生成器中的下一个数据会抛出一个 `StopIteration` 异常，表示停止迭代异常。
3. `while` 循环内部没有处理异常操作，需要手动添加处理异常操作。
4. `for` 循环内部自动处理了停止迭代异常。

### 生成器的使用场景

> [!tip]
>
> 生成斐波拉契数列（Fibonacci）

```python
def fibonacci(num):
    a = 0
    b = 1
    
    current_index = 0
    while current_index < num:
        result = a
        a, b = b, a + b
        current_index += 1
        yield result


fib = fibonacci(5)
for value in fib:
    print(value)
```

使用生成器产生斐波拉契数列，每次调用只生成一个数据，可以节省大量的内存。

## 拷贝

使用 `copy` 包来完成拷贝

### 浅拷贝

`copy.copy` 浅拷贝函数：只对可变类型的第一层对象进行拷贝，对拷贝的对象开辟新的内存空间进行存储，不会拷贝对象内部的子对象。

```python
import copy

num1 = 1
num2 = copy.copy(num1)
print("num1:", id(num1), "num2:", id(num2))

my_tuple1 = (3, 5)
my_tuple2 = copy.copy(my_tuple1)
print("my_tuple1:", id(my_tuple1), "my_tuple2:", id(my_tuple2))


my_list1 = [1, 3, [4, 6]]
my_list2 = copy.copy(my_list1)
print("my_list1:", id(my_list1), "my_list2:", id(my_list2))

my_list1.append(5)
print(my_list1, my_list2)
print("my_list1[2]:", id(my_list1[2]), "my_list2[2]:", id(my_list2[2]))

my_list1[2].append(3)
print(my_list1, my_list2)
```

1. 不可变类型进行浅拷贝不会给拷贝的对象开辟新的内存空间，而只是拷贝了这个对象的引用。
2. 可变类型进行浅拷贝只对可变类型的第一层对象进行拷贝，对拷贝的对象会开辟新的内存空间进行存储，子对象不进行拷贝。

### 深拷贝

`copy.deepcopy` 深拷贝函数：只要发现对象有可变类型就会对该对象到最后一个可变类型的每一层对象就行拷贝, 对每一层拷贝的对象都会开辟新的内存空间进行存储。

```python
import copy

num1 = 1
num2 = copy.deepcopy(num1)

print("num1:", id(num1), "num2:", id(num2))

str1 = 'hello'
str2 = copy.deepcopy(str1)

print("str1:", id(str1), "str2:", id(str2))

my_tuple1 = (1, [1, 2])
my_tuple2 = copy.deepcopy(my_tuple1)

print("my_tuple1:", id(my_tuple1), "my_tuple2:", id(my_tuple2))
print("my_tuple1[1]:", id(my_tuple1[1]), "my_tuple2[1]:", id(my_tuple2[1]))

my_tuple2[1].append(4)

print(my_tuple1, my_tuple2)

print("my_tuple1[0]:", id(my_tuple1[0]), "my_tuple2[0]:", id(my_tuple2[0]))

my_list1 = [1, [2, 3]]
my_list2 = copy.deepcopy(my_list1)
print("my_list1:", id(my_list1), "my_list2:", id(my_list2))

print("my_list1[1]:", id(my_list1[1]), "my_list2[1]:", id(my_list2[1]))
```

1. 不可变类型进行深拷贝如果子对象没有可变类型则不会进行拷贝，而只是拷贝了这个对象的引用。
2. 不可变类型进行深拷贝如果子对象有可变类型，会对该对象到最后一个可变类型的每一层对象就行拷贝, 对每一层拷贝的对象都会开辟新的内存空间进行存储。
3. 可变类型进行深拷贝会对该对象到最后一个可变类型的每一层对象就行拷贝，对每一层拷贝的对象都会开辟新的内存空间进行存储。

> [!warning]
>
> 实际应用中拷贝函数多使用深拷贝，但是深拷贝会耗费大量内存。
