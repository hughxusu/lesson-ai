# Numpy

<img src="https://numanal.com/wp-content/uploads/2021/06/NumPy_logo-768x346.png" style="zoom: 67%;" />

Numpy（Numerical Python）是一个开源的Python科学计算库，用于快速处理任意维度的数组。Numpy 支持常见的数组和矩阵操作。对于同样的数值计算任务，使用Numpy比直接使用Python要简洁的多。Numpy 使用ndarray对象来处理多维数组，该对象是一个快速而灵活的大数据容器。

## 基本概念

定义一个 Numpy 数组

```python
import numpy as np

# 创建ndarray
score = np.array(
[[80, 89, 86, 67, 79],
[78, 97, 89, 67, 81],
[90, 94, 78, 67, 74],
[91, 91, 90, 67, 69],
[76, 87, 75, 67, 86],
[70, 79, 84, 67, 84],
[94, 92, 93, 67, 64],
[86, 85, 83, 67, 80]])

score
```

### 测试 Numpy 的速度

```python
import random
import time
import numpy as np
a = []
for i in range(100000000):
    a.append(random.random())

%time sum1=sum(a)

b=np.array(a)
%time sum2=np.sum(b)
```

机器学习的最大特点就是大量的数据运算

Numpy专门针对ndarray的操作和运算进行了设计，所以数组的存储效率和输入输出性能远优于Python中的嵌套列表，数组越大，Numpy的优势就越明显。

![](http://jakevdp.github.io/images/array_vs_list.png)

numpy内置了并行运算功能，当系统有多个核心时，做某种计算时，numpy会自动做并行计算。

Numpy底层使用C语言编写，内部解除了GIL（全局解释器锁），其对数组的操作速度不受Python解释器的限制，所以，其效率远高于纯Python代码。

### ndarray的属性

数组属性反映了数组本身固有的信息。

|     属性名字     |          属性解释          |
| :--------------: | :------------------------: |
|  ndarray.shape   |       数组维度的元组       |
|   ndarray.ndim   |          数组维数          |
|   ndarray.size   |      数组中的元素数量      |
| ndarray.itemsize | 一个数组元素的长度（字节） |
|  ndarray.dtype   |       数组元素的类型       |

```python
a = np.array([[1,2,3],[4,5,6]])
a.shape
b = np.array([1,2,3,4])
b.shape
c = np.array([[[1,2,3],[4,5,6]],[[1,2,3],[4,5,6]]])
c.shape
```

### ndarray的类型

|     名称      |                       描述                        | 简写  |
| :-----------: | :-----------------------------------------------: | :---: |
|    np.bool    |      用一个字节存储的布尔类型（True或False）      |  'b'  |
|    np.int8    |             一个字节大小，-128 至 127             |  'i'  |
|   np.int16    |               整数，-32768 至 32767               | 'i2'  |
|   np.int32    |              整数，-2^31 至 2^32 -1               | 'i4'  |
|   np.int64    |              整数，-2^63 至 2^63 - 1              | 'i8'  |
|   np.uint8    |               无符号整数，0 至 255                |  'u'  |
|   np.uint16   |              无符号整数，0 至 65535               | 'u2'  |
|   np.uint32   |             无符号整数，0 至 2^32 - 1             | 'u4'  |
|   np.uint64   |             无符号整数，0 至 2^64 - 1             | 'u8'  |
|  np.float16   | 半精度浮点数：16位，正负号1位，指数5位，精度10位  | 'f2'  |
|  np.float32   | 单精度浮点数：32位，正负号1位，指数8位，精度23位  | 'f4'  |
|  np.float64   | 双精度浮点数：64位，正负号1位，指数11位，精度52位 | 'f8'  |
| np.complex64  |     复数，分别用两个32位浮点数表示实部和虚部      | 'c8'  |
| np.complex128 |     复数，分别用两个64位浮点数表示实部和虚部      | 'c16' |
|  np.object_   |                    python对象                     |  'O'  |
|  np.string_   |                      字符串                       |  'S'  |
|  np.unicode_  |                    unicode类型                    |  'U'  |

创建数组的时候指定类型

```python
a = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
a.dtype

arr = np.array(['python', 'tensorflow', 'scikit-learn', 'numpy'], dtype = np.string_)
arr
```

## 基本操作

### 生成数组

#### 生成0和1数组

- `np.ones(shape, dtype)` shape 形状，dtype类型
- `np.ones_like(a, dtype)` 根据已有形状构造数据，a 已有矩阵 dtype类型
- `np.zeros(shape, dtype)`
- `np.zeros_like(a, dtype)`

```python
ones = np.ones([4,8])
ones
np.zeros_like(ones)
```

#### 从现有数组生成

- `np.array(a, dtype)` object 已有数据，dtype类型
- `np.asarray(a, dtype)` 

```python
a = np.array([[1,2,3],[4,5,6]])


a1 = np.array(a) # 深拷贝
a2 = np.asarray(a) # 浅拷贝
```

#### 生成固定范围的数组

`np.linspace (start, stop, num, endpoint)`创建等差数组，`start` 序列的起始值，`stop` 序列的终止值，num 要生成的等间隔样例数量（默认为50），endpoint 序列中是否包含stop值，默认为ture。

```python
np.linspace(0, 100, 11)
```

`np.arange(start, stop, step, dtype)` 创建等差数组，step 步长（默认值为1）

```python
np.arange(10, 50, 2)
```

`np.logspace(start, stop, num)`  创建等比数列，num 要生成的等比数列数量，默认为50

```python
np.logspace(0, 2, 3) # 生成10^x
```