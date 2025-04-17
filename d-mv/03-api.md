# PyTorch常用API

## 自动求导（autograd）

在整个Pytorch框架中，所有的神经网络，本质上都是一个自动求导工具包（autograd package），它提供了一个，对Tensors上所有的操作，进行自动微分的功能。

### 关于`torch.tensor`

`torch.tensor`是整个package中的核心类：

*  将属性`requires_grad`设置为`True`，将追踪在这个类上定义的所有操作。
  * 当代码进行反向传播时，直接调用`backward()`，可以自动计算梯度。
  * `tensor`上的所有梯度，被累加进属性`grad`中。
* 如果终止一个`tensor`在计算图中的回溯，只需要执行`detach()`，就可以将该`tensor`从计算图中撤下。
* 如果想终止对整个计算图的回溯，也就是不再进行反向传播， 可以采用代码块的方式`with torch.no_grad():`，一般适用于模型推理阶段。

### 关于`torch.function`

`torch.function`是和`tensor`同等重要的一个核心类：

* 每一个`tensor`拥有一个`grad_fn`属性，代表引用了哪个函数创建了该`tensor`。
* 用户自定义的`tensor`是，`grad_fn`属性是`None`。

### 自动求导属性

```python
x1 = torch.ones(3, 3)
print(x1)

x = torch.ones(2, 2, requires_grad=True)
print(x)
```

对`requires_grad=True`的`tensor`执行加法操作

```python
y = x + 2
print(y)
```

打印`tensor`的`grad_fn`属性

```python
print(x.grad_fn)
print(y.grad_fn)
```

在`tensor`上执行更复杂的操作

```python
z = y * y * 3
out = z.mean()
print(z, out)
```

关于方法`requires_grad_()`方法，可以原地改变`tensor.requires_grad`的属性值，默认值为`False`

```python
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)
```

在PyTorch中，反向传播是依靠`backward()`实现

```python
out.backward()
print(x.grad)
```

关于自动求导的属性，可以通过`requires_grad=True`设置，也可以通过代码块来停止自动求导

```python
print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)
```

可以通过`detach()`获得一个新的`tensor`，拥有相同内容，不需要自动求导

```python
print(x.requires_grad)
y = x.detach()
print(y.requires_grad)
print(x.eq(y).all())
```

