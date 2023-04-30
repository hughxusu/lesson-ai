

# 程序的执行原理

## 计算机中的三大件

计算机中包含有较多的硬件，但是一个程序要运行，有三个核心的硬件，分别是：

1. CPU
   * 中央处理器，是一块超大规模的集成电路
   * 负责处理数据和计算
2. 内存
   * 临时存储数据（断电之后，数据会消失）
   * 速度快
   * 空间小（单位价格高）
3. 硬盘
   * 永久存储数据
   * 速度慢
   * 空间大（单位价格低）

|                             CPU                              |                             内存                             |                             硬盘                             |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![](https://gimg2.baidu.com/image_search/src=http%3A%2F%2Fimg.jbzj.com%2Ffile_images%2Farticle%2F201706%2F201706171617215.jpg&refer=http%3A%2F%2Fimg.jbzj.com&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=auto?sec=1669305010&t=17081edf171f479617ee2a77c8f62275) | <img src="https://gimg2.baidu.com/image_search/src=http%3A%2F%2Fwww.jicong.net%2Fwp-content%2Fuploads%2F2021%2F03%2F20210315230121-604fe74153474.jpg&refer=http%3A%2F%2Fwww.jicong.net&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=auto?sec=1669305077&t=08bf9db47b41d2a20daac2409e9803e1" alt="内存条-w200" style="zoom:80%;" /> | <img src="https://gimg2.baidu.com/image_search/src=http%3A%2F%2Fnews.mydrivers.com%2FImg%2F20110130%2F01084245.jpg&refer=http%3A%2F%2Fnews.mydrivers.com&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=auto?sec=1669305485&t=ae0e9ce2467ac1b80c318d21e755dcf7" alt="硬盘-w200" style="zoom:25%;" /> |

## 程序执行的原理

![](https://s1.ax1x.com/2023/02/26/pppcHEV.jpg)

1. 程序运行之前，程序是保存在硬盘中的。
2. 当要运行一个程序时：
   * 操作系统会首先让 CPU 把程序复制到内存中。
   * CPU 执行内存中的程序代码。

## 程序在内存中的运行过程

![](https://s1.ax1x.com/2023/02/26/pppcq4U.jpg)

## Python 程序执行原理

![](https://s1.ax1x.com/2023/02/26/pppcX34.jpg)

1. 操作系统会首先让 CPU 把 Python 解释器的程序复制到内存中。
2. Python 解释器根据语法规则，从上向下让 CPU 翻译Python 程序中的代码。
3. CPU 负责执行翻译完成的代码。

### Python 解释器的大小

```shell
ls -lh /Users/xusu/opt/anaconda3/bin/python3.9
```

## 程序的调试

在 PyCharm 中调试如下代码

```python
price = 8.5
number = 10
money = price * number # 计算金额
money *= 0.8

print(money)
```

<img src="https://s1.ax1x.com/2023/02/26/pppgxJS.jpg" style="zoom:50%;" />