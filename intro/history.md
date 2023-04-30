# 认识Python

> 人生苦短，我用 Python —— Life is short, you need Python



## Python 的起源

>  Python 的创始人为吉多·范罗苏姆（Guido van Rossum）

<img src="https://gvanrossum.github.io/images/guido-headshot-2019.jpg" style="zoom:40%;" />

1991年，在荷兰阿姆斯特丹，开发出一个新的解释程序Python。第一个 Python 解释器用 C 语言实现的，并能够调用 C 语言的库文件。

Python（蟒蛇）

<img src="https://gimg2.baidu.com/image_search/src=http%3A%2F%2Fpica.zhimg.com%2Fv2-8f51c35f5e8e48d3c8616a4353178689_1440w.jpg%3Fsource%3D172ae18b&refer=http%3A%2F%2Fpica.zhimg.com&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=auto?sec=1669026945&t=c6e16904aaa196132b953c73e4db0ade" style="zoom: 15%;" />

### 编译语言与解释语言

计算机不能直接理解任何除机器语言以外的语言，必须要把程序员所写的程序语言翻译成机器语言，计算机才能执行。将其他语言翻译成机器语言的工具，被称为编译器。

编译器翻译的方式有两种：一个是编译，另外一个是解释。

![](https://pic2.zhimg.com/v2-09614038877b06dd2cfa17d55dbf6652_1440w.jpg?source=172ae18b)

* 编译型语言：程序在执行之前经过编译，转换为为机器语言的文件，运行时直接使用编译的结果。程序执行效率高，依赖编译器，跨平台性差些。如 C、C++
* 解释型语言：解释型语言编写的程序不进行预先编译，以文本方式存储程序代码，会将代码一句一句直接运行。在发布程序时，需要依赖于解释器环境。

编译型语言和解释型语言对比

* 编译型语言执行速度快更快。
* 解释型语言开发效率更高跨平台性好。

**Python 是一门解释性语言**，Python 的解释器包含多个语言的实现：

* `CPython` —— 官方版本的 C 语言实现
* `Jython` —— 可以运行在 Java 平台
* `IronPython` —— 可以运行在 .NET 和 Mono 平台
* `PyPy` —— Python 实现的，支持 JIT 即时编译

### Python 的设计目标

吉多为 Python 语言制定的目标是：

* 一门简单直观的语言并与主要竞争者一样强大
* 开源，以便任何人都可以为它做贡献
* 代码像纯英语那样容易理解
* 适用于短期开发的日常任务

### Python 的设计哲学

>  优雅、明确、简单

**Python开发者的哲学是**：用一种方法，最好是只有一种方法来做一件事；如果面临多种选择，选择明确没有或者很少有歧义的语法。

## Python 特点

**优点**

1. 代码量少

2. Python 是完全面向对象的语言
   * 函数、模块、数字、字符串都是对象，在 Python 中一切皆对象。

   * 完全支持继承、重载、多重继承

   * 支持重载运算符，也支持泛型设计

3. Python 拥有强大的标准库，标准库提供了系统管理、网络通信、文本处理、数据库接口、图形系统、XML 处理 等额外的功能。

4. Python 社区提供了大量的第三方模块，覆盖：科学计算**、**人工智能**、**机器学习**、**Web 开发**、**数据库接口**、**图形系统多个领域：

   * Numpy 支持大量的维度数组与矩阵运算，此外也针对数组运算提供大量的数学函数库

   * Matplotlib 2D-绘图工具

   * Scikit-Learn 机器学习工具

   * Pandas 一个强大的分析结构化数据的工具集

   * 其它……

5. 可扩展性（胶水语言）Python可以调用其它语言的函数如：C/C++

**缺点**

* 运行速度慢

## 为什么选择 Python

1. 简单易学、免费、开源、无惧限制。
2. [TIOBE Index](https://www.tiobe.com/tiobe-index/) 计算机语言流行度标准，排名靠前。
3. 深度学习、数据分析等工作利器。
4. 对于数学专业可以完全代替 Matlab 工具。

