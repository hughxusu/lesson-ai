# Matplotlib

<img src="https://camo.githubusercontent.com/109927a15915074d15313889468aa9aa688de3b9e38cc4359a01f665d351114e/68747470733a2f2f6d6174706c6f746c69622e6f72672f5f7374617469632f6c6f676f322e737667" style="zoom: 45%;" />

- 是专门用于开发2D图表（包括3D图表）
- 以渐进、交互式方式实现数据可视化

## 基本操作

导入模块

```python
import matplotlib.pyplot as plt
```

绘制图像

1. 创建画布

`plt.figure(figsize=(), dpi=) `  figsize: 指定图的长宽；dpi: 图像的清晰度

2. 绘制图像

`plt.plot(x, y) ` x 轴坐标（数组），y 轴坐标（数组）x轴和y轴数据必须一致

3. 显示图像

`plt.show()`

绘制一条直线

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10), dpi=80)
plt.plot([1, 2, 3, 4, 5, 6 ,7], [11,12,13,14,15,16,17])
plt.show()
```

### Matplotlib图像结构

![](https://s1.ax1x.com/2023/05/15/p9ggozq.jpg)

## 绘制单幅图像

> [!tip]
>
> 画出某城市11点到12点1小时内每分钟的温度变化折线图，温度范围在15度~18度

1. 准备数据并画出初始折线图

```python
import matplotlib.pyplot as plt
import random

x = range(60)
y_beijing = [random.uniform(5, 10) for i in x]

plt.figure(figsize=(20, 8), dpi=80)
plt.plot(x, y_beijing)
plt.show()
```

2. 解决中文显示问题

```python
plt.rcParams['font.sans-serif']=['Arial Unicode MS']
plt.rcParams['axes.unicode_minus']=False
```

3. 添加自定义 x，y 刻度

`plt.xticks(x, labels)` x 要显示的刻度值，labels x轴刻度标签

`plt.yticks(y, labels)` x 要显示的刻度值，labels x轴刻度标签

```python
x_ticks_label = ["11点{}分".format(i) for i in x]
y_ticks = range(40)

plt.xticks(x[::5], x_ticks_label[::5])
plt.yticks(y_ticks[::5])
```

4. 添加网格显示

`plt.grid(flag, **kwargs)` flag 是否显示网格，**kwargs 配置网格样式

```python
plt.grid(True, linestyle='--', alpha=0.5)
```

> [!note]
>
> linestyle 通常用于设置曲线样式： `-` 实线，`--` 虚线，`-.` 点划线，`: ` 点虚线

5. 添加描述信息

```python
plt.xlabel("时间")
plt.ylabel("温度")
plt.title("中午11点0分到12点之间的温度变化图示", fontsize=20) # 通过fontsize参数可以修改图像中字体的大小
```

6. 图像保存

```python
plt.savefig("test.png") # 保存图片到指定路径
```

> [!warning]
>
> `plt.show()` 会释放 figure 资源，如果在显示图像之后保存图片将只能保存空图片。

7. 绘制多条曲线

```python
y_shanghai = [random.uniform(18, 20) for i in x]
plt.plot(x, y_shanghai, color='r', linestyle='--')
```

> [!note]
>
> color 通常用于设置线条颜色：r 红色，g 绿色，b 蓝色，w 白色，c 青色，m 洋红，y 黄色，k 黑色

8. 显示图例

```python
plt.plot(x, y_shanghai, label="上海")
plt.plot(x, y_beijing, color='r', linestyle='--', label="北京")

plt.legend(loc="center") # 显示图例
```

`plt.legend(loc)` loc 表示图例显示的位置：best，upper right，upper left，lower left，lower right，right，center left，center right，

lower center，upper center，center

## 绘制多幅图像

在同一画布上绘制多幅图像

`plt.subplots(rows, cols, figsize, dpi)`  nrows, ncols：设置有几行几列坐标系，返回 fig 和 axes

```python
x = range(60)
y_beijing = [random.uniform(5, 10) for i in x]
y_shanghai = [random.uniform(18, 20) for i in x]

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8), dpi=100)

axes[0].plot(x, y_beijing, label="北京")
axes[1].plot(x, y_shanghai, color="r", linestyle="--", label="上海")

x_ticks_label = ["11点{}分".format(i) for i in x]
y_ticks = range(40)

axes[0].set_xticks(x[::5])
axes[0].set_yticks(y_ticks[::5])
axes[0].set_xticklabels(x_ticks_label[::5])
axes[1].set_xticks(x[::5])
axes[1].set_yticks(y_ticks[::5])
axes[1].set_xticklabels(x_ticks_label[::5])

axes[0].grid(True, linestyle="--", alpha=0.5)
axes[1].grid(True, linestyle="--", alpha=0.5)

axes[0].set_xlabel("时间")
axes[0].set_ylabel("温度")
axes[0].set_title("中午11点--12点某城市温度变化图", fontsize=20)
axes[1].set_xlabel("时间")
axes[1].set_ylabel("温度")
axes[1].set_title("中午11点--12点某城市温度变化图", fontsize=20)

axes[0].legend(loc=0)
axes[1].legend(loc=0)

plt.show()
```

> [!warning]
>
> 设置标题等方法不同：`set_xtick` `set_yticks` `set_xticklabels` `set_xlabel` `set_ylabel` `set_title`

优化图例设置

```python
for one in axes:
    one.set_xticks(x[::5])
    one.set_yticks(y_ticks[::5])
   	one.set_xticklabels(x_ticks_label[::5])
    one.grid(True, linestyle="--", alpha=0.5)
    one.set_xlabel("时间")
    one.set_ylabel("温度")
    one.set_title("中午11点--12点某城市温度变化图", fontsize=20)
    one.legend(loc=0)
```

