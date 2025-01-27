# 人工智能与计算机视觉

## CIFAR-10数据集

CIFAR-10/100是从8000万个微小图像中提取的分类任务。

CIFAR-10数据集包含60000张32x32像素的彩色图像，共分为10个类。以下代码展示了如何加载数据、预处理数据、定义模型、编译模型以及训练模型。

```python
  from keras.datasets import cifar10

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
```

![](https://raw.githubusercontent.com/hughxusu/lesson-ai/developing/_images/cnn/4fdf2b82-2bc3-4f97-ba51-400322b228b1.png) +  
