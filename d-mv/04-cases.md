# 实践案例

## 神经网络

使用手写数字的MNIST数据集，该数据集包含60000个用于训练的样本和10000个用于测试的样本，图像是固定大小（28$\times$28像素），其值为0到255。

### 加载数据集

```python
from torchvision import datasets

train = datasets.MNIST(root='./data', train=True, download=True)
test = datasets.MNIST(root='./data', train=False, download=True)
```

* `root='./data'`指定下载路径。
* 分别读取训练数据了测试数据`train`和`test`。
