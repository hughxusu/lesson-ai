# Transformer

Transformer是一种深度学习架构，2017年由Google Brain团队发表在论文["Attention Is All You Need"](https://arxiv.org/pdf/1706.03762)中。Transformer架构的出现彻底改变了NLP领域的发展。目前的主流大语言模型都是基于Transformer架构。

基于seq2seq架构的transformer模型可以完成NLP领域研究的典型任务，如：机器翻译、文本生成等。同时又可以构建预训练语言模型，用于不同任务的迁移学习。Transformer有两个显著的优势：

* Transformer能够利用分布式GPU进行并行训练，提升模型训练效率。
* 在分析预测更长的文本时，捕捉间隔较长的语义关联效果更好。

## 认识Transformer架构

<img src="https://raw.githubusercontent.com/hughxusu/lesson-ai/develop/images/nlp/v2-4b53b731a961ee467928619d14a5fd44_r.png" style="zoom:85%;" />

Transformer总体架构包括：

* 输入部分
  * 源文本嵌入层，及其位置编码器
  * 目标文本嵌入层，及其位置编码器
* 输出部分：线性层和softmax层
* 编码器
  * 由N个编码器层堆叠而成
  * 每个编码器层由两个子层连接结构组成
  * 第一个子层包括：一个多头**自注意力**子层和规范化层以及一个残差连接
  * 第二个子层包括：一个前馈全连接子层和规范化层以及一个残差连接
* 解密器
  * 由N个解码器层堆叠而成
  * 每个解码器层由三个子层连接结构组成
  * 第一个子包括：一个多头**自注意力**子层和规范化层以及一个残差连接
  * 第二个子包括：一个多头**注意力**子层和规范化层以及一个残差连接
  * 第三个子包括：一个前馈全连接子层和规范化层以及一个残差连接

## 输入部分

无论是源文本嵌入，还是目标文本嵌入，都是为了将文本中词汇的数字表示转变为向量表示，希望在这样的高维空间捕捉词汇间的关系。

使用[`nn.Embedding`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html#embedding)转化词向量

```python
import torch
from torch import nn

embedding = nn.Embedding(10, 3)
input = torch.LongTensor([[1, 2, 4, 5],[4, 3, 2, 9]])
embedding(input)
```

* `embedding`空间共包含10个词，每个词是3维向量。
* `input`输入数据，包含两个句子，每个句子有4个词，数字表示词在整个词空间的编号。

```python
embedding = nn.Embedding(10, 3, padding_idx=0)
input = torch.LongTensor([[0, 2, 0, 5]])
embedding(input)
```

* `padding_idx=0`如果单词在词空间的编号等于该值，则向量为0。

对`nn.Embedding`进行封装，创建嵌入类

```python
import math

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
    
d_model = 512
vocab = 1000
x = torch.LongTensor([[100,2,421,508],[491,998,1,221]])
emb = Embeddings(d_model, vocab)
embr = emb(x)
print("embr:", embr)
```

* 将词嵌入乘以`math.sqrt(self.d_model)`可以增大词嵌入的幅度，使其与位置编码的尺度大致保持一致，从而确保在两者相加时，它们能够对模型提供均衡的信息。
* 改善注意力机制的稳定性。这种缩放操作是一种数值稳定性和信息平衡的策略。

### 位置编码器

因为在Transformer的编码器结构中，并没有针对词汇位置信息的处理，因此需要在Embedding层后加入位置编码器，将词汇位置不同可能会产生不同语义的信息加入到词嵌入张量中，以弥补位置信息的缺失。

[`torch.unsqueeze`](https://docs.pytorch.org/docs/stable/generated/torch.unsqueeze.html#torch-unsqueeze)返回在指定位置插入一个维度为 1 的新张量。

```python
x = torch.tensor([1, 2, 3, 4])
print(torch.unsqueeze(x, 0))
print(torch.unsqueeze(x, 1))
```

位置编码的计算利用了正弦和余弦函数的周期性，能够让模型轻松学习到相对位置信息。
$$
\begin{array}{lll} 
\text{PE}_{\left(pos,i\right)}=\sin{\left(\frac{pos}{10000^{\frac{2i}{d}}}\right)} \\
\text{PE}_{\left(pos,i+1\right)}=\cos{\left(\frac{pos}{10000^{\frac{2i}{d}}}\right)}
\end{array}
$$ {\begin{array}{lll} \text{PE}_{\left(pos,i\right)}=\sin{\left(\frac{pos}{10000^{\frac{2i}{d}}}\right)} \\\text{PE}_{\left(pos,i+1\right)}=\cos{\left(\frac{pos}{10000^{\frac{2i}{d}}}\right)}\end{array}
其中10000是一个经验值的超参数。位置编码第$i$个维度使用的频率是不同：

* 位置编码是有多个正余弦函数组成的。
* 在位置编码中，维度索引$i$越接近0，对应的频率越高。反之，频率越低。
* 因为维度足够多，所以各个频率组合在一起，形成了一种唯一指纹式的编码。

![](https://raw.githubusercontent.com/hughxusu/lesson-ai/develop/images/nlp/transformer-pos.png)

位置编码的实现

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) 
                             * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

dropout = 0.1
max_len=60

x = embr
pe = PositionalEncoding(d_model, dropout, max_len)
pe_result = pe(x)
print("pe_result:", pe_result)
```

`pe`是一个`max_len`$\times$`d_model`的矩阵，偶数行被$\sin$值填充，奇数行被$\cos$值填充。`div_term`的计算公式
$$
w_k=\exp\left(2k \cdot \left(-\frac{\ln(1000)}{d_{\text{model}}} \right)\right)
=1000^{-\frac{2k}{d_{model}}}
$$
绘制词向量中特征的分布曲线

```python
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(15, 5))
pe = PositionalEncoding(20, 0)
y = pe(torch.zeros(1, 100, 20))
plt.plot(np.arange(100), y[0, :, 4:8].data.numpy(), linewidth=2)
plt.legend(["dim %d"%p for p in [4,5,6,7]], fontsize=15)
plt.show()
```

## 编码器部分

编码器部一般是由N个编码器层堆叠而成。

### 掩码张量

掩码张量（Mask Tensor）主要作用是控制模型在计算注意力时对序列中某些位置的“可见性”或“关注度”。它告诉模型哪些部分应该被“看到”并参与注意力计算，哪些部分应该被“忽略”。

<img src="https://raw.githubusercontent.com/hughxusu/lesson-ai/develop/images/nlp/Xnip2025-05-29_14-39-39.jpg" style="zoom:45%;" />

使用[`np.triu`](https://numpy.org/doc/stable/reference/generated/numpy.triu.html#numpy.triu)来实现掩码功能

```python
mat = [[1,2,3], [4,5,6], [7,8,9], [10,11,12]]

print(np.triu(mat, k=-1))
print(np.triu(mat, k=0))
print(np.triu(mat, k=1))
```

掩码张量的实现

```python
def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(1 - subsequent_mask)

size = 4
sm = subsequent_mask(size)
print("sm:", sm)
```

### 注意力机制

[注意力机制的根本目标](https://zh.d2l.ai/chapter_attention-mechanisms/attention-cues.html)：想办法构造一个权重分布，使得模型在进行预测或生成某个输出时，有选择地赋予输入序列中某些部分更大的权重，而对其他不那么相关的部分赋予较小的权重。注意力的计算规则
$$
\text{Attention}\left(Q, K, V\right)=
\text{softmax}\left(\frac{QK^T}{\sqrt{d_k} }\right)V
$$
注意力机制在网络中表示为

<img src="https://raw.githubusercontent.com/hughxusu/lesson-ai/develop/images/nlp/qkv.jpg" style="zoom:35%;" />

自注意力机制就是$Q,K,V$的值全部来自于同一个序列，自注意力机制确实代替了 RNN 的序列性，并且能够同时关注到当前词前面和后面的所有词。下面的例子中判断“它”指代的内容

```
她把水从水壶倒进杯子里，直到它满了。
她把水从水壶倒进杯子里，直到它空了。
```

* $Q$表示每个词向量的查询，当分析到“它”字时，与查询矩阵$W_Q$相乘进行计算，得到查询$Q$。
* “它”字意思的估计需要结合“水壶”、“杯子”、“满了”或“空了”等上下文进行分析。在RNN中只能连接到上文，不能连接下文，而在自注意力机制中可以，结合下文词汇“满了”或“空了”进行分析。句子中全部词向量与$W_K$相乘得到索引$K$。
* $K$是句子中词汇”她“、”把“、“水壶”……等内容和当前查询（Query）的相关度（或理解为“摘要”或“元数据”）。
* $Q$与$K$计算得到一个权重值。
* 矩阵中的所有词汇向量都与$W_V$相乘到内容本身$V$。
* 将权重和内容本身相乘，得到的就是分析的结果。
* 在例子中”她“、”把“、”直到“等词汇对”它“意思的分析显然权重较小。

$W_Q,W_K,W_V$三个矩阵在训练中学习到将高维词向量压缩到更小的“注意力子空间”的能力：

1. 投影去冗余：它们将原始高维词向量投影到维度更小的空间，保留对当前任务（如下一词预测）最关键的语义/句法特征，去除冗余信息。
2. 训练目标驱动： 通过海量文本和预测任务（如：完形填空或下一句预测），模型根据预测误差不断调整这三个矩阵的权重。
3. $W_Q,W_K,W_V$的梯度：某个token的表示在当前注意力头里没法很好地预测上下文，它就会被微调，以降低损失。
4. 压缩语言规律：经过成百上千个epoch后，这些矩阵最终将语料中的语言规律（如句法模式、语义关联、指代关系）高效地“压缩”到其固定的权重模式里。

注意力机制的实现，其中使用了`tesnor`的[`masked_fill`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.masked_fill.html#torch.Tensor.masked_fill)函数

```python
import torch.nn.functional as F

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


query = key = value = pe_result
mask = torch.zeros(2, 4, 4)
attn, p_attn = attention(query, key, value, mask=mask)
print("attn:", attn)
print("p_attn:", p_attn)
```

#### 注意力的计算

$Q$与$K$计算得到一个权重值，有不同的计算方式

1. 加性注意力（Additive Attention），也常被称为[Bahdanau Attention](https://arxiv.org/pdf/1409.0473)

$$
\text{Attention}\left(Q,K,V\right)
=\text{softmax}\left(\text{Linear}\left(\left[Q,K\right]\right)\right)\cdot V
$$

2. 加性注意力的另一种形式

$$
\text{Attention}\left(Q,K,V\right)
=\text{softmax}\left(\text{sum}\left(\tanh\left(\text{Linear}\left(\left[Q,K\right]\right)\right)\right)\right)
$$

### 多头注意力机制

多头注意力机制就是通过并行地执行多个注意力计算，每个计算关注不同的信息方面，然后将这些独立的信息整合起来，以获得对输入序列更全面、更丰富的理解。
$$
\begin{array}{lll} 
\text{MultiHead}\left(Q,K,V\right)=\text{Concat}\left( 
\text{head}_1, \cdots ,\text{head}_h
\right)W^O \\
\text{where} \quad \text{Attention}\left(QW_i^Q, KW_i^K, VW_i^V\right) 
\\ 
\text{where} \quad
W_i^Q\in R^{d_{\text{model}}\times d_k},
W_i^K\in R^{d_{\text{model}}\times d_k},
W_i^V\in R^{d_{\text{model}}\times d_v},
W_i^O\in R^{d_{\text{hd}_v}\times d_{model}}
\end{array}
$$
结构图如下

<img src="https://raw.githubusercontent.com/hughxusu/lesson-ai/develop/images/nlp/13.jpg" style="zoom:70%;" />

假设输入的词嵌入向量是1000维，包含4个头：

1. 首先通过线性变换将1000维的向量转换为4个250的向量。
2. 通过4个自注意力机制的处理，得到4个250的输出向量。
3. 将4个250的输出向量拼接得到新的1000维向量。

通过使用多个头，每个头可以学习并专注于捕获不同类型的关系或从不同的表示子空间中提取信息。例如：

* 一个头可能关注句法关系（发现“它”与“满了”的联系更直接，从而判断“它”是“满”的主语）
* 另一个头可能关注语义关系（发现“它”与“倒进”和“满了”的联系）
* 再一个头可能关注指代关系（发现“它”与“杯子”和“水壶”的联系）

这种多样性使得模型能够更全面地理解文本的复杂性。

多头注意力机制的实现

```python
import copy

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MultiHeadedAttention(nn.Module):
    def __init__(self, head, embedding_dim, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert embedding_dim % head == 0
        self.d_k = embedding_dim // head
        self.head = head
        self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(0)
        batch_size = query.size(0)
        query, key, value = \
           [model(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
            for model, x in zip(self.linears, (query, key, value))]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)
        return self.linears[-1](x)
```

调用多头注意力机制计算

```python
head = 8
embedding_dim = 512
dropout = 0.2

query = value = key = pe_result
mask = torch.zeros(8, 4, 4)

mha = MultiHeadedAttention(head, embedding_dim, dropout)
mha_result = mha(query, key, value, mask)
print(mha_result)
```

###  前馈全连接层

具有两层线性层的全连接网络，通过增加两层网络来增强模型的能力，使用Rule激活函数。 前馈全连接层实现为

```python
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(F.relu(self.w1(x))))
    
    
d_model = 512
d_ff = 64
dropout = 0.2
x = mha_result
ff = PositionwiseFeedForward(d_model, d_ff, dropout)
ff_result = ff(x)
print(ff_result)
```

* 原论文中的前馈全连接层，输入和输出的维度`d_model=512`，层内的连接维度`d_ff = 2048`

### 规范化层

随着网络层数的增加，通过多层的计算后参数可能开始出现过大或过小的情况，模型可能收敛非常的慢。通过数值的规范化，使其特征数值在合理范围内，从而增加速度，这就是规范化层。规范化层的实现

```python
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a2 = nn.Parameter(torch.ones(features))
        self.b2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a2 * (x - mean) / (std + self.eps) + self.b2
    
features = d_model = 512
eps = 1e-6
ln = LayerNorm(features, eps)
ln_result = ln(x)
print(ln_result)
```

### 子层连接结构

将注意力层或前馈全连接层，与规范化层结合起来，并加上残差结构的网络。代码实现如下

```python
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
size = 512
dropout = 0.2
head = 8
d_model = 512
x = pe_result
mask = torch.zeros(8, 4, 4)
self_attn =  MultiHeadedAttention(head, d_model)
sublayer = lambda x: self_attn(x, x, x, mask)

sc = SublayerConnection(size, dropout)
sc_result = sc(x, sublayer)
print(sc_result)
print(sc_result.shape)
```

### 编码器层

组合了多头注意力机制和前馈全连接层。代码实现如下

```python
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()

        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
    
size = 512
head = 8
d_model = 512
d_ff = 64
x = pe_result
dropout = 0.2
self_attn = MultiHeadedAttention(head, d_model)
ff = PositionwiseFeedForward(d_model, d_ff, dropout)
mask = torch.zeros(8, 4, 4)
el = EncoderLayer(size, self_attn, ff, dropout)
el_result = el(x, mask)

print(el_result)
print(el_result.shape)
```

### 编码器

一个编码器由N个编码器层堆叠而成。代码实现如下

```python
class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
size = 512
head = 8
d_model = 512
d_ff = 64
c = copy.deepcopy
attn = MultiHeadedAttention(head, d_model)
ff = PositionwiseFeedForward(d_model, d_ff, dropout)
dropout = 0.2
layer = EncoderLayer(size, c(attn), c(ff), dropout)
N = 8
mask = torch.zeros(8, 4, 4)
en = Encoder(layer, N)
en_result = en(x, mask)

print(en_result)
print(en_result.shape)
```

* 原论文的编码器模块，包含6个编码器。

## 解码器部分

### 解码器层

在解码器层中，交叉注意力机制中

* Q（Query）：来自于解码器在前一个时间步的输出。
* K（Key）和V（Value）：来自于编码器的输出。

其他的组件与编码器一致。代码实现如下

```python
class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, source_mask, target_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, target_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, source_mask))
        return self.sublayer[2](x, self.feed_forward)
    
head = 8
size = 512
d_model = 512
d_ff = 64
dropout = 0.2
self_attn = src_attn = MultiHeadedAttention(head, d_model, dropout)
ff = PositionwiseFeedForward(d_model, d_ff, dropout)
x = pe_result
memory = en_result
mask = torch.zeros(8, 4, 4)
source_mask = target_mask = mask
dl = DecoderLayer(size, self_attn, src_attn, ff, dropout)
dl_result = dl(x, memory, source_mask, target_mask)

print(dl_result)
print(dl_result.shape)
```

### 解码器

一个解码器由N个解码器层堆叠而成。代码实现如下

```python
class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, source_mask, target_mask): 
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
        return self.norm(x)
    
size = 512
d_model = 512
head = 8
d_ff = 64
dropout = 0.2
c = copy.deepcopy
attn = MultiHeadedAttention(head, d_model)
ff = PositionwiseFeedForward(d_model, d_ff, dropout)
layer = DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout)
N = 8
x = pe_result
memory = en_result
mask = torch.zeros(8, 4, 4)
source_mask = target_mask = mask
de = Decoder(layer, N)
de_result = de(x, memory, source_mask, target_mask)

print(de_result)
print(de_result.shape)
```

* 原论文的解码器模块，包含6个解码器。

解密器层中：

* Output Embedding输出的是预测的词向量。
  * 训练时输入：`["<start>", "四海", "之内", "皆", "兄弟"]`，训练数据加开始符。
  * 预测时输入：上一个时间步累积到当前的结果。
* Output Probability模型预测的下一个词的概率分布，用于损失函数的计算。

## 输出部分

由线性层和softmax层组成。代码实现如下

```
class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.project = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return F.log_softmax(self.project(x), dim=-1)
    
d_model = 512
vocab_size = 1000
x = de_result
gen = Generator(d_model, vocab_size)
gen_result = gen(x)

print(gen_result)
print(gen_result.shape)
```

## 模型构建

### 编解码器的实现

```python
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, source_embed, target_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = source_embed
        self.tgt_embed = target_embed
        self.generator = generator

    def forward(self, source, target, source_mask, target_mask):
        return self.decode(self.encode(source, source_mask), 
                           source_mask, target, target_mask)

    def encode(self, source, source_mask):
        return self.encoder(self.src_embed(source), 
                            source_mask)

    def decode(self, memory, source_mask, target, target_mask):
        return self.decoder(self.tgt_embed(target), 
                            memory, source_mask, target_mask)
    
vocab_size = 1000
d_model = 512
encoder = en
decoder = de
source_embed = nn.Embedding(vocab_size, d_model)
target_embed = nn.Embedding(vocab_size, d_model)
generator = gen
source = target = torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]])
source_mask = target_mask = torch.zeros(8, 4, 4)
ed = EncoderDecoder(encoder, decoder, source_embed, target_embed, generator)
ed_result = ed(source, target, source_mask, target_mask)
print(ed_result)
print(ed_result.shape)
```

### Tansformer构建

```python
def make_model(source_vocab, target_vocab, 
               N=6, d_model=512, d_ff=2048, head=8, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadedAttention(head, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), 
                N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), 
                N),
        nn.Sequential(Embeddings(d_model, source_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, target_vocab), c(position)),
        Generator(d_model, target_vocab))

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

source_vocab = 11
target_vocab = 11 
N = 6
res = make_model(source_vocab, target_vocab, N)
print(res)
```

