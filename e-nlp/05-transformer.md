# Transformer

Transformer是一种深度学习架构，2017年由Google Brain团队发表在论文["Attention Is All You Need"](https://arxiv.org/pdf/1706.03762)中。Transformer架构的出现彻底改变了NLP领域的发展。目前的主流大语言模型都是基于Transformer架构。

基于seq2seq架构的transformer模型可以完成NLP领域研究的典型任务，如：机器翻译、文本生成等。同时又可以构建预训练语言模型，用于不同任务的迁移学习。Transformer有两个显著的优势：

* Transformer能够利用分布式GPU进行并行训练，提升模型训练效率。
* 在分析预测更长的文本时，捕捉间隔较长的语义关联效果更好。

## 认识Transformer架构

下面以文本翻译为例来解释Transformer架构

<img src="https://raw.githubusercontent.com/hughxusu/lesson-ai/developing/_images/nlp/v2-4b53b731a961ee467928619d14a5fd44_r.png" style="zoom:85%;" />

Transformer总体架构包括：

* 输入部分
  * 源文本嵌入层，及其位置编码器
  * 目标文本嵌入层，及其位置编码器
* 输出部分：线性层和softmax层
* 编码器
  * 由N个编码器层堆叠而成
  * 每个编码器层由两个子层连接结构组成
  * 第一个子层包括：一个多头自注意力子层和规范化层以及一个残差连接
  * 第二个子层包括：一个前馈全连接子层和规范化层以及一个残差连接
* 解密器
  * 由N个解码器层堆叠而成
  * 每个解码器层由三个子层连接结构组成
  * 第一个子包括：一个多头**自注意力**子层和规范化层以及一个残差连接
  * 第二个子包括：一个多头注意力子层和规范化层以及一个残差连接
  * 第三个子包括：一个前馈全连接子层和规范化层以及一个残差连接

