# Pytorch 入门

[Pytorch](https://pytorch.org/)是由Facebook AI Research (FAIR)开发的开源深度学习框架，是一个基于Numpy的科学计算包，向它的使用者提供了两大功能。与Tensorflow对比，PyTorch在GitHub上的开源项目，数量和社区活跃度方面，略占优势，尤其在研究和学术领域。

* Hugging Face Transformers提供BERT、GPT-2、T5、RoBERTa等预训练模型，支持文本分类、翻译、生成等任务。
* 常用的开源大语言模型基于PyTorch
  * Deepseek官方推荐PyTorch。
  * Facebook开源模型LLaMA基于PyTorch。
* YOLO开源工具V5、V8、V11均基于PyTorch。
* Stable Diffusion基于 扩散模型（Diffusion Models）的文本到图像生成工具。
* FaceFusion高度真实感的换脸工具。

核心特点

* 动态计算图：计算图，在代码运行时动态构建，灵活性高，适合调试和研究。对比 TensorFlow 1.x 的静态图（Define-and-Run），PyTorch 更直观，适合快速实验。
* 张量计算：作为Numpy的替代者，向用户提供使用GPU强大功能的能力。
* 自动微分：自动计算梯度，简化反向传播。
* 模块化神经网络：做为一款深度学习的平台，向用户提供最大的灵活性和速度。
* GPU 加速（CUDA 支持）：只需简单命令即可实现GPU加速。
* 数据加载与预处理：使用`Dataset` 和 `DataLoader`方便数据批处理和多线程加载。

工作流

<img src="../_images/mv/01_a_pytorch_workflow.png" style="zoom:75%;" />

可视化

* 支持Tensorboard可视化分析。

模型仓库

* PyTorch提供了[PyTorch Hub](https://pytorch.org/hub/)官方模型仓库。
* [HuggingFace](https://huggingface.co/)：开源的AI工具库中的预训练模型。

学习资料

* 学习网站
  * [官方教程](https://pytorch.org/tutorials/)
  * [20天吃掉那只Pytorch](https://jackiexiao.github.io/eat_pytorch_in_20_days/)
  * [深入浅出PyTorch](https://datawhalechina.github.io/thorough-pytorch/index.html#)
* 学习书籍
  * [《PyTorch深度学习实战》](https://book.douban.com/subject/35776474/)

Pytorch的安装

* [安装命令](https://pytorch.org/get-started/locally/)

## Pytorch基本语法

### 张量及其操作

张量（Tensors）是一种多为数组，它可以看做是矩阵和向量的推广。

<img src="https://raw.githubusercontent.com/hughxusu/lesson-ai/developing/_images/mv/0ca2fd5a6590d22027e3058b497fdff1.jpeg" style="zoom:55%;" />

在Pytorch中，张量的概念类似于Numpy中的ndarray数据结构，最大的区别在于Tensor可以利用GPU的加速功能。