# 简介

计算机视觉是指用摄像机和电脑及其他相关设备，对生物视觉的一种模拟，它的主要任务让计算机理解图片或者视频中的内容。

![](https://raw.githubusercontent.com/hughxusu/lesson-ai/develop/images/cv/2762311-20220321161940607-2135747316.png)

其任务目标可以拆分为：

* 让计算机理解图片中的场景（办公室，教室等）
* 让计算机识别场景中包含的物体（交通工具，人等）
* 让计算机定位物体在图像中的位置（物体的大小，边界等）
* 让计算机理解物体之间的关系或行为，以及图像表达的意义。

机器学习工作流

* 确定任务目标，确定方法和模型。
* 使用OpenCV对图像的处理，比如平滑，缩放等。
* 收集训练和测试数据。
* 训练和调优模型。
* 上线模型完成任务。

## 常见任务

计算机视觉目标任务的分解，可将其分为三大经典任务：图像分类、目标检测、图像分割。

<img src="https://raw.githubusercontent.com/hughxusu/lesson-ai/develop/images/cv/image-20201013161400022.png" style="zoom:50%;" />

- 图像分类（Classification）：将图像结构化为某一类别，用事先确定好的类别（category）来描述图片。
- 目标检测（Detection）：关注特定的物体目标，要求同时获得这一目标的类别信息和位置信息。
- 图像分割（Segmentation）：分割是对图像的像素级描述，需要找打目标物体在图像中的边界。

## 应用场景

1. 人脸识别：人脸识别技术广泛应用于金融、司法、军队、公安、边检、政府、航天、电力、工厂、教育、医疗等行业。
2. 视频监控：在大量人群流动的交通枢纽，用于人群分析、防控预警等。
3. 图片识别分析：主要应用是以图搜图，物体分析。
4. 辅助驾驶：在辅助驾驶领域机器视觉是重要的手段。

## 发展历史

* 1963年，Larry Roberts发表了CV领域的第一篇专业论文，用以对简单几何体进行边缘提取和三维重建。

<img src="https://raw.githubusercontent.com/hughxusu/lesson-ai/develop/images/cv/image-20201013164341893.png" style="zoom:90%;" />

* 1982年，学者David Marr发表的著作《Vision》从严谨又长远的角度给出了CV的发展方向和一些基本算法，标志着计算机视觉成为了一门独立学科。

* 1999年David Lowe提出了尺度不变特征变换（SIFT, Scale-invariant feature transform）目标检测算法，用于匹配不同拍摄方向、纵深、光线等图片中的相同元素。

  <img src="https://raw.githubusercontent.com/hughxusu/lesson-ai/develop/images/cv/image-20201013164647298.png" style="zoom:65%;" />

* 2009年，由Felzenszwalb教授在提出基于HOG的deformable parts model，可变形零件模型开发，它是深度学习之前最好的最成功的物体识别算法。
* Everingham等人在2006年至2012年间搭建了一个大型图片数据库，供机器识别和训练，称为PASCAL Visual Object Challenge，该数据库中有20种类别的图片，每种图片数量在一千至一万张不等。
* 2009年，李飞飞教授等在CVPR2009上发表了一篇名为《ImageNet: A Large-Scale Hierarchical Image Database》的论文，发布了ImageNet数据集，这是为了检测计算机视觉能否识别自然万物，回归机器学习，克服过拟合问题。
* 2012 年，Alex Krizhevsky、Ilya Sutskever 和 Geoffrey Hinton 创造了一个“大型的深度卷积神经网络”，也即现在众所周知的 AlexNet，赢得了当年的 ILSVRC。这是史上第一次有模型在 ImageNet 数据集表现如此出色。

<img src="https://raw.githubusercontent.com/hughxusu/lesson-ai/develop/images/cv/image-20201013165127393.png" style="zoom:50%;" />

* 2014年，蒙特利尔大学提出生成对抗网络（GAN）：拥有两个相互竞争的神经网络可以使机器学习得更快。一个网络尝试模仿真实数据生成假的数据，而另一个网络则试图将假数据区分出来。随着时间的推移，两个网络都会得到训练，生成对抗网络（GAN）被认为是计算机视觉领域的重大突破。
* 2018年末，英伟达发布的视频到视频生成（Video-to-Video synthesis），它通过精心设计的发生器、鉴别器网络以及时空对抗物镜，合成高分辨率、照片级真实、时间一致的视频，实现了让AI更具物理意识，更强大，并能够推广到新的和看不见的更多场景。
* 2019，更强大的GAN，BigGAN，是拥有了更聪明的学习技巧的GAN，可以生成更逼真的图像。
