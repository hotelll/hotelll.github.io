---
title: "从神经网络到VQ-VAE模型"
description: "这是一篇介绍从神经网络（Neural Network）开始谈起，依次介绍自编码器（AutoEncoder）、变分自编码器 （VAE）和矢量量化变分自编码器（VQ-VAE）的文章。"
pubDate: "Jul 8 2020"
layout: "/src/layouts/MarkdownLyaout.astro"
heroImage: "/blog/From-NN-to-VQ-VAE/Head.png"
tags: ["生成模型", "人工智能"]
---

这是一篇介绍从神经网络（Neural Network）开始谈起，依次介绍自编码器（AutoEncoder）、变分自编码器 （VAE）和矢量量化变分自编码器（VQ-VAE）的文章。

因为本人刚刚入门机器学习领域，又在项目中遇到了 VQ-VAE 模型的应用。在学习模型的时候又不断地遇到更基础的问题，于是一步步递归地进行学习，就产生了一条这样的学习路径。这篇博客的目的也是记录本周的学习进展，同时介绍以下本人对各个概念与模型的初步认识，之后可能会加入更详细的数学推导与代码实现。



## 神经网络

> 参考博客：[https://www.cnblogs.com/subconscious/p/5058741.html](https://www.cnblogs.com/subconscious/p/5058741.html)

神经网络是一门重要的机器学习技术，也是深度学习的基础。下面我们按照从部分到整体的顺序来介绍神经网络的基本概念。

### 神经元

神经元是组成神经网络的基本单元，具有以下属性：

1. 包含多个**输入**；
2. 结构中的箭头称为**连接**；
3. 在每个输入的连接上会赋对应的**权值**，权值就是神经网络的训练目标；
4. 所有赋权的输入会被求和；
5. 求和后的结果经过非线性函数进行计算处理后输出。
6. 输出的端口数量不定，但是所有端口的值是**固定**的。

下图为神经元模型：

<img src="\blog\From-NN-to-VQ-VAE\unit.jpg" alt="神经元模型" style="max-width: 600px" />

神经元之间通过相互连接形成神经网络。

### 单层神经网络

神经网络层数指的是网络中有计算操作的层次，简称**计算层**。单层神经网络又称为**感知器**，只有输入层和输出层。一个3输入2输出的感知器结构如下：

<img src="\blog\From-NN-to-VQ-VAE\perceptron.jpg" alt="感知器结构" style="max-width: 600px" />

这里权值 $w_{i,j}$ 意思是从第 $i$ 个输入到第 $j$ 个输出的权值。

如果我们把输入记作向量 $\mathbf{a}=[a_1,a_2,a_3]^\mathrm{T}$，权值记作 $2\times 3$ 的矩阵 $\mathbf{W}$，输出公式可以写成：
$$
z = g(\mathbf{W*a})
$$
感知器可以完成简单的线性分类的任务，如下所示：

<img src="\src\content\blog\From-NN-to-VQ-VAE\classifier.png" alt="线性分类图示" style="max-width: 600px" />

### 多层神经网络

随着算力的提升以及解决更复杂问题的需求，科学家将多个感知器组成了多层感知器，也就是多层神经网络。一个 3-2-2 的两层神经网络结构如下：

<img src="\blog\From-NN-to-VQ-VAE\multiNN.jpg" alt="两层神经网络结构" style="max-width: 600px" />

结构中，有如下注意点：

1. 在输入层和输出层之间加入了**中间层**，又称**隐藏层**，隐藏层的结点数是任意的，好的结点数设置可以优化神经网络的效果；
2. 这里加入了新的概念：**偏置**。偏置节点的值永远为+1，特点是节点无输入，偏置大小由向量 $\mathbf{b}$ 决定；
3. 由于包含多层，$\mathbf{W}^{(k)}$代表的是第 k 层的权值矩阵，定义输入层后为第一层。

这样，上图结构中的运算式可表达为：
$$
g(\mathbf{W}^{(1)}*\mathbf{a}^{(1)}+\mathbf{b}^{(1)})=\mathbf{a}^{(2)}
$$

$$
g(\mathbf{W}^{(2)}*\mathbf{a}^{(2)}+\mathbf{b}^{(2)})=\mathbf{z}
$$



### 训练

机器学习的目的，就是使得参数尽可能逼近真实的模型。具体方法是给神经网络的所有参数（权值）赋随机值来进行初始化，来预测训练数据中的样本。假设样本预测目标为 $y_p$，真实目标为 $y$，我们定义损失 loss 如下：
$$
\mathrm{loss} = (y_p-y)^2
$$
训练目标是让 loss 最小。因此训练的问题变成了：

> 如何通过改变参数（权值），让损失函数的值最小化。

一般解这个优化问题的方法是**梯度下降**法，即每次计算参数当前的梯度，然后让参数沿梯度下降的方向变化一定距离，经过多次迭代，直到梯度接近零为止。

由于每次计算所有参数的梯度代价很大，我们还需要**反向传播**算法，从后往前。对于上面的例子，计算梯度的顺序为：输出层 —> 参数矩阵2 —> 中间层 —> 参数矩阵1 —> 输入层。这样就获得了所有参数的梯度。



## 自编码器 AutoEncoder

> 参考博客：[https://zhuanlan.zhihu.com/p/68903857](https://zhuanlan.zhihu.com/p/68903857)

**自编码器**是神经网络中的一类模型。它的结构如下所示。

<img src="\blog\From-NN-to-VQ-VAE\AntoEncoder.jpg" alt="自编码器" style="max-width: 600px" />

自编码器框架由**编码过程**和**解码过程**组成。首先，通过 encoder g 将输入样本 x 映射到特征空间z；之后，再通过 decoder f 将抽象特征 z 映射回原始空间得到重构样本 x'。

在编码过程，神经网络逐层降低神经元个数来对数据进行**压缩**；在解码过程，神经网络基于数据的抽象表示，逐层提升神经元数量，来实现对输入样本的**重构**。

在训练过程中不需要样本的 label，因为它本质上是把输入同时作为输入和输出，输出是和输入相比较的，因此是一种**无监督学习**的方法。



## 变分自编码器 Variational AutoEncoder

> 参考博客：[https://zhuanlan.zhihu.com/p/34998569](https://zhuanlan.zhihu.com/p/34998569)
>
> ​				  [https://zhuanlan.zhihu.com/p/91434658](https://zhuanlan.zhihu.com/p/91434658)
>
> 相关论文：[https://arxiv.org/pdf/1606.05908.pdf](https://arxiv.org/pdf/1606.05908.pdf)

VAE是一类强大的生成模型。VAE较大的不同点在于，它假设了样本 x 的抽象特征 z 服从 $(\mu,\ \sigma^2)$ 的高斯分布（一般会假设为标准正态分布 $\mathcal{N}(0,1)$）。这样做的优势是，训练结束后，我们可以抛弃 Encoder 部分，而直接从高斯分布中采样一个 z，然后根据 z 来生成一个 x，这是将是一个很棒的生成模型。VAE模型的示意图如下：

<img src="\blog\From-NN-to-VQ-VAE\VAE.jpg" alt="VAE结构" style="max-width: 600px" />

VAE模型中，针对每个 $X_k$，通过神经网络训练出一个专属于 $X_k$ 的正态分布 $p(Z|X_k)$ 的均值和方差。这样，我们就得到了一个生成器 $\hat{X}_k=g(Z_k)$。总结来说，VAE为每个样本构造专属的正态分布，之后进行采样来重构。

在训练的过程中，我们希望最小化 $D(\hat{X}_k,X_k)^2$。 同时，VAE让所有的 $p(Z|X)$ 向标准正态分布$\mathcal{N}(0,1)$看齐。这样就防止了方差（噪声）为零时 VAE 退化为 AE 的情况，保证了模型的生成能力。

在 VAE 中，有两个 encoder，分别用来计算均值和方差。

VAE 本质上是在常规的自编码器的基础上，对 encoder 的结果（在 VAE 中对应计算均值的神经网络）加入高斯噪声，从而使 decoder 结果对噪声具有鲁棒性。

在实际情况中，我们需要在模型的准确率上与隐含向量服从标准正态分布之间做一个权衡。原论文中直接采用各**一般正态分布**与**标准正态分布**的KL散度作为 loss 来进行计算：
$$
KL[\mathcal{N}(\mu,\sigma^2)\ ||\ \mathcal{N}(0,I)]
$$
计算的结果如下：
$$
L_{\mu,\sigma^2}=\frac{1}{2}\sum_{i=1}^d(\mu_{(i)}^2+\sigma_{(i)}^2-\log\sigma_{(i)}^{2}-1)
$$
此处的 $d$ 是隐变量 Z 的维度，$\mu_{(i)}$ 和 $\sigma_{(i)}^2$ 分别代表一般正态分布的均值向量和方差向量的第 $i$ 个分量。这个综合的式子帮我们考虑了均值损失和方差损失之间的协调问题。



## 矢量量化变分自编码器 

## Vector Quantised-Variational AutoEncoder

> 参考博客：[https://kexue.fm/archives/6760](https://kexue.fm/archives/6760)
>
> ​				   [https://zhuanlan.zhihu.com/p/91434658](https://zhuanlan.zhihu.com/p/91434658)

VQ-VAE的特点是编码出的编码向量是离散的（向量的每个元素都是整数）。其总体结构如下：

<img src="\blog\From-NN-to-VQ-VAE\VQVAE.jpg" alt="VQ-VAE结构" style="max-width: 600px" />

### 最邻近重构

最邻近重构是将输入编码成为离散向量的方法。输入向量 $\mathbf{x}$ 传入 encoder 后得到的应是连续的 $d$ 维向量 $\mathbf{z}$。同时，VQ-VAE 模型会维护一个 编码表（codebook），记为：
$$
E=[e_1,e_2,...,e_K]
$$
其中，每个 $e_i$ 都是一个大小为 $d$ 的向量。VQ-VAE 模型通过最邻近搜索，将 $\mathbf{z}$ 映射为这 K 个向量中最邻近的一个：
$$
z\to e_k,\quad k=\arg\min_{j}||z-e_j||_2
$$
我们将 $\mathbf{z}$ 对应的离散向量记为 $\mathbf{z}_q$ 。因为每个 $\mathbf{z}_q$ 对应着编码表中的一个索引编号，因此 VQ-VAE 将输入编码成为 $1,2,...,K$ 中的一个整数。

最后，我们将 $\mathbf{z}_q$ 传入一个 decoder，希望重构原图 $\hat{x}=\mathrm{decoder}(\mathbf{z_q})$。

整个流程如下：
$$
\mathbf{x}\stackrel{\mathrm{encoder}}\longrightarrow
\mathbf{z} \stackrel{最邻近}\longrightarrow
\mathbf{z}_q \stackrel{\mathrm{decoder}}\longrightarrow
\hat{\mathbf{x}}
$$
在实际情况中，为了避免重构失真，通常通过多层卷积将 $\mathbf{x}$ 编码为 $m\times m$ 个大小为 $d$ 的向量，因此最终相当于将 $\mathbf{x}$ 编码成为 $z_q$，也就是一个 $m\times m$ 的整数矩阵。 

### 自行梯度下降

对于一个普通的自编码器，损失函数为：
$$
||x-\mathrm{decoder}(z)||_2^2
$$
但是在 VQ-VAE 中，损失函数应当是：
$$
||x-\mathrm{decoder}(z_q)||_2^2
$$

但是由于 $z_q$ 在构建的时候有取最小值操作，$\arg\min$ 是没有梯度的，因此难以更新 encoder。

VQ-VAE 使用了**直通估计**（Straight-Through Estimator）的方法，在前向传播的时候使用真正的损失函数，而在反向传播的时候，可以使用自己设计的梯度来对 encoder 进行优化。根据这个思想所设计出的目标函数为：
$$
||x-\mathrm{decoder}(z+sg[z_q-z])||_2^2
$$
其中 $sg$ 代表不计算其梯度。在此情况下，前向传播时的目标函数为 $\left\|x-\mathrm{decoder}(z_q)\right\|_2^2$，反向传播的目标函数为 $\left\|x-\mathrm{decoder}(z)\right\|_2^2$。

### 维护编码表

根据 VQ-VAE 的最邻近搜索的设计，我们期望 $z_q$ 和 $z$ 尽可能接近。因此在损失函数中，我们可以直接加入相关项一起进行优化：
$$
||x-\mathrm{decoder}(z+sg[z_q-z])||_2^2+\beta||z-z_q||_2^2
$$
严谨来说，模型应当尽量让 $z_q$ 去靠近 $z$，而不是让 $z$ 去接近 $z_q$。因此最终的 loss 设计如下：
$$
\mathrm{loss}=||x-\mathrm{decoder}(z+sg[z_q-z])||_2^2+\beta||sg[z]-z_q||_2^2+\gamma||z-sg[z_q]||_2^2
$$
其中 $\beta$ 的项代表固定 $z$，让 $z_q$ 去靠近 $z$；$\gamma$ 的项代表固定 $z_q$，让 $z$ 去靠近 $z_q$。通过令 $\gamma < \beta$ 表示我们重视前者甚于后者。

### 拟合编码分布

对于一个编码后的 $m\times m$ 的整数矩阵，利用自回归模型，如 PixelCNN （这里没有再做了解）对其进行拟合，得到编码分布，并根据分布可以生成一个新的编码矩阵。将编码矩阵根据编码表映射为矩阵 $z_q$，再经过 decoder 既可得到新的生成。



## 总结

以上就是本周的学习进展，总体来说是比较磕磕碰碰的，“万事开头难”。因为主要是通过博客学习，本人的基础和理解能力也有限，可能会有理解不到位或表达不周之处，也欢迎大家指出。