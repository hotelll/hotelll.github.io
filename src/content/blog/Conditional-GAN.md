---
title: "基于条件的GAN生成"
description: "我们如何将条件信息加入GAN？"
pubDate: "Feb 17 2021"
layout: "/src/layouts/MarkdownLyaout.astro"
heroImage: "/blog/Conditional-GAN/cGAN_results.png"
tags: ["生成模型", "人工智能"]
---

我们谈一谈如何将条件限制信息加入 GAN。

GAN 是一种创新的训练生成模型的方式，但是对于没有条件信息的生成模型，我们无法控制所生成数据的模式。我们通过将额外的信息作为模型的条件，那么就能够引导数据生成的过程。由此，有条件的生成对抗网络（conditional Generative Adversarial Nets，cGAN）就诞生了。

## 原理

我们知道 GAN 中由生成器 $G$ 和鉴别器 $D$ 组成。生成器的目的是从数据 $x$ 中学习出一个分布 $p_g$，生成器建立了从先验噪声分布 $p_z(z)$ 到输出数据 $G(z;\theta_g)$ 的映射关系，鉴别器 $D$ 负责输出输入数据的真实性。二者是同时训练的，优化函数如下：
$$
\min_{G}\max_{D} V(D,G)=E_{x\sim p_{data}(x)}\left[\log D(x)\right]+E_{z\sim p_z(z)}\left[\log(1-D(G(z)))\right]
$$
现在，在 GAN 的基础上，我们同时在生成器和鉴别器中加入额外信息 $y$ 作为条件。$y$ 可以是任何形式的辅助信息，例如类别标签等。我们将条件信息 $y$ 作为额外输入层给到生成器和鉴别器。

在生成器中，输入的先验噪音 $p_z(z)$ 和 $y$ 组合成为联合隐藏层。在这个条件下，cGAN 的目标函数如下：
$$
\min_{G}\max_{D} V(D,G)=E_{x\sim p_{data}(x)}\left[\log D(x|y)\right]+E_{z\sim p_z(z)}\left[\log(1-D(G(z|y)))\right]
$$
<img src="/blog/Conditional-GAN/cGAN_structure.png" alt="cGAN-structure" style="max-width: 600px" />

## 实验

### MNIST

在 cGAN 的帮助下，针对 MNIST 数据集，我们能够通过额外信息 $y$（分类标签 0~9）来指定生成器输出的数字。下面是实验结果：

<img src="/blog/Conditional-GAN/cGAN_results.png" alt="Generated Results" style="max-width: 600px" />

我们可以看到生成的结果是有序排列的，而普通的 GAN 只能生成无序随机数字。

## 总结

很简单的思想，但是实现了在 GAN 中加入条件引导生成过程，好极了。