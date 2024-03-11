---
title: "认认真真写一篇LSTM"
description: 想爬上树就得从底下开始。
pubDate: "Feb 10 2021"
layout: "/src/layouts/MarkdownLyaout.astro"
heroImage: "http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png"
tags: ["人工智能", "神经网络"]
---

长短期记忆（Long Short-Term Memory, LSTM）是一种事件循环神经网络，适合于处理和预测时间序列中间隔和延迟非常长的重要事件。

<!-- more -->

> 参考博客：[http://colah.github.io/posts/2015-08-Understanding-LSTMs/](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

## 循环神经网络

神经网络尽力用一种模仿人脑的算法操作，来得到大量数据之中隐含的关系。传统的神经网络通过模拟神经元来实现。然而我们发现，人脑的思维过程并不仅仅基于当下的信息，而是具有持续性，受先前数据和结果影响的，这一特征却无法由传统神经网络实现。

由此，循环神经网络（Recurrent Neural Network, RNN）诞生了。它通过在网络中加入循环结构，实现不同 step 之间的信息传递，示意图如下：

<img src="http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png" alt="An unrolled recurrent neural network." style="zoom: 33%;" />

从上图，我们发现循环神经网络展开后与普通的神经网络是相同的。循环神经网络相当于将原始网络复制好几份，每一份将信息传递给下一份。循环神经网络在语音识别、语言建模、翻译、图像标注等领域大放异彩。

## 长期相关性的局限性

循环神经网络的优势就在于它试图能够将先前的信息与当前任务连接在一起，那样之前的信息可能可以指导当前的任务。然而，循环神经网络是否能够真正实现这一优势，要视情况而定。

一些情况下，任务可能只需要之前短期的信息就能很好完成。我们设想一个预测词语的语言模型想要完成如下任务：

> The clouds are in the <u>sky</u>.

我们只需要**短期**的信息（cloud）即可进行预测，信息的间隔很小，因此 RNN 可以学习过去的信息。

<img src="http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-shorttermdepdencies.png" alt="img" style="zoom: 33%;" />

但是，如果我们需要使用**长期**的信息作为依据，RNN 可能力不从心，如下例：

> I grew up in **France** and I lived there for about ten years during my childhood. Therefore, I am very familiar to the culture there and I speak  fluent <u>French</u>.

这个例子中，我们作为人类可以从 France 推断出 French，但是对于模型来说这个信息的间隔太大（这个例子中体现为两个单词间距太大）。在实际实践中，随着间隔的增大，RNN 将无法再连接这个长期信息到当前任务，这也就是 RNN 在处理长期信息相关性中的局限性。

<img src="http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-longtermdependencies.png" alt="Neural networks struggle with long term dependencies." style="zoom: 33%;" />

然而，LSTMs 并没有这个局限性！

## LSTM 网络

长短期记忆网络，简称 LSTMs，是一种独特的循环神经网络。它由 Hochreiter & Schmidhuber 提出并经过无数人的优化改进。LSTM 对于很多问题能够很好解决，并被广泛使用。

LSTMs 的设计直指**长期信息相关性**的问题。标准的 RNN 中重复的模块只有一个简单结构，例如 tanh 层：

<img src="http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-SimpleRNN.png" alt="img" style="zoom: 33%;" />

LSTM 将重复模块改成了一个特殊的四层结构：

<img src="http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png" alt="A LSTM neural network." style="zoom: 33%;" />

LSTMs 的关键就是**细胞状态**（cell state），也就是图中顶部水平穿过的这条线。

<img src="http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-C-line.png" alt="img" style="zoom: 50%;" />

cell state 就像是一个传送带，径直穿过整个链式结构，过程中只会经过几个线性操作。LSTM 通过**门**（gates）对 cell state 的信息进行增减。例如图中 sigmoid 层和点乘 gate 组合能够选择信息通过的量。sigmoid 输出 0-1 的值，0 即禁止通过，1 即全部通过。

<img src="http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-gate.png" alt="img" style="zoom: 80%;" />

一个 LSTM 中有三个这样的 gates 来控制和保护 cell state。下面我们来一步步走一下 LSTM 的整个流程。

LSTM 中第一步是决定 cell state 中要丢弃的信息。这个决定是由一个叫做 “**遗忘门**” 的 sigmoid 层完成的。其位置如下图所示：

<img src="http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-f.png" alt="img" style="zoom: 40%;" />

它接受 $h_{t-1}$ 和 $x_t$ 后，对应 $C_{t-1}$ 中的每个数字输出一个 0-1 之间的数字，这个数字就代表了接收信息的比例（0为全拒绝，1为全接受）。

下一步，LSTM 需要决定在 cell state 中加入什么新信息，这包括两个步骤。第一步由 sigmoid 实现的 “**输入门**”，决定我们要更新的值，第二步由一个 tanh 层创造一个包含候选值的新向量 $\tilde{C}_{t}$，也可以被加入 cell state 来代替原来被我们选择遗忘的数值。具体结构如下：

<img src="http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-i.png" alt="img" style="zoom: 40%;" />

要更新 $C_{t-1}$ 为 $C_t$，我们只需要将之前步骤付诸实践即可。

<img src="http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-C.png" alt="img" style="zoom:40%;" />

最后，我们需要决定我们的输出。我们的输出是将 cell state 过滤后的版本。首先，我们使用一个 sigmoid 层来选择 cell state 中要输出的部分。接着，我们让 cell state 通过一个 tanh 层并且将结果与之前的 sigmoid 门，从而实现只输出我们想要的部分。

<img src="http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-o.png" alt="img" style="zoom: 40%;" />

## 长短期记忆的变体

之前所介绍的是普通的 LSTM。但是 LSTM 有很多变体，这里我们接受几种种最常用的变体。

### Peephole Connections

这种变体由 Gers & Schimidhuber 提出，在其中加入了 “窥视孔连接”（peephole connections），使得门能够看到 cell state，结构如下所示：

<img src="http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-var-peepholes.png" alt="img" style="zoom:40%;" />

我们可以给所有门加上窥视孔，也可以选择其中的几个添加。

### Coupled forget and input gates

在这个变体中，我们不再分别决定删除和增加信息，而是一起进行决策，也就是说，我们只在该位置添加信息时才选择遗忘该信息，或者只在遗忘老信息之后才输入新信息。也就是说，遗忘和添加信息是同时考虑，联动发生的。

<img src="http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-var-tied.png" alt="img" style="zoom:40%;" />

## Gate Recurrent Unit

这是一个由 Cho 提出的变体，它将遗忘门和输入门组合成一个门，称为 “更新门”（update gate）。它也合并了 cell state 和 hidden state，以及一些其他的改变。这个变体比标准 LSTM 模型更简明，也在科研中越来越流行。

<img src="http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-var-GRU.png" alt="A gated recurrent unit neural network." style="zoom:40%;" />

LSTM 还有很多别的变体，例如 Depth Gated RNNs by Yao，以及别的用于处理长期相关性的模型，如 Clockwork RNNs by Koutnik。虽然总体上大致相似，但它们在特定任务下表现不同的性能。

## 总结

LSTMs 是 RNNs 的一种很棒的突破与实现方式，对于很多任务都展现出其强大的能力。与此同时，Grid LSTMs by Kalchbrenner 也具有远大前景，RNNs 在生成模型中的应用也十分有趣，例如：[Gregor, *et al.* (2015)](http://arxiv.org/pdf/1502.04623.pdf), [Chung, *et al.* (2015)](http://arxiv.org/pdf/1506.02216v3.pdf), or [Bayer & Osendorfer (2015)](http://arxiv.org/pdf/1411.7610v3.pdf)。可以说，RNNs 将会在未来保持它的前景和生命力。