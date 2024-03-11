---
title: "网络流I：详解最大流最小割"
description: 网络流(Network-Flows)是一种类比水流的解决问题方法，是图论中的热门问题。网络流部分充满复杂的概念、算法以及奇妙的证明，对于初学者很不友好。因此本博客的目标是总结和梳理网络流的基础知识。
pubDate: "Oct 28 2020"
layout: "/src/layouts/MarkdownLyaout.astro"
heroImage: "/blog/网络流I/head.png"
tags: ["算法", "网络流"]
---

网络流(Network-Flows)是一种类比水流的解决问题方法，是图论中的热门问题。网络流部分充满复杂的概念、算法以及奇妙的证明，对于初学者很不友好。因此本博客的目标是总结和梳理网络流的基础知识。网络流的知识将分为多个部分，在这一部分中我们主要讨论最大流最小割的定义，最大流问题的算法以及最大流最小割定理的证明。

## 最大流和最小割

### 网络流图的概念
<img src="\blog\网络流I\1.png" alt="网络流I-1" style="max-width: 600px" />

网络流图（Flow Network）是对于物质流动的一种抽象。它的定义如下：

- 是一张有向图 $G=(V,E)$，它包含源点 $s\in V$ 和汇点 $t\in V$。
- 对于图上每条边 $e$，都有非负整数容量 $c(e)$，容量是指同一时间能够流过边的最大的量。



### 最小割问题

<img src="\blog\网络流I\2.png" alt="网络流I-2" style="max-width: 600px" />

**割：**割（Cut）是对图上节点的分割 $(A,B)$，其中 $s\in A$ 且 $t\in B$。

**割的流量：**所有从点集 $A$ 到点集 $B$ 的边的流量之和（注意一定从 $A$ 指向 $B$）。
$$
cap(A,B)=\sum_{\mathrm{e\ out\ of\ A}} c(e)
$$
**最小割问题：**找到一个流量最小的割。



### 最大流问题

**流(Flow)：**流是一个满足下列条件的函数：

- 【流量限制】每条边的流小于该边容量

  对于每一条边 $e\in E$，有 $0\leq f(e)\leq c(e)$。

- 【流量守恒】除了源点与汇点，每个点流入量等于流出量

  对于每一个点 $v\in V-\{s,t\}$，有

$$
\sum_{e\ \mathrm{into}\ v}f(e)=\sum_{e\ \mathrm{out\ of}\ v}f(e)​
$$

<img src="\blog\网络流I\3.png" alt="网络流I-3" style="max-width: 600px" />

**流的值：**从源点流出的流量总和。
$$
val(f)=\sum_{e\ \mathrm{out\ of}\ s}f(e)
$$

<img src="\blog\网络流I\4.png" alt="网络流I-4" style="max-width: 600px" />

**最大流问题：**找到**值最大**的流函数。



## 最大流问题：Ford-Fulkerson 方法

### 错误思想：贪心算法

首先提一种错误的算法来抛砖引玉，那就是贪心算法，它的流程如下所示。

- 初始化：对于所有边 $e\in E$，$f(e)=0$。 
- 找到任意一条 $s\to t$ 的路径 $P$，路径上的边满足 $f(e)<c(e)$。
- 沿着路径 $P$ 在每条边上添加流。
- 重复此过程直到找不到满足条件的路径 $P$。

<img src="\blog\网络流I\greedyDemo1.png" alt="greedyDemo1" style="max-width: 600px" />

<img src="\blog\网络流I\greedyDemo2.png" alt="greedyDemo2" style="max-width: 600px" />

<img src="\blog\网络流I\greedyDemo3.png" alt="greedyDemo3" style="max-width: 600px" />

<img src="\blog\网络流I\greedyDemo4.png" alt="greedyDemo4" style="max-width: 600px" />

<img src="\blog\网络流I\greedyDemo5.png" alt="greedyDemo5" style="max-width: 600px" />


通过贪心算法得到的最大流的值为16，但是我们发现最大流的值可以达到19，如下图所示。

<img src="\blog\网络流I\answer.png" alt="answer" style="max-width: 600px" />

事实上，单纯的贪心算法无法解决最大流问题，因为贪心算法中的每一个选择是无法回退的，很可能使算法达不到最优解。



### 残存网络 Residual graph

既然贪心算法无法回退，那么我们就在图上增加回退的边，构成一张新的网络——残存网络。对于网络中的每条边 $e=(u,v)$，添加一条反向边 $e^R=(v,u)$。残存网络 $G_f$ 中各边的容量称为**残存容量**（Residual capacity），残存容量的大小定义为：

<img src="\blog\网络流I\residual.png" alt="residual" style="max-width: 600px" />

下图展示了残边的生成过程。

<img src="\blog\网络流I\residualEdge.png" alt="residualEdge" style="max-width: 600px" />

**关键性质：**$f'$ 是残存网络 $G_f$ 的流函数$\iff$ $f+f'$ 是原网络 $G$ 的流函数。



### 增广路径 Augmenting path

**简单路径 Simple path：**路径上经过的结点不重复的路径。

**增广路径 Augmenting path：**残存网络上一条从 $s$ 到 $t$ 的简单路径 $P$。

**瓶颈容量 Bottleneck capacity：**增广路径上所有边的残存容量的最小值。

**关键性质：**令 $f$ 是流，$P$ 是残存网络 $G_f$ 中的一条增广路径，则存在另一个流 $f'$，满足:
$$
val(f')=val(f)+bottleneck(G_f,P)
$$


### Ford-Fulkerson 算法

准备铺垫完成，正式进入正题。Ford-Fulkerson 算法的流程如下：

算法伪代码：

<img src="\blog\网络流I\FF.png" alt="FF" style="max-width: 600px" />

算法流程：

- 初始化：对于网络 $G$ 上所有边 $e$，令 $f(e)=0$。
- 在残存网络 $G_f$ 中任意寻找一条增广路径 $P$。
- 在路径 $P$ 上添加流。
- 重复直到找不到增广路径。

下面展示一个 Ford-Fulkerson 算法运行的 demo：


<img src="\blog\网络流I\F12.png" alt="F12" style="max-width: 600px" />

<img src="\blog\网络流I\F34.png" alt="F34" style="max-width: 600px" />

<img src="\blog\网络流I\F56.png" alt="F56" style="max-width: 600px" />



## 相关定理

### 流值引理

令 $f$ 是任意流并令 $(A,B)$ 是任意割。则穿过 $(A,B)$ 的净流量等于流 $f$ 的值。
$$
\sum_{e\ \mathrm{out\ of}\ A}f(e)-\sum_{e\ \mathrm{in\ to}\ A}f(e)=val(f)
$$
**证明：**首先根据流值的定义：
$$
val(f)=\sum_{e\ \mathrm{out\ of}\ s}f(e)\\
$$
根据**流量守恒**定理，在 $v\neq s$ 时有：
$$
\sum_{e\ \mathrm{out\ of}\ v}f(e)-\sum_{e\ \mathrm{in\ to}\ v}f(e)=0
$$
所以进一步可得：
$$
val(f)=\sum_{v\in A}\left(\sum_{e\ \mathrm{out\ of}\ v}f(e)-\sum_{e\ \mathrm{in\ to}\ v}f(e) \right)
$$
最终证明完毕：
$$
val(f)=\sum_{e\ \mathrm{out\ of}\ A}f(e)-\sum_{e\ \mathrm{in\ to}\ A}f(e)
$$


### 弱对偶性

令 $f$ 为任意流且 $(A,B)$ 为任意割，则 $val(f)\leq cap(A,B)$。

**证明：**

根据流值引理可得：
$$
\begin{aligned}
	val(f)
	&=\sum_{e\ \mathrm{out\ of}\ A}f(e)-\sum_{e\ \mathrm{in\ to}\ A}f(e)\\
	&\leq \sum_{e\ \mathrm{out\ of}\ A}f(e)\\
	&\leq \sum_{e\ \mathrm{out\ of}\ A}c(e)\\
	&=cap(A,B)
\end{aligned}
$$

### 最大流最小割定理

令 $f$ 为流网络 $G=(V,E)$ 中的一个流，该网络的源点为 $s$，汇点为 $t$，则下面条件等价：

- 存在 $(A,B)$ 是流网络 $G$ 的一个割，使得 $val(f)=cap(A,B)$。

- $f$ 是 $G$ 的一个最大流。
- 残存网络 $G_f$ 不包括任何增广路径。

**证明：1 $\to$ 2**

- 假设 $(A,B)$ 是一个割且满足 $cap(A,B)=val(f)$。

- 然后，对于任意流 $f'$，根据弱对偶性，$val(f')\leq cap(A,B)=val(f)$

- 因此，$f$ 是一个最大流。

**证明：2 $\to$ 3（反证法）**

- 假设对于流 $f$ 存在一条增广路径。
- 那么我们可以通过在这条路径上加流量来增加 $val(f)$。
- 因此，$f$ 不是最大流。

**证明：3 $\to$ 1**

- 令 $f$ 是没有增广路径的流。
- 令 $A$ 是残存网络 $G_f$ 中源点 $s$ 可达的点的集合。
- 根据割 $A$ 的定义，$s\in A$。
- 根据流 $f$ 的定义，因为没有增广路径，所以 $t\notin A$。

根据上述条件，我们可以得到两个有趣的结论：

- 对于任何从 $B$ 到 $A$ 的边 $e=(v,w)$ 其中 $v\in B,\ w\in A$，有 $f(e)=0$。
  - 原因：如果 $f(v,w)\neq 0$，则在残存网络中，残存边 $(w,v)$ 的残存容量 $c_f(w,v)=e(v,w)>0$。这样在残存网络 $G_f$ 中 $w$ 可达 $v$，即 $s$ 可达 $v$，说明 $v$ 应当在 $A$ 中，与条件矛盾。

- 对于任何从 $A$ 到 $B$ 的边 $e=(v,w)$ 其中 $v\in A,\ w\in B$，有 $f(e)=c(e)$。
  - 原因：如果 $f(v,w)<c(v,w)$，则在残存网络 $G_f$ 中有 $c_f(v,w)=c(v,w)-f(v,w)>0$，这意味着 $G_f$ 中 $v$ 可达 $w$，即 $s$ 可达 $w$，$w$ 应该在 $A$ 中，与条件矛盾。

结论的示意图如下：

<img src="\blog\网络流I\proof3to1.png" alt="proof3to1" style="max-width: 600px" />

由此进行推导即可证明：
$$
\begin{aligned}
	val(f)
	&=\sum_{e\ \mathrm{out\ of}\ A}f(e)-\sum_{e\ \mathrm{in\ to}\ A}f(e)\\
	&=\sum_{e\ \mathrm{out\ of}\ A}c(e)\\
	&=cap(A,B)
\end{aligned}
$$

事实上，最大流最小割定理就证明了：**最大流的值等于最小割的容量**。

因为根据弱对偶性，对于任意割 $(A,B)$ 我们有：
$$
val(f)\leq cap(A,B)
$$
根据最大流最小割定理3，可得：
$$
val(f^*)=cap(A,B)
$$


因此 $(A,B)$ 一定是容量最小的割，且其容量等于最大流的值。

