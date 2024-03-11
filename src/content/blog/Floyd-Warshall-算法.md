---
title: "Floyd-Warshall 算法"
description: "Floyd-Warshall 算法使用一种不同的动态规划公式来解决所有结点对最短路径问题，图上可以存在负权重的边，但是不存在负权重的环。本篇将按照动态规划的过程阐述 Floyd 算法，并且拓展如何利用 Floyd 算法找出有向图的传递闭包。"
pubDate: "Nov 11 2020"
layout: "/src/layouts/MarkdownLyaout.astro"
heroImage: "/blog/Floyd-Warshall-算法/FWDP.jpg"
tags: ["算法", "动态规划"]
---

Floyd-Warshall 算法使用一种不同的动态规划公式来解决所有结点对最短路径问题，运行时间为 $\Theta(|V|^3)$，图上可以存在负权重的边，但是不存在负权重的环。本篇将按照动态规划的过程阐述 Floyd 算法，并且拓展如何利用 Floyd 算法找出有向图的传递闭包。

## 全源最短路径

Floyd 算法考虑的是一条最短路径上的**中间结点**。

**中间结点**：简单路径 $p=<v_1,v_2,\cdots,v_l>$ 上的中间结点指的是路径 $p$ 上除了 $v_1$ 和 $v_l$ 之外的任意结点，也就是集合 $\{v_2,v_3\cdots,v_{l-1}\}$ 中的结点。

假定图 $G$ 的所有结点为 $V=\{1,2,\cdots,n\}$，考虑其中一个子集 $\{1,2,\cdots,k\}$，对于任意结点对 $i,j\in V$，考虑从 $i$ 到 $j$ 的所有中间结点均取自集合 $\{1,2,\cdots,k\}$ 的那些路径，并设 $p$ 为其中权重最小的路径（$p$ 是简单路径）。我们分别考虑结点 $k$ 是否是路径 $p$ 上的一个中间结点的情况。

- 如果 $k$ 不是 $p$ 上的中间结点，则 $p$ 上所有中间结点都属于集合 $\{1,2,\cdots, k-1\}$。因此，从 $i$ 到 $j$ 且中间结点均取自 $\{1,2,\cdots,k-1\}$ 的一条最短路径也同时是从 $i$ 到 $j$ 且中间结点均取自 $\{1,2,\cdots,k\}$ 的一条最短路径。
- 如果结点 $k$ 是路径 $p$ 上的中间结点，则将路径 $p$ 分解成 $p_1:i\to k$ 和 $p_2: k\to j$。可得 $p_1$ 是从结点 $i$ 到结点 $k$ 的，中间结点全部取自集合 $\{1,2,\cdots, k-1\}$ 的一条最短路径（因为 $k$ 是末尾结点）。类似的，$p_2$ 是从结点 $k$ 到结点 $j$ 的，中间结点全部取自集合 $\{1,2,\cdots, k-1\}$ 的一条最短路径。

下图很好的展示了两种不同情况。
<img src="/blog/Floyd-Warshall-算法/FWDP.jpg" alt="FWDP" style="max-width: 600px" />

我们假设 $d_{ij}^{(k)}$ 是从结点 $i$ 到结点 $j$ 的所有中间结点全部取自 $\{1,2,\cdots,k\}$ 的最短路径权重。$k=0$ 时路径只由一条边构成。根据如上定义，我们可以递归定义：

<img src="\blog\Floyd-Warshall-算法\dp.png" alt="DP" style="max-width: 600px" />

此定义下，矩阵 $D^{(n)}=(d_{ij}^{(n)})$ 就是我们想要的最终答案，因为所有结点都在 1~n 中。

我们可以自底向上计算最短路径权重，算法输入为 $n\times n$ 的矩阵 $W$，返回最短路径权重矩阵 $D^{(n)}$。算法伪代码如下：

<img src="\src\content\blog\Floyd-Warshall-算法\FW.jpg" alt="FW" style="max-width: 600px" />

该算法包含三层 for 循环，运行时间为 $\Theta(n^3)$。

## 有向图的传递闭包

### 传递闭包

给定有向图 $G=(V,E)$，结点集合为 $V=\{1,2,\cdots,n\}$，我们希望判断所有结点对之间是否包含一条 $i\to j$ 的路径。我们定义图 $G$ 的传递闭包为图 $G'=(V,E')$，其中 $E'=\{(i,j)\}$，如果 G 中包含从 $i$ 到 $j$ 的路径。

### 思路

如果图 $G$ 中存在一条从结点 $i$ 到 $j$ 的所有中间结点都取自集合 $\{1,2,\cdots,k\}$ 的路径，则 $t_{ij}^{(n)}=1$，否则 $t_{ij}^{(n)}=0$。我们构建传递闭包的方法为：将边 $(i,j)$ 置于集合 $E'$ 当且仅当 $t_{ij}^{(n)}=1$，递归定义如下：

<img src="\blog\Floyd-Warshall-算法\transit.png" alt="FW" style="max-width: 600px" />

$$
t_{ij}^{(k)}=t_{ij}^{(k-1)}\vee(t_{ik}^{(k-1)}\land t_{kj}^{(k-1)})\quad\quad k\geq 1
$$

我们同样使用递增的次序计算矩阵 $T^{(k)}=(t_{ij}^{(k)})$。

<img src="\blog\Floyd-Warshall-算法\tc.jpg" alt="FW" style="max-width: 600px" />

此算法的时间复杂度仍然是 $\Theta(n^3)$。