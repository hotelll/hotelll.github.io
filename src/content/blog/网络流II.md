---
title: "网络流II：Capacity-scaling 算法"
description: 这是网络流的第二部分。在网络流 I 中我们讨论了最大流最小割的定义、求解方法以及定理证明。在第二部分中，我们将讨论一种优化 Ford-Fulkerson 算法时间复杂度的方法——Capacity-scaling 算法。
pubDate: "Oct 29 2020"
layout: "/src/layouts/MarkdownLyaout.astro"
heroImage: "/src/content/blog/网络流I/head.png"
tags: ["算法", "网络流"]
---

这是网络流的第二部分。在网络流 I 中我们讨论了最大流最小割的定义、求解方法以及定理证明。在第二部分中，我们将讨论一种优化 Ford-Fulkerson 算法时间复杂度的方法——Capacity-scaling 算法。Capacity-scaling 算法通过不断搜寻残存网络子图中瓶颈容量最大的增广路径，将算法时间复杂度从 $O(mnC)$ 优化到 $O(mn\log C)$。

## Ford-Fulkerson 算法缺陷

### 回顾 Ford-Fulkerson 算法

我们这里再次写一下 FF 算法的流程：

- 初始化：对于网络 $G$ 上所有边 $e$，令 $f(e)=0$。
- 在残存网络 $G_f$ 中任意寻找一条增广路径 $P$。
- 在路径 $P$ 上添加流。
- 重复直到找不到增广路径。

首先，我们来分析一下为什么需要 Capacity-scaling 算法，直接用 Ford-Fulkerson 算法的缺陷是什么？

### Ford-Fulkerson 算法分析

**假设：**网络中边的容量均为 $1$ 到 $C$ 之间的整数。

**整数不变性**（Integrality invariant）：通过 Ford-Fulkerson 算法，网络中流的值 $f(e)$ 以及残存容量 $c_f(e)$ 同样是整数。

**定理：**算法最多在 $val(f^*)\leq nC$ 次迭代后终止。

- 证明：Ford-Fulkerson 算法每一次迭代至少给流的值增加1，且根据定义，流的值上限为 $nC$。

**推论：**Ford-Fulkerson 算法的时间复杂度是 $O(mnC)$。

**整数定理**（Integrality theorem）：存在一个最大流 $f^*$ ，它的所有边的流值 $f^*(e)$ 均为整数。

- 证明：因为 Ford-Fulkerson 算法会终止，由 Integrality invariant 可以直接推得。

### Ford-Fulkerson 算法的 Bad case

我们可以发现，Ford-Fulkerson 算法的时间复杂度不仅基于输入规模 $(m,n)$ 的多项式时间，还与网络最大容量 $C$ 有关。如果网络的最大容量为 $C$，那么算法一定会进行 $\geq C$ 次迭代。因此，即便输入规模很小，如果输入网络边的最大容量 $C$ 很大，那么 FF 算法的时间复杂度依然会很大。

### 选择好的增广路径

避免这个问题的方法就是在算法过程中**选择更好的增广路径**。那么我们该如何高效地找到增广路径并且使 FF 算法的迭代更少呢？

记得在上一节我们将瓶颈容量的时候提到，如果 $f$ 是原来的流，且找到一条增广路径 $P$，那么更新的流 $f'$ 满足 $val(f')=val(f)+bottleneck(G_f, P)$。所以如果能够每次挑选**瓶颈容量最大**的增广路径，我们就可以保证每次迭代中 $val(f)$ 增加的量最大，从而减少迭代次数。

直接搜索瓶颈容量最大的路径计算量较大，因此我们采用 Capacity-scaling 算法进行寻找。

## Capacity-scaling 算法

在 Capacity-Scaling 算法中，我们记录一个缩放参数 $\Delta$，在每次迭代中，我们不关注整个 $G_f$，只关注 $G_f(\Delta)$。$G_f(\Delta)$ 是 $G_f$ 的子图，只包括 $G_f$ 中残存容量 $\geq\Delta$ 的边。我们初始化 $\Delta$ 为不大于最大容量 $C$ 的最大2次幂，且在每轮迭代中缩小 $\Delta$ 为 $\Delta /2$。

### 算法伪代码
<img src="\src\content\blog\网络流II\CSalgorithm.png" alt="CSalgorithm" style="max-width: 600px" />

### 算法正确性与复杂性分析

假设：所有边的容量是 $1$ 到 $C$ 的整数。

**整数不变性：**所有流和残存容量都是整数。

**定理：如果 Capacity-scaling 算法终止，那么 $f$ 是一个最大流。**

证明：

- 根据整数不变性，当 $\Delta=1$时，$G_f(\Delta)=G_f$。
- 根据伪代码，算法在 $\Delta=1$ 阶段终止时，图上将不再有增广路径。 

##### 引理1：外层 While 循环重复 $1+\lceil \log_2 C\rceil$ 次。

证明：算法初始化 $C/2<\Delta\leq C$，且 $\Delta$ 每次迭代减少 $1/2$，由此可得结果。

##### 引理2：令 $f$ 是某个 $\Delta$-scaling 阶段后的流，则最大流的值 $val(f^*)\leq val(f)+m\Delta$（$m$ 为边数）。

证明：（类比最大流最小割 $3\to 1$ 的证明方法）

- 假设存在割 $(A,B)$ 满足 $cap(A,B)\leq val(f)+m\Delta$。
- $A$ 为源点 $s$ 在 $G_f(\Delta)$ 中可达的所有点的集合。
- 根据割 $A$ 的定义，$s\in A$。
- 根据流 $f$ 的定义，因为是迭代后的结果，所以一定没有增广路径，所以 $t\notin A$。

- 由上述条件可得（证明类比最大流最小割定理 $3\to 1$，在残存网络中用反证法）：
  - 对于任意边 $e=(v,w)$，$v\in B$ 且 $w\in A$，有 $f(e)<\Delta$。
  - 对于任意边 $e=(v,w)$，$v\in A$ 且 $w\in B$，有 $f(e)>c(e)-\Delta$。

<img src="\src\content\blog\网络流II\algorithm_val.png" alt="algorithm_val" style="max-width: 600px" />

##### 引理3：流的值在每个 Scaling 阶段最多增加 $2m$ 次。

证明：

- 令 $f'$ 是上个阶段（$\Delta'$-$scaling$）得到的流。
- 令 $f$ 是当前阶段（$\Delta$-$scaling$）得到的流，其中 $\Delta=\Delta'/2$。
- 根据引理2，可得 $val(f^*)\leq val(f')+m\Delta'$。
- 也就是说，$val(f^*)\leq val(f')+2m\Delta$。
- 所以说，在 $\Delta$-$scaling$ 阶段，流的值最多增加 $2m\Delta$。
- 根据算法，在 $\Delta$-$scaling$ 阶段，每次流值的增量至少为 $\Delta$。
- 所以，流值最多的增加次数为 $2m$。

##### 定理：Capacity-scaling 算法需要在 $O(m\log C)$ 次增加中找到最大流，每一次增加所需要的时间为 $O(m)$，包括建立网络以及寻找路径。因此 Capacity-scaling 算法总体时间复杂度为 $O(m^2\log C)$。

证明：由引理1和引理3可得。

又学会了一个算法，激不激动233。