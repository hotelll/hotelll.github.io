---
title: "网络流III：Edmonds-Karp 算法"
description: 这一章中，我们将选择**边数最小**的增广路径。基于这种选择的 Ford-Fulkerson 算法称为 Edmonds-Karp 算法。
pubDate: "Nov 16 2020"
layout: "/src/layouts/MarkdownLyaout.astro"
heroImage: "/blog/网络流I/head.png"
tags: ["算法", "网络流"]
---

上一章提到，Ford-Fulkerson 算法效率的突破点就在于寻找更好的增广路径。上一章中提到的 Capacity-scaling 算法选择的是瓶颈容量最大的增广路径。这一章中，我们将选择**边数最小**的增广路径。基于这种选择的 Ford-Fulkerson 算法称为 Edmonds-Karp 算法。

Edmonds-Karp 算法就是在 Ford-Fulkerson 方法的基础上，将每条边上权重视为1的情况下，寻找最短增广路径，也就是边数最少的路径。我们可以很自然地想到利用**广度优先搜索**（BFS）的方法来寻找边数最小的增广路径，算法伪代码如下：

<img src="\blog\网络流III\bfs.png" alt="bfs" style="max-width: 600px" />


## 算法分析

**引理：**如果 Edmonds-Karp 算法运行在流网络 $G=(V,E)$ 上，该网络的源点为 $s$，汇点为 $t$。则对于所有结点 $v\in V-\{s,t\}$，残存网络 $G_f$ 中从结点 $u$ 到结点 $v$ 的最短路径距离（**边权均为1**）$\delta_f(u,v)$ 随着每次流量的递增而**单调递增**。

证明（反证法）

- 假设对于结点 $v$，存在一个流量递增操作，导致从源点 $s$ 到结点 $v$ 的最短路径距离减少。

- 设 $f$ 为第一个这样的流量操作之前的流量，$f'$ 是该操作之后的流量。

- 设 $v$ 是所有流量递增操作下最短路径被减少的结点中，$\delta_{f'}(s,v)$ 最小的结点，可得 $\delta_{f'}(s,v)<\delta_{f}(s,v)$。

- 设 $p=s\to (u,v)$ 是残存网络 $G_{f'}$ 中从源点 $s$ 到结点 $v$ 的一条最短路径，因此，$(u,v)\in E_{f'}$，且
  $$
  \delta_{f'}(s,u)=\delta_{f'}(s,v)-1
  $$

- 因为无论如何选择结点 $v$，我们知道从源点 $s$ 到结点 $u$ 的距离并没有减少（因为 $s\to v$ 是减少的路径中最短的一条），即
  $$
  \delta_{f'}(s,u)\geq \delta_{f}(s,u)
  $$
  
- 我们断言 $(u,v)\notin E_f$。因为如果有 $(u,v)\in E_f$，则
  $$
  \delta_{f}(s,v)\leq \delta_{f}(s,u)+1\leq \delta_{f'}(s,u)+1=\delta_{f'}(s,v)
  $$
  此结果与我们假设的 $\delta_{f'}(s,v)<\delta_f(s,v)$ 矛盾。

- 也就是说，$(u,v)\notin E$ 且 $(u,v)\in E_{f'}$，由此可以推断，流量递增操作一定增加了从结点 $v$ 到结点 $u$ 的流量。

- 所以残存网络 $G_f$ 中从源点 $s$ 到结点 $u$ 的最短路径上的最后一条边是 $(v,u)$。因此
  $$
  \delta_{f}(s,v)=\delta_f(s,u)-1\leq \delta_{f'}(s,u)-1=\delta_{f'}(s,v)-2
  $$
  此结论与假设 $\delta_{f'}(s,v)<\delta_f(s,v)$ 矛盾，因此不存在这样的结点 $v$，证毕。



**定理：**如果 Edmonds-Karp 算法运行在源点为 $s$ 且汇点为 $t$ 的流网络 $G=(V,E)$ 上，则该算法所执行的流量增加操作的总次数为 $O(VE)$。

**证明：**

- 残存网络 $G_f$ 中，如果一条路径 $p$ 的残存容量是该路径上边 $(u,v)$ 的残存容量，则称 $(u,v)$ 为**关键边**。

- 沿一条增广路径增加流后，该条路径上的所有关键边都会从 $G_f$ 中消失，且每条增广路径至少有一条关键边。

- 假设边 $(u,v)$，其第一次成为关键边时，我们有
  $$
  \delta_{f}(s,v)=\delta_{f}(s,u)+1
  $$

- 一旦增加流后，$(u,v)$ 将从 $G_f$ 中消失，以后也不能出现在另一条增广路径上，直到从 $u$ 到 $v$ 的网络流减小，并且 $(u,v)$ 出现在增广路径上。假设这一事件发生时流为 $f'$，则
  $$
  \delta_{f'}(s,u)=\delta_{f'}(s,v)+1
  $$

- 根据引理，我们可知 $\delta_{f}(s,v)\leq \delta_{f'}(s,v)$，因此有
  $$
  \delta_{f'}(s,u)=\delta_{f'}(s,v)+1\geq \delta_{f}(s,v)+1=\delta_{f}(s,u)+2
  $$

- 因此，$(u,v)$ 在两次成为关键边的间隔中，从 $s$ 到 $u$ 的距离至少增加 2 个单位，且距离最初至少为零。同时，从 $s$ 到 $u$ 的最短路径上中间结点不可能包括 $s$，$u$ 和 $t$，因此距离最多增加至 $|V|-2$。所以一条边最多成为关键边 $|V|/2$ 次。
- 由于一共有 $|E|$ 条边，因此在 Edmonds-Karp 算法过程中关键边的总数为 $O(VE)$。

- 因为每条增广路径至少有一条关键边，因此流量增加操作总次数（增广路径数）为 $O(VE)$。

由于 Ford-Fulkerson 算法的每次迭代可以在 $O(E)$ 时间内完成，因此 Edmonds-Karp 算法总运行时间为 $O(VE^2)$。



### 番外：水平图 Level graph

给定一个有向图 $G=(V,E)$，源点为 $s$，则它的水平图 $L_G=(V,E_G)$ 定义为：

- $l(v)=$ 从 $s$ 到 $v$ 的最短路径的边的数量。
- $L_G=(V,E_G)$ 是 $G$ 的子图，只包含满足 $l(w)=l(v)+1$ 的边 $(v,w)\in E$。

<img src="\blog\网络流III\level_graph.png" alt="level_graph" style="max-width: 600px" />


我们可以通过运行 BFS 在 $O(m+n)$ 的时间内计算出水平图。

**性质：**$P$ 是 $G$ 中 $s\to v$ 的一条最短路径，当且仅当 $P$ 是 $L_G$ 中 $s\to v$ 的一条路径。