---
title: "LLM推理加速 Lesson3：FlashInfer"
description: "从I/O视角优化Transformer的基础模块Attention。"
pubDate: "Jun 3 2025"
layout: "/src/layouts/MarkdownLyaout.astro"
heroImage: "/blog/LLM推理加速Lesson2-FlashAttention/FlashAttention.png"
tags: ["大语言模型", "推理加速"]
---

FlashInfer是2024年由NVIDIA团队提出的一个面向LLM的部署推理库，论文为[FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving](https://arxiv.org/abs/2501.01005)。FlashInfer能够提供高性能的LLM算子，在多种场景实现了领先的性能表现。FlashInfer建立在[Lesson2中介绍的FlashAttention](llm推理加速lesson2-flashattention)之上，并且也是我们后续重点：**SGLang** 的默认AttentionBackend，因此我将它作为Lesson3的主角。

## 1. LLM部署推理的挑战
LLM推理的核心就是它的Attention。Attention机制读取会读取包含历史内容的KV Cache，并根据当前输入的query计算输出结果。但是，为LLM构建一个高性能的attention支持库具有数个挑战。

**A. LLM推理过程存在多样的Attention运作模式**：
LLM推理过程中包含**Prefill**计算和**Batched Decoding**计算。除此基本情况以外，多个requests共同处理时，还可以应用`prefix-reuse`提升KV Cache效率。更进一步，投机采样（speculative decoding）中所采用的树形解码（tree decoding）也创造了新的attention模式。

**B. LLM推理具有动态的输入形式**：
LLM同一batch内每个request的query长度以及KV Cache不同，且会随着推理变化，因此要求kernel能够动态地适配输入形式来达到最优性能。

**C. Attention库需要支持自定义Attention操作**：
对于不断涌现的与Attention相关的改进和优化（内存相关：paged attention, radix tree；计算相关：针对特定GPU硬件优化的attention；多样的attention改进：multi-query attention，grouped attention，specialized mask，customed attention score等），需要强大的可拓展性。

## 2. FlashInfer的主要贡献
### 2.1 KV Cache存储
> 总结：采用统一的BSR格式，大block存储共享前缀的KV，小block存储独特的KV，从而减少多requests共享前缀的IO成本。

FlashInfer采用统一的Block-Sparse Row（BSR）格式存储KV Cache。在batch并行的情况下，由于所有requests的KV Cache是存储在一起的，因此如果去看每个request对应的KV Cache，是一个稀疏的矩阵。传统的稀疏矩阵保存采用Compressed Sparse Row（CSR）的方式，这种方式虽然节省大量的存储空间，但是每次独立的元素访问会带来很大的IO延迟。而BSR将CSR推广到“块”层面，按照固定block大小进行分配，对于如结构化剪枝以及分块数据，可以大幅减少DRAM访问，是一种硬件效率更高的稀疏矩阵存储格式。
在这种存储结构下，FlashInfer可以更好地实现一个batch推理中的重要功能：共享前缀（prefix-reuse）。其整体思想如下图所示：
<img src="\blog\LLM推理加速Lesson3-FlashInfer\composable_BSR.png" alt="composable BSR" style="zoom:50%;"/>

FlashInfer设计了一种Composable Block-Sparse Matrix，这种方式不再使用单一的格式存储稀疏矩阵，而是采用了多种block-sparse格式。一个batch中，对于共享同一个前缀的requests，它们对应的前缀KV Cache形成了一个子块，我们可以用一个更大的block-sparse matrix来保存，它可以利用更大的shared memory，并且在不同requests之间共享KV Cache。而对于非共享的独特前缀，则依然采用细粒度的单一元素作为block。如此设计，多个requests共享的前缀部分KV Cache只需要从DRAM读取一次到Shared Memory，并且多个requests之间可以共享通信，从而减少IO overhead。

### 2.2 计算抽象层（Compute Abstraction）
FlashInfer设计了特定的CUDA/CUTLASS计算抽象层，来适配前面提到的Sparse/Dense多种存储方式，并且支持任意的Block行列划分以及多种Attention变种。

#### 2.2.1 Global到Shared Memory的数据移动方式
LLM推理时，需要将KV Cache从Global的DRAM移动到Shared Memory中，再送入Tensor Core进行Matrix-Multiply Accumulation的计算。FlashInfer的attention支持任意的block大小，但是block大小不一定能符合Tensor Core的大小，因此需要设计一套专门的数据移动方案，如下图所示：
<img src="\blog\LLM推理加速Lesson3-FlashInfer\global_to_shared_memory.png" alt="global to shared memory" style="zoom:50%;"/>

图中展示了FlashInfer将tiles从BSR/dense KV Cache加载到Shared Memory的过程。稀疏的KV-Cache的访问地址可根据BSR格式的INDICES数组计算得到，而密集的KV-Cache的访问地址可直接根据DRAM地址加offset得到。

#### 2.2.2 Kernel兼容不同的FlashAttention Tile大小
由于不同计算平台/架构的最优FA tile大小各不相同，因此FlashInfer提供了$(1,16,32,64,128)\times (32, 64, 128)$的tile大小，并设计启发式算法根据硬件资源（register + shared memory）& 负载密度（平均query长度），来选取最优的tile形状。

#### 2.2.3 JIT Complier实现Attention变体
FlashInfer设计了CUDA template（tile大小为1）和JIT compiler（更大的tile大小），可以输入不同Attention变体的具体设定（包含QueryTransform，KeyTransform，ValueTransform，OutputTransform，LogitsTransform，LogitsMask），自动生成优化的对应Attention Kernel，方便用户适配自定义的Attention形式。

### 2.3 负载平衡的调度策略
FlashInfer设计了load-balanced的调度策略，目标是最小化Shared Memory的idle时间。调度算法如下图所示：
<img src="\blog\LLM推理加速Lesson3-FlashInfer\scheduling.png" alt="scheduling" style="zoom:50%;"/>

算法的输入是query长度$l_{qo}$，KV-Cache的长度$l_{kv}$，tile大小$T_{q}$，以及Cooperative Thread Array（CTA）。算法根据输入信息计算出合适的KV-Cache分段大小，并将分段不断分配给当前负载最小的CTA进行处理。下图展示了一次处理流程的可视化：
<img src="\blog\LLM推理加速Lesson3-FlashInfer\scheduler_visualize.png" alt="scheduler visualize" style="zoom:50%;"/>

算法将一些长度很长的KV-Cache切分成多个chunks，在输入Attention Kernel计算后通过Contraction Kernel将多个chunks的结果聚合成最终结果。同时，FlashInfer将attention和contraction kernels合并为一个persistent kernel来消除跨kernel的overhead，并保证兼容CUDAGraphs。

# 3. 总结
以上就是FlashInfer的主要设计，它在KV-Cache存储和调度策略上实现了优化，并且提供了更加user-friendly的抽象层定义以及编程界面，适配了一系列LLM服务框架，如vLLM，MLC-Engine以及SGLang，是它们加速的重要基础。了解了FlashInfer的框架细节后，在后续优化时，我们可以更好地则其善者而从之，其不善者而改之。




