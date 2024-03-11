---
title: "Google Jukebox 项目复现"
description: "OpenAI 的项目 Jukebox 所实现的功能是：输入艺术家、曲风以及歌词，生成一首歌曲。在本周，笔者成功跑通了 Sample 部分的代码来生成音乐。写下这篇博文主要是大致记录整个过程，以及踩过的坑，希望能够帮助到有相同需求的朋友。"
pubDate: "Jul 24 2020"
layout: "/src/layouts/MarkdownLyaout.astro"
heroImage: "/src/content/blog/Jukebox/jukebox.png"
tags: ["生成模型", "音乐"]
---

OpenAI 的项目 Jukebox 所实现的功能是：输入艺术家、曲风以及歌词，生成一首歌曲。在本周，笔者成功跑通了 Sample 部分的代码来生成音乐。写下这篇博文主要是大致记录整个过程，以及踩过的坑，希望能够帮助到有相同需求的朋友。

> 项目地址：[https://github.com/openai/jukebox](https://github.com/openai/jukebox)

## 基本要求

### Linux 系统

GitHub上的 sample.py 代码在 Linux 系统可以成功跑通，而在 Windows 系统则会报错（Windows 报错的原因和解决办法我们在最后会提到，感兴趣的朋友可以参考）。因此更加推荐使用 Linux 系统。此外，使用 Linux 虚拟机也无法跑通，因为虚拟机无法使用 GPU，而代码中使用了 GPU。因此使用 Linux 系统的电脑或者服务器是最佳选择。

### GPU

笔者使用的是 2080Ti。Jukebox 项目本身使用的GPU 是V100。在项目的 README 中提到了所采用的三个模型对 GPU 显存的需求如下：

| 模型      | 显存需求 |
| --------- | -------- |
| 1b_lyrics | 3.8GB    |
| 5b        | 10.3GB   |
| 5b_lyrics | 11.5GB   |

根据我的个人经验，使用 2080Ti 进行 Sample 时，虽然显存有 11GB（>10.3GB），但是用 5b 跑代码还是会报错 CUDA out of memory，因此最后还是选择了 1b_lyrics 模型。所以一个更加宽裕的显存大小还是有必要的，如 V100 具有 16GB 显存，可以很好的满足所有模型的需求。



## 环境搭建

这部分按照 Jukebox 项目的 README 就可以完成。为了完整性，这里笔者复述一遍。

首先，我们需要安装 conda 包管理工具，相关教程这里不再赘述。

下面，在命令行运行下列代码进行环境搭建。

```shell
conda create --name jukebox python=3.7.5
conda activate jukebox
conda install mpi4py=3.0.3 # if this fails, try: pip install mpi4py==3.0.3
conda install pytorch=1.4 torchvision=0.5 cudatoolkit=10.0 -c pytorch
git clone https://github.com/openai/jukebox.git
cd jukebox
pip install -r requirements.txt
pip install -e .
```

注意事项：

- git clone [远程仓库] [本地目录] 可以指定克隆位置；

- 如果 conda 下载太慢，推荐使用北外 conda 镜像（目前感受是最快的）

> [https://mirrors.bfsu.edu.cn/help/anaconda/](https://mirrors.bfsu.edu.cn/help/anaconda/)



## 运行 Sampling

### 初次采样

如果是初次采样，在命令行运行以下代码：

```bash
python jukebox/sample.py --model=5b_lyrics --name=sample_5b --levels=3 --sample_length_in_seconds=20 --total_sample_length_in_seconds=180 --sr=44100 --n_samples=6 --hop_fraction=0.5,0.5,0.125
```

我们这里讲一下几个相对重要的参数含义：

| 参数名                   | 作用                                                         |
| ------------------------ | ------------------------------------------------------------ |
| model                    | 根据 GPU 显存以及个人需求选择使用的模型，可以是 1b_lyrics、5b 或 5b_lyrics |
| name                     | 本次采样的名称，采样结果会放在 jukebox/{name} 文件夹中       |
| sample_length_in_seconds | 最后采样生成的歌曲长度，单位为秒                             |
| n_samples                | 一次采样运行所生成的歌曲数量                                 |

#### 注意事项

- 生成的结果中 level_0 是最终结果，从音质上也可以听得出 level_0 比 level_1 和 level_2 更佳；

- 在运行 sample.py 的过程中，会下载对应的模型到项目同级的 .cache 文件夹中，文件在 13~14GB 左右，需要保证网络通畅与充足的空间。下载这一步也是导致 Windows 端运行报错的原因，下文会提到。

### 继续采样

如果想基于之前的采样结果继续采样，来延长歌曲的长度，在命令行运行下列代码：

```shell
python jukebox/sample.py --model=5b_lyrics --name=sample_5b --levels=3 --mode=continue --codes_file=sample_5b/level_0/data.pth.tar --sample_length_in_seconds=40 --total_sample_length_in_seconds=180 --sr=44100 --n_samples=6 --hop_fraction=0.5,0.5,0.125
```

| 参数                     | 作用                                                 |
| ------------------------ | ---------------------------------------------------- |
| --codes_file             | 作为基础的代码的路径，在它的基础上继续采样           |
| sample_length_in_seconds | 新的生成歌曲的长度，注意这里是总长度，不是增加的长度 |



## 运行结果

运行结果会保存在 jukebox/{name} 文件夹下，用浏览器打开 index.html 就可以浏览生成的结果。



**艺术家**：Alan Jackson	**曲风**：Country 
<iframe width=200 height=60 src="\src\content\blog\Jukebox\item_0.wav" sandbox></iframe>

**歌词**：

I met a traveler from an antique land,

Who said Two vast and trunkless legs of stone.

Stand in the desert. . . . Near them, on the sand,

Half sunk a shattered visage lies, whose frown,

And wrinkled lip, and sneer of cold command



**艺术家**：Joe Bonamassa	**曲风**：Blues Rock 
<iframe width=200 height=60 src="\src\content\blog\Jukebox\item_1.wav" sandbox></iframe>

**歌词**： 

It's Christmas time, and you know what that means,

Ohh, it's hot tub time!

As I light the tree, this year we'll be in a tub,

Ohh, it's hot tub time!


**艺术家**：Céline Dion	**曲风**：Pop 
<iframe width=200 height=60 src="\src\content\blog\Jukebox\item_2.wav" sandbox></iframe>

**歌词**： 

Don't you know it's gonna be alright

Let the darkness fade away

And you, you gotta feel the same

Let the fire burn

Just as long as I am there

I'll be there in your night

I'll be there when the condition's right



## 避坑指南

### 选择空闲的GPU

在实验室使用共享的服务器时，提前在命令行输入 nvidia-smi 来观察一下服务器的使用情况，选择空闲的 GPU；

### 应对 CUDA out of memory 报错

- 如果使用的是共享的服务器，首先查看当前 GPU 是否有人正在使用；

- 在 sample.py 中降低 max_batch_size 的大小，最低为1;

- 在运行命令的参数中，--model 选择 1b_lyrics，降低 --n_samples 的大小。

### 在 Windows 端运行

在 Windows 端运行会报错 FileNotFoundError，原因可能是一条 Download 指令在 Linux 端是可用的，但是在 Windows 端是错误的，因此在 Windows 端没有将模型下载到 .cache 文件夹中。我的解决办法就是自己手动地去下载所需要的模型文件（参考jukebox:issue）。步骤如下：

- 安装 wget，教程此处不再赘述；

- 在命令行中运行下列命令，注意，下面的 .cache 文件路径可能需要改成你自己的路径。

```shell
wget http://storage.googleapis.com/jukebox-assets/models/5b/vqvae.pth.tar --no-check-certificate -O C:/Users/<you>/.cache/jukebox-assets/models/5b/vqvae.pth.tar

wget http://storage.googleapis.com/jukebox-assets/models/5b/prior_level_0.pth.tar --no-check-certificate -O C:/Users/<you>/.cache/jukebox-assets/models/5b/prior_level_0.pth.tar

wget http://storage.googleapis.com/jukebox-assets/models/5b/prior_level_1.pth.tar --no-check-certificate -O C:/Users/<you>/.cache/jukebox-assets/models/5b/prior_level_1.pth.tar
```

如果你想要使用 5b_lyrics 模型，运行：

```shell
wget http://storage.googleapis.com/jukebox-assets/models/5b/prior_level_2.pth.tar --no-check-certificate -O C:/Users/<you>/.cache/jukebox-assets/models/5b/prior_level_2.pth.tar
```

如果你想要使用 1b_lyrics 模型，运行：

```shell
wget http://storage.googleapis.com/jukebox-assets/models/1b/prior_level_2.pth.tar --no-check-certificate -O C:/Users/<you>/.cache/jukebox-assets/models/1b/prior_level_2.pth.tar
```

这样应该就可以跑通 sample.py 的代码了，笔者没有进行完整的验证，因为文件实在太大了 QAQ。



## 总结

以上就是笔者跑通 Jukebox 的Sampling 部分代码的经验，希望能够帮助到各位朋友们。