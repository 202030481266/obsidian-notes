# Related Links

[Karpathy's tutorial youtube](https://www.youtube.com/watch?v=TCH_1BHY58I&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=5)
[Karpathy's google colab notebook](https://colab.research.google.com/drive/1YIfmkftLrz6MPTOO9Vwqrop2Q5llHIGK?usp=sharing)
[Karpathy's github notebooks](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part2_mlp.ipynb)
[The Ref Paper: A Nerual Probabilistic Model](obsidian://open?vault=obsidian-notes&file=assets%2FA%20Neural%20Probabilistic%20Language%20Model.pdf)
[The Ref Pytorch internals Blog](http://blog.ezyang.com/2019/05/pytorch-internals/)

# Paper Reading Reflection

## Two Challenges

[[A Neural Probabilistic Language Model.pdf#page=3&selection=15,29,62,18|A Neural Probabilistic Language Model, 页面 3]]

两个当时看来非常重要的问题（实际上这篇文章是Word2Vec的开山论文）：
1. 没有处理更加长的上下文。（因为维度诅咒的原因，维度诅咒指的是序列的表达空间增长速度是指数型的，比如随着序列长度的增加，不同的序列种类数是非常多的）
2. 没有考虑句子的语法结构和语义信息。

## Smooth Probability Function

> The proposed model, it will so generalize because “similar” words are expected to have a similar feature vector, and because the probability function is a smooth function of these feature values, a small change in the features will induce a small change in the probability. Therefore, the presence of only one of the above sentences in the training data will in- crease the probability, not only of that sentence, but also of its combinatorial number of “neighbors” in sentence space (as represented by sequences of feature vectors).

这是论文中一个很重要但是不好理解的一段话。首先可以看到这里面讲了主要的几件事情：
- **特征向量的相似性**： 每个词都被表示为一个特征向量。相似的词（在语义或语法上）会有相似的特征向量。这就创造了一种"聚类"效应，但它是在**连续空间**中的，而不是离散的类别。
- 概率函数的平滑性： 模型使用这些特征向量来计算词序列的概率。这个概率函数被设计成对特征向量的变化是平滑的，==这意味着特征向量的小变化只会导致概率的小变化==。
- 泛化能力： 由于上述两个特性，模型能够泛化到训练中没有见过的句子。例如，如果模型在训练中看到了"The cat is walking in the bedroom"，它不仅会增加这个确切句子的概率，还会增加类似句子的概率，如"A dog was running in a room"，因为这些词的特征向量是相似的。
- 组合式泛化： 这种方法允许模型进行"组合式"的泛化。它可以将学到的关于单个词的知识组合起来，形成对新句子的预测，即使这些新句子的确切组合从未在训练中出现过。
- 克服维度诅咒： 这种方法有助于克服语言建模中的"维度诅咒"问题。在传统的n-gram模型中，我们需要看到每个特定的词序列多次才能做出好的预测。但在这个模型中，每个训练样本能够影响大量相关的词序列的概率估计。

论文中提到概率函数是"平滑的"，==它指的是概率函数对输入的小变化会产生小的、连续的变化，而不是突变==。让我通过一个例子来解释这个概念：

假设我们有一个简化的模型，其中每个词只有两个特征。我们可以将词的特征向量visualize在一个二维平面上。

例子：

1. "cat" 的特征向量可能是 $[0.9, 0.2]$
2. "dog" 的特征向量可能是 $[0.8, 0.3]$
3. "tiger" 的特征向量可能是 $[0.85, 0.15]$

现在，假设我们的概率函数$f(x, y)$计算一个词在某个上下文中出现的概率，其中$x$和$y$是词的两个特征值。

如果这个函数是平滑的，那么：

1. $f(0.9, 0.2) ≈ f(0.8, 0.3) ≈ f(0.85, 0.15)$

这意味着"cat"，"dog"和"tiger"在类似的上下文中出现的概率会相近，因为它们的特征向量相似。

2. 如果我们稍微改变输入，比如从$[0.9, 0.2]$变成$[0.91, 0.21]$，f的输出也只会有很小的变化。

平滑函数的特性：

1. 连续性：没有突然的跳跃或断点。
2. 可微性：在每一点都有定义良好的斜率。
3. 对输入的小变化敏感：能够反映输入的细微差别。

***在神经网络中，这种平滑性主要通过使用连续的激活函数（如sigmoid或tanh）来实现。这些函数可以将输入空间平滑地映射到输出空间***。

这种平滑性对模型的泛化能力至关重要，因为它允许模型在处理未见过的输入时做出合理的预测，只要这些新输入与训练数据中的例子相似。
# Code

[Google Drive's link](https://drive.google.com/drive/folders/1dlmTg-rbVvrxN2dc3eUDIvCs_Cns9DPB)

首先需要获取其中的数据集文件以及导入相对应的环境：

```python
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

!wget https://raw.githubusercontent.com/karpathy/makemore/master/names.txt # 下载数据集
```

然后开始构建数据集，使用的方法和[[makemore-bigram]]中的方法基本上是一致的，但是我们这里还要进行一个验证集和训练集、开发集的划分，这是为了防止过拟合或者测试欠拟合的。

```python
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

block_size = 5 # 上下文长度，我这里使用 5-gram

def build_dataset(words):
  X, Y = [], []
  for w in words:
    context = [0] * block_size
    for ch in w + '.':
      ix = stoi[ch]
      X.append(context)
      Y.append(ix)
      context = context[1:] + [ix]

  X = torch.tensor(X)
  Y = torch.tensor(Y)
  print(X.shape, X.dtype, Y.shape, Y.dtype)
  return X, Y

import random
random.shuffle(words)

n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])
```

下面开始构建神经网络，我们按照Karpathy的思路，就是使用论文中的MLP架构：

![[Figrure 1 Algorithm description.png]]

我们这里的共线参数（特征向量集合）$C$的大小就是$[27,10]$，其中特征向量维度是$10$，增强表达能力。第一层隐藏层参数设置为$300$个，增强模型的表达能力。论文使用的上下文是$3$，但是我这边使用$5$作为上下文，毕竟数据集还是有比较长的序列。那么就可以推导出我们的网络：

$$
h_1=XW_1+b_1
$$
$$
h_2=h_1W_2+b_2
$$
其中$X\in \mathbb{R}^{B\times 50},W_1\in \mathbb{R}^{50\times 300},W_2\in \mathbb{R}^{300\times 27}$，最后的一层输出用于归一化（softmax算法）+ 对数似然计算，不过这里采用了更加高效和简介的`Pytorch`的函数`cross_entropy`，这个函数的计算相当高效，其中主要有几个原因：（参考Karpathy和Ezyang的博客）
1. 数学的聚合计算优化。即使不同的数学的表达式可以进行聚合，化简得到一个相当简单的形式，从而大幅减少计算量。
2. 不需要copy新的内存空间，这对于大的向量来说非常有效。

![[pytorch cross_entropy function parameters.png]]

学习率的选取实际上也是一个很大的学问，==工程上一般会选择网格搜索——设置很多不同组的学习率、隐藏层参数的数量等等超参数，然后在小步骤数上进行一个实验，观察模型的能力变化==。这里的学习率可以简单的用一个逐渐递增的方法找到一个最合适的位置，因为学习率在很小的时候对于LOSS整体很小，逐渐增大，LOSS开始减少，但是会存在一个阈值，此时LOSS恰好不在减少反而开始上升，此时说明学习率过大开始震荡甚至开始大幅增加，此时的阈值往往是最佳的学习率。下面的实验我的设置$batchsize=64, lr=0.1$，应该是不错的一个配置了。

```python
C = torch.randn((27, 10))
W1 = torch.randn((50, 300))
b1 = torch.randn(300) # 广播
W2 = torch.randn((300, 27))
b2 = torch.randn(27)

parameters = [C, W1, b1, W2, b2]

print(sum(p.nelement() for p in parameters))
steps, losses = [], []
for p in parameters:
  p.requires_grad = True # 需要显式声明

for i in range(400000):
  # minibatch，我们需要随机一些样本进行梯度计算
  # torch.randint(low, high, size: tuple)
  ix = torch.randint(0, Xtr.shape[0], (64,))

  # forward pass
  emb = C[Xtr[ix]] # 64 * 5 * 10，相当于X[ix]中每一个数字做了一次emb
  h = torch.tanh(emb.view(-1, 50) @ W1 + b1)
  logits = h @ W2 + b2
  loss = F.cross_entropy(logits, Ytr[ix])

  # backward
  for p in parameters:
    p.grad = None
  loss.backward()
  
  lr = 0.1 if i < 200000 else 0.01
  # update
  for p in parameters:
    p.data += -0.1 * p.grad
  steps.append(i)
  losses.append(loss.log10().item()) # 更加平滑的表示
```

最后的就是评测数据集的开发集和验证集上的损失，以及推理效果！！！👍👍👍👍👍

```python
plt.plot(steps, losses)

emb = C[Xdev]
h = torch.tanh(emb.view(-1, 50) @ W1 + b1)
logits = h @ W2 + b2
loss1 = F.cross_entropy(logits, Ydev)
emb = C[Xdev]
h = torch.tanh(emb.view(-1, 50) @ W1 + b1)
logits = h @ W2 + b2
loss2 = F.cross_entropy(logits, Ydev)

print(loss1.item(), loss2.item())

# 测试模型的效果

for _ in range(20):
  out = []
  context = [0] * block_size
  while True:
    emb = C[torch.tensor(context)] # b * 10
    h = torch.tanh(emb.view(1, -1) @ W1 + b1)
    logits = h @ W2 + b2 # 1 * 27
    probs = F.softmax(logits, dim=1) # 1 * 27
    ix = torch.multinomial(probs, num_samples=1).item()
    context = context[1:] + [ix]
    out.append(ix)
    if ix == 0:
      break
  print(''.join(itos[i] for i in out))
```

最后的结果显然比[[makemore-bigram]]中的结果好得多，因为参数量上来了，模型的拟合能力大大增强，同时使用了验证集和开发集、训练集拆分的方法来防止过拟合。