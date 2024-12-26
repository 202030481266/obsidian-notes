
# Related Link

[Karpathy's youtube tutorial](https://www.youtube.com/watch?v=P6sfmUTpUmc&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=5)
[Karpathy's github notebook](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part3_bn.ipynb)
[Karpathy's google colab notebook](https://colab.research.google.com/drive/1H5CSy-OnisagUgDUXhHwo1ng2pjKHYSN?usp=sharing)
[My google colab notebook for exercise](https://colab.research.google.com/drive/1PD-OVzcZamEqaW0C8ZPI6YA_Cu9_SLsR)
[李宏毅的BN课程讲解](https://www.youtube.com/watch?v=BABPWOkSbLE)
[Batch Norm Explained Visually — How it works, and why neural networks need it](https://towardsdatascience.com/batch-norm-explained-visually-how-it-works-and-why-neural-networks-need-it-b18919692739)
[Batch Norm Explained Visually — Why does it work?](https://towardsdatascience.com/batch-norm-explained-visually-why-does-it-work-90b98bcc58a0)
[Kaiming Paper of Nerual Network Initialization](https://arxiv.org/abs/1502.01852)
[BatchNorm Paper](https://arxiv.org/abs/1502.03167)
[Rethinking in BatchNorm](https://arxiv.org/abs/2105.07576)

# BatchNorm

Batch Normalization 是一种在深度学习中常用的技术，旨在提高训练过程的效率和稳定性。它由Sergey Ioffe和Christian Szegedy在2015年提出，论文名为《Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift》。

## 基本原理

Batch Normalization 的主要思想是在神经网络的每一层输入之前，对输入进行标准化处理。具体来说，对于每一个mini-batch的数据，计算其均值和方差，然后对数据进行标准化，使其均值为0，方差为1。之后，再通过两个可学习的参数（缩放因子$\gamma$和平移因子$\beta$）对标准化后的数据进行线性变换。

## 公式

假设输入数据为 $x$，Batch Normalization 的计算过程如下：

1. 计算mini-batch的均值和方差：
   $$
   \mu_B = \frac{1}{m} \sum_{i=1}^m x_i
   $$
   $$
   \sigma_B^2 = \frac{1}{m} \sum_{i=1}^m (x_i - \mu_B)^2
   $$

2. 标准化数据：
   $$
   \hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
   $$
   其中，$\epsilon$ 是一个很小的常数，用于防止除零错误。

3. 线性变换：
   $$
   y_i = \gamma \hat{x}_i + \beta
   $$
   我们并不想每一次输入都是高斯分布（just for initialization），而是想让网络自己调整。其中，$\gamma$和 $\beta$ 是可学习的参数。==注意这个是逐个元素乘法，不是矩阵乘法==！假设 $\hat{x}$ 是一个形状为 $(n, d)$ 的矩阵，其中 $n$ 是batch size，$d$ 是特征维度。$\gamma$ 和 $\beta$ 是形状为 $(d,)$ 的向量。那么，对于每个样本 $i$ 和每个特征维度 $j$，假设这里的 $\gamma_j$ 和 $\beta_j$ 是向量 $\gamma$ 和 $\beta$ 中的第 $j$ 个元素，则有：
$$
y_{ij} = \gamma_j \cdot \hat{x}_{ij} + \beta_j
$$


### 总结

线性变换步骤中的操作是逐元素的标量乘法和加法，而不是矩阵乘法。这种逐元素的操作确保了每个特征维度可以独立地进行缩放和偏移，从而更好地适应数据的分布。

## 优点

1. **加速收敛**：Batch Normalization 减少了梯度消失和梯度爆炸的问题，使得网络训练更快。
2. **提高模型稳定性**：通过标准化输入，减少了内部协变量偏移（Internal Covariate Shift），使得每一层的输入分布更加稳定。
3. **允许更高的学习率**：Batch Normalization 使得可以使用更高的学习率，进一步加速训练过程。
4. **正则化效果**：Batch Normalization 在一定程度上具有正则化效果，可以减少对Dropout等正则化方法的依赖。

## 缺点

1. **对batch size敏感**：Batch Normalization 的效果依赖于batch size的大小，当batch size较小时，效果可能不佳。
2. **不适合RNN**：在循环神经网络（RNN）中使用Batch Normalization 比较复杂，因为不同时间步的输入分布可能不同。

总的来说，Batch Normalization 是一种**非常有效**的技术，广泛应用于各种深度学习模型中，特别是在卷积神经网络（CNN）和线性层中。

## BN层可视化

就像网络中的任何参数一样，Batch Norm层也有自己的参数。

1. 两个可以学习的参数$\gamma$和$\beta$，用来调节变量的分布。
2. 两个不可学习的参数Moving mean Avg和Moving Var Avg，这两个参数主要是用来进行推理当个sample的时候直接计算。

![[Batch Norm Layer Image.png]]

算法更新的时候也是非常简单，一般来说就是根据原论文给出的公式。

![[Batch Norm Updata Image.png]]

使用Vector的视角看BN层：

![[Batch Norm in Vector View.png]]

由于训练的时候是批量进行训练，这会产生一些很微妙的变化，因为一个sample会影响整个Batch的sample（因为顾名思义嘛，归一化是按照整个Batch进行的）。首先这样做带来的第一个后果是如何进行推理？（因为很多时候你不能强制等到队列有了Batch个sample你才去进行推理），所以这个时候就需要Moving Avg的两个参数（论文中提出的，Pytorch也是这样实现的，**意义就是用这个Moving Avg参数去近似整体数据集的均值和归一化**）来进行Testing：

![[Batch Norm Testing Inference.png]]

BN层放置的顺序是一个争议性问题！Karpathy认为这个不会给神经网络带来非常明显的差别，

![[Batch Norm Layer Sequence.png]]

## Why it works?

这也是一个充满争议的话题，目前来看主要有两种理论支持：**内部协变量偏移**和**损失和梯度平滑**。

### 内部协变量偏移

![[内部协变量偏移例子.png]]

这个术语看起来很吓人，实际上非常简单。它的意思其实就是：==**每一层的网络输入总是存在分布差异**==。（举个例子，你在训练一个图像分类，私家车和军用车的特征分布肯定存在较大差异，但是网络都要接收这种输入，并且输出类别——车）由于这种差异，网络需要更长久的训练时间去收敛。而此时如果使用归一化，就能将这种分布差异大幅减弱，减弱了网络的训练难度，从而更快收敛。

### 损失和梯度平滑


![[Batch Norm Change Error Surface.png | center]]


在典型的神经网络中，“损失景观”并不是光滑的凸面。这里非常崎岖，有陡峭的悬崖和平坦的表面。这给梯度下降带来了挑战——因为它可能会在它认为有希望遵循的方向上突然遇到障碍。为了弥补这一点，==学习率保持较低，以便我们在任何方向上只采取小步==。Batch Norm 的作用是通过改变网络权重的分布来平滑损失（参考上图）。这意味着梯度下降可以自信地朝某个方向迈出一步，因为它知道沿途不会发现突然的中断（change the error suface，更加平滑了）。因此，它可以通过使用更大的学习率来采取更大的步骤，加快神经网络的收敛。

![[Batch Norm make Train Stable and Faster.png]]

## 另外的两个重要点（By Karpathy)

### BN之前的bias大多数情况是无效的

在进行线性层（全连接层）和Batch Normalization（批归一化）操作时，对线性层的输出 $h = XW + b$进行归一化，其中 $b$ （偏置项）对归一化结果没有影响。让我们通过一个具体的例子来详细说明这一点。

假设我们有以下线性层的输入和参数：

- 输入矩阵 $X$ 形状为 $(N, D)$，即 $N$个样本，每个样本有 $D$ 维特征。
- 权重矩阵 $W$ 形状为 $(D, M)$，即将 $D$ 维特征映射到 $M$ 维输出。
- 偏置项 $b$ 形状为 $(M)$，即对每个输出维度都有一个偏置项。

线性层的输出为：

$$h = XW + b$$

接下来对 \( h \) 进行批归一化。批归一化的公式为：

$$
\text{BN}(h) = \gamma \frac{h - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta 
$$

其中：

- $\mu$ 是批均值，计算公式为 $\mu = \frac{1}{N} \sum_{i=1}^N h_i$
- $\sigma^2$ 是批方差，计算公式为 $\sigma^2 = \frac{1}{N} \sum_{i=1}^N (h_i - \mu)^2$
- $\gamma$ 和 $\beta$ 是可学习的缩放和平移参数
- $\epsilon$ 是防止除零的小数值

假设有一个简单的例子：

- 输入矩阵 $X$ 为：

$$
X = \begin{pmatrix}
1 & 2 \\
3 & 4 \\
5 & 6 \\
7 & 8 \\
\end{pmatrix}
$$

- 权重矩阵 $W$ 为：

$$
W = \begin{pmatrix}
0.1 & 0.2 \\
0.3 & 0.4 \\
\end{pmatrix}
$$

- 偏置项 $b$ 为：

$$
b = \begin{pmatrix}
0.5 & 0.5
\end{pmatrix}
$$

首先计算线性层的输出 $h$：

$$
h = XW + b = \begin{pmatrix}
1 & 2 \\
3 & 4 \\
5 & 6 \\
7 & 8 \\
\end{pmatrix} \begin{pmatrix}
0.1 & 0.2 \\
0.3 & 0.4 \\
\end{pmatrix} + \begin{pmatrix}
0.5 & 0.5
\end{pmatrix} = \begin{pmatrix}
1 \times 0.1 + 2 \times 0.3 + 0.5 & 1 \times 0.2 + 2 \times 0.4 + 0.5 \\
3 \times 0.1 + 4 \times 0.3 + 0.5 & 3 \times 0.2 + 4 \times 0.4 + 0.5 \\
5 \times 0.1 + 6 \times 0.3 + 0.5 & 5 \times 0.2 + 6 \times 0.4 + 0.5 \\
7 \times 0.1 + 8 \times 0.3 + 0.5 & 7 \times 0.2 + 8 \times 0.4 + 0.5 \\
\end{pmatrix} = \begin{pmatrix}
1.2 & 2.1 \\
2.6 & 3.7 \\
4.0 & 5.3 \\
5.4 & 6.9 \\
\end{pmatrix}
$$

接下来，对 $h$ 进行批归一化。计算批均值 $\mu$ 和批方差 $\sigma^2$：

$$
\mu = \begin{pmatrix}
\frac{1.2 + 2.6 + 4.0 + 5.4}{4} & \frac{2.1 + 3.7 + 5.3 + 6.9}{4}
\end{pmatrix} = \begin{pmatrix}
3.3 & 4.5
\end{pmatrix}
$$

$$
\sigma^2 = \begin{pmatrix}
\frac{(1.2-3.3)^2 + (2.6-3.3)^2 + (4.0-3.3)^2 + (5.4-3.3)^2}{4} & \frac{(2.1-4.5)^2 + (3.7-4.5)^2 + (5.3-4.5)^2 + (6.9-4.5)^2}{4}
\end{pmatrix} = \begin{pmatrix}
2.5425 & 2.5425
\end{pmatrix} 
$$

对 $h$ 进行批归一化：

$$\text{BN}(h) = \gamma \frac{h - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta $$

对于任意的偏置项 $b$，由于 $h$ 中已经包含了偏置项 $b$，均值和方差的计算都会包含偏置项的贡献。然而，在标准化步骤中，均值 $\mu$ 和方差 $\sigma^2$ 的计算会抵消掉偏置项 $b$ 的影响，使得批归一化后的结果不受偏置项 $b$ 的影响。

例如，当 $b$ 为0时，均值 $\mu$ 和方差 $\sigma^2$ 是由 $XW$ 的输出计算得来的；而当 $b$ 不为0时，均值 $\mu$ 和方差 $\sigma^2$ 仍然会因为包含了偏置项的结果而变化，但在归一化后的结果中，偏置项 $b$ 的影响会被消除。形式化我们可以得到：

$$\mu_B = \frac{1}{m} \sum_{i=1}^m (X_iW + b) = \frac{1}{m} \sum_{i=1}^m X_iW + b = \mu_{XW} + b$$

在标准化步骤中，偏置项 $b$ 会被减去：
$$
\hat{h}_i = \frac{(X_iW + b) - (\mu_{XW} + b)}{\sqrt{\sigma_B^2 + \epsilon}} = \frac{X_iW - \mu_{XW}}{\sqrt{\sigma_B^2 + \epsilon}}
$$

因此，偏置项 $b$ 在标准化过程中被消除了，同理可以类比方差的计算，同样也会被消除，因为方差$\sigma$是由$x-\mu$决定的。

因此，偏置项 $b$ 对于批归一化后的结果没有实际影响。这也是为什么在使用批归一化时，通常会省略线性层中的偏置项 $b$。

### 正则化的影响

Batch Normalization 引入噪声的原因主要与它在训练过程中对小批量数据（batch）进行归一化操作有关。具体来说：

1. **小批量统计量的变化**：
    
    - 在训练过程中，Batch Normalization 在每个小批量数据上计算均值和方差。这些均值和方差会因为不同的小批量数据而有所不同。==因此，对于同一训练样本，归一化后的值会随着不同的小批量而变化。这种变化引入了噪声，类似于数据增强或 Dropout 中的随机性==。（在进行采样的时候是随机性质的，这样就会引入小批量样本中的分布差异）
    
1. **训练和推理的不同**：
    
    - 在训练过程中，Batch Normalization 使用的是小批量数据的均值和方差，而在推理（测试）时，使用的是在整个训练集上累积的全局均值和方差。这种差异进一步增加了训练过程中的噪声，因为训练和推理时的归一化方式不完全相同。

这种噪声有助于防止模型过拟合，因为它迫使模型在训练过程中学习更加鲁棒和广泛的特征，而不是记住训练数据中的特定模式。


# Nerual Network Initialization

虽然Karpathy说了，由于现代的神经网络训练高级技术，比如归一化，Adam优化器和残差连接等等使得现代的深度神经网络更加容易训练，但是我们仍然有必要了解神经网络的初始化的关键性以及学会如何有效调整神经网络的初始化参数。

**在没有高级的神经网络训练技术的时候，微调这些初始化参数是一种权衡的艺术，You need to be very very careful!**。（就是不断测试，然后通过一些工具方法去进行一个半原则判断）Kaiming Init 是一种相对经验主义的方法，即针对不同的激活函数采取一种可以定量计算的公式来进行网络参数的初始化。具体可以看上面的论文链接，也可以参考[Pytorch的初始化方法实现](https://pytorch.org/docs/stable/nn.init.html)。

# Code

这篇文章的重点放在对上一次的代码的重构上，并且进行Pytorch API Style的神经网络编程，加上了神经网络初始化的增益和BatchNorm代码。

首先，`torch.randn`会返回标准的高斯分布，即
$$ \text{out}=\mathcal{N}(0,1)$$
参考了Pytorch的`kaiming_normal_`函数实现，其中对于标准的高斯分布，我们将其转换为标准差等于$std$的高斯分布（本质上是$X'=X\times std$），其中
$$std=\frac{gain}{\sqrt{fan_{in}}}$$

由于我们的激活函数使用$\tanh$ ，所以我们使用的增益$gain=\frac{3}{5}$，这个是Pytorch里面推荐的值。

```python
# refactored code
# pytorch API style

class Linear:

  def __init__(self, fan_in, fan_out, bias=True):
    self.weight = torch.randn((fan_in, fan_out)) / fan_in**0.5  # He init for normal distribution
    self.bias = torch.zeros(fan_out) if bias else None

  def __call__(self, x):
    # forward pass
    self.out = x @ self.weight
    if self.bias is not None:
      self.out += self.bias
    return self.out

  def parameters(self):
    return [self.weight] + ([] if self.bias is None else [self.bias])
```

BatchNorm层的实现参考`torch.nn.BatchNorm1d`的实现，[点击此处查看更加详细的内容](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html#torch.nn.BatchNorm1d)。
实际上大多数对于BatchNorm的实现遵循了论文中给出的定义和方法。

```python
class BatchNorm1d:

  def __init__(self, dim, eps=1e-5, momentum=0.1):

    self.momentum = momentum
    self.eps = eps
    self.training = True # 控制神经网络的行动
    # initailly make normalization
    self.gamma = torch.ones(dim)
    self.beta = torch.zeros(dim)
    # 缓冲区，前向更新，不进行反向传播
    self.running_mean = torch.zeros(dim)
    self.running_var = torch.ones(dim)

  def __call__(self, x):
    if self.training:
      xmean = x.mean(0, keepdim=True) # batch mean
      xvar = x.var(0, keepdim=True)  # batch var
    else:
      xmean = self.running_mean
      xvar = self.running_var
    xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize
    self.out = self.gamma * xhat + self.beta
    # update
    self.running_mean = (1-self.momentum) * self.running_mean + self.momentum * xmean
    self.running_var = (1-self.momentum) * self.running_var + self.momentum * xvar
    return self.out

  def parameters(self):
    return [self.gamma, self.beta]
```

接着我们简单实现一个$\tanh$激活函数层：

```python
class Tanh:

  def __call__(self, x):
    self.out = torch.tanh(x)
    return self.out

  def parameters(self):
    return []
```

接着，就可以像Pytorch那样写神经网络的代码了（真的是太舒服了啊），我们将所有的网络层对象一样一层一层写出来，同时做好网络的参数初始化（**在这里只有添加增益，因为我们用了BatchNorm，其实对于很多初始化设置是不太敏感的**）：

```python
n_embd = 10
n_hidden = 100

C = torch.randn((vocab_size, n_embd))
layers = [
  Linear(n_embd * block_size, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
  Linear(n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
  Linear(n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
  Linear(n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
  Linear(n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
  Linear(n_hidden, vocab_size, bias=False), BatchNorm1d(vocab_size),
]

with torch.no_grad():
  layers[-1].gamma *= 0.1 # 避免损失过大
  for layer in layers[:-1]:
    if isinstance(layer, Linear):
      layer.weight *= 5/3 # 添加gain，tanh函数使用5/3

parameters = [C] + [p for layer in layers for p in layer.parameters()]
print(sum(p.nelement() for p in parameters)) # number of parameters in total
for p in parameters:
  p.requires_grad = True
```

这里Karpathy介绍了一些有效的观察神经网络的工具，==其中的本质就是将神经网络的参数属性打印出来：观察是否存在一个变化的趋势（shrink or diffusion ?)==，如果全部的网络层能够表示出良好的对称性和稳定性，那么就说明是一个相当好的初始化。具体参加colab上的代码。
