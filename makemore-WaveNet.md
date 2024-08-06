
# Related Resources

[Karpathy's Youtube tutorial](https://www.youtube.com/watch?v=t3YJ5hKiMQ0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=7)
[Karpathy's jupyter notebook in Youtube](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part5_cnn1.ipynb)
[Karpathy's google colab notebook](https://colab.research.google.com/drive/1CXVEmCO_7r7WYZGb5qnjfyxTvQa13g5X?usp=sharing#scrollTo=7lDFibCgbHwr)

# Architecture Update

为了进一步改进之前构建的字符级别的语言模型，这一次开始考虑修改一下模型的架构。首先观察下面的模型架构图：

![[Figrure 1 Algorithm description.png]]

我们可以观察到有一些瓶颈存在这个架构里面：
1. $C$ 是一个全局的embedding层，其中的维度大小反应了词表示能力，维度越大显然效果越好，但是同样可能存在一个瓶颈限制了$C$的能力——MLP的第一层。
2. 这是一个三层的Sandwich架构（MLP），网络的深度可能还不够深，所以表达能力有待进一步增强。同时第一层至关重要，我们可以观察到第一层的网络将所有的维度向量压缩了，这可能导致 了数据中的信号被压缩的过于快速，所以可以考虑分层增加网络深度而不是用一个更加宽线性层来进行压缩。

# Code

首先因为更加复杂的网络结构会导致代码更加的复杂，所以一个好的面向对象设计可以大大简化代码结构，同时使得整个构建更加轻松。所以我们首先选择重构我们的代码：

```python
# 重写了网络层和容器，但是和之前的是非常像的

class Linear:

  def __init__(self, fan_in, fan_out, bias=True):
    self.weight = torch.randn((fan_in, fan_out)) / fan_in**0.5 # kaiming init
    self.bias = torch.zeros(fan_out) if bias else None

  def __call__(self, x):
    self.out = x @ self.weight
    if self.bias is not None:
      self.out += self.bias
    return self.out

  def parameters(self):
    return [self.weight] + ([] if self.bias is None else [self.bias])

#------------------------------------------------------------------------------------------------

class BatchNorm1d:

  def __init__(self, dim, eps=1e-5, momentum=0.1):
    # 小批量使用大的momentum
    self.dim = dim
    self.momentum = momentum
    self.eps = eps
    self.training = True # 控制模型的行为
    self.gamma = torch.ones(dim)
    self.beta = torch.zeros(dim)
    self.running_mean = torch.zeros(dim)
    self.running_var = torch.zeros(dim)
  
  def __call__(self, x):
    if self.training:
      if x.ndim == 2:
        dim = 0
      elif x.ndim == 3:
        dim = (0,1)
      xmean = x.mean(dim, keepdim=True)
      xvar = x.var(dim, keepdim=True) # 这里实际上使用了修正偏差
    else:
	  xmean = self.running_mean
	  xvar = self.running_var
    xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
    self.out = self.gamma * xhat + self.beta
    if self.training:
      with torch.no_grad():
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
    return self.out
  
  def parameters(self):
    return [self.gamma, self.beta]

#------------------------------------------------------------------------------------------------

class Tanh:
  
  def __call__(self, x):
    self.out = torch.tanh(x)
    return self.out
  
  def parameters(self):
    return []

#------------------------------------------------------------------------------------------------

class Embedding: 

  def __init__(self, num_embeddings, embedding_dim):
    self.weight = torch.randn((num_embeddings, embedding_dim))

  def __call__(self, x):
    self.out = self.weight[x]
    return self.out
  
  def parameters(self):
    return [self.weight]

#------------------------------------------------------------------------------------------------

class FlattenConsecutive:

  def __init__(self, n):
    self.n = n
  
  def __call__(self, x):
    B, T, C = x.shape
    x = x.view(B, T//self.n, C*self.n) # 扁平化操作，实际上很简单，就是改变了视图
    if x.shape[1] == 1:
      # 相当于直接扁平化了dim1
      x = x.squeeze(1)
    self.out = x
    return self.out
  
  def parameters(self):
    return []

#------------------------------------------------------------------------------------------------

class Sequential:

  def __init__(self, layers):
    self.layers = layers

  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    self.out = x
    return self.out
  
  def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]
```

这里着重说说修改的几个点。

BatchNorm层修改了`mean()`和`var()`的求解方法，因为考虑到输入的向量可能不仅仅是二维的，所以如果不修改之前的代码（`dim=0`）就可能发生意想不到的情况。（因为广播的机制，不会报错，但是代码并没有按照我们的设计那样运行）在新的代码中，引入了一个维度的判断，从而根据不同的输入选择不同的维度参数进行归一化，在torch的框架中其实也是有这个实现。

在这里新添加了两个网络层`Embedding`、`FlattenConseutive`和一个线性网络层容器类`Sequential`。其中`Embedding`层其实就是对$C$的抽象，而`FlattenConsecutive`就是对于之前的扁平化的操作的抽象，不过更加灵活，可以根据维度选择。

然后就可以开始构建神经网络了，其中不妨一边推导shape一边写代码（这是我手写的维度变化）：

$$
\begin{align*}
& \text{[B, 8, 24]} \\
& \text{[B, 4, 48]} \\
& \text{[B, 4, 48]} \quad @ \quad \text{[48, 128]} \quad \rightarrow \quad \text{[B, 4, 128]} \\
& y \quad * \quad \text{[B, 4, 128]} \quad + \quad b \quad \rightarrow \quad \text{[B, 4, 128]} \\
& \text{[B, 2, 256]} \\
& \text{[B, 2, 256]} \quad @ \quad \text{[256, 128]} \quad \rightarrow \quad \text{[B, 2, 128]} \\
& y \quad * \quad \text{[B, 2, 128]} \quad + \quad b \quad \rightarrow \quad \text{[B, 2, 128]} \\
& \text{[B, 1, 256]} \quad \rightarrow \quad \text{[B, 256]} \\
& \text{[B, 256]} \quad @ \quad \text{[256, 128]} \quad \rightarrow \quad \text{[B, 128]} \\
& y \quad * \quad \text{[B, 128]} \quad + \quad b \quad \rightarrow \quad \text{[B, 128]} \\
& \text{[B, 128]} \quad @ \quad \text{[128, 27]} \quad \rightarrow \quad \text{[B, 27]}
\end{align*}
$$

神经网络结构的代码：

```python
# 构建神经网络

n_embd = 24
n_hidden = 128

model = Sequential([
    Embedding(vocab_size, n_embd),
    FlattenConsecutive(2), Linear(2*n_embd, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    FlattenConsecutive(2), Linear(2*n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    FlattenConsecutive(2), Linear(2*n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, vocab_size)
])

with torch.no_grad():
  model.layers[-1].weight *= 0.1 #不要让一开始的损失过大

parameters = model.parameters()
print(sum(p.nelement() for p in parameters))
for p in parameters:
  p.requires_grad = True
```

其中有一个很细节的地方需要说明一下。torch的矩阵乘法具有非常强大的功能：

```python
a = torch.tensor([2, 2]) # [2]
b = torch.tensor([[1,2],[3,4]]) # [2,2]
print(a.shape, b.shape)
c = a @ b # [1, 2] @ [2, 2] = [1, 2] delete the first dimension -> [2]
print(c, c.shape)
d = b @ a # [2, 2] @ [2] = [2, 2] @ [2, 1] delete the 1 dimension -> [2]
print(d, d.shape)
A = torch.randn((3, 4, 5))
B = torch.randn((3, 5, 7))
C = A @ B # [3, 4, 5] @ [3, 5, 7] -> [3, 4, 7]
print(C.shape)
```

详细可以参考文档：[torch.matmul](https://pytorch.org/docs/stable/generated/torch.matmul.html)

接着，开始训练！因为上面做了大量的封装和抽象，所以这里的train代码就会异常简单：

```python
# 训练模型

max_steps = 300000
batch_size = 64
lossi = []

for i in range(max_steps):
  # mini-batch
  ix = torch.randint(0, Xtr.shape[0], (batch_size,))
  Xb, Yb = Xtr[ix], Ytr[ix]
  
  # forward pass
  logits = model(Xb)
  loss = F.cross_entropy(logits, Yb) # input: (N, C), target: (C)

  lossi.append(loss.log10().item())

  # backward pass
  for p in parameters:
    p.grad = None
  loss.backward()

  # update simple SGD
  lr = 0.1 if i < max_steps//2 else 0.01
  for p in parameters:
    p.data += -lr*p.grad
  
  # print track information
  if i % 10000 == 0:
    print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
```

观察一下loss曲线（取平均值），并且在验证集上测试模型性能。

```python
losses = torch.tensor(lossi)
losses = losses.view(-1,100).mean(1) # 100平均值，这样子的曲线更加平滑
plt.plot(losses)

# 关闭训练状态

for layer in model.layers:
  layer.training = False

# evaluate the loss
@torch.no_grad()
def split_loss(split):
  x, y = {
      'train': (Xtr, Ytr),
      'val': (Xdev, Ydev),
      'test': (Xte, Yte)
  }[split]
  logits = model(x)
  loss = F.cross_entropy(logits, y)
  print(split, loss.item())

split_loss('train')
split_loss('val')
```

Karpathy说了最后的数值`1.993`非常难以打破，其实有道理，我把batchsize调到了64，反而最后的性能还下降了，并且无论怎么scale，到后面都非常难以进步。我选择了scale up模型，针对上面分析的瓶颈，我认为其中隐藏层参数和向量维度$C$决定了整个模型的性能，下面是超参数：

```python
n_embd = 30
n_hidden = 256
max_steps = 100000 # 防止过拟合
batchsize = 32
```

最后我们可以得到超过Karpathy笔记本的性能：

```
train 1.782850742340088 
val 1.977455973625183
```
