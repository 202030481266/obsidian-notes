
# 相关资源

[Karpathy's Youtube tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=7)
[The repository of this lecture](https://github.com/karpathy/ng-video-lecture)
[Official version of this tiny GPT(nanoGPT)](https://github.com/karpathy/nanoGPT)
[Karpathy's google colab notebook](https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing)
[My Google colab notebook](https://colab.research.google.com/drive/19oQtSdXw7cXTpxU2_nFGfUBJ1bwB2FUI?authuser=0#scrollTo=WsY2AfSo2Yz8)

# 代码实现

这个课程包含了很多很多的内容（如果之前没学过Transformer大概率是GG的~），其中最最重要的是各种高级技巧了，包括self-attention的mask如何巧妙实现，以及高级模型训练技术：残差连接，dropout正则化，LayerNorm归一化，AdamW优化器。这些基本上是现代使用的工具了，非常实用有效，但是本篇笔记不会详细了解这些难点，而是单独开一些专题去学习。

## Tokenizer

对于构建语言模型来说第一个步骤就是构建词表，因为机器只能看到数字，所以我们要针对原始的语料文本进行一个Tokenize，将语料划分为Token的形式。成熟的库有OpenAI的Tiktoken，Google的Sentencepiece等等。不过我们构建的是玩具GPT，所以我们只会使用字符级别的tokenizer，也因此非常简单，和之前的makemore系列是一模一样的。

```python
# build the tokenizer

chars = sorted(list(set(text)))
vocab_size = len(chars)

print(''.join(chars))
print(vocab_size)

stoi = {s:i for i,s in enumerate(chars)}
itos = {i:s for i,s in enumerate(chars)}

encode = lambda x: [stoi[c] for c in x]
decode = lambda x: ''.join([itos[c] for c in x])

print(encode("hello world"))
print(decode(encode("hello world")))
```

然后开始构建训练数据集，其中这里GPT和之前的训练数据集有点不一样。对于一个长度为$blocksize$ 的序列$X$而言，它实际上包含了多个训练数据，以`abcdef`为例：

$$
\begin{align*}
X&=\text{abcdef} \\
\text{a} &\rightarrow \text{b} \\
\text{ab} &\rightarrow \text{c} \\
\text{abc} &\rightarrow \text{d} \\
\text{abcd} &\rightarrow \text{e} \\
\text{abcde} &\rightarrow \text{f}
\end{align*}
$$

看到这里大概就可以理解了GPT的生成能力为什么这么厉害？因为哪怕即使给GPT一个字符它也可以一直生成下去，相比makemore系列构建的模型，显然这个模型强太多了。那么这个数据集如何构造呢？实际上还是一个滑动窗口的技巧（之前使用zip函数构建过类似的），详细可以看代码：

```python
# 滑动窗口的数据集构造
batch_size = 4  # 训练数据批量大小
block_size = 8  # 上下文长度

def get_batch(split):
  
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data) - block_size, (batch_size,))
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  return x,y
```

## GPTLanguageModel(decoder only)

首先的话，还是先看看Transformer的架构图，不过由于我们实现的实GPT，所以只会用到其中的Decoder的结构，而对于Encoder则是没有必要。（因为原论文实现的是翻译任务）

![[Transformer架构图.png#pic_center]]

由于Transformer过于经典，很多文章都讲得很好，所以我这里就说说实现过程中或者Karpathy提到的一些很重要的点。

## $\sqrt{d_k}$ 很重要

在Transformer的论文《Attention is All You Need》中，Q（Query），K（Key），V（Value）这三个矩阵之间的Attention计算中确实有一个除以$\sqrt{d_k}$的操作。这个操作的主要目的是为了缓解点积在数值上可能过大的问题，从而稳定梯度和训练过程。

具体来说，Attention的计算公式如下：

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V $$

其中，$d_k$是Query和Key向量的维度。

### 原因分析

1. **点积的数值范围**：
   在没有$\sqrt{d_k}$的情况下，Query和Key向量的点积可能会随着维度的增加而变大。因为点积的期望值与维度成正比。如果维度很大，这些点积值会变得很大，导致softmax函数的输入值也很大。

2. **Softmax的饱和问题**：
   Softmax函数对于输入值的数值范围非常敏感。如果输入值很大，softmax函数的输出会非常接近0或1，这样就会导致梯度消失问题（gradient vanishing），使得模型难以训练。

3. **稳定性和训练效果**：
   通过除以$\sqrt{d_k}$，可以将点积的数值范围缩小到一个较为稳定的范围，从而使得softmax函数的输出更加平滑。这样，Attention机制在训练时更加稳定，梯度传播更加有效，最终提升模型的训练效果和收敛速度。

### 数学解释

假设Query和Key向量中的元素是独立同分布的随机变量，且均值为0，方差为1。那么，Query和Key向量的点积的期望值为0，方差为$d_k$。通过除以$\sqrt{d_k}$，我们将点积标准化，使得结果的方差为1。这种标准化可以防止数值过大导致的梯度问题，并==确保softmax函数的输入值在一个合理的范围==内。Karpathy说了大值会让softmax取到了两边的极端值，这样子会使得输出非常依赖于某一个值，数值分布是“尖”的，但是我们想要的是”平滑“的，特别是在一开始网络初始化的时候。下面是更加详细的数学推导：

#### 点积的期望值和维度成正比

假设Query $Q$ 和 Key $K$ 向量的维度为 $d_k$，且它们的每个元素是独立同分布的随机变量，均值为0，方差为1。那么，我们来看它们的点积：

$$ Q \cdot K = \sum_{i=1}^{d_k} Q_i K_i $$

其中 $Q_i$ 和 $K_i$ 是 $Q$ 和 $K$ 向量的第 $i$ 个元素。

由于 $Q_i$ 和 $K_i$ 均值为0，我们可以计算点积的期望值：

$$ \mathbb{E}[Q \cdot K] = \mathbb{E}\left[\sum_{i=1}^{d_k} Q_i K_i \right] $$

利用期望的线性性质，我们可以将期望分解：

$$ \mathbb{E}[Q \cdot K] = \sum_{i=1}^{d_k} \mathbb{E}[Q_i K_i] $$

因为 $Q_i$ 和 $K_i$ 是独立的随机变量，它们的乘积的期望是两个期望的乘积，而两个期望都是0：

$$ \mathbb{E}[Q_i K_i] = \mathbb{E}[Q_i] \cdot \mathbb{E}[K_i] = 0 \cdot 0 = 0 $$

因此，点积的期望值是0：

$$ \mathbb{E}[Q \cdot K] = 0 $$

#### 点积的方差

我们继续计算点积的方差。方差的计算涉及二阶矩（即平方的期望）。我们首先计算每个乘积的期望的平方：

$$ \text{Var}(Q \cdot K) = \mathbb{E}[(Q \cdot K)^2] - (\mathbb{E}[Q \cdot K])^2 $$

由于 $\mathbb{E}[Q \cdot K] = 0$，方差简化为：

$$ \text{Var}(Q \cdot K) = \mathbb{E}[(Q \cdot K)^2] $$

我们需要计算 $(Q \cdot K)^2$ 的期望值：

$$ (Q \cdot K)^2 = \left(\sum_{i=1}^{d_k} Q_i K_i \right)^2 $$

展开后：

$$ (Q \cdot K)^2 = \sum_{i=1}^{d_k} Q_i^2 K_i^2 + 2 \sum_{1 \le i < j \le d_k} Q_i K_i Q_j K_j $$

由于 $Q_i$ 和 $K_i$ 是独立同分布的随机变量，它们的二次项和交叉项独立：

$$ \mathbb{E}[(Q \cdot K)^2] = \sum_{i=1}^{d_k} \mathbb{E}[Q_i^2 K_i^2] + 2 \sum_{1 \le i < j \le d_k} \mathbb{E}[Q_i K_i Q_j K_j] $$

由于 $Q_i$ 和 $K_i$ 独立且方差为1，有：

$$ \mathbb{E}[Q_i^2] = \text{Var}(Q_i) = 1 $$
$$ \mathbb{E}[K_i^2] = \text{Var}(K_i) = 1 $$

所以：

$$ \mathbb{E}[Q_i^2 K_i^2] = \mathbb{E}[Q_i^2] \mathbb{E}[K_i^2] = 1 \cdot 1 = 1 $$

对于交叉项，由于 $Q_i$ 和 $K_i$ 的期望为0：

$$ \mathbb{E}[Q_i K_i Q_j K_j] = \mathbb{E}[Q_i K_i] \cdot \mathbb{E}[Q_j K_j] = 0 \cdot 0 = 0 $$

因此：

$$ \mathbb{E}[(Q \cdot K)^2] = \sum_{i=1}^{d_k} 1 = d_k $$

最后，点积的方差为 $d_k$：

$$ \text{Var}(Q \cdot K) = d_k $$

#### 为什么除以 $\sqrt{d_k}$

在Attention机制中，点积结果除以 $\sqrt{d_k}$ 是为了将点积的标准差标准化，使其与输入维度无关，从而防止数值过大导致的数值不稳定问题。

通过除以 $\sqrt{d_k}$，我们得到：

$$ \frac{Q \cdot K}{\sqrt{d_k}} $$

此时，该表达式的方差为1，标准化后的值更适合传入softmax函数，避免了数值过大的问题，从而提高了模型训练的稳定性。

## 代码细节

### 推理阶段和训练阶段

在 PyTorch 中，`model.eval()` 和 `model.train()` 是两种不同的模式，它们用于控制模型在训练和推理（测试）时的行为。具体来说，它们会影响模型中的一些层，比如 dropout 和 batch normalization 层。下面详细解释它们的作用：

#### model.train()

- **训练模式**：这是默认模式。在这种模式下，模型会启用 dropout 和 batch normalization 层，以便在训练时应用这些正则化技术。
- **dropout**：在训练模式下，dropout 层会随机地丢弃一些神经元（根据设定的概率），以防止过拟合。
- **batch normalization**：在训练模式下，batch normalization 层会计算当前 batch 的均值和方差，并使用这些统计量来进行标准化。此外，它会更新内部的均值和方差的移动平均值，这些值将在推理阶段使用。

#### model.eval()

- **推理模式**：在这种模式下，模型会禁用 dropout 层和 batch normalization 层的训练行为，以便进行推理或测试。
- **dropout**：在推理模式下，dropout 层将不会丢弃任何神经元（即，所有神经元都会被使用）。
- **batch normalization**：在推理模式下，batch normalization 层会使用在训练过程中计算并保存的均值和方差的移动平均值，而不是当前 batch 的统计量。

#### 具体使用场景

1. **训练模型**：在训练模型时，应调用 `model.train()` 以启用训练模式，从而使 dropout 和 batch normalization 层按照预期工作。

```python
model.train()
for data, target in train_loader:
	optimizer.zero_grad()
	output = model(data)
	loss = criterion(output, target)
	loss.backward()
	optimizer.step()
```

2. **评估模型**：在评估模型时，应调用 `model.eval()` 以启用推理模式，从而使模型在验证集或测试集上进行推理时行为正确。

```python
model.eval()
test_loss = 0
correct = 0
with torch.no_grad():
	for data, target in test_loader:
		output = model(data)
		test_loss += criterion(output, target).item()
		pred = output.argmax(dim=1, keepdim=True)
		correct += pred.eq(target.view_as(pred)).sum().item()
```

通过正确地在训练和推理时使用 `model.train()` 和 `model.eval()`，可以确保模型在不同阶段的表现符合预期，从而得到稳定和可靠的训练和评估结果。

### 完整代码

项目所有文件可以看路径`code/GPT from scratch`。

```python
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 10000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.1

with open('addition_dataset.txt', 'r', encoding='utf-8') as f:
  text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {s:i for i,s in enumerate(chars)}
itos = {i:s for i,s in enumerate(chars)}
encode = lambda x: [stoi[c] for c in x]
decode = lambda x: ''.join([itos[c] for c in x])
n = int(0.9 * len(text))
data = torch.tensor(encode(text), dtype=torch.long)
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data) - block_size, (batch_size,))
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  x, y = x.to(device), y.to(device)
  return x,y

@torch.no_grad()
def estimate_loss():
  out = {}
  model.eval()
  for split in ['train', 'val']:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      X, Y = get_batch(split)
      logits, loss = model(X, Y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  model.train()
  return out

class Head(nn.Module):

  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(n_embd, head_size, bias=False)  # [B, T, C] @ [C, H] -> [B, T, H]
    self.value = nn.Linear(n_embd, head_size, bias=False) # bias = False, 因为后面要LayerNorm
    self.query = nn.Linear(n_embd, head_size, bias=False) 
    
    # torch的缓冲区，不会进行反向传播
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) 
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, x):
    B, T, C = x.shape
    k = self.key(x) # [B, T, H]
    q = self.query(x) # [B, T, H]
    # self-attention
    dk = k.shape[-1]
    # [B, T, H] @ [B, H, T] -> [B, T, T]
    wei = q @ k.transpose(-2, -1) * (dk**-0.5) 
    # mask, 实际上这里可以广播
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # mask the bool matrix where is true
    wei = F.softmax(wei, dim=-1) # [B, T, T], dim是一个必要的参数，对最后一个维度进行softmax
    wei = self.dropout(wei) # dropout，舍弃掉一些参数
    v = self.value(x) # [B, T, H]
    out = wei @ v # [B, T, T] @ [B, T, H] -> [B, T, H]
    return out

class MultiHeadAttention(nn.Module):

  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.proj = nn.Linear(n_embd, n_embd)
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, x):
    # [B, T, H] * num_heads ---cat---> [B, T, H * num_heads] -> [B, T, C]
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    # dropout 正则化加上了一个残差连接
    out = self.dropout(self.proj(out)) # [B, T, C] @ [C, C] -> [B, T, C]
    return out

class FeedFoward(nn.Module):

  def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(n_embd, 4 * n_embd),   # paper 512 -> 2048 -> 512
        nn.ReLU(),
        nn.Linear(4 * n_embd, n_embd),
        nn.Dropout(dropout)
    )
  
  def forward(self, x):
    return self.net(x)  # 可以直接调用Module的方法

class Block(nn.Module):

  def __init__(self, n_embd, n_head):
    super().__init__()
    head_size = n_embd//n_head
    self.sa = MultiHeadAttention(n_head, head_size)
    self.ffwd = FeedFoward(n_embd)

    # LayerNorm 两次
    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)

  def forward(self, x): 
    # 输入前进行归一化，这和论文有点不一样
    x = x + self.sa(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x


class GPTLanguageModel(nn.Module):

  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # [voc, C]
    self.position_embedding_table = nn.Embedding(block_size, n_embd) # [block, C] 
    self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
    self.ln_f = nn.LayerNorm(n_embd)
    self.lm_head = nn.Linear(n_embd, vocab_size)

  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      torch.nn.init.normal_(module.weight, mean=0.0,std=0.02)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def forward(self, idx, targets=None):
    B, T = idx.shape
    # embedding
    # [B, T] -> [B, T, C], [*] ---embedding---> [*,C]
    tok_emb = self.token_embedding_table(idx)
    # [T] -> [T, C]
    pos_emb = self.position_embedding_table(torch.arange(T, device=device)) 
    x = tok_emb + pos_emb # [B, T, C]
    x = self.blocks(x) # [B, T, C]
    x = self.ln_f(x) # [B, T, C]
    logits = self.lm_head(x) # [B, T, C] @ [B, C, voc] -> [B, T, voc]

    if targets is None:
      loss = None # infer stage
    else:
      B, T, voc = logits.shape
      logits = logits.view(B*T, voc)
      targets = targets.view(B*T) # [B, T]
      loss = F.cross_entropy(logits, targets)

    return logits, loss

  def generate(self, idx, max_new_token):
    for _ in range(max_new_token):
      # 我们这里没有special_token，所以不会自动停止
      idx_second = idx[:, -block_size:] # 截断 [B, T]
      logits, loss = self(idx_second) # [B, T, voc] 因为是推理阶段
      logits = logits[:, -1, :] # [B, voc]
      probs = F.softmax(logits, dim=-1) # [B, voc]
      idx_next = torch.multinomial(probs, num_samples=1) # [B, voc]
      idx = torch.cat([idx, idx_next], dim=1) # [B, T + 1]
    return idx

def eval_generate():
    context = torch.zeros((1,1), dtype=torch.long, device=device)
    res = decode(m.generate(context, max_new_token=50)[0].tolist())
    with open('result.txt', 'a+', encoding='utf-8') as f:
        f.write(res)

model = GPTLanguageModel()
m = model.to(device)
eval_generate() # 没有训练

# 使用AdamW优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

  if iter % eval_interval == 0 or iter == max_iters-1:
    losses = estimate_loss()
    print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
  
  xb, yb = get_batch('train')
  logits, loss = model(xb, yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

eval_generate() # 经过训练
```

