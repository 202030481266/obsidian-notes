# Related Links
[Karpaphy's jupyter lab code](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part1_bigrams.ipynb)
[Karpaphy's lecture on Youtube](https://www.youtube.com/watch?v=PaCmpygFfXo)
[makemore/names.txt at master · karpathy/makemore (github.com)](https://github.com/karpathy/makemore/blob/master/names.txt)
[Shulin's colab notebook](https://drive.google.com/drive/folders/1dlmTg-rbVvrxN2dc3eUDIvCs_Cns9DPB)
# Jupyter Lab Code

我们在Google Colab完全复现这个二元模型（字符级别的二元语言模型，预测每一个字符的时候都只会关注前一个字符是什么）。

```python
import torch
import requests

# 下载数据集
url = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
response = requests.get(url)
if response.status_code == 200:
    with open("names.txt", "w") as file:
        file.write(response.text)  
    print("File 'names.txt' has been downloaded successfully.")
else:
    print(f"Failed to download the file. Status code: {response.status_code}") 
```

统计信息函数，查看其中的一些二元组信息。（得到一个初步的感知）

```python
def describe_dataset(words):
  # 显示数据集的统计信息
  b = {}
  for w in words:
    chs = ['<S>'] + list(w) + ['<E>'] # 添加起始和终止的token
    for c1, c2 in zip(chs, chs[1:]):
      bigram = (c1, c2)
      b[bigram] = b.get(bigram, 0) + 1
  mx = sorted(b.items(), key = lambda kv : -kv[1]) # 按照出现的次数从大到小排序
  print(mx[:10])
```

对于每一个英文字符使用一个索引作为编码，然后使用`.`作为special token，也就是开始的点和结束的点，因此我们可以进行一个统计得到所有的二元字符组数量，并且做一个可视化。

```python
# 第一种训练模型的方法，直接从数据集中进行统计（统计频率）

import matplotlib.pyplot as plt
%matplotlib inline

def count_matrix(words):
  N = torch.zeros((27, 27), dtype=torch.int32) # 统计频率表
  chars = list(set(''.join(words))) # 去重
  # 实际上，我们正在构建 tokenizer
  stoi = {ch : i + 1 for i, ch in enumerate(chars)}
  itos = {i + 1 : ch for i, ch in enumerate(chars)}
  stoi['.'] = 0 # special token
  itos[0] = '.'

  # 计数
  for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
      ix1 = stoi[ch1]
      ix2 = stoi[ch2]
      N[ix1, ix2] += 1
  
  # 可视化
  plt.figure(figsize=(16, 16))
  plt.imshow(N, cmap='Blues')

  for i in range(27):
    for j in range(27):
      chstr = itos[i] + itos[j]
      plt.text(j, i, chstr, ha='center', va='bottom', color='gray')
      plt.text(j, i, N[i, j].item(), ha='center', va='top', color='gray')
  plt.axis('off')
  plt.show()
  return N
```

下面就是训练模型的一般步骤，有几个关键的点：
1. `torch.Generator().manual_seed()` 用来手动设置随机生成数的种子，可以保证数字的一致性，一般用来测试不同的代码是否具有相同的输入和输出。
2. `sum()` 函数的计算属于聚合运算，并且要保持维度，不然后面因为广播机制会导致错误。
3. `torch.multinomial()` 用来根据给定的向量生成符合其中概率分布的索引（向量之和不一定等于1），执行次数越多分布就越符合。([details for multinomial](https://pytorch.org/docs/stable/generated/torch.multinomial.html))

```python
# normalize
p = (N+1).float()  # 让模型更加smooth，加1是为了防止爆0
p /= p.sum(1, keepdim=True) # keepdim必须是True，不然就是广播机制
g = torch.Generator().manual_seed(2147483647)  # 手动设置随机数种子
for i in range(5):
  out = []
  ix = 0
  while True:
    c = p[ix]
    # c = torch.rand((1, 27)) 如果想看对比的概率分布
    ix = torch.multinomial(c, num_samples=1, replacement=True, generator=g).item()
    out.append(itos[ix])
    if ix == 0:
      break
  print(''.join(out))
```

Karpathy也提到了这个二元模型实在是垃圾到难以置信，但是总是比随机的概率分布产生的单词好得多！

下面是基于神经网络+反向传播算法实现的模型。首先介绍最大似然函数的一些关键概念（几百年前学过了，忘得都差不多了）。

对数似然函数的数学原理基于概率论和统计学中的似然原理。假设我们有一个数据集 $$\mathbf{X} = \{x_1, x_2, \ldots, x_n\}$$，并且我们有一个参数化的概率模型 $p(x \mid \theta)$，其中 $\theta$ 是模型参数。

**似然函数** $L(\theta \mid \mathbf{X})$ 定义为给定数据集 $\mathbf{X}$ 下，参数 $\theta$ 的概率：

$$ L(\theta \mid \mathbf{X}) = \prod_{i=1}^n p(x_i \mid \theta) $$

这里，似然函数是所有单个数据点概率的乘积，因为假设每个数据点是独立同分布的。

**对数似然函数** $\log L(\theta \mid \mathbf{X})$ 是对似然函数取自然对数：

$$ \log L(\theta \mid \mathbf{X}) = \sum_{i=1}^n \log p(x_i \mid \theta) $$

取对数的好处是将乘法转换为加法，这在计算上更稳定，也更容易处理。

**最大对数似然估计**（Maximum Log-Likelihood Estimation, MLE）是找到使得对数似然函数最大的参数 $\theta$：

$$ \hat{\theta}_{\text{MLE}} = \arg \max_{\theta} \log L(\theta \mid \mathbf{X}) $$

在实际应用中，我们通常通过求解对数似然函数的梯度等于零的方程来找到这个最大值：

$$ \frac{\partial \log L(\theta \mid \mathbf{X})}{\partial \theta} = 0 $$

这个方程通常是非线性的，可能需要使用数值方法来求解。在深度学习中，通常将这个对数似然作为一个损失函数，但是深度学习中一般是**最小化损失**，所以上面的最大化对数似然等价于最小化**负对数似然函数** $-\log L(\theta \mid \mathbf{X})$ ，其实就是对数似然函数的负值：

$$-\log L(\theta \mid \mathbf{X}) = -\sum_{i=1}^n \log p(x_i \mid \theta)$$

$$\hat{\theta}_{\text{MLE}} = \arg \min_{\theta} -\log L(\theta \mid \mathbf{X}) $$
简单计算一下数据集中的整体的负数对数似然损失：

```python
log_likelihood = 0.0
n = 0

for w in words:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    prob = p[ix1, ix2]
    logprob = torch.log(prob)
    log_likelihood += logprob
    n += 1

print(f'{log_likelihood=}')
nll = -log_likelihood
print(f'{nll=}')
print(f'{nll/n}')
```

设计神经网络的架构，由于我们的输入是二元组$(x_i,y_i)$这种形式，不能直接应用到神经网络。一个常见的手段是使用`one-hot`编码。对于每一个$x_i$，我们将其编码成为

$$x=[0,0,0,1,0,0,...,0]$$
这种形式的向量。我们假设数据集的批次是$B$，那么输入就是$B\times27$的矩阵。同样的我们输出也应该是这个大小的矩阵。Karpathy大神用了最简单的线性回归方法，就是一层网络的感知机模型：

$$Y=XW$$
其中$Y,X\in\mathbb{R}^{B\times27}$，而其中的$W\in \mathbb{R}^{27\times27}$ 。思考一下参数矩阵里面的每一个元素到底是什么意思呢？实际上，参数矩阵反应的意义就是$N$ ，代码中的那个计数统计表。（可以动手算算）

```python
# 构造数据集并且进行训练
import torch.nn.functional as F

xs, ys = [], []
for w in words:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    xs.append(ix1)
    ys.append(ix2)
xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()
print(f'the number of this dataset is {num}')

g = torch.Generator().manual_seed(2147483647)
W = torch.rand((27, 27), generator=g, requires_grad=True) # 别忘了加上梯度

for k in range(200):
  
  # forward pass
  xenc = F.one_hot(xs, num_classes=27).float() # one-hot encode
  logits = xenc @ W # predict, [num, 27]
  counts = logits.exp() # soft-max [num, 27]
  probs = counts / counts.sum(1, keepdim=True) # normalize
  loss = -probs[torch.arange(num), ys].log().mean() + 0.01 * (W**2).mean()
  if k % 10 == 0:
    print(f'{loss.item()}')

  # backward pass
  W.grad = None  # 梯度清零
  loss.backward()

  # update
  W.data += -10 * W.grad
```

代码里面其实还包含了一个正则化损失`W**2` ，这个意思就是最后得到的参数会更加均匀，并且让数字趋向于0，这就能够让训练更加稳定。

最后，看看我们的模型成果！🔥🔥🔥

```python
for i in range(5):
  
  out = []
  ix = 0
  while True:
    
    # ----------
    # BEFORE:
    #p = P[ix]
    # ----------
    # NOW:
    xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
    logits = xenc @ W # predict log-counts
    counts = logits.exp() # counts, equivalent to N
    p = counts / counts.sum(1, keepdims=True) # probabilities for next character
    # ----------
    
    ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
    out.append(itos[ix])
    if ix == 0:
      break
  print(''.join(out))
```

