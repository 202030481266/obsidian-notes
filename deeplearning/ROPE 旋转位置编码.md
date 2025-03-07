# 推荐阅读

ROPE作者：[Transformer升级之路：10、RoPE是一种β进制编码](https://kexue.fm/archives/9675)
解读好文1：[RoPE 相对位置编码解读与外推性研究](https://blog.csdn.net/weixin_43378396/article/details/138977299)
解读好文2：[RoPE + 位置线性内插](https://blog.csdn.net/weixin_43378396/article/details/139010288)
解读好文3：[十分钟读懂旋转编码（RoPE）](https://zhuanlan.zhihu.com/p/647109286)
讲解视频：[你还不懂旋转位置编码吗？](https://www.bilibili.com/video/BV1F1421B7iv/)
拓展阅读：[Extending the RoPE](https://blog.eleuther.ai/yarn/)

---
# ROPE概念和介绍

### 1. ROPE 解决了什么问题？

在 Transformer 模型中，位置编码（Positional Encoding, PE）是一个关键部分，因为 Transformer 的自注意力机制本身不具备序列顺序信息（不像 RNN 或 LSTM）。传统的 Transformer（如 BERT）使用固定的正弦-余弦函数作为位置编码，这种方法虽然有效，但存在一些局限性：

1. **缺乏相对位置信息**：传统的位置编码是绝对的，无法很好地捕捉 token 之间的相对位置关系，而在自然语言处理中，相对位置往往比绝对位置更重要。
2. **扩展性问题**：固定的位置编码通常是为特定序列长度设计的，如果输入序列长度超过预设值，模型需要额外的处理（如截断或插值），这可能影响性能。
3. **动态性不足**：传统的 PE 是静态的，不能很好地适应不同的任务或动态调整。

ROPE 的提出（最早由 Su 等人在 2021 年的论文《RoFormer: Enhanced Transformer with Rotary Position Embedding》中引入）旨在解决这些问题。ROPE 通过将位置信息嵌入到注意力机制中，以旋转的方式编码位置，使得模型能够更好地捕捉相对位置关系，并且具有更好的扩展性。

---

### 2. ROPE 的数学原理

ROPE 的核心思想是：通过对查询（Query, $q$）和键（Key, $k$）向量施加一个基于位置的旋转变换，将位置信息融入到注意力计算中。这种旋转操作基于二维平面上的旋转矩阵，具有良好的数学性质。

#### 2.1 基本概念
假设输入序列中的某个 token 的嵌入向量为 $x_m$（位于位置 $m$），其维度为 $d$（通常是模型的隐藏维度）。ROPE 将这个向量分成多对二维子向量（假设 $d$ 是偶数），然后对每一对子向量施加旋转操作。

注意力机制中的点积计算为：
$$ q_m^T k_n $$
其中 $q_m$ 是位置 $m$ 的查询向量，$k_n$ 是位置 $n$ 的键向量。ROPE 的目标是让这个点积不仅依赖于 $q_m$ 和 $k_n$ 的内容，还依赖于它们的相对位置 $m - n$。

#### 2.2 旋转矩阵
对于二维向量，旋转矩阵定义如下：
$$
R(\theta) = \begin{bmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{bmatrix}
$$
其中 $\theta$ 是旋转角度。在 ROPE 中，$\theta$ 与位置相关，通常定义为：
$$ \theta_i = m \cdot \Theta_i $$
其中：
- $m$ 是 token 的位置；
- $\Theta_i$ 是与维度 $i$ 相关的旋转频率，通常设置为 $\Theta_i = 10000^{-2i/d}$（与原始 Transformer 的正弦-余弦 PE 频率一致）。

对于更高维的向量（$d$ 维），ROPE 将向量分成 $d/2$ 个二维子向量，然后对每一对子向量应用旋转矩阵，形成一个块对角矩阵：
$$
R_m = \text{diag}(R(\theta_0), R(\theta_1), ..., R(\theta_{d/2-1}))
$$

#### 2.3 应用到查询和键
将旋转矩阵应用到 $q_m$ 和 $k_n$ 上，得到旋转后的向量：
$$ q_m' = R_m q_m $$
$$ k_n' = R_n k_n $$

然后，注意力机制中的点积变为：
$$ (q_m')^T k_n' = (R_m q_m)^T (R_n k_n) $$

通过数学推导，这个点积可以表示为：
$$ (q_m')^T k_n' = q_m^T R_m^T R_n k_n $$
由于 $R_m^T R_n$ 是一个只依赖于相对位置 $m - n$ 的旋转矩阵，这就实现了相对位置编码的目标。

#### 2.4 优点
- **相对性**：点积的结果只依赖于 $m - n$，而不是绝对位置 $m$ 和 $n$。
- **可扩展性**：ROPE 不需要预先定义最大序列长度，可以动态适应任意长度的输入。
- **高效性**：旋转操作可以直接融入到注意力计算中，不增加额外的参数。

---

### 3. 代码实现

下面将会以 HuggingFace transformers 库的实现为例，代码文件 `src/transformers/models/llama/modeling_llama.py`。

```python
class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # For BC we register cos and sin cached
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
        t = t / self.scaling_factor
        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("_cos_cached", emb.cos().to(torch.get_default_dtype()), persistent=False)
        self.register_buffer("_sin_cached", emb.sin().to(torch.get_default_dtype()), persistent=False)

    @property
    def sin_cached(self):
        logger.warning_once(
            "The sin_cached attribute will be removed in 4.39. Bear in mind that its contents changed in v4.38. Use "
            "the forward method of RoPE from now on instead. It is not used in the `LlamaAttention` class"
        )
        return self._sin_cached

    @property
    def cos_cached(self):
        logger.warning_once(
            "The cos_cached attribute will be removed in 4.39. Bear in mind that its contents changed in v4.38. Use "
            "the forward method of RoPE from now on instead. It is not used in the `LlamaAttention` class"
        )
        return self._cos_cached

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

```

---
# 上下文窗口和外推

### 1. 上下文长度和固定窗口的限制

在 Transformer 模型中，上下文长度通常由训练时设定的最大序列长度（比如 2048 个 token）决定。这是因为：
- **注意力机制的计算**：自注意力需要计算每个 token 与其他所有 token 的关系。如果训练时模型只见过固定长度（比如 2048），它的参数和位置编码都是针对这个长度优化的。
- **位置编码的限制**：传统的位置编码（如正弦-余弦 PE）是为固定长度预计算的。如果输入序列超过这个长度，模型不知道如何为超出部分生成位置信息。

由于模型在训练时只能看到固定大小的上下文窗口。如果直接给它一个更长的序列（比如 4096 个 token），它可能会出现性能下降甚至完全失效的情况。这就引出了“外推”的概念。

---

### 2. 外推（Extrapolation）是什么意思？

外推是指模型在训练时未见过的情况下，尝试处理超出训练数据范围的输入。简单来说：
- **内插（Interpolation）**：在训练数据范围内进行预测。例如，模型训练时见过 1 到 2048 的位置，它可以在这个范围内很好地工作。
- **外推（Extrapolation）**：处理训练范围之外的数据。例如，模型训练时最大长度是 2048，但现在要处理 4096 的序列，这就超出了训练时的范围。

在传统的 Transformer 中，外推性能很差。因为位置编码是固定的，超出预设长度的位置没有定义，导致模型无法理解这些位置的相对关系。这就像你在地图上只知道 0 到 100 公里的路标，突然要你判断 150 公里的位置，你会觉得茫然。

---

### 3. 线性插值如何扩大上下文长度？

线性插值（Linear Interpolation）是一种简单的方法，用于扩展位置编码的范围，从而让模型能够处理更长的上下文。它的核心原理是：通过“拉伸”或“压缩”位置编码的频率，使其适配新的序列长度。

#### 3.1 传统正弦-余弦位置编码
在原始 Transformer 中，位置编码基于正弦和余弦函数：
$$ PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d}}\right) $$
$$ PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d}}\right) $$
其中：
- $pos$ 是位置（0, 1, 2, ...）；
- $i$ 是维度索引；
- $d$ 是模型维度。

这些函数的频率（$\frac{1}{10000^{2i/d}}$）是固定的，设计时假设最大长度是 $L$（比如 2048）。如果 $pos > L$，编码值会继续增长，但模型没见过这么大的 $pos$，无法正确理解。

#### 3.2 线性插值的原理
线性插值通过调整位置索引的比例，将原始长度 $L$ 的位置编码“映射”到新的长度 $L'$（比如 4096）。具体方法是：
- 假设原始训练长度是 $L$，新输入长度是 $L'$；
- 对位置 $pos$ 进行缩放：$pos' = pos \cdot \frac{L}{L'}$；
- 将缩放后的 $pos'$ 代入位置编码公式。

数学上：
$$ PE'(pos, 2i) = \sin\left(\frac{pos \cdot \frac{L}{L'}}{10000^{2i/d}}\right) $$
$$ PE'(pos, 2i+1) = \cos\left(\frac{pos \cdot \frac{L}{L'}}{10000^{2i/d}}\right) $$

#### 3.3 举个例子
- 训练时最大长度 $L = 2048$，现在输入长度 $L' = 4096$。
- 对于位置 $pos = 4096$（新序列的末尾），缩放后 $pos' = 4096 \cdot \frac{2048}{4096} = 2048$。
- 然后用 $pos' = 2048$ 计算位置编码，相当于把新序列“压缩”到模型熟悉的范围内。

#### 3.4 为什么有效？
线性插值本质上是假设位置编码的频率分布可以线性扩展。虽然这种方法不完美（因为频率分布并非完全线性），但它提供了一种简单的方式，让模型能够“猜测”超出训练长度的位置信息，从而实现一定程度的外推。

---

### 4. ROPE 如何增强外推性能？

ROPE（旋转位置编码）相比传统位置编码，在外推性能上有显著优势。原因在于它的设计天然支持相对位置信息，并且对序列长度不敏感。

#### 4.1 ROPE 的回顾
ROPE 通过旋转矩阵将位置信息融入查询 $q$ 和键 $k$：
$$ q_m' = R_m q_m, \quad k_n' = R_n k_n $$
其中旋转角度 $\theta_i = m \cdot 10000^{-2i/d}$，注意力点积变为：
$$ (q_m')^T k_n' = q_m^T R_m^T R_n k_n $$
关键点是：$R_m^T R_n$ 只依赖于相对位置 $m - n$，而不是绝对位置 $m$ 或 $n$。

#### 4.2 ROPE 的外推优势
1. **相对位置编码**：
   - 传统 PE 是绝对位置编码，$pos > L$ 时模型无法处理。
   - ROPE 只关心 $m - n$，即使 $m$ 和 $n$ 很大，只要它们的差值在合理范围内（比如小于训练时的最大长度），模型依然能理解。

2. **动态性**：
   - ROPE 的旋转角度 $\theta_i = m \cdot 10000^{-2i/d}$ 是连续函数，随着 $m$ 增加自然延伸，不需要预定义最大长度。
   - 相比之下，传统 PE 是离散的，超出 $L$ 后无法直接计算。

3. **与线性插值的结合**：
   - 如果需要进一步增强 ROPE 的外推能力，可以结合线性插值。例如，将 $\theta_i$ 缩放为 $\theta_i' = m \cdot \frac{L}{L'} \cdot 10000^{-2i/d}$，让旋转频率适配新长度。

#### 4.3 为什么能增强外推？
- **直观理解**：ROPE 就像一个“指南针”，它不关心你走了多远（绝对位置），只关心你相对的方向（相对位置）。**即使序列长度超出训练范围，token 之间的相对关系依然可以通过旋转角度捕捉到**（**本质上是压缩，两个token之间的位置缩小了，见下图**）。
- **数学支持**：$R_m^T R_n$ 的形式保证了外推时注意力机制的稳定性，只要相对距离 $m - n$ 不超过模型的理解范围（通常与训练长度相关），结果仍然有意义。

一图胜千言，下面就表示了ROPE外推的基本原理：

![[ROPE 线性插值.png#center]]

---

### 5.代码实现

以HuggingFace 的 transformers 库 `models/llama/modeling_llama.py`的实现方式为例。

```python
class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        
        # ......

        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
        t = t / self.scaling_factor
        freqs = torch.outer(t, self.inv_freq)

        # ......
```

其中，`self.scaling_factor`是缩放比`L/L'`，`inv_freq`是$\theta_i$。

---
# NTK方法增强外推

### 1. NTK 外推的背景

NTK 外推最初与神经切空间理论（Neural Tangent Kernel）相关，但在这里我们关注它在位置编码中的应用，特别是在 ROPE 的改进版本中。它的核心目标是解决 Transformer 在训练长度（比如 2048）之外的外推问题，与线性插值类似，但方法不同：
- **线性插值**：通过缩放位置索引（$pos$），将新长度压缩到训练范围内。
- **NTK 外推**：通过调整位置编码的频率基数（base），让模型“感知”更长的序列，而不需要显式缩放位置。

NTK 外推特别适用于 ROPE，因为 ROPE 的旋转角度与频率直接相关，调整频率可以自然影响外推性能。

---

### 2. ROPE 的回顾

先简单回顾一下 ROPE 的数学基础，因为 NTK 外推是基于它的：
- ROPE 的旋转角度为：
  $$ \theta_i = pos \cdot \Theta_i $$
  其中 $\Theta_i = 10000^{-2i/d}$，$pos$ 是位置，$d$ 是模型维度，$i$ 是维度索引。
- 查询 $q$ 和键 $k$ 被旋转后，注意力点积依赖于相对位置 $m - n$：
  $$ (R_m q_m)^T (R_n k_n) = q_m^T R_{m-n} k_n $$

问题在于：如果 $pos$ 超出训练时的最大长度（比如 2048），$\theta_i$ 会变得很大，导致旋转频率过高，模型可能无法正确捕捉长距离关系。

---

### 3. NTK 外推的原理

NTK 外推的核心思想是调整 ROPE 的频率基数（base，原为 10000），使旋转角度 $\theta_i$ 在更长的序列中保持合理的频率分布。具体来说：
- 原始 ROPE 的频率基数是固定的（通常是 10000），这决定了 $\Theta_i$ 的变化速度。
- 当序列长度增加（比如从 2048 到 4096），原始基数可能导致 $\theta_i$ 增长过快，使得高维度的旋转过于频繁，模型难以泛化。

NTK 外推通过**增大基数**，减缓 $\theta_i$ 的增长速度，让位置编码在长序列中依然平滑可控。

#### 3.1 数学调整
原始 ROPE 的频率：
$$ \Theta_i = 10000^{-2i/d} $$

NTK 外推将基数从 10000 改为一个更大的值（记为 $\text{base}'$），通常根据目标长度动态计算：
$$ \Theta_i' = (\text{base}')^{-2i/d} $$
其中 $\text{base}' > 10000$，具体值可以通过公式推导或实验确定。

一个常见的经验公式是：
$$ \text{base}' = 10000 \cdot \left(\frac{L'}{L}\right)^\alpha $$
- $L$：训练时的最大长度（比如 2048）；
- $L'$：目标长度（比如 4096）；
- $\alpha$：调节因子，通常接近 1（具体值可能根据模型调整）。

例如，若 $L = 2048$，$L' = 4096$，$\alpha = 1$：
$$ \text{base}' = 10000 \cdot \frac{4096}{2048} = 10000 \cdot 2 = 20000 $$

新的旋转角度变为：
$$ \theta_i' = pos \cdot (20000)^{-2i/d} $$

#### 3.2 为什么调整基数有效？

- **减缓频率增长**：增大基数后，$\Theta_i'$ 变小，$\theta_i'$ 的增长速度变慢。这样，即使 $pos$ 达到 4096，旋转角度也不会变得过于极端，保持在模型训练时学到的范围内。
- **保持相对性**：ROPE 的核心是相对位置 $m - n$。调整基数后，$\theta_i$ 的分布更平滑，长距离 token 的相对关系仍然可识别。

---

### 4. NTK 外推与线性插值的对比

| 方法       | 操作方式                     | 效果                              |
|------------|------------------------------|-----------------------------------|
| 线性插值   | 缩放位置 $pos' = pos \cdot \frac{L}{L'}$ | 将新长度压缩到训练范围，保持编码分布 |
| NTK 外推   | 增大基数 $\text{base}'$       | 减缓旋转频率，适应更长序列         |

- **线性插值**：相当于“挤压”位置，让 4096 个位置看起来像 2048。
- **NTK 外推**：相当于“拉伸”频率范围，让模型自然适应 4096，而不压缩位置。

---

### 5. 为什么它能工作？

NTK 外推的“奇迹”与线性插值类似，依赖于以下几点：

1. **频率的可调性**：
   - ROPE 的旋转频率决定了 token 间关系的“分辨率”。原始基数 10000 适合短序列，但长序列需要更低的频率，NTK 通过增大基数实现这一点。
   - 例如，$pos = 4096$ 时，原始 $\theta_i$ 可能旋转几十圈（高频震荡），而新基数让它只旋转几圈，更接近训练时的模式。

2. **模型的泛化能力**：
   - 模型在训练时学会了如何处理不同频率的旋转角度。NTK 外推调整后的 $\theta_i'$ 仍然落在模型可理解的范围内。

3. **相对位置的稳定性**：
   - ROPE 的注意力依赖 $m - n$，调整基数后，相对角度 $\theta_{m-n}$ 的分布依然合理，长距离关系不会失真。

---

### 6. 代码实现

下面是一个简单的 PyTorch 实现，展示如何在 ROPE 中应用 NTK 外推：

```python
import torch
import torch.nn as nn

class NTKRotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000, max_seq_len=2048):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len

        # 原始频率
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def adjust_base(self, new_max_len):
        # 根据新长度调整基数
        scale = new_max_len / self.max_seq_len
        new_base = self.base * scale
        self.inv_freq = 1.0 / (new_base ** (torch.arange(0, self.dim, 2).float() / self.dim))

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]
        if seq_len > self.max_seq_len:
            self.adjust_base(seq_len)

        positions = torch.arange(seq_len, device=x.device).float()
        theta = positions[:, None] * self.inv_freq[None, :]

        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)

        x1 = x[..., 0::2]  # 奇数维度
        x2 = x[..., 1::2]  # 偶数维度
        x1_rot = x1 * cos_theta - x2 * sin_theta
        x2_rot = x1 * sin_theta + x2 * cos_theta

        return torch.stack([x1_rot, x2_rot], dim=-1).reshape_as(x)

# 示例
dim = 64
x = torch.randn(1, 4096, dim)  # 输入长度 4096
rope = NTKRotaryEmbedding(dim=dim, max_seq_len=2048)
x_rot = rope(x)
print(x_rot.shape)  # torch.Size([1, 4096, 64])
```

#### 代码说明：
- `adjust_base`：根据新长度动态调整基数。
- 当输入长度超过训练长度（2048）时，基数增大，频率减缓。

---

### 7. 总结

NTK 外推是一种简单而强大的方法，通过增大 ROPE 的频率基数，让模型适应更长的上下文长度。它的“奇迹”在于：
- 调整频率分布，使旋转角度在长序列中保持平滑；
- 利用 ROPE 的相对位置特性，增强外推能力；
- 不需要改变模型结构，只需一个参数调整。

相比线性插值，NTK 外推更自然地“拉伸”位置编码，而不是“压缩”位置，适用于需要保留原始位置分辨率的场景。

---
# YaRN

### 1. RoPE 的数学基础

回顾一下，RoPE 是一种高效的位置编码方法，通过旋转变换将 token 的位置信息嵌入到查询（Query）和键（Key）向量中。相比传统正弦位置编码，RoPE 更适合 Transformer 的自注意力机制。

#### 1.1 RoPE 的核心公式
给定一个 $d$ 维向量 $x = [x_1, x_2, \dots, x_d]$（例如 Query 或 Key 的表示），其在序列中的位置为 $m$，RoPE 对 $x$ 应用旋转变换：

$$
x'_m = R_m \cdot x
$$

其中，$R_m$ 是一个旋转矩阵。假设 $d$ 为偶数，RoPE 将 $x$ 分成 $d/2$ 对，每一对由一个 $2 \times 2$ 的旋转矩阵处理。对于第 $i$ 对（对应维度 $2i-1$ 和 $2i$），旋转矩阵定义为：

$$
R_m^i = \begin{bmatrix}
\cos(m \theta_i) & -\sin(m \theta_i) \\
\sin(m \theta_i) & \cos(m \theta_i)
\end{bmatrix}
$$

旋转频率 $\theta_i$ 通常定义为：

$$
\theta_i = 10000^{-2(i-1)/d}
$$

此频率随 $i$ 递减，反映了近距离 token 需要更高分辨率、远距离 token 需要较低分辨率的特性。

#### 1.2 应用示例
对于向量 $x = [x_1, x_2, x_3, x_4, \dots]$，经过 RoPE 编码后：

$$
x'_m = [x_1 \cos(m \theta_1) - x_2 \sin(m \theta_1), x_1 \sin(m \theta_1) + x_2 \cos(m \theta_1), x_3 \cos(m \theta_2) - x_4 \sin(m \theta_2), \dots]
$$

#### 1.3 RoPE 的优势与局限
RoPE 的旋转机制天然编码了相对位置关系。在自注意力中，点积 $q_m^T k_n$ 只依赖相对距离 $m - n$，而非绝对位置。然而，当输入长度超出训练时的上下文窗口（例如从 $L_{train} = 4096$ 到 $L_{target} = 16384$），直接外推 $\theta_i$ 会导致：
1. **高频维度（小 $i$）旋转过快，位置信息失真**。
2. 注意力分布过于集中，模型难以有效处理长序列。

---

### 2. YaRN 的改进机制

YaRN 通过两项关键改进解决 RoPE 的外推问题：**动态插值（Dynamic Interpolation）** 和 **温度缩放（Temperature Scaling）**。以下详细介绍其数学原理。

#### 2.1 动态插值
##### 2.1.1 问题描述
当上下文长度从 $L_{train}$ 扩展到 $L_{target}$，扩展比例为 $s = L_{target} / L_{train}$（例如 $s = 4$），位置 $m$ 的增加会导致 $m \theta_i$ 的角度过大，尤其在高频维度（小 $i$），从而破坏位置信息的可区分性。

##### 2.1.2 解决方案
YaRN 通过缩放位置索引 $m$ 来调整旋转角度，定义缩放后的位置为：

$$
m' = \frac{m}{\alpha}
$$

旋转矩阵变为：

$$
R_m^i = \begin{bmatrix}
\cos(m' \theta_i) & -\sin(m' \theta_i) \\
\sin(m' \theta_i) & \cos(m' \theta_i)
\end{bmatrix}
$$

其中，$\alpha$ 是缩放因子，且对不同维度 $i$ 动态调整：
- **低频维度（大 $i$）**：$\alpha = s$，适应长距离信息。
- **高频维度（小 $i$）**：$\alpha = 1$，保留局部精度。
- **中间维度**：$\alpha$ 在 $1$ 和 $s$ 之间平滑过渡。

##### 2.1.3 $\alpha$ 的选择
YaRN 对 $\alpha$ 采用分段或线性插值策略。例如：
- 前 $k$ 个维度（高频）：$\alpha_i = 1$。
- 后 $d/2 - k$ 个维度（低频）：$\alpha_i = s$。
- 中间维度：$\alpha_i = 1 + (s - 1) \cdot \frac{i - k}{d/2 - k}$。

超参数 $k$（例如 $k = d/4$）通过实验确定。

##### 2.1.4 数学效果
以 $m = 16384$、$s = 4$ 为例：
- 若 $\alpha = 4$，则 $m' = 4096$，低频维度角度回到训练范围。
- 若 $\alpha = 1$，则 $m' = m$，高频维度保留局部分辨率。

此方法比简单位置插值（所有维度 $\alpha = s$）更灵活，能同时优化近距离和远距离信息。

#### 2.2 温度缩放
##### 2.2.1 问题描述（下面有详细解释）
长上下文下，注意力机制的 softmax 输出分布过于尖锐（熵降低），因为 $q_m^T k_n$ 的值范围随 $m$ 增加而扩大，导致模型只关注少数 token。

##### 2.2.2 解决方案
YaRN 引入温度因子 $t$，调整旋转角度：

$$
R_m^i = \begin{bmatrix}
\cos\left(\frac{m' \theta_i}{t}\right) & -\sin\left(\frac{m' \theta_i}{t}\right) \\
\sin\left(\frac{m' \theta_i}{t}\right) & \cos\left(\frac{m' \theta_i}{t}\right)
\end{bmatrix}
$$

等价地，可直接缩放 $q$ 和 $k$：

$$
q' = \frac{q}{\sqrt{t}}, \quad k' = \frac{k}{\sqrt{t}}
$$

点积变为：

$$
q'^T k' = \frac{q^T k}{t}
$$

softmax 输出更平滑，熵增加。

##### 2.2.3 $t$ 的选择
YaRN 提出经验公式：

$$
\sqrt{t} \approx 0.1 \cdot \ln(s) + 1
$$

示例：
- $s = 1$：$\sqrt{t} \approx 1$，$t \approx 1$。
- $s = 4$：$\ln(4) \approx 1.386$，$\sqrt{t} \approx 1.1386$，$t \approx 1.3$。
- $s = 16$：$\ln(16) \approx 2.773$，$\sqrt{t} \approx 1.2773$，$t \approx 1.63$。

##### 2.2.4 数学效果
若 $q^T k$ 原始范围为 $[-a, a]$，缩放后为 $[-a/t, a/t]$，softmax 输出从 $\exp(a)/Z$ 变为 $\exp(a/t)/Z'$，分布更均匀。

---

### 3. YaRN 的实现流程

对于输入长度从 $L_{train} = 4096$ 扩展到 $L_{target} = 16384$：
1. 计算 $s = 16384 / 4096 = 4$。
2. 动态插值：
   - 对每个 $i$ 计算 $\alpha_i$（从 $1$ 到 $4$ 过渡）。
   - 计算 $m' = m / \alpha_i$。
3. 温度缩放：
   - 计算 $t \approx 1.3$（基于 $\sqrt{t} \approx 0.1 \ln(4) + 1$）。
   - 调整角度 $\theta_i / t$。
4. 应用到模型：
   - 用 $R_m^i$ 计算 $q$ 和 $k$。

---

### 4. 伪代码实现

```python
def yarn_rope(q, m, d, s, base=10000):
    theta = 1.0 / (base ** (torch.arange(0, d, 2) / d))  # RoPE 频率
    alpha = torch.ones(d//2)  # 动态插值因子
    k = d//4  # 高频维度数
    alpha[k:] = torch.linspace(1, s, d//2 - k)  # 低频插值
    m_prime = m / alpha  # 缩放位置
    t = (0.1 * math.log(s) + 1) ** 2  # 温度因子
    angles = m_prime * theta / t  # 调整角度
    cos, sin = torch.cos(angles), torch.sin(angles)
    q_rot = apply_rotation(q, cos, sin)  # 应用旋转
    return q_rot
```

---

### 5. 详细解释为什么长上下文softmax会导致问题

在长上下文场景下，注意力机制的 softmax 输出分布变得过于“尖锐”（即熵降低），这是一个由 Transformer 架构和位置编码特性共同导致的现象。
### 1. 注意力机制的基本原理

Transformer 的自注意力机制通过计算查询（Query, $q$）和键（Key, $k$）之间的点积，并经过 softmax 归一化，生成注意力权重。给定序列长度为 $L$，位置 $m$ 的查询向量 $q_m$ 和位置 $n$ 的键向量 $k_n$，注意力分数为：

$$
\text{score}(m, n) = q_m^T k_n
$$

注意力权重通过 softmax 计算：

$$
\alpha_{m,n} = \text{softmax}(\text{score}(m, n)) = \frac{\exp(q_m^T k_n)}{\sum_{j=1}^L \exp(q_m^T k_j)}
$$

softmax 的输出 $\alpha_{m,n}$ 表示位置 $m$ 对位置 $n$ 的关注程度，其分布的熵（entropy）定义为：

$$
H_m = -\sum_{n=1}^L \alpha_{m,n} \log \alpha_{m,n}
$$

熵 $H_m$ 反映注意力分布的均匀性：熵越高，分布越平滑；熵越低，分布越集中（尖锐）。

---

### 2. 长上下文下熵降低的原因

当上下文长度 $L$ 显著增加（例如从 $4096$ 到 $64k$），注意力分布的熵倾向于降低，表现为 softmax 输出过于尖锐，即少数 $\alpha_{m,n}$ 接近 1，大多数接近 0。以下是具体原因：

#### 2.1 点积值范围的扩大

在 RoPE（旋转位置嵌入）或类似位置编码下，$q_m^T k_n$ 的值与位置 $m$ 和 $n$ 的相对距离相关。对于短上下文（如 $L = 4096$），$m$ 和 $n$ 的最大距离较小，$q_m^T k_n$ 的值分布在一个有限范围内（例如 $[-a, a]$）。但在长上下文（如 $L = 64k$）下：
- 最大距离 $m - n$ 增加，导致 $q_m^T k_n$ 的值范围扩大（例如 $[-A, A]$，其中 $A \gg a$）。
- 这是因为 RoPE 通过旋转角度 $\theta_i m$ 编码位置，角度随 $m$ 线性增长，未经调整的 $\theta_i$ 会使远距离 token 的点积值显著偏离。

数学上，若 $q_m$ 和 $k_n$ 是 $d$ 维向量，RoPE 编码后：

$$
q_m^T k_n = \sum_{i=1}^{d/2} \left[ q_{2i-1} k_{2i-1} \cos((m - n) \theta_i) + q_{2i} k_{2i} \cos((m - n) \theta_i) + \text{交叉项} \right]
$$

当 $|m - n|$ 变大时，$\cos((m - n) \theta_i)$ 的振荡加剧，点积值的方差增加，使得每一个维度上的值更有可能取到极端值而不是均匀分布。

#### 2.2 Softmax 的指数放大效应

Softmax 函数对输入的指数放大效应使得点积值范围的扩大直接影响输出分布。假设 $q_m^T k_n$ 的值分布在 $[-A, A]$：
- 最大值 $A$ 对应的 $\exp(A)$ 在分母 $\sum_j \exp(q_m^T k_j)$ 中占主导地位。
- 其他较小值（例如 $0$ 或负值）对应的 $\exp(0) = 1$ 或 $\exp(-A)$ 变得微不足道。

例如，若 $L = 3$，点积值为 $[10, 0, -10]$：
$$
\alpha = \text{softmax}([10, 0, -10]) = \frac{[\exp(10), \exp(0), \exp(-10)]}{\exp(10) + \exp(0) + \exp(-10)} \approx [0.9999, 0.0001, 0.0]
$$

熵 $H \approx 0$，分布极尖锐。若 $L$ 增加到 $100$，且点积值范围进一步扩大（例如 $[-50, 50]$），最大值的影响更显著，熵进一步降低。

在长上下文下，$L$ 的增加使得分母项数量激增，但点积值的最大值（由模型权重和位置编码决定）仍会被 softmax 放大，导致分布集中于少数 token。

#### 2.3 位置编码的外推效应

对于 RoPE，未经调整的 $\theta_i = 10000^{-2(i-1)/d}$ 是为训练长度 $L_{train}$ 设计的。当 $L > L_{train}$ 时，$m \theta_i$ 的角度超出训练范围：
- 高频维度（小 $i$）：$\theta_i$ 大，角度 $m \theta_i$ 变化快，$q_m^T k_n$ 的值波动剧烈。
- 这种波动使某些 token 的点积值意外变大，softmax 进一步放大这些“异常”值。

例如，若 $L_{train} = 4096$，$\theta_1 \approx 1$，则 $m = 4096$ 时角度为 $4096$ 弧度。但若 $L = 16384$，$m = 16384$ 时角度为 $16384$ 弧度，远超设计范围，导致点积值分布失控。


