# Paper

[FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
# 论文解读

## 核心思想

> The main idea is that we split the inputs Q, K, V into blocks, load them from slow HBM to fast SRAM, then compute the attention output with respect to those blocks. By scaling the output of each block by the right normalization factor before adding them up, we get the correct result at the end.

简单来讲，FlashAttention就是优化了注意力计算过程中的***存储***，将注意力的$Q$, $K$, $V$矩阵计算变成一种分块的计算形式，因为是小块所以可以完全装在SRAM中。最后使用一种在线的$\operatorname{softmax}$算法，分别将计算的结果组合起来。虽然在后面的$\operatorname{softmax}$过程中产生了额外的计算量，但是整体来说，计算存储发生在内核和SRAM之间，效率能够得到极大的提高！论文的图很好地讲了这个过程：

![[FlashAttention.png]]

---
# 算法流程

这里我只关注最为核心的前向过程，因为反向过程本质上是一样大差不差的。

![[flash attention forward pass.png]]

---
### 块大小设置的细节

这里重点提一下关于块大小的细节，由于在该算法计算的过程中，SRAM里面总共有四个最大的占用内存的块：$K_j,V_j,Q_i,S_{ij}$，它们的大小满足：

$$
B_c\cdot d+B_c\cdot d+B_r\cdot d+B_r\cdot B_c < M
$$
首先可以设置$B_c < \frac{M}{4\cdot d}$，然后因为存在中间变量$S_{ij}$，该变量的大小是$B_r\cdot B_c$，因此需要设置$B_r < \min(\frac{M}{4\cdot d}, d)$，否则可能会产生过大的中间变量。

更加详细的说明参考：
https://github.com/Dao-AILab/flash-attention/issues/618

---
# 代码实现

这个代码其实是FlashAttention2的实现，但是无所谓，大体的思想是没有变的。

```python
@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    Bias,
    Out,
    Lse,
    TMP,  # NOTE: TMP is a scratchpad buffer to workaround a compiler bug
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_bb,
    stride_bh,
    stride_bm,
    stride_ob,
    stride_oh,
    stride_om,
    nheads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    BIAS_TYPE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # off_b = tl.program_id(1)
    # off_h = tl.program_id(2)
    # off_hb = off_b * nheads + off_h
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    # Initialize pointers to Q, K, V
    # Adding parenthesis around indexing might use int32 math instead of int64 math?
    # https://github.com/openai/triton/issues/741
    # I'm seeing a tiny bit of difference (5-7us)
    q_ptrs = (
        Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])
    )
    k_ptrs = (
        K + off_b * stride_kb + off_h * stride_kh + (offs_n[:, None] * stride_kn + offs_d[None, :])
    )
    v_ptrs = (
        V + off_b * stride_vb + off_h * stride_vh + (offs_n[:, None] * stride_vn + offs_d[None, :])
    )
    if BIAS_TYPE == "vector":
        b_ptrs = Bias + off_b * stride_bb + off_h * stride_bh + offs_n
    elif BIAS_TYPE == "matrix":
        b_ptrs = (
            Bias
            + off_b * stride_bb
            + off_h * stride_bh
            + (offs_m[:, None] * stride_bm + offs_n[None, :])
        )
    # initialize pointer to m and l
    t_ptrs = TMP + off_hb * seqlen_q_rounded + offs_m
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
    # load q: it will stay in SRAM throughout
    # [2022-10-30] TD: Triton bug - in the case of EVEN_M=True and EVEN_N=False, if we just call
    # tl.load(q_ptrs), we get the wrong output!
    if EVEN_M & EVEN_N:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs)
        else:
            q = tl.load(q_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
        else:
            q = tl.load(
                q_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0
            )
    # loop over k, v and update accumulator
    end_n = seqlen_k if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M, seqlen_k)
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        if EVEN_N & EVEN_M:  # If we just do "if EVEN_N", there seems to be some race condition
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs + start_n * stride_kn)
            else:
                k = tl.load(k_ptrs + start_n * stride_kn, mask=offs_d[None, :] < headdim, other=0.0)
        else:
            if EVEN_HEADDIM:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
            else:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                    other=0.0,
                )
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k, trans_b=True)
        # Trying to combine the two masks seem to make the result wrong
        if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
            qk += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0, float("-inf"))
        if IS_CAUSAL:
            qk += tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], 0, float("-inf"))
        if BIAS_TYPE != "none":
            if BIAS_TYPE == "vector":
                if EVEN_N:
                    bias = tl.load(b_ptrs + start_n).to(tl.float32)
                else:
                    bias = tl.load(
                        b_ptrs + start_n, mask=(start_n + offs_n) < seqlen_k, other=0.0
                    ).to(tl.float32)
                bias = bias[None, :]
            elif BIAS_TYPE == "matrix":
                if EVEN_M & EVEN_N:
                    bias = tl.load(b_ptrs + start_n).to(tl.float32)
                else:
                    bias = tl.load(
                        b_ptrs + start_n,
                        mask=(offs_m[:, None] < seqlen_q)
                        & ((start_n + offs_n)[None, :] < seqlen_k),
                        other=0.0,
                    ).to(tl.float32)
            # Slightly faster to multiply the softmax_scale in the tl.exp below since the compiler
            # can then fuse the mult and add into an fma instruction. But if we have bias we need to
            # to multiply with softmax_scale here.
            qk = qk * softmax_scale + bias
            m_ij = tl.maximum(tl.max(qk, 1), lse_i)
            p = tl.exp(qk - m_ij[:, None])
        else:
            m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, lse_i)
            p = tl.exp(qk * softmax_scale - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        # scale acc_o
        acc_o_scale = tl.exp(m_i - m_ij)

        # # -- update output accumulator --
        # BUG: have to store and immediately load
        tl.store(t_ptrs, acc_o_scale)
        acc_o_scale = tl.load(t_ptrs)
        acc_o = acc_o * acc_o_scale[:, None]
        # update acc_o
        if EVEN_N & EVEN_M:  # If we just do "if EVEN_N", there seems to be some race condition
            if EVEN_HEADDIM:
                v = tl.load(v_ptrs + start_n * stride_vn)
            else:
                v = tl.load(v_ptrs + start_n * stride_vn, mask=offs_d[None, :] < headdim, other=0.0)
        else:
            if EVEN_HEADDIM:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
            else:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                    other=0.0,
                )
        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)

        # -- update statistics
        m_i = m_ij
        l_i_new = tl.exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl.log(l_i_new)

    o_scale = tl.exp(m_i - lse_i)
    # BUG: have to store and immediately load
    tl.store(t_ptrs, o_scale)
    o_scale = tl.load(t_ptrs)
    acc_o = acc_o * o_scale[:, None]
    # rematerialize offsets to save registers
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # write back l and m
    lse_ptrs = Lse + off_hb * seqlen_q_rounded + offs_m
    tl.store(lse_ptrs, lse_i)
    # initialize pointers to output
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    out_ptrs = (
        Out
        + off_b * stride_ob
        + off_h * stride_oh
        + (offs_m[:, None] * stride_om + offs_d[None, :])
    )
    if EVEN_M:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o)
        else:
            tl.store(out_ptrs, acc_o, mask=offs_d[None, :] < headdim)
    else:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o, mask=offs_m[:, None] < seqlen_q)
        else:
            tl.store(
                out_ptrs, acc_o, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim)
            )
```

里面有很多细节，但是挑着看就行了，又不是做AI Infra的！

首先，这段代码是用 Triton 语言实现的 Flash Attention 算法的核心前向传播内核函数（`_fwd_kernel`）。Flash Attention 是一种高效的注意力机制实现，主要用于加速 Transformer 模型中的自注意力计算。它通过分块计算和减少内存访问来优化性能。以下是对代码的逐行中文解释：

---
一开始的时候，其实就已经按照`BLOCK_M`，批次，头和维度进行了线程块的划分。下面的内核函数只代表了一个线程块。

### 函数定义

```python
@triton.jit
def _fwd_kernel(
    Q, K, V, Bias, Out, Lse, TMP,  # 输入和输出张量
    softmax_scale,  # softmax 缩放因子
    stride_qb, stride_qh, stride_qm,  # Q 张量的步幅
    stride_kb, stride_kh, stride_kn,  # K 张量的步幅
    stride_vb, stride_vh, stride_vn,  # V 张量的步幅
    stride_bb, stride_bh, stride_bm,  # Bias 张量的步幅
    stride_ob, stride_oh, stride_om,  # 输出张量的步幅
    nheads, seqlen_q, seqlen_k, seqlen_q_rounded, headdim,  # 超参数
    CACHE_KEY_SEQLEN_Q, CACHE_KEY_SEQLEN_K,  # 缓存键
    BIAS_TYPE: tl.constexpr,  # 偏置类型（常量表达式）
    IS_CAUSAL: tl.constexpr,  # 是否为因果注意力（常量表达式）
    BLOCK_HEADDIM: tl.constexpr,  # 头的维度分块大小
    EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, EVEN_HEADDIM: tl.constexpr,  # 是否均匀分块
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,  # 分块大小
):
```

- @triton.jit`: 表示这是一个 Triton JIT（即时编译）函数，运行时会被编译为 GPU 核函数。
- 函数参数包括输入张量 `Q`（查询）、`K`（键）、`V`（值）、`Bias`（偏置）、输出张量 `Out`、统计量 `Lse`（log-sum-exp）、临时缓冲区 `TMP`，以及各种步幅（stride）和超参数。
- `BIAS_TYPE`：指定偏置类型（"none"、"vector" 或 "matrix"）。
- `IS_CAUSAL`：指示是否为因果注意力（即只关注前面的序列）。
- `BLOCK_M` 和 `BLOCK_N`：定义查询和键的分块大小。
- `EVEN_M`、`EVEN_N` 和 `EVEN_HEADDIM`：布尔值，表示序列长度或头维度是否能被分块大小整除。

---

### 初始化程序 ID 和偏移

```python
start_m = tl.program_id(0)
off_hb = tl.program_id(1)
off_b = off_hb // nheads
off_h = off_hb % nheads
```

- `tl.program_id(0)` 和 `tl.program_id(1)`：获取当前线程块的 ID。`start_m` 是查询序列的分块索引，`off_hb` 是批次和头的组合索引。
- `off_b`：批次索引（batch index），通过 `off_hb // nheads` 计算。
- `off_h`：头索引（head index），通过 `off_hb % nheads` 计算。

```python
offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
offs_n = tl.arange(0, BLOCK_N)
offs_d = tl.arange(0, BLOCK_HEADDIM)
```

- `offs_m`：查询序列的分块偏移量，从 `start_m * BLOCK_M` 开始，范围为 `[0, BLOCK_M)`。
- `offs_n`：键序列的分块偏移量，范围为 `[0, BLOCK_N)`。
- `offs_d`：头维度的偏移量，范围为 `[0, BLOCK_HEADDIM)`。

---

### 初始化指针（本质上就是计算对应的地址）

```python
q_ptrs = (
    Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])
)
k_ptrs = (
    K + off_b * stride_kb + off_h * stride_kh + (offs_n[:, None] * stride_kn + offs_d[None, :])
)
v_ptrs = (
    V + off_b * stride_vb + off_h * stride_vh + (offs_n[:, None] * stride_vn + offs_d[None, :])
)
```

- `q_ptrs`、`k_ptrs`、`v_ptrs`：分别为 `Q`、`K` 和 `V` 张量的内存指针。
- 每个指针通过批次偏移（`off_b * stride_*b`）、头偏移（`off_h * stride_*h`）和序列/维度偏移（`offs_m/n * stride_*m/n + offs_d`）计算。
- `[:, None]` 和 `[None, :]`：扩展维度以进行广播。

```python
if BIAS_TYPE == "vector":
    b_ptrs = Bias + off_b * stride_bb + off_h * stride_bh + offs_n
elif BIAS_TYPE == "matrix":
    b_ptrs = (
        Bias
        + off_b * stride_bb
        + off_h * stride_bh
        + (offs_m[:, None] * stride_bm + offs_n[None, :])
    )
```

- 如果 `BIAS_TYPE` 是 "vector"，偏置是一个一维向量，指针为 `offs_n`。
- 如果是 "matrix"，偏置是一个二维矩阵，指针为 `offs_m` 和 `offs_n` 的组合。
- 这里是扩展了标准的Attention，比如这里可以加上ROPE旋转编码的矩阵（大小就是$[b, h,seq_q, seq_k]$。

```python
t_ptrs = TMP + off_hb * seqlen_q_rounded + offs_m
lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
```

- `t_ptrs`：指向临时缓冲区 `TMP`，用于存储中间结果。
- `lse_i`：log-sum-exp 统计量，初始化为负无穷。
- `m_i`：最大值统计量，初始化为负无穷。
- `acc_o`：输出累加器，初始化为零。

---

### 加载 Q 张量

```python
if EVEN_M & EVEN_N:
    if EVEN_HEADDIM:
        q = tl.load(q_ptrs)
    else:
        q = tl.load(q_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
else:
    if EVEN_HEADDIM:
        q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
    else:
        q = tl.load(
            q_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0
        )
```

- `tl.load`：从 `q_ptrs` 加载查询张量 `Q`。
- 如果 `EVEN_M` 和 `EVEN_N` 为真（序列长度均匀），直接加载。
- 如果 `EVEN_HEADDIM` 为假，应用掩码确保不超过头维度 `headdim`。
- 如果 `EVEN_M` 为假，应用掩码确保不超过查询序列长度 `seqlen_q`。
- `other=0.0`：超出掩码范围的值填充为 0。

---

### 主循环：计算注意力

```python
end_n = seqlen_k if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M, seqlen_k)
for start_n in range(0, end_n, BLOCK_N):
    start_n = tl.multiple_of(start_n, BLOCK_N)
```

- `end_n`：键序列的结束位置。如果是因果注意力，取 `start_m + 1` 块的最小值与 `seqlen_k` 的较小者。
- 循环以 `BLOCK_N` 为步长遍历键序列。

#### 加载 K 张量

```python
if EVEN_N & EVEN_M:
    if EVEN_HEADDIM:
        k = tl.load(k_ptrs + start_n * stride_kn)
    else:
        k = tl.load(k_ptrs + start_n * stride_kn, mask=offs_d[None, :] < headdim, other=0.0)
else:
    if EVEN_HEADDIM:
        k = tl.load(
            k_ptrs + start_n * stride_kn,
            mask=(start_n + offs_n)[:, None] < seqlen_k,
            other=0.0,
        )
    else:
        k = tl.load(
            k_ptrs + start_n * stride_kn,
            mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
            other=0.0,
        )
```

- 加载当前块的键张量 `K`，逻辑与加载 `Q` 类似，但根据 `start_n` 偏移。

#### 计算 QK

```python
qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
qk += tl.dot(q, k, trans_b=True)
```

- `qk`：初始化为零的注意力分数矩阵。
- `tl.dot(q, k, trans_b=True)`：计算 `Q` 和 `K` 的点积，`trans_b=True` 表示转置 `K`。

#### 应用掩码

```python
if not EVEN_N:
    qk += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0, float("-inf"))
if IS_CAUSAL:
    qk += tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], 0, float("-inf"))
```

- 如果 `EVEN_N` 为假，超出 `seqlen_k` 的部分设为负无穷。
- 如果是因果注意力，`offs_m < start_n + offs_n` 的部分（未来位置）设为负无穷。

#### 应用偏置

```python
if BIAS_TYPE != "none":
    if BIAS_TYPE == "vector":
        if EVEN_N:
            bias = tl.load(b_ptrs + start_n).to(tl.float32)
        else:
            bias = tl.load(
                b_ptrs + start_n, mask=(start_n + offs_n) < seqlen_k, other=0.0
            ).to(tl.float32)
        bias = bias[None, :]
    elif BIAS_TYPE == "matrix":
        if EVEN_M & EVEN_N:
            bias = tl.load(b_ptrs + start_n).to(tl.float32)
        else:
            bias = tl.load(
                b_ptrs + start_n,
                mask=(offs_m[:, None] < seqlen_q) & ((start_n + offs_n)[None, :] < seqlen_k),
                other=0.0,
            ).to(tl.float32)
    qk = qk * softmax_scale + bias
    m_ij = tl.maximum(tl.max(qk, 1), lse_i)
    p = tl.exp(qk - m_ij[:, None])
else:
    m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, lse_i)
    p = tl.exp(qk * softmax_scale - m_ij[:, None])
l_ij = tl.sum(p, 1)
```

- 如果有偏置，加载并加到 `qk` 上，然后应用 `softmax_scale`。
- `m_ij`：当前块的最大值，与全局最大值 `lse_i` 比较。
- `p`：softmax 分数，通过指数运算计算。
- 如果无偏置，直接应用 `softmax_scale`。
- `l_ij`：softmax 分数的和。

#### 更新输出累加器

```python
acc_o_scale = tl.exp(m_i - m_ij)
tl.store(t_ptrs, acc_o_scale)
acc_o_scale = tl.load(t_ptrs)
acc_o = acc_o * acc_o_scale[:, None]
```

- `acc_o_scale`：缩放因子，用于调整累加器。
- 由于 Triton 的编译器 bug，需先存储再加载 `acc_o_scale`。

#### 加载 V 并更新输出

```python
if EVEN_N & EVEN_M:
    if EVEN_HEADDIM:
        v = tl.load(v_ptrs + start_n * stride_vn)
    else:
        v = tl.load(v_ptrs + start_n * stride_vn, mask=offs_d[None, :] < headdim, other=0.0)
else:
    if EVEN_HEADDIM:
        v = tl.load(
            v_ptrs + start_n * stride_vn,
            mask=(start_n + offs_n)[:, None] < seqlen_k,
            other=0.0,
        )
    else:
        v = tl.load(
            v_ptrs + start_n * stride_vn,
            mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
            other=0.0,
        )
p = p.to(v.dtype)
acc_o += tl.dot(p, v)
```

- 加载当前块的值张量 `V`。
- `p` 转换为 `V` 的数据类型。
- `acc_o`：通过点积更新输出累加器。

#### 更新统计量

```python
m_i = m_ij
l_i_new = tl.exp(lse_i - m_ij) + l_ij
lse_i = m_ij + tl.log(l_i_new)
```

- 更新全局最大值 `m_i`。
- 更新 log-sum-exp `lse_i`，通过指数和对数计算。

---

### 写回结果

```python
o_scale = tl.exp(m_i - lse_i)
tl.store(t_ptrs, o_scale)
o_scale = tl.load(t_ptrs)
acc_o = acc_o * o_scale[:, None]
```

- 计算最终输出缩放因子 `o_scale`，并应用于 `acc_o`。

```python
start_m = tl.program_id(0)
offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
lse_ptrs = Lse + off_hb * seqlen_q_rounded + offs_m
tl.store(lse_ptrs, lse_i)
```

- 重新计算 `offs_m`，将 `lse_i` 写回到 `Lse`。

```python
offs_d = tl.arange(0, BLOCK_HEADDIM)
out_ptrs = (
    Out
    + off_b * stride_ob
    + off_h * stride_oh
    + (offs_m[:, None] * stride_om + offs_d[None, :])
)
if EVEN_M:
    if EVEN_HEADDIM:
        tl.store(out_ptrs, acc_o)
    else:
        tl.store(out_ptrs, acc_o, mask=offs_d[None, :] < headdim)
else:
    if EVEN_HEADDIM:
        tl.store(out_ptrs, acc_o, mask=offs_m[:, None] < seqlen_q)
    else:
        tl.store(
            out_ptrs, acc_o, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim)
        )
```

- 计算输出指针 `out_ptrs`。
- 根据 `EVEN_M` 和 `EVEN_HEADDIM` 的值，将 `acc_o` 写回到 `Out`，应用适当的掩码。







