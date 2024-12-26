
# 参考资源

- [Karpathy's Youtube tutorial](https://www.youtube.com/watch?v=l8pRSuU81PU&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
- [Karpathy's github notebook](https://github.com/karpathy/build-nanogpt)

# CausalSelfAttention算法优化

首先经典的实现如下：

1、首先就是将输入的矩阵中的词嵌入维度拆成很多个头，其中 $H = C / num\_head$ ，可得：

$$[B,T,C] \rightarrow [B,T,H]$$

2、这其实就是做一个线性变化，得到了不同的头之间的 $Q, K, V$，然后就是计算注意力，softmax之后得到一个掩码矩阵，然后在加权，最后将所有的头按照词嵌入的维度进行一个拼接，比较直观的看法是从矩阵角度来看，每一个头都是在做下面的操作：

$$
\begin{align}
[B,T,H]\ @\ [B,H,T] &\rightarrow [B,T,T] \\
[B,T,T]\ @\ [B,T,H] &\rightarrow [B,T,H] \\
\operatorname{Concat}_{i=1}^{num\_heads}[B,T,H] &\rightarrow [B,T,C]
\end{align}
$$

更加高效率的算法。

1、可以观察到一个事实：**这些头全部可以并行计算**，而其中的关键是批量矩阵乘法。首先我们可以容易推导出一个头的并行计算过程（**我们尊重样本独立性，所以Batch总是放到前面**）：

$$
\begin{align}
[B,T,C] &\rightarrow [B,1,T,C] \\
[B,1,T,C]\ @\ [3,C,H] &\rightarrow [B,3,T,H]
\end{align}
$$

因此在一个头，我们完全可以并行计算其中的 $Q,K,V$，本质上它们是按照第二个维度切分的结果。那么对于所有的头是否可以这样做呢？完全可以。

$$
[B,1,T,C]\ @\ [3*num\_head,C,H] \rightarrow [B,3*num\_head,T,H]
$$

如果我们按照其中的第二维度切分就可以得到所有头组成的$Q$矩阵，$K$矩阵，$V$矩阵，形状分别是$[B,num\_head,T,H]$，这些矩阵后面可以直接用来处理所有乘法。

这里我们是直接先提升了输入矩阵的维度，然后去乘法，最后拆分。而Karpathy的做法是先直接乘法，然后再拆分，最后提升维度（最后调整了一下位置）。仔细看本质完全是一样的，殊途同归。

![[Transformer多头注意力矩阵乘法.svg#center]]

以上是Karpathy做法的配图，如果看矩阵变换的角度就是：

$$
\begin{align}
[B,T,C]\ @\ [C,3C] &\rightarrow [B,T,3C] \\
[B,T,3C]\ split\ &\rightarrow 3 * [B,T,C] \\
[B,T,C] &\rightarrow [B,T,num\_head,H] \\
[B,T,num\_head,H] &\rightarrow [B,num\_head,T,H]
\end{align}
$$

3、优化后面的算法流程。得到了上面的 $Q,K,V$ 矩阵后，后面的算法流程就简单很多了（后面的两个操作就是调了一下维度顺序并且降维，和一开始的拆分和升维刚好是相反的操作）：

$$
\begin{align}
[B,num\_head,T,H]\ @\ [B,num\_head,H,T] &\rightarrow [B,num\_head,T,T] \\
[B,num\_head,T,T]\ @\ [B,num\_head,T,H] &\rightarrow [B,num\_head,T,H] \\
[B,num\_head,T,H] &\rightarrow [B,T,num\_head,H] \\
[B,T,num\_head,H] &\rightarrow [B,T,C]
\end{align}
$$


