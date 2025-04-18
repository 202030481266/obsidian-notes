
# Related Link

[Karpathy's youtube tutorial](https://www.youtube.com/watch?v=q8SA3rM6ckI&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=5)
[Karpathy's github notebook](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part4_backprop.ipynb)
[Karpathy's google colab notebook](https://colab.research.google.com/drive/1WV2oi2fh9XXyldh02wupFQX0wh5ZC-z-?usp=sharing)
[My google colab notebook](https://colab.research.google.com/drive/1gLmxKD6vNhbE4Pw9gL-PmLY5uvOdxFl0#scrollTo=-znMQSkdbZKA)
[Yes, you should understand backpropgation](https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b)
[Bessel's Correction](https://math.oxford.emory.edu/site/math117/besselCorrection/)

# Matrix Math

这个视频就是手动计算反向传播并且进行优化（Karpathy说这个练习非常有价值）。但是在做的过程中发现这个梯度的计算还真的是很不简单：==涉及到各种广播操作的矩阵，各种不同维度的矩阵以及各种骚操作（聚合函数）==。最后再使用这些数学表达式进行一个化简得到更加简单的反向传播方法，但是需要一定的矩阵代数基础。

## Recommend Materials

[[Matrix Calculus - Notes on the derivative of a Trace.pdf]]
[[Matrix Cookbook.pdf]]
[矩阵求导的本质与分子布局、分母布局的本质（矩阵求导——本质篇）](https://zhuanlan.zhihu.com/p/263777564)
[矩阵求导公式的数学推导（矩阵求导——基础篇）](https://zhuanlan.zhihu.com/p/273729929)
[矩阵求导公式的数学推导（矩阵求导——进阶篇）](https://zhuanlan.zhihu.com/p/288541909)
[矩阵求导术（上）](https://zhuanlan.zhihu.com/p/24709748)
[矩阵求导术（下）](https://zhuanlan.zhihu.com/p/24863977)
[Matrix Calculus Website](https://www.matrixcalculus.org/)

## 矩阵求导的本质

实际上应用的时候不会就去查吧，矩阵求导真的非常复杂（绝大部分情况都是直接看代码然后可以输出，或者Pytorch的AutoGrad已经做完了和反向传播的所有的事情）。下面是一些碎碎念、文章阅读总结和一些数学推导的精华。

矩阵求导的本质就是：**本质就是** $function$ 中的**每个** $f$ **分别对变元中的每个元素逐个求偏导，只不过写成了向量、矩阵形式而已。如果 $function$ 中有 $m$ 个 $f$ ，变元中有 $n$ 个元素，那么，每个 $f$ 对变元中的每个元素逐个求偏导后，我们就会产生 $m×n$ 个结果**。

## 指标法和矩阵的迹

复杂的矩阵求导直接使用定义来求解真的非常麻烦，使用**矩阵整体来进行推导**真的是一大进步！而这个就是矩阵的迹发挥的作用，而且在求导的过程中通常会使用到各种下标的变换，这也就是指标法的一大妙用。（虽然具体数学已经无数次强调了orz~）

本质上，矩阵的高级求导术就是下面的关系（导数和全微分的关系）：

$$\mathrm{d}f=\operatorname{tr}\left({\frac{\partial{f}}{\partial{X}}}^T\mathrm{d}X\right)$$
首先如果$f$是一个标量的实值函数，那么由于trace的性质（这个性质超有用），就可以得到：
$$f=\operatorname{tr}\left(f\right)$$
$$\mathrm{d}f=\mathrm{d}\left(\operatorname{tr}f\right)=\operatorname{tr}\left(\mathrm{d}f\right)$$
所以**若标量函数f是矩阵X经加减乘法、逆、行列式、逐元素函数等运算构成，则使用相应的运算法则对$f$求微分，再使用迹技巧给$\mathrm{d}f$套上迹并将其它项交换至$\mathrm{d}$左侧，对照导数与微分的联系即能得到导数。**

**特别地，若矩阵退化为向量，对照导数与微分的联系，也同样的能得到导数。** 如果是复合的函数情形，那么就要使用基于微分的链式法则来进行求导。（==强烈推荐看上面的Recommend Materials==)。

### 一些常用的矩阵微分运算法则

1. 加法法则：$d(X \pm Y) = dX \pm dY$

2. 矩阵乘法：$d(XY) = (dX)Y + X(dY)$

3. 转置：$d(X^T) = (dX)^T$

4. 迹：$d \operatorname{tr}(X) = \operatorname{tr}(dX)$

5. 矩阵乘积：$d(X \circ Y) = dX \circ Y + X \circ dY$，※表示只对相同的矩阵 $X, Y$ 逐元素相乘。

6. 逆：$dX^{-1} = -X^{-1}(dX)X^{-1}$。此式只在 $XX^{-1} = I$ 两侧矩阵分解运用于。

7. 行列式：$d|X| = \operatorname{tr}(X^\# dX)$，其中 $X^\#$ 表示 $X$ 的伴随矩阵，在 $X$ 可逆的只可应用
   $d|X| = |X| \operatorname{tr}(X^{-1} dX)$。

8. 矩阵跟踪微分：$d\sigma(X) = \sigma'(X) \circ dX$，$\sigma(X) = [\sigma(X_{ij})]$ 是矩阵特征值函数，
   $\sigma'(X) = [\sigma'(X_{ij})]$ 是矩阵特征导数。

例如：
$$
X = \begin{bmatrix} X_{11} & X_{12} \\ X_{21} & X_{22} \end{bmatrix}, d \sin(X) = \begin{bmatrix} \cos X_{11} dX_{11} & \cos X_{12} dX_{12} \\ \cos X_{21} dX_{21} & \cos X_{22} dX_{22} \end{bmatrix} = \cos(X) \circ dX
$$
### 迹的运用技巧

标量的迹等于自身：$a = \operatorname{tr}(a)$。

转置：$\operatorname{tr}(A^T) = \operatorname{tr}(A)$。

线性：$\operatorname{tr}(A \pm B) = \operatorname{tr}(A) \pm \operatorname{tr}(B)$。

交换性：$\operatorname{tr}(AB) = \operatorname{tr}(BA)$，其中 $A$ 与 $B^T$ 尺寸相同。两侧矩阵等于 $\sum_{i,j} A_{ij} B_{ji}$。

矩阵乘法/矩阵乘积交换性：$\operatorname{tr}(A^T (B \circ C)) = \operatorname{tr}((A \circ B)^T C)$，其中 $A, B, C$ 尺寸相同。两侧矩阵等于 $\sum_{i,j} A_{ij} B_{ij} C_{ij}$。

