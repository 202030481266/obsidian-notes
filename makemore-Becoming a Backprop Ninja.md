
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

## 矩阵求导的本质

实际上应用的时候不会就去查吧，矩阵求导真的非常复杂（绝大部分情况都是直接看代码然后可以输出，或者Pytorch的AutoGrad已经做完了和反向传播的所有的事情）。下面是一些碎碎念、文章阅读总结和一些数学推导的精华。



