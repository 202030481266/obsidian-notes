
# 参考文章

[Continuous-Batching-llm-inference](https://www.anyscale.com/blog/continuous-batching-llm-inference)
[Optimization History of BLOOM](https://huggingface.co/blog/zh/bloom-inference-optimization)
[大白话解释continuous batching](https://zhuanlan.zhihu.com/p/680123256)

# 关键事实

- LLM的推理速度瓶颈在于内存或者显存带宽，而不是计算能力。
- LLM的Prefill阶段非常迅速（这个主要是基于GPU的并行计算能力，所以很快）。
- LLM的显存容量很重要，越大的显存意味着可以计算更长的序列和更大的Batch。

# 传统的静态批处理

![[diagram-static-batching.png]]

就像上面这张图所反应的一样，由于生成的序列长度不一，导致了GPU无法被充分利用，哪怕是并行计算了一批的序列，但是效果还是很糟糕。（这是经典的Transformer带来的推理问题）

# 连续批处理

这个名词其实是VLLM和TGI的叫法，原始论文叫做“Iteration Batching”，下面就是论文作者给的一张图，非常好解释了连续批处理的精华：

![[continuous batching orca.gif]]

