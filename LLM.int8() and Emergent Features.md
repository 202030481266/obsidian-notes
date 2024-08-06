
这篇笔记是对模型量化大神Tim Dettmers的博文[LLM.int8() and Emergent Features](https://timdettmers.com/2022/08/17/llm-int8-and-emergent-features/)的思考。

## 一些有用的认知

1. 量化误差不只是存在特定的位置，这种误差会传播和累积，这在深度神经网络非常麻烦，较大的量化误差可能将给模型带来巨大的性能下降。
2. 