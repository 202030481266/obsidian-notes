
# 相关链接

[Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking](https://arxiv.org/abs/2403.09629)
[yiyibooks翻译](https://yiyibooks.cn/arxiv/2403.09629v1/index.html)
[Quiet-STaR Official Code](https://github.com/ezelikman/quiet-star)

# 前沿后训练技术

| viv 学习策略               | 优点                    | 缺点                   | 代表                    |
| ---------------------- | --------------------- | -------------------- | --------------------- |
| Behaviour Clone Expert | 1. 更像人、专家，并且有人类的偏好    | 1. 实际能力由于数据分布有偏      | 各种游戏陪玩 AI，LLM SFT     |
|                        | 2. 可以通过单 Agent 的方式训练  | 2. 无法探索出人类行为之外的行为    |                       |
|                        | 3. 当数据量无限多的时候可以取得完美表现 | 3. 无法利用错误数据          |                       |
| RLHF                   | 1. 可以对齐人类偏好及价值观       | 1. 偏好建模困难，容易 hacking | ChatGPT，目前主流的Chat模型玩法 |
|                        | 2. 能力利用错误数据           | 2. 训练成本高             |                       |
|                        | 3. 数据利用效率高            |                      |                       |
| Self-Play              | 1. 绝对强度更高，甚至超越最强人类、专家 | 1. 有时候无法理解人类，行为不像人   | AlphaGo, OpenAI o1    |
|                        | 2. 可以实现双人对垒和博弈的最优     | 2. 训练及推理成本极高         |                       |

# 论文解读

论文的核心思想简而言之，就是一句话：在说话之前先思考（三思而后行）。但是很多人会反驳这一点：推理和思考不会发生在文本中，而是在文本的各个token之间，所以再长的思维链其实不会从本质上提升模型的智能。但是COT能够显著改善模型的效果，这一点是不言而喻的，本文也是基于了这个观察。不过这篇论文最大的创新点不是这个IDEA，而是一些关键问题的解决：

1、如何解决推理前动态产生思维链（COT）？
2、如何高效解决第一个问题（因为计算量很多，如果不并行计算）
3、如何让模型能够自己学会思考？（论文中是强化学习）

## 算法

![[Quiet-STaR image 1.png]]

总的来看，这个算法具有一个十分强大的特点——**一般适用性**（不针对特定的领域或者任务）。下面主要来看看算法的这一块是怎么做的：

Algorithm 1: Quiet Self-Taught Reasoner (Quiet-STaR)

Input: Language model $\theta_0$, training steps $\text{num\_steps}$, sequence length $l$, thought length $t$, learning rate $\alpha$, batch size $b$, number of thoughts $n_{\text{thoughts}}$, number of ground truth tokens used for supervising each thought $n_{\text{true}}$

Output: Language model $\theta$ that generates rationales to predict future text

for $i = 0$ to $\text{num\_steps}$ do
  Sample batch of sequences $X$ of length $l$
  
  $h_{\text{init}} \leftarrow \text{hidden\_states}_{\theta_i}(X)$
  
  for $j = 1$ to $l$ in parallel using attention mask do
  $$
  \log p_{\text{init}}^{j:j+n_{\text{true}}} \leftarrow \text{lm\_head}_{\theta_i}(h_{\text{init}}^{j:j+n_{\text{true}}})
  $$
  // Predict next tokens
  
  $$
  T_j \leftarrow \text{generate\_tokens}_{\theta_i}([X_{:j}; \texttt{<start\_thought>}], t, n_{\text{thoughts}})
  $$
  // Generate thought
  $$
  T_j \leftarrow [T_j; \texttt{<end\_thought>}]
  $$
  $$
  h_{\text{thought}}^{j:j+n_{\text{true}}} \leftarrow \text{hidden\_states}_{\theta_i}([X_{:j}; T_j; X_{j:j+n_{\text{true}}-1}])
  $$
  $$
  \log p_{\text{thought}}^{j:j+n_{\text{true}}} \leftarrow \text{lm\_head}_{\theta_i}(h_{\text{thought}}^{j:j+n_{\text{true}}})
  $$
  // Predict next tokens w/ thought
  $$
  w_{j:j+n_{\text{true}}} \leftarrow \text{mixing\_head}_{\theta_i}(h_{\text{init}}^{j:j+n_{\text{true}}}, h_{\text{thought}}^{j:j+n_{\text{true}}})
  $$
  $$
  \log p_j \leftarrow w_{j:j+n_{\text{true}}} \cdot \log p_{\text{init}}^{j:j+n_{\text{true}}} + (1 - w_{j:j+n_{\text{true}}}) \cdot \log p_{\text{thought}}^{j:j+n_{\text{true}}}
  $$
  // Mix logits
  $$
  \mathcal{L}_{\text{NLL}}^j \leftarrow -\log p_{\text{talk}}(X_{j:j+n_{\text{true}}})
  $$
  $$
  r_j = \log p_{\text{talk}}(X_{j:j+n_{\text{true}}+1}) - \log \overline{p}_{\text{talk}}(X_{j:j+n_{\text{true}}})
  $$
  $$
  \nabla_{\theta_i}\mathcal{L}_{\text{REINFORCE}} \leftarrow -r_j[\mathbf{r}_j > 0] \cdot \nabla_{\theta_i}\log p_{\theta_i}(T_j | [X_{:j}, \texttt{<start\_thought>}])
  $$
  $$
  \nabla_{\theta_i}\mathcal{L}_j \leftarrow \nabla_{\theta_i}\mathcal{L}_{\text{NLL}}^j + \nabla_{\theta_i}\mathcal{L}_{\text{REINFORCE}}
  $$
  
  $\theta_{i+1} \leftarrow \theta_i - \alpha \sum_{j=1}^l \nabla_{\theta_i}\mathcal{L}_j$
  
return $\theta_{\text{num\_steps}}$

其实认真看这个算法其实是很简单的，首先就是生成思考推理，然后根据思考推理再生成后续的推理，最后的推理是一个混合的结果，其中的奖励值定义为这个思考推理的贡献。



