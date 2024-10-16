
### 参考博客

[DPO 是如何简化 RLHF 的](https://zhuanlan.zhihu.com/p/671780768)

### 阅读笔记

#### 引言
在深度学习的优化策略中，**RLHF**（Reinforcement Learning from Human Feedback, 基于人类反馈的强化学习）是常见的方法，而 **DPO**（Direct Preference Optimization, 直接偏好优化）通过简化 RLHF 训练步骤，达到类似的效果。本文通过对公式的详细推导，深入解析 DPO 的原理，并与 RLHF 进行对比。

#### 一、KL 散度与 RLHF 中的最优策略

KL 散度用于衡量两个概率分布之间的差异，公式如下：

$$
D_{KL}(P || Q) = \sum P(x) \log \frac{P(x)}{Q(x)}
$$

我们希望通过优化 KL 散度，找到一个尽可能接近真实分布的模型。在 RLHF 中，目标是通过强化学习获得能够反映人类偏好的最优策略分布 $\pi^*(y|x)$。

### 1.1 推导 KL 散度的最优解

从以下表达式的最大化出发（RLHF）：

$$
\max_{\pi_{\theta}} \mathbb{E}_{x \sim D, y \sim \pi_{\theta}(y|x)} \left[ r_{\phi}(x, y) \right] - \beta D_{KL} \left( \pi_{\theta}(y|x) || \pi_{\text{ref}}(y|x) \right]
$$

其中 $r_{\phi}(x, y)$ 是通过人类反馈训练的奖励模型，目标是让好的回答得分更高，而 $\pi_{\text{ref}}(y|x)$ 是参考策略。通过最大化期望，我们可以得到最优策略。

#### 二、归一化分布

为了确保这个策略是合法的概率分布，引入归一化项 $Z(x)$，定义为：

$$
Z(x) = \sum_y \pi_{\text{ref}}(y|x) e^{r_{\phi}(x, y)/\beta}
$$

有了这个归一化常数后，可以构造一个新的概率分布 $\pi^*(y|x)$：

$$
\pi^*(y|x) = \frac{\pi_{\text{ref}}(y|x) e^{r_{\phi}(x, y)/\beta}}{Z(x)}
$$

这样，$\pi^*(y|x)$ 就是一个合法的概率分布，满足概率分布的和为 1。

### 2.1 归一化的推导步骤

我们希望找到新分布 $\pi^*(y|x)$，它与原始参考分布 $\pi_{\text{ref}}(y|x)$ 以及奖励模型 $r_{\phi}(x, y)$ 相关联。通过归一化，确保新分布的合法性，并将问题转化为 KL 散度的最小化形式：

$$
\min_{\pi_{\theta}} \mathbb{E}_{x \sim D, y \sim \pi_{\theta}(y|x)} \left[ D_{KL} \left( \pi_{\theta}(y|x) || \pi^*(y|x) \right) \right]
$$

这意味着，我们希望找到一个分布 $\pi_{\theta}(y|x)$，使其与通过人类反馈训练出的最优分布 $\pi^*(y|x)$ 尽可能接近。

#### 三、DPO 的推导与简化

在 RLHF 中，训练通常分为两步：
1. 训练 **reward model**，使其能够区分好的回答和不好的回答。
2. 使用强化学习（RL）优化模型输出，使其尽可能符合人类偏好。

而 **DPO** 通过将这两步简化为一步，直接从 reward model 中推导出最优策略 $\pi^*(y|x)$，从而避免复杂的 RL 训练过程。

### 3.1 DPO 中 reward model 的优化

首先，训练 reward model，使其能够区分两个回答的优劣，优化目标如下：

$$
\max_{r_{\phi}} \mathbb{E}_{x, y_{\text{win}}, y_{\text{lose}} \sim D} \left[ \log \left( r_{\phi}(x, y_{\text{win}}) - r_{\phi}(x, y_{\text{lose}}) \right) \right]
$$

这个 loss 函数表示，好的回答（$y_{\text{win}}$）的得分应该显著高于不好的回答（$y_{\text{lose}}$），使得模型能够有效区分回答的优劣。

### 3.2 DPO 的目标优化

接下来，使用以下目标优化策略：

$$
\max_{\pi_{\theta}} \mathbb{E}_{x, y \sim \pi_{\theta}(y|x)} \left[ r_{\phi}(x, y) \right] - \beta D_{KL} \left( \pi_{\theta}(y|x) || \pi_{\text{ref}}(y|x) \right)
$$

其中 $\pi_{\theta}$ 是正在训练的 LLM 策略，$\pi_{\text{ref}}(y|x)$ 是训练开始时的参考策略。这个 loss 函数的意义在于希望最终的 LLM 策略 $\pi_{\theta}(y|x)$ 不偏离参考策略 $\pi_{\text{ref}}(y|x)$，同时又能够通过 reward model 优化输出的质量。

#### 四、DPO 的优势

**DPO** 方法通过简化 **RLHF** 的训练过程，不再需要同时训练多个模型（如 reward model、actor、critic 等），只需要一个参考策略和一个 actor 模型。其核心思想是将 RLHF 中的强化学习过程转化为一种监督学习，通过 reward model 直接优化 LLM。

相比传统的 RLHF，DPO 具有以下优势：
- **训练更简单**：只需要训练 reward model 和 actor，不再需要复杂的强化学习。
- **避免在线更新**：不需要在线获取数据进行强化学习，可以直接利用离线数据进行优化。
- **计算效率高**：减少了多次模型训练与更新的步骤，显著降低计算量。

#### 结语

通过以上推导和分析，DPO 方法通过引入 KL 散度和 reward model，将 RLHF 中复杂的强化学习优化过程简化为监督学习，从而提高了训练效率。在实际应用中，DPO 使得模型训练变得更加简单易实现，同时保留了与 RLHF 相似的效果。
