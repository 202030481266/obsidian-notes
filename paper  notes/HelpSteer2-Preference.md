
## 相关资料

[Llama3.1-70B-Nemotron-Instruct](https://huggingface.co/nvidia/Llama-3.1-Nemotron-70B-Instruct-HF)
[HelpSteer2 DataSet](https://huggingface.co/datasets/nvidia/HelpSteer2)

## 经典的两种训练方法

### 1. Bradley-Terry模型

Bradley-Terry模型是一种用来处理成对比较(pairwise comparison)的方法。**它用于估计两个选项之间的相对偏好**。假设在一对比较中，用户更偏好选项 $A$ 而非 $B$，Bradley-Terry模型可以通过估计每个选项的"技能评分"(skill rating)来建模这种偏好。

#### Bradley-Terry模型公式

假设我们有两个选项 $A$ 和 $B$，其相应的评分为 $s_A$ 和 $s_B$。Bradley-Terry模型定义了 $A$ 被选择的概率为：
$$
P(A > B) = \frac{\exp(s_A)}{\exp(s_A) + \exp(s_B)}
$$

其中 $s_A$ 和 $s_B$ 是对应选项的评分。这个概率函数可以理解为选择 $A$ 的概率是其评分的指数函数与两者评分指数函数之和的比值。

#### Bradley-Terry模型的训练过程

1. **初始化评分**：对每个选项初始化其评分，一般可以设置为相同的初始值（如0）。
2. **数据收集**：收集成对比较的数据，例如，用户选择 $A$ 胜于 $B$ 的次数。
3. **最大似然估计**：通过最大化似然函数来更新评分。对于一组观察数据，似然函数可以表示为：
   $$
   L = \prod_{(A,B) \in D} P(A > B)^{n_{AB}} \cdot P(B > A)^{n_{BA}}
   $$
   其中，$n_{AB}$ 和 $n_{BA}$ 分别表示 $A$ 胜于 $B$ 和 $B$ 胜于 $A$ 的次数。
4. **梯度更新**：通过梯度下降法更新评分，直到收敛。

Bradley-Terry模型适用于处理相对偏好信息，其优点是直观且容易解释。

### 2. SteerLM Regression

SteerLM regression是一种回归方法，通常用于语言模型的奖励建模。它的目标是通过回归拟合来估计选项的评分，**使得更符合偏好的选项得分更高**。

#### SteerLM回归的公式

假设有一对选项 $A$ 和 $B$，回归的目标是使得选择偏好的一致性最大化。例如，如果 $A$ 被认为优于 $B$，那么奖励模型应当给 $A$ 一个更高的评分。

可以定义一个回归损失函数，如：
$$
L = \max(0, 1 - (R_A - R_B))
$$
其中 $R_A$ 和 $R_B$ 分别是奖励模型对 $A$ 和 $B$ 的评分。这个损失函数表示，如果 $R_A$ 比 $R_B$ 大1以上，则不再有损失；否则会有一个损失值，激励模型继续优化。

#### SteerLM的训练过程

1. **数据准备**：收集成对比较的数据，即每对选项之间的偏好信息。
2. **初始化模型参数**：奖励模型的参数通常初始化为随机值或预训练的权重。
3. **定义损失函数**：根据成对比较的偏好关系定义一个损失函数，通常是基于差值的hinge loss或log loss。
4. **优化过程**：通过梯度下降或其他优化算法最小化损失函数，调整奖励模型的参数。
5. **验证与迭代**：在验证集上评估模型性能，必要时调整超参数或改进训练数据。

### 总结

- **Bradley-Terry模型**适用于直接建模成对比较的相对偏好，主要通过最大似然估计来优化评分。
- **SteerLM回归**方法则直接对评分进行回归，使得奖励模型符合成对比较的排序，采用基于差值的损失函数进行优化。

这两种方法都能有效地用来训练奖励模型，根据实际场景选择合适的方法会带来更好的性能和效果。

## 核心训练方法创新

本文中将SteerLM回归和Bradley-Terry模型结合起来，以构建更强大的奖励模型。具体来说，这种结合方法利用了两种方法的互补性来提高模型的性能。以下是结合的主要步骤和原理：

### 1. SteerLM回归与Bradley-Terry模型的单独比较

首先，研究人员分别对SteerLM回归和Bradley-Terry模型进行了评估，并发现它们各自有优点。SteerLM回归在仅使用“有用性”属性进行训练时，表现相对较好，且简化了训练和推理的设置。而对于Bradley-Terry模型，他们发现缩放版本（Scaled BT）能够更有效地利用偏好幅度信息。

在这种单独比较中，最佳的SteerLM回归（仅帮助）和缩放后的Bradley-Terry（Scaled BT）在RewardBench数据集上的表现几乎相同（93.0 vs 92.7），这说明两者的建模能力非常接近。

### 2. 结合方法的核心思想

尽管单独的SteerLM回归和Scaled BT都表现出色，但研究人员认为可以通过组合这两种方法来增强性能。结合的核心思想是利用每种方法的优势来弥补另一种方法的不足。例如，SteerLM回归能够较好地处理单一属性（如有用性）的训练，而Bradley-Terry可以更好地利用偏好强度信息。

这种组合方法的灵感来自于“监督微调 + 偏好调优”的两阶段方法，即先进行监督学习（训练回归模型），然后再通过偏好优化（例如近端策略优化）来微调模型。

### 3. 具体的结合步骤

以下是具体的结合步骤：

1. **初始化**：首先，用SteerLM回归（仅有用性）训练的模型作为初始模型。这意味着使用SteerLM回归训练得到的参数来初始化Bradley-Terry模型的参数。
2. **Bradley-Terry优化**：然后在此基础上，使用Scaled BT方法继续训练模型。这一步利用Bradley-Terry的偏好比较优势，在已有的回归模型基础上进一步优化模型参数。
3. **ExPO外推优化**：为了进一步改进模型，研究人员使用了一种称为ExPO的外推优化方法。ExPO的核心思想是通过增量调整权重来提升模型性能。具体做法是：以1.1到2.0之间的0.1为步长进行外推因子的搜索，发现1.6是最优值后，再以0.01为步长在1.51到1.69之间进一步微调，最终选择1.52作为最佳外推因子。

### 4. 协同效应的解释

这种结合方法的协同效应在于，SteerLM回归模型捕捉了一些基础的属性信息，而Bradley-Terry模型则能够利用偏好强度来精细地调整这些信息。因此，使用回归模型初始化后再进行偏好优化，相当于将两种方法的优点结合起来。这种方法类似于在强化学习中的策略优化阶段进行更精细的调优，从而实现更好的性能。

