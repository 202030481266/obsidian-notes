## 官方文档

[torch.Tensor.expand](https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html)
[torch.flatten](https://pytorch.org/docs/stable/generated/torch.flatten.html#torch.flatten)
## PyTorch中的flatten函数

### 1. 操作原理

flatten函数用于将张量的一部分维度展平成一维。它允许指定起始维度和结束维度，这两个维度之间的所有维度将被合并成一个维度。

### 2. 输入输出的例子和数学表达

假设有一个形状为(2, 3, 4)的张量x：

```python
import torch
x = torch.randn(2, 3, 4)
```

使用flatten函数将第1维到第2维展平：

```python
y = torch.flatten(x, start_dim=1, end_dim=2)
```

输出结果y的形状为(2, 12)，即将原来的(3, 4)变为12。

数学表达：如果原始张量形状为(d₀, d₁, ..., dₙ)，使用flatten(x, start_dim=s, end_dim=e)后，新张量形状变为: (d₀, d₁, ..., dₛ₋₁, dₛ × dₛ₊₁ × ... × dₑ, dₑ₊₁, ..., dₙ)

### 3. 常见的应用场景及相关代码

**场景1: 在神经网络中连接全连接层** 最常见的应用是在卷积神经网络中，将卷积层的输出展平后传递给全连接层：

```python
class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3)
        self.fc1 = torch.nn.Linear(16 * 28 * 28, 10)  # 假设卷积后的特征图大小为28x28
        
    def forward(self, x):
        x = self.conv1(x)
        # 将所有维度展平，除了批次维度
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        return x
```

**场景2: 计算批次数据的统计信息** 当需要计算批次中每个样本的统计信息时：

```python
batch = torch.randn(32, 3, 224, 224)  # 批次图像数据
# 对每个样本计算均值，先将空间维度展平
flattened = torch.flatten(batch, start_dim=2)  # 形状变为(32, 3, 50176)
mean_per_channel = flattened.mean(dim=2)  # 每个通道的均值，形状为(32, 3)
```

## PyTorch中的expand函数

### 1. 操作原理

expand函数用于扩展张量的维度，但不会分配新的内存，而是创建一个新的视图，共享原始数据。它只能扩展大小为1的维度，或者添加新的维度。

### 2. 输入输出的例子和数学表达

假设有一个形状为(1, 3)的张量x：

```python
import torch
x = torch.tensor([[1, 2, 3]])  # 形状为(1, 3)
```

使用expand函数将第一维扩展：

```python
y = x.expand(4, 3)
```

输出结果y的形状为(4, 3)，相当于将x在第一维上复制了4次。

数学表达：如果原始张量某一维度大小为1，使用expand可以将该维度扩展到任意大小，而如果维度大小不为1，则只能保持原来的大小。

### 3. 常见的应用场景及相关代码

**场景1: 广播运算** 在需要进行元素级别的运算，但张量形状不一致时：

```python
# 创建一个向量和一个标量
vector = torch.tensor([1, 2, 3])  # 形状为(3)
scalar = torch.tensor([5])        # 形状为(1)

# 将标量扩展为向量大小
expanded_scalar = scalar.expand(3)

# 现在可以进行元素级别的加法
result = vector + expanded_scalar  # 结果为[6, 7, 8]
```

**场景2: 批处理中重复使用同一个张量** 在处理批次数据时，可能需要将一个张量应用到批次中的每个元素：

```python
# 批次数据
batch = torch.randn(32, 5)  # 32个样本，每个样本有5个特征

# 一个需要应用到每个样本的向量
vector = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]).unsqueeze(0)  # 形状为(1, 5)

# 扩展向量以匹配批次大小
expanded_vector = vector.expand(32, 5)

# 现在可以进行批处理操作
weighted_batch = batch * expanded_vector
```

**场景3: 注意力机制中的矩阵操作** 在Transformer等架构的注意力机制计算中：

```python
# 查询矩阵和键矩阵
queries = torch.randn(8, 10, 64)  # 批大小为8，序列长度为10，嵌入维度为64
keys = torch.randn(8, 15, 64)     # 批大小为8，序列长度为15，嵌入维度为64

# 计算注意力得分矩阵
scores = torch.bmm(queries, keys.transpose(1, 2))  # 形状为(8, 10, 15)

# 创建一个掩码来忽略某些位置
mask = torch.ones(1, 10, 15)  # 初始掩码，形状为(1, 10, 15)
mask[:, :, 5:] = 0  # 将后10列设为0

# 扩展掩码以匹配批次大小
expanded_mask = mask.expand(8, 10, 15)

# 应用掩码
masked_scores = scores * expanded_mask + -1e9 * (1 - expanded_mask)
```

这两个函数在深度学习中非常实用，特别是在处理不同形状的张量和构建复杂的神经网络架构时。