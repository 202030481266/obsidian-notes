## 官方文档

[torch.polar](https://pytorch.org/docs/stable/generated/torch.polar.html)
[torch.view_as_complex](https://pytorch.org/docs/stable/generated/torch.view_as_complex.html)
## torch.polar 函数

### 1. 操作原理

`torch.polar`函数用于根据给定的幅度（magnitude）和相位角（phase angle）创建复数张量。这个函数将极坐标形式的数据转换为复数的直角坐标形式。

数学上，复数可以表示为： z = a + bi（直角坐标形式） z = r * e^(iθ)（极坐标形式）

其中，r是幅度，θ是相位角，两种表示形式的转换关系是： a = r * cos(θ) b = r * sin(θ)

`torch.polar`函数正是执行这个转换过程。

### 2. 输入输出的例子和数学表达

**语法：**

```python
torch.polar(abs, angle)
```

**参数：**

- `abs`：表示复数的幅度（magnitude/absolute value）的张量
- `angle`：表示复数的相位角（单位为弧度）的张量

**输出：**

- 返回一个复数张量，其实部和虚部通过上述的数学关系计算得出

**数学表达：** 对于输入的幅度r和相位角θ，输出的复数z = r * (cos(θ) + i * sin(θ))

**例子：**

```python
import torch

# 创建幅度张量和相位角张量
abs = torch.tensor([1.0, 2.0, 3.0])
angle = torch.tensor([0.0, torch.pi/2, torch.pi])

# 使用polar创建复数张量
complex_tensor = torch.polar(abs, angle)
print(complex_tensor)
```

输出：

```
tensor([1.0000+0.0000j, 0.0000+2.0000j, -3.0000+0.0000j])
```

解释：

- 当角度为0时，得到的是纯实数：1 + 0j
- 当角度为π/2时，得到的是纯虚数：0 + 2j
- 当角度为π时，得到的是负实数：-3 + 0j

### 3. 常见的应用场景及相关代码

1. **信号处理中的频域操作**
    
    在信号处理中，经常需要在频域中操作，而频域数据通常是以幅度和相位的形式表示的。
    

```python
import torch
import matplotlib.pyplot as plt

# 生成一个信号
t = torch.linspace(0, 1, 1000)
signal = torch.sin(2 * torch.pi * 5 * t) + torch.sin(2 * torch.pi * 10 * t)

# 进行傅里叶变换
fft_result = torch.fft.fft(signal)
magnitude = torch.abs(fft_result)
phase = torch.angle(fft_result)

# 做一些频域处理
# ...

# 使用polar重建复数信号
reconstructed_freq = torch.polar(magnitude, phase)

# 逆傅里叶变换回时域
reconstructed_signal = torch.fft.ifft(reconstructed_freq).real

plt.figure(figsize=(10, 6))
plt.plot(t, signal, label='Original')
plt.plot(t, reconstructed_signal, label='Reconstructed', linestyle='--')
plt.legend()
plt.title('Signal Reconstruction using polar')
plt.show()
```

2. **相位恢复问题**
    
    在某些情况下，我们可能只有幅度信息而没有相位信息（如某些成像技术），需要恢复相位：
    

```python
import torch

# 原始复数数据
original_data = torch.complex(torch.randn(10), torch.randn(10))

# 只保留幅度信息
magnitude = torch.abs(original_data)

# 假设我们通过某种算法估计得到了相位
estimated_phase = torch.randn(10)  # 这里只是示例

# 使用polar重建复数数据
reconstructed_data = torch.polar(magnitude, estimated_phase)

print("原始数据:", original_data)
print("重建数据:", reconstructed_data)
```

## torch.view_as_complex 函数

### 1. 操作原理

`torch.view_as_complex`函数用于将形状为(..., 2)的实数张量转换为复数张量，其中最后一个维度表示复数的实部和虚部。这个函数不改变底层数据，只是改变了数据的解释方式。

### 2. 输入输出的例子和数学表达

**语法：**

```python
torch.view_as_complex(input)
```

**参数：**

- `input`：最后一个维度大小为2的实数张量，表示[实部, 虚部]

**输出：**

- 返回一个复数张量，其形状比输入少一个维度

**数学表达：** 对于输入张量的每一对值[a, b]，输出的复数为z = a + bi

**例子：**

```python
import torch

# 创建一个形状为(3, 2)的实数张量，表示3个复数
real_tensor = torch.tensor([[1.0, 0.0], 
                            [0.0, 2.0], 
                            [-3.0, 0.0]])

# 将其视为复数张量
complex_tensor = torch.view_as_complex(real_tensor)
print(complex_tensor)
```

输出：

```
tensor([1.+0.j, 0.+2.j, -3.+0.j])
```

### 3. 常见的应用场景及相关代码

1. **复数操作中的内存优化**
    
    当需要高效处理复数数据时，有时会使用实数张量存储，然后在需要时转换为复数视图：
    

```python
import torch

# 创建一个存储复数数据的实数张量
batch_size = 100
features = 50
complex_features_real = torch.randn(batch_size, features, 2)

# 在需要进行复数运算时，转换为复数视图
complex_features = torch.view_as_complex(complex_features_real)

# 进行复数操作
result = complex_features * torch.complex(torch.tensor(2.0), torch.tensor(0.0))

# 转回实数表示（如果需要）
result_real = torch.view_as_real(result)
```

2. **神经网络中的复数层**
    
    在一些处理复数数据的神经网络中：
    

```python
import torch
import torch.nn as nn

class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features, 2))
        self.bias = nn.Parameter(torch.randn(out_features, 2))
        
    def forward(self, x):
        # 输入x是形状为[batch, in_features, 2]的实数张量
        # 转换为复数进行计算
        weight_complex = torch.view_as_complex(self.weight)
        bias_complex = torch.view_as_complex(self.bias)
        x_complex = torch.view_as_complex(x)
        
        # 复数矩阵乘法
        output_complex = torch.matmul(x_complex, weight_complex.t()) + bias_complex
        
        # 转回实数表示
        return torch.view_as_real(output_complex)
```

3. **傅里叶变换的实现**
    
    在自定义FFT实现或频域处理中：
    

```python
import torch

# 原始信号
signal = torch.sin(torch.linspace(0, 10, 1000) * 2 * torch.pi)

# 准备FFT输入（实部和虚部）
fft_input_real = torch.zeros(signal.size(0), 2)
fft_input_real[:, 0] = signal  # 实部是信号值，虚部为0

# 转换为复数表示
fft_input_complex = torch.view_as_complex(fft_input_real)

# 使用PyTorch的FFT
fft_result = torch.fft.fft(fft_input_complex)

# 获取幅度和相位
magnitude = torch.abs(fft_result)
phase = torch.angle(fft_result)

print("前10个频率的幅度:", magnitude[:10])
print("前10个频率的相位:", phase[:10])
```

这两个函数在信号处理、图像处理、复数神经网络和科学计算等领域都有广泛的应用。`polar`主要用于从极坐标形式创建复数，而`view_as_complex`则用于优化复数数据的存储和处理。