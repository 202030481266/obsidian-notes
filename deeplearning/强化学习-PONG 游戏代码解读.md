# Background

PONG 游戏是一个经典的ATARI Game，其中游戏的规则很简单，而且非常适合用来做强化学习入门，Karpathy的博客就使用这个例子来讲解了强化学习的策略梯度，而且还完整实现了代码（没有使用Pytorch，from scratch级别的实现）。但是代码不容易看懂，所以记录一下。
# Code

[UCLA-COURSE pg-pong.py](https://github.com/ucla-rlcourse/RLexample/blob/master/pg-pong.py)

```python
# From http://karpathy.github.io/2016/05/31/rl/
""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import pickle

import gymnasium as gym
import numpy as np

# hyperparameters
H = 200  # number of hidden layer neurons
batch_size = 10  # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99  # discount factor for reward
decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2
resume = True  # resume from previous checkpoint?
test = True  # test mode, turn off epsilon-greedy and render the scene
save_file = 'pong_model_bolei.p'

if test == True:
    render = True
else:
    render = False

# model initialization
D = 80 * 80  # input dimensionality: 80x80 grid
if resume:
    model = pickle.load(open(save_file, 'rb'))
else:
    model = {}
    model['W1'] = np.random.randn(H, D) / np.sqrt(D)  # "Xavier" initialization
    model['W2'] = np.random.randn(H) / np.sqrt(H)

grad_buffer = {k: np.zeros_like(v) for k, v in model.items()}  # update buffers that add up gradients over a batch
rmsprop_cache = {k: np.zeros_like(v) for k, v in model.items()}  # rmsprop memory


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))  # sigmoid "squashing" function to interval [0,1]


def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float32).ravel()


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def policy_forward(x):
    h = np.dot(model['W1'], x)
    h[h < 0] = 0  # ReLU nonlinearity
    logp = np.dot(model['W2'], h)
    p = sigmoid(logp)
    return p, h  # return probability of taking action 2, and hidden state


def policy_backward(eph, epdlogp):
    """ backward pass. (eph is array of intermediate hidden states) """
    dW2 = np.dot(eph.T, epdlogp).ravel()
    dh = np.outer(epdlogp, model['W2'])
    dh[eph <= 0] = 0  # backpro prelu
    dW1 = np.dot(dh.T, epx)
    return {'W1': dW1, 'W2': dW2}


if render:
    env = gym.make("ale_py:ALE/Pong-v5" , render_mode='human')
else:
    env = gym.make("ale_py:ALE/Pong-v5")
    
observation, _ = env.reset()
prev_x = None  # used in computing the difference frame
xs, hs, dlogps, drs = [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 0
while True:
    if render: env.render()

    # preprocess the observation, set input to network to be difference image
    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = cur_x

    # forward the policy network and sample an action from the returned probability
    aprob, h = policy_forward(x)
    if test == True:
        action = 2 if aprob > 0.5 else 3
    else:
        action = 2 if np.random.uniform() < aprob else 3  # roll the dice!

    # record various intermediates (needed later for backprop)
    xs.append(x)  # observation
    hs.append(h)  # hidden state
    y = 1 if action == 2 else 0  # a "fake label"
    dlogps.append(
        y - aprob)  # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

    # step the environment and get new measurements
    observation, reward, terminated, truncated, info = env.step(action)
    done = np.logical_or(terminated, truncated)
    reward_sum += reward

    drs.append(reward)  # record reward (has to be done after we call step() to get reward for previous action)

    if done:  # an episode finished
        episode_number += 1

        # stack together all inputs, hidden states, action gradients, and rewards for this episode
        epx = np.vstack(xs)
        eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(drs)
        xs, hs, dlogps, drs = [], [], [], []  # reset array memory

        # compute the discounted reward backwards through time
        discounted_epr = discount_rewards(epr)
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        epdlogp *= discounted_epr  # modulate the gradient with advantage (PG magic happens right here.)
        grad = policy_backward(eph, epdlogp)
        for k in model: grad_buffer[k] += grad[k]  # accumulate grad over batch

        # perform rmsprop parameter update every batch_size episodes
        if episode_number % batch_size == 0 and test == False:
            for k, v in model.items():
                g = grad_buffer[k]  # gradient
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g ** 2
                model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                grad_buffer[k] = np.zeros_like(v)  # reset batch gradient buffer

        # boring bookkeeping
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
        if episode_number % 100 == 0: pickle.dump(model, open(save_file, 'wb'))
        reward_sum = 0
        observation, _ = env.reset()  # reset env
        prev_x = None

    if reward != 0:  # Pong has either +1 or -1 reward exactly when game ends.
        print('ep %d: game finished, reward: %f' % (episode_number, reward)), ('' if reward == -1 else ' !!!!!!!!')
```

# 维度变化

个人的习惯，看代码之前一定先搞懂输入输出，能够清晰很多。下面我详细说明代码中各个变量在前向传播和反向传播过程中，其矩阵和向量的形状（维度）如何变化。为便于理解，我们记：

- **D = 80×80 = 6400**：输入图像预处理后展开得到的向量维度
- **H = 200**：隐藏层神经元数
- **T**：一个回合（episode）中的时间步数

---

### 前向传播

1. **输入预处理与构造差分图像**
    
    - **原始输入**：游戏的一个帧，经过 `prepro()` 处理后，裁剪、下采样并二值化，最终展平为一个 1D 浮点向量。
    - **x**：经过预处理后的当前帧（或与上一帧作差后的差分图像），形状为  
        **(D,) = (6400,)**
        
2. **第一层：输入层到隐藏层**
    
    - **权重矩阵 W1**：初始化时为  
        **(H, D) = (200, 6400)**
    - **隐藏层激活计算**：
        
        ```python
        h = np.dot(model['W1'], x)
        ```
        
        其中 `x` 形状为 (6400,)，故得到的 `h` 形状为  
        **(200,)**  
        接下来通过 ReLU 非线性激活：
        
        ```python
        h[h < 0] = 0
        ```
        
        这一步对每个元素独立作用，形状不变。
        
3. **第二层：隐藏层到输出层**
    
    - **权重向量 W2**：初始化时为  
        **(H,) = (200,)**  
        注意这里 W2 其实是一个 1D 数组，每个元素对应隐藏层一个神经元的输出权重。
    - **输出计算**：
        
        ```python
        logp = np.dot(model['W2'], h)
        ```
        
        其中 `h` 形状为 (200,)，`W2` 形状为 (200,)，点乘结果是一个标量。  
        接着应用 sigmoid 函数：
        
        ```python
        p = sigmoid(logp)
        ```
        
        得到的 `p` 就是采取动作2的概率（标量）。

**总结前向传播维度**：

- **x**： (6400,)
- **W1**： (200, 6400)
- **h = W1 · x**： (200,)
- **W2**： (200,)
- **logp = W2 · h**： scalar
- **p = sigmoid(logp)**： scalar

---

### 反向传播

在一个回合（episode）内，我们在每个时间步都保存了：

- **xs**：输入 x 的集合，每个 x 形状为 (6400,)
- **hs**：隐藏层激活 h 的集合，每个 h 形状为 (200,)
- **dlogps**：每步梯度信息（即 “标签” y 与实际输出 p 的差，即 `y - p`），为标量
- **drs**：每步的奖励

在回合结束后，这些列表被堆叠成矩阵：

- **epx**：将所有 x 堆叠，形状为  
    **(T, D) = (T, 6400)**
- **eph**：将所有 h 堆叠，形状为  
    **(T, H) = (T, 200)**
- **epdlogp**：将所有 dlogp 堆叠，形状为  
    **(T, 1)**  
    （也可能是 (T,) 形式，但在后续计算中通常视为列向量）

接下来调用 `policy_backward(eph, epdlogp)` 计算梯度。

1. **计算 W2 的梯度**
    
    ```python
    dW2 = np.dot(eph.T, epdlogp).ravel()
    ```
    
    - **eph.T** 的形状为 (H, T) = (200, T)
    - **epdlogp** 的形状为 (T, 1)
    
    点乘后得到的结果形状为 (200, 1)，再用 `.ravel()` 拉平成 (200,)，与 W2 的形状匹配。
    
2. **反向传播至隐藏层**
    
    ```python
    dh = np.outer(epdlogp, model['W2'])
    ```
    
    - **epdlogp** 的形状为 (T, 1)
    - **W2** 的形状为 (200,)
    
    外积运算结果 `dh` 形状为 (T, 200)。
    
    然后执行：
    
    ```python
    dh[eph <= 0] = 0
    ```
    
    这一步根据 ReLU 的性质，在 `eph` 中小于等于0的地方，其梯度置为0，形状保持不变。
    
3. **计算 W1 的梯度**
    
    ```python
    dW1 = np.dot(dh.T, epx)
    ```
    
    - **dh.T** 的形状为 (H, T) = (200, T)
    - **epx** 的形状为 (T, D) = (T, 6400)
    
    点乘结果 `dW1` 的形状为 (200, 6400)，与 W1 的形状一致。
    

**总结反向传播维度**：

- **eph**： (T, 200)
- **epdlogp**： (T, 1)
- **dW2 = eph.T · epdlogp**： (200,)
- **dh = outer(epdlogp, W2)**： (T, 200)
- **epx**： (T, 6400)
- **dW1 = dh.T · epx**： (200, 6400)

这些梯度随后被累积到 `grad_buffer` 中，等累计一定回合数后（batch_size 个 episode）再用 RMSProp 算法更新模型参数。

---

### 小结

- **前向传播**：  
    输入 (6400,) → 经过 W1 (200×6400) 得到隐藏层 (200,) → 经过 W2 (200,) 得到输出标量 → 应用 sigmoid 得到动作概率。
    
- **反向传播**：  
    收集每个时间步的 x（(6400,)）、h（(200,)）和 dlogp（scalar），堆叠成矩阵（(T,6400)、(T,200) 和 (T,1)）。  
    对 W2 的梯度为：`dW2 = eph.T (200×T) · epdlogp (T×1)` → (200,)；  
    对 W1 的梯度为：先计算 `dh = outer(epdlogp, W2)` → (T,200)，再反向传播并与 epx (T,6400) 相乘：`dW1 = dh.T (200×T) · epx (T×6400)` → (200,6400)。
    

# 反向传播（难点）

无论什么模型，最难的永远是那个反向传播啊！数学真的掉脑袋！
## 优化目标建模

### 神经网络结构

- **输入层**: $x$（维度为 $D = 6000$），表示预处理后的图像差分向量。
- **隐藏层**: $h$（维度为 $H = 200$），通过 $h = \text{ReLU}(W_1 \cdot x)$ 计算。
- **输出层**: $p$（标量，范围[0,1]），通过 $p = \sigma(W_2 \cdot h)$ 计算，表示选择动作"向上"（动作2）的概率。

### 策略梯度目标

- 策略梯度方法的目的是优化神经网络参数（$W_1$ 和 $W_2$），最大化期望累积奖励 $J(\theta)$。
- 根据策略梯度定理，参数的梯度为： $$\nabla_\theta J(\theta) = \mathbb{E} [\nabla_\theta \log \pi(a|s; \theta) \cdot R]$$
    - $\pi(a|s; \theta)$: 策略函数，给定状态 $s$ 和参数 $\theta$ 输出动作 $a$ 的概率。
    - $R$: 折扣奖励（即`discounted_epr`）。
    - 在实践中，用样本估计梯度：$\nabla_\theta J(\theta) \approx \sum_t \nabla_\theta \log \pi(a_t|s_t; \theta) \cdot R_t$。

### 代码中的实现

- `policy_backward`函数计算梯度 $\nabla_{W_1} J$ 和 $\nabla_{W_2} J$，基于整个回合的数据：
    - `epx`: 输入向量 $x$ 的堆叠矩阵（形状 $T \times D$）。
    - `eph`: 隐藏状态 $h$ 的堆叠矩阵（形状 $T \times H$）。
    - `epdlogp`: 梯度调整因子（形状 $T \times 1$），即 $y - p$ 乘以折扣奖励。

## 数学推导

### 1. 前向传播公式

先明确前向传播的计算过程：

1. 隐藏层：$h = \text{ReLU}(W_1 \cdot x)$，其中 $\text{ReLU}(z) = \max(0, z)$。
2. 输出（logits）：$z = W_2 \cdot h$。
3. 动作概率：$p = \sigma(z) = \frac{1}{1 + e^{-z}}$。

### 2. 损失函数(下面会多提一嘴)

策略梯度没有显式的损失函数，而是通过调整 $\log \pi(a|s)$ 来间接优化期望奖励。我们定义：

- $\pi(a|s) = p$（如果 $a = 2$），或 $1 - p$（如果 $a = 3$）。
- 实际动作的"伪标签" $y$：若 $a = 2$ 则 $y = 1$，若 $a = 3$ 则 $y = 0$。

代码中的`dlogps`（即`epdlogp`的初始值）计算为： $$d\log p = y - p$$

- 这实际上是交叉熵损失的梯度形式：
    - 若 $y = 1$：$-\log(p)$ 的梯度为 $1 - p$。
    - 若 $y = 0$：$-\log(1-p)$ 的梯度为 $-p$。
- 然后，`epdlogp *= discounted_epr` 将其乘以折扣奖励 $R$，实现策略梯度的加权。

### 3. 反向传播推导

我们需要计算 $\nabla_{W_2} J$ 和 $\nabla_{W_1} J$。反向传播从输出层开始，逐步向前传播误差。

#### (1) 对 $W_2$ 的梯度

- **输出层的梯度**：
    
    - 策略的对数概率：$\log \pi(a|s) = y \log p + (1 - y) \log (1 - p)$（交叉熵损失）。
    - 对 $z$ 的偏导： $$\frac{\partial \log \pi}{\partial z} = \frac{\partial}{\partial z} [y \log p + (1 - y) \log (1 - p)]$$
        - $p = \sigma(z)$，$\frac{d\sigma(z)}{dz} = \sigma(z) (1 - \sigma(z)) = p (1 - p)$。
        - 使用链式法则： $$\frac{\partial \log \pi}{\partial z} = \frac{\partial \log \pi}{\partial p} \cdot \frac{\partial p}{\partial z}$$
            - $\frac{\partial \log \pi}{\partial p} = \frac{y}{p} - \frac{1 - y}{1 - p} = \frac{y - p}{p (1 - p)}$。
            - $\frac{\partial p}{\partial z} = p (1 - p)$。
            - 因此： $$\frac{\partial \log \pi}{\partial z} = (y - p)$$
        - 策略梯度加权：$\frac{\partial J}{\partial z} = (y - p) \cdot R$（代码中的`epdlogp`）。
- **对 $W_2$ 的梯度**：
    
    - $z = W_2 \cdot h$，$W_2$ 是 $1 \times H$ 向量。
    - 对每个时间步 $t$： $$\frac{\partial J}{\partial W_2} = \frac{\partial J}{\partial z_t} \cdot \frac{\partial z_t}{\partial W_2} = (y_t - p_t) \cdot R_t \cdot h_t$$
    - 整个回合的梯度（$T$ 个时间步）： $$\nabla_{W_2} J = \sum_{t=1}^T (y_t - p_t) R_t h_t$$
    - **代码实现**：
        - `eph.T`（形状 $H \times T$）：隐藏状态的转置。
        - `epdlogp`（形状 $T \times 1$）：$(y - p) \cdot R$。
        - `dW2 = np.dot(eph.T, epdlogp)`：矩阵乘法实现上述**求和**，得到 $H \times 1$ 的梯度。
        - `.ravel()`：将结果展平为向量。

#### (2) 对 $W_1$ 的梯度

- **隐藏层的梯度**：
    
    - $h = \text{ReLU}(W_1 \cdot x)$。
    - 对 $h$ 的偏导： $$\frac{\partial J}{\partial h} = \frac{\partial J}{\partial z} \cdot \frac{\partial z}{\partial h}$$
        - $z = W_2 \cdot h$，$\frac{\partial z}{\partial h} = W_2$（$1 \times H$）。
        - 因此： $$\frac{\partial J}{\partial h} = (y - p) \cdot R \cdot W_2$$
        - **代码实现**：
            - `np.outer(epdlogp, model['W2'])`：外积计算 $(y - p) R \cdot W_2$，形状 $T \times H$。
- **ReLU的梯度**：
    
    - $h = \text{ReLU}(W_1 \cdot x)$，其导数为： $$\frac{\partial h}{\partial (W_1 \cdot x)} =
\begin{cases} 
1 & \text{if } W_1 \cdot x > 0 \\ 
0 & \text{if } W_1 \cdot x \leq 0
\end{cases}
$$
    - 因此，隐藏层梯度在 $h \leq 0$ 的位置被置为0：
        - 代码中：`dh[eph <= 0] = 0`。
- **对 $W_1$ 的梯度**：
    
    - $W_1 \cdot x$ 是 $H \times 1$ 向量，$W_1$ 是 $H \times D$ 矩阵。
    - 对每个时间步： $$\frac{\partial J}{\partial W_1} = \frac{\partial J}{\partial h} \cdot \frac{\partial h}{\partial (W_1 \cdot x)} \cdot \frac{\partial (W_1 \cdot x)}{\partial W_1}$$
        - $\frac{\partial (W_1 \cdot x)}{\partial W_1} = x$（$D \times 1$）。
        - 因此： $$\frac{\partial J}{\partial W_1} = [(y - p) R W_2] \cdot [1_{h>0}] \cdot x^T$$
        - 整个回合： $$\nabla_{W_1} J = \sum_{t=1}^T [(y_t - p_t) R_t W_2] \cdot [1_{h_t>0}] \cdot x_t^T$$
    - **代码实现**：
        - `dh`（形状 $T \times H$）：隐藏层梯度。
        - `epx`（形状 $T \times D$）：输入矩阵。
        - `dW1 = np.dot(dh.T, epx)`：矩阵乘法，$(H \times T) \cdot (T \times D) = H \times D$，计算总梯度。

## 代码与推导的对应

```python
def policy_backward(eph, epx, epdlogp):
  dW2 = np.dot(eph.T, epdlogp).ravel()  # W2的梯度
  dh = np.outer(epdlogp, model['W2'])   # 隐藏层梯度
  dh[eph <= 0] = 0                      # ReLU反向传播
  dW1 = np.dot(dh.T, epx)               # W1的梯度
  return {'W1': dW1, 'W2': dW2}
```

- **`dW2`**:
    - $\nabla_{W_2} J = \sum_t (y_t - p_t) R_t h_t$。
    - `eph.T @ epdlogp` 实现矩阵形式的求和。
- **`dh`**:
    - $\frac{\partial J}{\partial h} = (y - p) R \cdot W_2$。
    - `np.outer` 计算每个时间步的梯度。
- **`dh[eph <= 0] = 0`**:
    - ReLU的反向传播特性。
- **`dW1`**:
    - $\nabla_{W_1} J = \sum_t [(y_t - p_t) R_t W_2] \cdot [1_{h_t>0}] \cdot x_t^T$。
    - `dh.T @ epx` 实现矩阵求和。

## 总结

- **数学原理**：
    - 反向传播通过链式法则，从输出概率 $p$ 开始，依次计算 $W_2$ 和 $W_1$ 的梯度。
    - 策略梯度用奖励 $R$ 加权，强化"好"动作的概率。
- **代码高效性**：
    - 使用矩阵运算（`np.dot`, `np.outer`）批量处理整个回合的数据，避免逐时间步循环。
- **强化学习特性**：
    - $y - p$ 表示动作概率的调整方向，$R$ 表示调整幅度。

### 策略梯度与 $dlogp = y - p$ 的数学与直觉

One more Thing!! 当然不是库克的那个:）我想多说一句策略梯度的数学直觉。这个我觉得是相当重要的东西，因为它和SFT真的太像了。

#### **1. 目标与背景**

- **目标**: 训练一个策略网络 $\pi(a|s; \theta)$，最大化期望累积奖励 $J(\theta) = \mathbb{E} \left[ \sum_t R_t \right]$。
  - $s$: 状态（游戏画面）。
  - $a$: 动作（$a = 2$ 表示向上，$a = 3$ 表示向下）。
  - $\theta$: 网络参数（$W_1$ 和 $W_2$）。
  - $R_t$: 折扣奖励（$discounted_epr$）。
- **策略梯度定理**: 参数梯度为：
  $$
  \nabla_\theta J(\theta) = \mathbb{E} \left[ \nabla_\theta \log \pi(a|s; \theta) \cdot R \right]
  $$
  - 实践中通过采样估计：$\nabla_\theta J(\theta) \approx \sum_t \nabla_\theta \log \pi(a_t|s_t; \theta) \cdot R_t$。

#### **2. 神经网络结构**

- **输入**: $x$（维度 $D = 6000$），预处理后的图像差分向量。
- **隐藏层**: $h = \text{ReLU}(W_1 \cdot x)$，$W_1$ 形状为 $H \times D$，$h$ 维度为 $H = 200$。
- **输出层**: 
  - Logits: $z = W_2 \cdot h$，$W_2$ 形状为 $1 \times H$。
  - 概率: $p = \sigma(z) = \frac{1}{1 + e^{-z}}$，表示选择 $a = 2$（向上）的概率。
- **策略定义**:
  - $\pi(a=2|s) = p$。
  - $\pi(a=3|s) = 1 - p$。

#### **3. 代码中的关键步骤**

```python
aprob, h = policy_forward(x)  # $p = aprob$
action = 2 if np.random.uniform() < aprob else 3
y = 1 if action == 2 else 0  # 伪标签 $y$
dlogps.append(y - aprob)     # $dlogp = y - p$
```

- $p$: 网络输出的向上概率。
- $y$: 实际动作的伪标签（$y = 1$ 表示 $a = 2$，$y = 0$ 表示 $a = 3$）。
- $dlogp = y - p$: 策略梯度的核心部分，后续乘以 $R$。

#### **4. 为什么 $dlogp = y - p$？**
##### **(1) 数学推导**
- **对数概率**:
  - 若 $a = 2$（$y = 1$）：$\log \pi(a|s) = \log p$。
  - 若 $a = 3$（$y = 0$）：$\log \pi(a|s) = \log (1 - p)$。
  - 统一形式：
    $$
    \log \pi(a|s) = y \log p + (1 - y) \log (1 - p)
    $$
- **对 $z$ 的偏导**:
  - $p = \sigma(z)$，导数为：
    $$
    \frac{\partial p}{\partial z} = p (1 - p)
    $$
  - 计算 $\frac{\partial \log \pi}{\partial z}$：
    $$
    \frac{\partial \log \pi}{\partial z} = \frac{\partial \log \pi}{\partial p} \cdot \frac{\partial p}{\partial z}
    $$
  - 其中：
    $$
    \frac{\partial \log \pi}{\partial p} = \frac{y}{p} + (1 - y) \cdot \left(-\frac{1}{1 - p}\right)
    $$
    $$
    = \frac{y}{p} - \frac{1 - y}{1 - p} = \frac{y (1 - p) - (1 - y) p}{p (1 - p)} = \frac{y - p}{p (1 - p)}
    $$
  - 代入：
    $$
    \frac{\partial \log \pi}{\partial z} = \frac{y - p}{p (1 - p)} \cdot p (1 - p) = y - p
    $$
- **策略梯度**:
  - $\frac{\partial J}{\partial z} = \frac{\partial \log \pi}{\partial z} \cdot R = (y - p) \cdot R$。
  - 代码中：$dlogp = y - p$，后续 $epdlogp *= discounted_epr$ 实现乘以 $R$。

##### **(2) 直觉解释**

- $y - p$ 表示“预测概率与实际动作的偏差”：
  - 若 $y = 1$（选了向上），$p = 0.8$：
    - $y - p = 1 - 0.8 = 0.2 > 0$，表示 $p$ 还不够高，应增加。
  - 若 $y = 0$（选了向下），$p = 0.8$：
    - $y - p = 0 - 0.8 = -0.8 < 0$，表示 $p$ 太高，应减少。
- 乘以 $R$ 后：
  - $R > 0$（正奖励）：放大调整方向，强化该动作。
  - $R < 0$（负奖励）：反转方向，削弱该动作。

##### **(3) 与交叉熵损失的关系**

- 交叉熵损失：$L = -[y \log p + (1 - y) \log (1 - p)]$。
- 对 $z$ 的导数：
  $$
  \frac{\partial L}{\partial z} = -(y - p)
  $$
- 监督学习中，最小化 $L$ 使 $p$ 接近 $y$。
- 策略梯度中，$(y - p) \cdot R$ 根据奖励调整 $p$，而不是强制匹配 $y$。

# 代码详细拆解

### **代码总体目标**

这个程序的目标是通过强化学习中的**策略梯度方法**训练一个神经网络（称为"策略网络"），让它学会玩Pong游戏。Pong是一个简单的双人乒乓球游戏，AI控制一个挡板，通过上下移动来接住球并得分。策略网络会根据游戏画面（输入）直接输出动作（向上或向下移动）的概率，然后通过试错和奖励反馈来优化网络参数。

---

### **1. 导入和超参数设置**

```python
import numpy as np
import cPickle as pickle
import gym
from gym import wrappers
```

- **作用**: 导入必要的库。
    - `numpy`: 用于数值计算，尤其是矩阵操作。
    - `cPickle`: 用于序列化（保存和加载）模型参数（Python 3中改为`pickle`）。
    - `gym`: OpenAI Gym库，提供Pong游戏环境。
    - `wrappers`: 用于记录游戏过程（例如保存视频）。

```python
H = 200  # 隐藏层神经元数量
batch_size = 10  # 每多少步更新一次参数
learning_rate = 1e-3  # RMSProp的学习率
gamma = 0.99  # 奖励的折扣因子
decay_rate = 0.99  # RMSProp的衰减率
```

- **超参数解释**:
    - $H$: 神经网络隐藏层的神经元数量，决定了模型容量。
    - $batch_size$: 每10个回合（episodes）累积一次梯度并更新参数，平衡了计算效率和更新频率。
    - $learning_rate$: 学习率，控制参数更新的步长，太大可能不收敛，太小收敛慢。
    - $gamma$: 折扣因子，衡量未来奖励的重要性（0~1之间，接近1表示更重视长期奖励）。
    - $decay_rate$: RMSProp优化器的衰减率，用于平滑梯度的平方平均值。

```python
resume = True  # 是否从已有模型恢复训练
render = False  # 是否渲染游戏画面
```

- **配置标志**:
    - $resume$: 如果为True，从`save.p`文件加载已有模型继续训练。
    - $render$: 如果为True，显示游戏画面（训练时通常关闭以加速）。

---

### **2. 模型初始化**

```python
D = 75 * 80  # 输入维度：75x80的网格
if resume:
  model = pickle.load(open('save.p', 'rb'))
else:
  model = {}
  model['W1'] = np.random.randn(H, D) / np.sqrt(D)  # "Xavier" 初始化
  model['W2'] = np.random.randn(H) / np.sqrt(H)
```

- **输入维度**:
    - $D = 75 * 80$: Pong游戏的原始画面经过预处理后变成75x80的网格，展平为6000维向量。
- **模型结构**:
    - 这是一个两层神经网络：
        - 第一层权重$W1$（形状$H \times D$）：连接输入层（$D$维）到隐藏层（$H$个神经元）。
        - 第二层权重$W2$（形状$H$）：连接隐藏层到输出层（1个标量，表示向上移动的概率）。
    - **Xavier初始化**: 用随机正态分布生成权重并除以输入维度的平方根，防止梯度过大或过小。
- **恢复训练**:
    - 如果$resume=True$，加载已有模型；否则，随机初始化。

```python
grad_buffer = { k : np.zeros_like(v) for k,v in model.iteritems() }  # 梯度累积缓冲区
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.iteritems() }  # RMSProp缓存
```

- **$grad_buffer$**: 用于累积多个回合的梯度，每$batch_size$次更新时使用。
- **$rmsprop_cache$**: RMSProp优化器的缓存，存储梯度的平方移动平均，用于自适应学习率。

---

### **3. 辅助函数**

#### **sigmoid激活函数**

```python
def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x))  # sigmoid函数，将值压缩到[0,1]
```

- **作用**: 将神经网络的输出（logits）转化为概率，表示选择某个动作（向上移动）的可能性。

#### **图像预处理**

```python
def prepro(I):
  """ 将210x160x3的uint8图像帧预处理为6000（75x80）的1D浮点向量 """
  I = I[35:185]  # 裁剪，去除顶部35像素和底部25像素
  I = I[::2, ::2, 0]  # 降采样，每2个像素取1个，仅保留R通道（灰度）
  I[I == 144] = 0  # 擦除背景类型1
  I[I == 109] = 0  # 擦除背景类型2
  I[I != 0] = 1  # 其他部分（球、挡板）设为1
  return I.astype(np.float).ravel()
```

- **原理**:
    - Pong环境的原始图像是210x160x3（RGB），太大且包含冗余信息。
    - **裁剪**: 去掉顶部和底部的无用区域（得分区域等）。
    - **降采样**: 每2像素取1个，缩小到75x80，减少计算量。
    - **灰度化**: 只取红色通道（R），并将背景像素（144和109）设为0，其他（球和挡板）设为1，简化输入。
    - **展平**: 用`ravel()`将75x80的矩阵变成6000维的1D向量。

#### **折扣奖励计算**

```python
def discount_rewards(r):
  """ 计算折扣奖励，从最后一步向前累积 """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(xrange(0, r.size)):  # Python 3中用range替代xrange
    if r[t] != 0: running_add = 0  # 游戏结束时重置（Pong特定）
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r
```

- **原理**:
    - 强化学习中，奖励需要考虑时间折扣：未来的奖励对当前决策的影响逐渐减小。
    - **公式**: $$ R_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots $$
    - **Pong特性**: 当$r[t] \neq 0$（+1或-1）时，表示游戏结束，重置累积值。
    - **实现**: 从最后一个时间步向前遍历，计算每个时间步的折扣奖励。

#### **前向传播**

```python
def policy_forward(x):
  h = np.dot(model['W1'], x)  # 隐藏层计算
  h[h < 0] = 0  # ReLU非线性
  logp = np.dot(model['W2'], h)  # 输出logits
  p = sigmoid(logp)  # 概率
  return p, h  # 返回向上移动的概率和隐藏状态
```

- **神经网络结构**:
    - 输入$x$（6000维图像向量）乘以$W1$（200x6000），得到200维隐藏层$h$。
    - **ReLU**: $h = \max(0, h)$，引入非线性。
    - 隐藏层$h$乘以$W2$（1x200），得到标量$logp$。
    - **Sigmoid**: 将$logp$转化为[0,1]的概率$p$，表示选择动作2（向上移动）的概率。
- **输出**: 返回概率$p$和隐藏状态$h$（后者用于反向传播）。

#### **反向传播**

```python
def policy_backward(eph, epx, epdlogp):
  """ 反向传播，计算梯度 """
  dW2 = np.dot(eph.T, epdlogp).ravel()  # W2的梯度
  dh = np.outer(epdlogp, model['W2'])  # 隐藏层的梯度
  dh[eph <= 0] = 0  # ReLU的反向传播
  dW1 = np.dot(dh.T, epx)  # W1的梯度
  return {'W1': dW1, 'W2': dW2}
```

- **原理**:
    - 策略梯度方法通过调整网络参数，使采取"好"动作的概率增加，"坏"动作的概率减少。
    - **输入**:
        - $eph$: 所有隐藏状态的堆叠矩阵。
        - $epx$: 所有观测值的堆叠矩阵。
        - $epdlogp$: 梯度调整因子（奖励加权）。
    - **计算**:
        - $dW2$: 输出层权重的梯度，通过隐藏状态和调整因子计算。
        - $dh$: 隐藏层的梯度，通过输出层梯度和$W2$反推。
        - ReLU的反向传播：当隐藏状态$\leq 0$时，梯度为0。
        - $dW1$: 输入层权重的梯度。

---

### **4. 主循环**

#### **环境初始化**

```python
env = gym.make("Pong-v0")
env = wrappers.Monitor(env, 'tmp/pong-base', force=True)  # 记录游戏视频
observation = env.reset()
prev_x = None  # 用于计算差分帧
xs, hs, dlogps, drs = [], [], [], []  # 存储中间结果
running_reward = None
reward_sum = 0
episode_number = 0
```

- **环境**:
    - `Pong-v0`: Gym提供的Pong游戏环境。
    - `Monitor`: 记录游戏过程并保存到`tmp/pong-base`目录。
    - $observation$: 初始游戏画面。
- **变量**:
    - $prev_x$: 前一帧图像，用于计算差分。
    - $xs, hs, dlogps, drs$: 分别存储输入、隐藏状态、梯度调整因子和奖励。
    - $running_reward$: 运行平均奖励，用于监控训练效果。

#### **游戏循环**

```python
while True:
  if render: env.render()
```

- **渲染**: 如果$render=True$，显示游戏画面。

```python
  cur_x = prepro(observation)
  x = cur_x - prev_x if prev_x is not None else np.zeros(D)
  prev_x = cur_x
```

- **差分图像**:
    - $cur_x$: 当前帧预处理后的向量。
    - $x$: 当前帧与前一帧的差分，捕捉运动信息（第一帧时为零向量）。
    - 更新$prev_x$为当前帧。

```python
  aprob, h = policy_forward(x)
  action = 2 if np.random.uniform() < aprob else 3  # 随机选择动作
```

- **动作选择**:
    - $aprob$: 网络输出选择动作2（向上）的概率。
    - 用均匀分布随机数与$aprob$比较，引入探索性：
        - 若随机数 < $aprob$，选择2（向上）。
        - 否则选择3（向下）。

```python
  xs.append(x)  # 记录输入
  hs.append(h)  # 记录隐藏状态
  y = 1 if action == 2 else 0  # 伪标签
  dlogps.append(y - aprob)  # 计算梯度调整因子
```

- **记录中间结果**:
    - $xs$: 存储输入图像。
    - $hs$: 存储隐藏状态。
    - $y$: 伪标签，表示实际选择的动作（1=向上，0=向下）。
    - $dlogps$: 策略梯度的核心，$y - aprob$表示预测概率与实际动作的偏差。

```python
  observation, reward, done, info = env.step(action)
  reward_sum += reward
  drs.append(reward)
```

- **环境交互**:
    - $env.step(action)$: 执行动作，获取新画面、奖励、是否结束和其他信息。
    - $reward$: Pong中为+1（得分）、-1（失分）或0（进行中）。
    - 累积奖励并记录。

#### **回合结束处理**

```python
  if done:  # 一个回合结束
    episode_number += 1
    epx = np.vstack(xs)
    eph = np.vstack(hs)
    epdlogp = np.vstack(dlogps)
    epr = np.vstack(drs)
    xs, hs, dlogps, drs = [], [], [], []  # 重置
```

- **堆叠数据**:
    - 将一个回合内的所有数据堆叠为矩阵，便于批量处理。
    - 重置列表，准备下一回合。

```python
    discounted_epr = discount_rewards(epr)
    discounted_epr -= np.mean(discounted_epr)
    discounted_epr /= np.std(discounted_epr)
```

- **奖励处理**:
    - 计算折扣奖励。
    - **标准化**: 减均值除以标准差，使奖励分布更稳定，减少梯度方差。

```python
    epdlogp *= discounted_epr  # 用优势调制梯度
    grad = policy_backward(eph, epx, epdlogp)
    for k in model: grad_buffer[k] += grad[k]
```

- **策略梯度**:
    - $epdlogp *= discounted_epr$: 用折扣奖励（优势）调制梯度，强化"好"动作，削弱"坏"动作。
    - 计算梯度并累积到$grad_buffer$。

```python
    if episode_number % batch_size == 0:
      for k, v in model.iteritems():
        g = grad_buffer[k]
        rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
        model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
        grad_buffer[k] = np.zeros_like(v)
```

- **RMSProp更新**:
    - 每$batch_size$次回合更新一次参数。
    - **公式**:
        - 缓存更新: $$ cache = decay \cdot cache + (1-decay) \cdot g^2 $$
        - 参数更新: $$ W += lr \cdot g / (\sqrt{cache} + \epsilon) $$
    - $1e-5$防止除零。

```python
    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
    print 'resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward)
    if episode_number % 100 == 0: pickle.dump(model, open('save.p', 'wb'))
    reward_sum = 0
    observation = env.reset()
    prev_x = None
```

- **监控与保存**:
    - $running_reward$: 平滑平均奖励，反映训练进展。
    - 每100回合保存模型。
    - 重置环境和变量。

```python
  if reward != 0:
    print ('ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!')
```

- **奖励反馈**: 游戏结束时（得分或失分），打印结果。

---

### **总结与原理**

1. **强化学习核心**:
    
    - 这是一个**无模型的策略梯度方法**，直接优化动作概率，不需要Q值或价值函数。
    - 通过试错（随机动作）和奖励反馈调整网络，使AI学会"接球"。

2. **神经网络**:
    
    - 输入：差分图像（捕捉运动）。
    - 输出：向上移动的概率。
    - 训练：用折扣奖励调制梯度，优化参数。

3. **关键技巧**:
    
    - **差分图像**: 聚焦动态信息。
    - **奖励折扣**: 重视长期效果。
    - **标准化**: 稳定训练。
    - **RMSProp**: 自适应学习率。

# GPU运行的代码

使用CPU跑强化学习实在是太慢了，但是好在google colab上提供了免费的T4，而且我们的模型参数量并不是很大，所以完全可以放心玩！

首先就是需要安装好环境，（很简单，都是pip install）：

```shell
pip install gymnasium[atari] ale-py torch
```

然后直接运行下面的使用torch重写的代码就可以了：

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import pickle
import os

# 超参数
H = 200               # 隐藏层神经元数
batch_size = 32       # 可调整为2的幂次方
learning_rate = 1e-4
gamma = 0.99          # 奖励折扣因子
resume = True         # 是否从检查点恢复
test = False          # 测试模式关闭（Colab中不支持渲染）
save_file = 'pong_model_pytorch.pt'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

# 定义策略网络
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# 初始化模型
D = 80 * 80  # 输入维度
model = PolicyNetwork(D, H).to(device)

if resume and os.path.exists(save_file):
    model.load_state_dict(torch.load(save_file))
    print("Loaded model from checkpoint.")

optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.99, eps=1e-5)

# 预处理函数
def prepro(I):
    """ 将210x160x3图像预处理为80x80的一维 float 数组 """
    I = I[35:195]         # 裁剪
    I = I[::2, ::2, 0]     # 降采样：步长为2
    I[I == 144] = 0       # 消除背景1
    I[I == 109] = 0       # 消除背景2
    I[I != 0] = 1         # 其他部分置1
    return I.astype(np.float32).ravel()

# 折扣奖励函数
def discount_rewards(r):
    r = np.array(r)
    discounted_r = np.zeros_like(r, dtype=np.float32)
    running_add = 0
    for t in reversed(range(r.size)):
        if r[t] != 0:
            running_add = 0  # 游戏边界，重置累加器（Pong特有）
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

# 创建环境（在Colab中不启用渲染）
env = gym.make("ale_py:ALE/Pong-v5")  # 不指定 render_mode

# 初始化变量
observation, _ = env.reset()
prev_x = None
# xs 用于记录观察（如果后续需要），rewards记录每步奖励，probs记录每步预测的概率，actions记录实际采取的动作标签（1代表action 2，0代表action 3）
xs, rewards, probs, actions = [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 0

while True:
    # 预处理当前观察，并计算与上一帧的差值
    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = cur_x

    # 转为tensor并传入网络得到动作概率
    x_tensor = torch.FloatTensor(x).to(device)
    prob = model(x_tensor)

    # 根据概率选择动作，并记录对应的标签 y（1: 动作2, 0: 动作3）
    if test:
        if prob.item() > 0.5:
            action = 2
            y = 1
        else:
            action = 3
            y = 0
    else:
        if np.random.uniform() < prob.item():
            action = 2
            y = 1
        else:
            action = 3
            y = 0

    xs.append(x)
    probs.append(prob)
    actions.append(y)  # 记录当前采取的动作标签

    # 与环境交互
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    reward_sum += reward
    rewards.append(reward)

    if done:
        episode_number += 1

        # 计算折扣奖励并标准化
        discounted_rewards = discount_rewards(np.array(rewards))
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)
        discounted_rewards = torch.FloatTensor(discounted_rewards).to(device)

        if not test:
            optimizer.zero_grad()
            policy_loss = []
            # 注意：此处遍历每一步，使用记录下来的 prob、动作标签和对应折扣奖励
            for prob, y, r in zip(probs, actions, discounted_rewards):
                # 若 y==1 则取 log(prob)，否则取 log(1-prob)
                log_prob = torch.log(prob if y == 1 else 1 - prob)
                policy_loss.append(-log_prob * r)
            policy_loss = torch.stack(policy_loss).sum()
            policy_loss.backward()
            optimizer.step()

        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print(f'ep {episode_number}: reward: {reward_sum}, running mean: {running_reward:.2f}')

        if episode_number % 100 == 0 and not test:
            torch.save(model.state_dict(), save_file)

        # 重置环境和存储变量
        reward_sum = 0
        observation, _ = env.reset()
        prev_x = None
        xs, rewards, probs, actions = [], [], [], []

    if reward != 0:
        print(f'ep {episode_number}: game finished, reward: {reward}' + (' !!!!!!!!' if reward == 1 else ''))
```

