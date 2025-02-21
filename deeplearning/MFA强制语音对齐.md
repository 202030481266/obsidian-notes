# 场景

最近公司内部接了一个单子，业务需求简单来讲就是给一个长音频，然后一段大致准确的参考文本，需要按照文本的句子划分将音频切块。这个看起来像是一个ASR任务，又不太像，因为本质上并不是为了获取文本而是为了切分音频，而且已经有了精确的文本参考。（但是后来得知给定的参考文本不一定是完全正确的）后来在网上搜了一下解决方案，发现可以使用一种叫做强制语音对齐的技术来解决，而这个技术最主流的方法库是Montreal Forced Aligner (MFA)。

官方文档：[MFA官方文档](https://montreal-forced-aligner.readthedocs.io/)
官方代码：[MFA官方开源代码](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner)
官方提供的模型：[MFA字典、预训练模型下载](https://github.com/MontrealCorpusTools/mfa-models)

---
# 安装依赖环境

在介绍MFA的原理之前，那肯定是先要解决最最恶心的环境问题。MFA 是一个基于 Kaldi 的强制对齐工具，**安装过程依赖于 Python 环境（推荐使用 Anaconda）和一些必要的依赖**项。然而实际上除了使用conda我几乎没有找到任何有效的方法安装这个玩意。

***切记：使用Python3.9！！！***

首先即使需要有一个conda环境，不妨称为aligner：

```shell
conda create -n aligner -c conda-forge python=3.9
conda activate aligner
```

然后使用conda来安装，这一步会自动安装 MFA 以及必要的依赖（如 Kaldi、OpenFST 等），下载需要一定的时间。

```shell
conda install -c conda-forge montreal-forced-aligner
```

如果上面的命令能够成功跑通基本上就下载完成了，可以使用下面的命令来验证是否下载成功：

```bash
mfa version
```

如果下载过程除了差错或者出现了网络问题，多跑几次就行了。

# MFA的原理

## 什么是强制语音对齐？

强制语音对齐是一种技术，目的是把一段语音和对应的文本“对齐”。简单来说，就是弄清楚这句话里的每个字、每个音是在音频的哪个时间点说出来的。比如，你有一句录音“今天天气很好”，对齐之后，就可以知道“今”是从第0.2秒到0.5秒，“天”是从0.5秒到0.8秒，等等。
## MFA是怎么做到的？

MFA的核心是用计算机模型把音频和文本“强行匹配”起来。它主要依赖以下几个步骤和技术：

 1. **准备阶段：声音和文本的输入**

- 这个过程，需要提供两样东西：一段音频文件（比如MP3或WAV，通常来说是wav文件）和对应的文本（比如一句完整的话）。
- MFA会把文本分解成更小的单位，比如一个个单词，甚至是音素（音素是语言里最小的发音单位，比如“t”、“a”这样的基本声音）。

 2. **声音的特征提取**

- 计算机听不懂音频波形，所以MFA先把音频变成一种“数字语言”。它会分析音频的声波，提取出一些特征，比如声音的频率、响度变化等。这些特征就像音频的“指纹”，能帮助计算机理解声音的模式。
- 这有点像把一段音乐变成乐谱，方便计算机去“读”它。

3. **语言模型和声学模型**

- MFA用到了两个关键的“助手”：  
    - **声学模型**：这个模型知道每个音素听起来是什么样子。比如，它知道“b”听起来是短促的爆破音，“a”是个长点的元音。它会根据音频特征去猜测每个时间段可能对应哪个音素。
    - **语言模型**：这个模型了解语言的规则，比如中文里“天”后面可能会跟“气”，而不是“猫”。它帮着确保对齐结果符合语言逻辑。
- 这两个模型通常是通过大量语音数据训练出来的，MFA默认会用预训练好的模型（比如英语或中文的），但是也可以完全自己训练。

 4. **强制对齐的过程**

- 现在，MFA会把音频和文本“强行”匹配起来。它的做法是：  
    - 把文本拆成音素序列（比如“今天”可能是“j-in t-ian”）。  
    - 然后拿音频的特征去比对，看看哪个时间段的声音最像“j”，哪个像“in”，以此类推。
    - 这里用到了一个叫 **隐马尔可夫模型（HMM）** 的东西，虽然名字听起来复杂，但实际上可以把它想象成一个“时间分配器”。它会根据概率计算 出每个音素最可能出现的时间范围。
- 因为是“强制”对齐，**MFA不会改变文本内容，它只会调整时间点，确保音频和文本完全吻合**。

5. **输出结果**

- 对齐完成后，MFA会给你一个文件（通常是TextGrid格式），里面记录了每个音素或单词的起始时间和结束时间。比如：  
    - “今”：0.2秒 - 0.5秒  
    - “天”：0.5秒 - 0.8秒
- 可以用可视化工具（比如Praat软件）打开这个文件，看看对齐效果，甚至拿来做进一步分析。
## 关键技术

其实本质上，MFA的技术依赖于HMM+GMM的声学建模。HMM负责时序建模，GMM则负责判断每一帧的状态概率分布，比方说对于第x帧，能够凭借第x帧的特征向量判断是属于哪个状态（一般来说有三个状态，开始，中间，结束）。而HMM干的啥事情呢？

HMM则负责建模整个序列的状态转移，哎，这不就是和jieba分词一样了吗？假设当前状态只会收到上一个状态的影响，那么整个HMM的建模实际上和jieba的一模一样的。

下面说一个例子吧，比如对于单词`cat`，有30帧数据，每一帧都对应一个13维的MFCC信号特征向量。顺便科普一下，**梅尔频率倒谱系数 (MFCC)** 作为声学特征，捕捉语音信号的频谱特性。MFCC 的提取包括预加重、分帧、傅里叶变换、梅尔滤波器组、对数运算和离散余弦变换等步骤，生成低维特征向量（通常是 13 维，加上速度和加速度后扩展到 39 维）。

首先将`cat`转换为音素序列：`/k/、/æ/、/t/`，不妨假设其中的一帧MFCC向量为：

```
[0.1, 0.3, 0.5, ..., 0.2]  （13维）
```

- 对于/k/的中间状态，GMM计算该特征属于该状态的概率。假设GMM由以下高斯分布组成：
    - 高斯1：均值=[0.2, 0.4, 0.6, ...]，方差=0.1
    - 高斯2：均值=[0.0, 0.1, 0.3, ...]，方差=0.2
    - ...
- GMM计算该特征属于/k/中间状态的总概率（各高斯分布的加权和）
GMM可以得到改特征下对于/k/的所有状态的概率，也就是说GMM维护了一个特征到状态的概率分布，输入特征向量就可以得到对应的状态是开始、还是中间，或者是结束。

虽然说可以用GMM得到每一帧可能的状态，但是要明白音频是一个序列，当前状态可不是单纯的说GMM中概率最高的谁就是谁，还和上一个状态有很大关系的！因此HMM要发挥作用了，假设当前状态只和上一个状态有关，那么我们就可以维护一个状态转移的概率分布了！假设三个状态：S1（开始）、S2（中间）、S3（结束）。HMM的状态转移概率如下：

```
S1 -> S2: 0.9
S2 -> S2: 0.1
S2 -> S3: 0.8
S3 -> S3: 0.2
```

- 如果输入语音的某一段对应/k/音素，HMM会计算每一帧属于哪个状态的概率，并结合状态转移概率，找到最可能的状态序列。
- 例如，可能的状态序列是：$S1 \rightarrow S2 \rightarrow S2 \rightarrow S3 \rightarrow S3$。

知道了HMM和GMM的作用，那么要怎么训练呢？这里就要和jieba分词打个招呼了，因为方法就是使用动态规划算法（维特比算法）找到最大的可能路径。

这是一个很巧妙的算法，所以下面详细讲讲，并且会结合例子计算。
### Viterbi算法的核心思想

Viterbi算法的目标是找到一条最可能的状态序列，使得：
1. 状态序列符合HMM的状态转移概率。
2. 每一帧的观测特征（由GMM计算）与状态序列匹配。

算法的核心是**动态规划**，通过递推的方式计算每一帧每个状态的最大概率，并记录路径。

---
### Viterbi算法的步骤

假设：
- 有 $N$ 个状态（如/k/的开始、中间、结束状态）。
- 有 $T$ 帧观测特征（如MFCC特征）。
- 状态转移概率矩阵为 $A$，其中 $A_{ij}$ 表示从状态 $i$ 转移到状态 $j$ 的概率。
- 观测概率矩阵为 $B$，其中 $B_j(o_t)$ 表示在状态 $j$ 下观测到特征 $o_t$ 的概率。
- 初始状态概率为 $\pi$，其中 $\pi_i$ 表示初始状态为 $i$ 的概率。

#### 1. **初始化**

对于第一帧（$t=1$），计算每个状态 $i$ 的概率：
$$
\delta_1(i) = \pi_i \cdot B_i(o_1)
$$
其中：
- $\delta_1(i)$ 表示在第一帧处于状态 $i$ 的概率。
- $\pi_i$ 是初始状态概率。
- $B_i(o_1)$ 是在状态 $i$ 下观测到第一帧特征 $o_1$ 的概率。

同时，记录路径：
$$
\psi_1(i) = 0
$$
（因为第一帧没有前驱状态。）

---

#### 2. **递推**

对于每一帧 $t = 2, 3, \dots, T$，计算每个状态 $j$ 的最大概率：
$$
\delta_t(j) = \max_{1 \leq i \leq N} \left[ \delta_{t-1}(i) \cdot A_{ij} \right] \cdot B_j(o_t)
$$
其中：
- $\delta_t(j)$ 表示在第 $t$ 帧处于状态 $j$ 的最大概率。
- $\delta_{t-1}(i)$ 是前一帧状态 $i$ 的概率。
- $A_{ij}$ 是从状态 $i$ 转移到状态 $j$ 的概率。
- $B_j(o_t)$ 是在状态 $j$ 下观测到第 $t$ 帧特征 $o_t$ 的概率。

同时，记录路径：
$$
\psi_t(j) = \arg\max_{1 \leq i \leq N} \left[ \delta_{t-1}(i) \cdot A_{ij} \right]
$$
（即选择使 $\delta_t(j)$ 最大的前一帧状态 $i$。）

---

#### 3. **终止**

在最后一帧（$t=T$），找到最大概率的状态：
$$
P^* = \max_{1 \leq i \leq N} \delta_T(i)
$$
以及对应的状态：
$$
q_T^* = \arg\max_{1 \leq i \leq N} \delta_T(i)
$$

---

#### 4. **回溯**

从最后一帧开始，回溯找到完整的状态序列：
$$
q_t^* = \psi_{t+1}(q_{t+1}^*), \quad t = T-1, T-2, \dots, 1
$$

---

### 举个例子

假设：
- 有2个状态：S1和S2。
- 有3帧观测特征：$o_1, o_2, o_3$。
- 状态转移概率矩阵 $A$：

```
S1 -> S1: 0.6 
S1 -> S2: 0.4 
S2 -> S1: 0.3 
S2 -> S2: 0.7
```

- 观测概率矩阵 $B$：

```
S1: B1(o1)=0.5, B1(o2)=0.4, B1(o3)=0.7 
S2: B2(o1)=0.1, B2(o2)=0.6, B2(o3)=0.2
```

- 初始状态概率 $\pi$：

```
S1: 0.8 
S2: 0.2
```

#### 1. **初始化**

对于第一帧：
$$
\delta_1(S1) = 0.8 \cdot 0.5 = 0.4
$$
$$
\delta_1(S2) = 0.2 \cdot 0.1 = 0.02
$$

#### 2. **递推**

对于第二帧：
$$
\delta_2(S1) = \max(0.4 \cdot 0.6, 0.02 \cdot 0.3) \cdot 0.4 = 0.24 \cdot 0.4 = 0.096
$$
$$
\delta_2(S2) = \max(0.4 \cdot 0.4, 0.02 \cdot 0.7) \cdot 0.6 = 0.16 \cdot 0.6 = 0.096
$$

对于第三帧：
$$
\delta_3(S1) = \max(0.096 \cdot 0.6, 0.096 \cdot 0.3) \cdot 0.7 = 0.0576 \cdot 0.7 = 0.04032
$$
$$
\delta_3(S2) = \max(0.096 \cdot 0.4, 0.096 \cdot 0.7) \cdot 0.2 = 0.0672 \cdot 0.2 = 0.01344
$$

#### 3. **终止**

在第三帧，最大概率的状态是 S1，概率为 0.04032。

#### 4. **回溯**

通过回溯路径，找到完整的状态序列。
### 总结

Viterbi算法通过动态规划的方式，高效地找到最可能的状态序列。它的核心是：
1. **初始化**：计算第一帧每个状态的概率。
2. **递推**：计算每一帧每个状态的最大概率，并记录路径。
3. **终止**：找到最后一帧的最大概率状态。
4. **回溯**：回溯找到完整的状态序列。

---
### 如何强制对齐文本

所谓“强制”，是指MFA假设输入的文本是正确的，它不会尝试识别音频中未出现在文本中的内容，也不会忽略文本中的任何部分。换句话说，**它强制音频与文本一一对应**，哪怕这会导致对齐质量下降（比如音频和文本实际不匹配时）文本信息一定是完全匹配的，这是如何做到的呢？实际上，这个并不是很难。MFA在对齐的时候，**路径是固定死的**！比如序列：`/k/、/æ/、/t/`，这个序列永远不会改变。如果第10帧正确的是`k`，但是GMM认为最大可能是`t`，那么也不会采样为`t`，因为这里会发生强制采样。HMM构建的最大可能路径始终都是符合一开始的序列顺序，不会出现在某个位置上采样的字符和给定的文本字符不一致的情况。

---
## 使用MFA对齐葡萄牙语

说这么多，其实还是要练，使用MFA其实非常简单，主要就是下载好声学模型（可以自己训练，但是大多数时候使用官方提供的）和对应字典。

下载葡萄牙语的声学模型（`portuguese_mfa`）：

```shell
mfa model download acoustic portuguese_mfa
```

然后下载对应的语言发音字典：（`portuguese_mfa`，有多个版本，要调和模型一致的）

```shell
mfa model download dictionary portuguese_mfa
```

然后按照下面的格式就可以进行对齐了：

```shell
mfa align 输入文件夹 葡萄牙语词典 葡萄牙语声学模型 输出文件夹
```

比如我的命令就是：

```shell
mfa align ./my_corpus portuguese_mfa portuguese_mfa ./output
```

如果正常运行（无需GPU，速度还挺快的），就可以在输出的文件夹下得到TextGrid文件了，这个文件包含了所有的词级别的音频对齐信息和发音级别的音频对齐信息。

---
## 数据分析处理

回到公司一开始的业务，由于我们只拿到了词级别的对齐信息，但是如何将其匹配到原始文本断句呢？这个其实写一个简单的双滑动窗口就行了，一个窗口维护单词，一个窗口维护当前句子的开始和结束。下面是一个示例代码：

```python
import re

def parse_textgrid(textgrid_content):
    """
    解析 TextGrid 文件内容，提取 items 信息并转换为列表格式
    
    参数:
    textgrid_content (str): TextGrid 文件的内容
    
    返回:
    list: 包含所有 items 信息的列表，每个 item 是一个字典
    """
    items = []
    current_item = {}
    in_item = False
    
    # 按行分割内容
    lines = textgrid_content.split('\n')
    
    for line in lines:
        line = line.strip()
        
        # 开始新的 item
        if line.startswith('item ['):
            if current_item:
                items.append(current_item)
            current_item = {'intervals': []}
            in_item = True
            continue
            
        if in_item:
            # 解析 item 类型
            if 'class' in line:
                current_item['class'] = line.split('=')[1].strip(' "')
                
            # 解析 name
            elif 'name' in line:
                current_item['name'] = line.split('=')[1].strip(' "')
                
            # 解析时间点
            elif 'xmin' in line:
                current_interval = {}
                current_interval['xmin'] = float(line.split('=')[1].strip())
            elif 'xmax' in line:
                current_interval['xmax'] = float(line.split('=')[1].strip())
                
            # 解析文本内容
            elif 'text' in line:
                current_interval['text'] = line.split('=')[1].strip(' "')
                current_item['intervals'].append(current_interval)
    
    # 添加最后一个 item
    if current_item:
        items.append(current_item)
        
    return items

def filter_mute_chars(chunk_text: str) -> str:
    special_chars = [ # lab文件中可能出现的标记，但是不出现在语音中
        re.compile(r'<.*?>'),
    ]
    for pattern in special_chars:
        chunk_text = pattern.sub('', chunk_text)
    return chunk_text

def main():
    # 读取 TextGrid 文件
    with open('./output/chunk_1.TextGrid', 'r', encoding='utf-8') as f:
        content = f.read()

    # 解析TextGrid文件内容
    items = parse_textgrid(content)
    text = []
    times = []
    for item in items[1]['intervals']:
        if item['text'] != '':
            text.append(item['text'])
            times.append((item['xmin'], item['xmax']))
    
    # 三元组的形式写入到文件中
    with open('./output/chunk_1_triple.txt', 'w', encoding='utf-8') as f:
        for i in range(len(text)):
            f.write(f"{text[i]} {times[i][0]} {times[i][1]}\n")
    
    # 读取原始的chunk
    with open('./my_corpus/mid_corpus/chunk_1.lab', 'r', encoding='utf-8') as f:
        chunk_text = f.read()
    
    # 标准化
    invalid_chars = ' .,?!-;:()*"“”/' # 分隔符
    sentence_end_chars = ".?!" # 句子结束符
    special_chars = [
        re.compile(r'<.*?>'),
    ]
    special_tokens = [ # MFA词典中的特殊符号
        '<eps>',
        '<unk>',
        '[bracketed]',
        '<cutoff>',
        '[laughter]',
        '#0',
        '<s>',
        '</s>'
    ]
    
    def invalid(c: str) -> bool:
        return c.isspace() or c in invalid_chars

    def is_sentence_end(c: str) -> bool:
        return c in sentence_end_chars

    def is_special_char(c: str) -> bool:
        return any(pattern.match(c) for pattern in special_chars)
    
    def is_special_token(c: str) -> bool:
        return c in special_tokens

    if not is_sentence_end(chunk_text[-1]):
        chunk_text += '.' # 确保chunk_text以句子结束符结尾

    # 将chunk_text中的词提取出来
    j = 0
    n = len(chunk_text)
    pre = -1
    left = -1
    st_time = -1
    ed_time = -1
    sentences = []
    for i in range(n):
        if not invalid(chunk_text[i]):
            if pre == -1:
                pre = i
            if left == -1:
                left = i
        else:
            if pre != -1:
                word = chunk_text[pre:i]
                if word.lower() == text[j]:
                    if st_time == -1:
                        st_time = times[j][0]
                    ed_time = times[j][1]
                    j += 1
                    while j < len(text) and is_special_token(text[j]):
                        # 忽略特殊符号
                        j += 1
                pre = -1
            if is_sentence_end(chunk_text[i]):
                # [left, i)
                if left != -1:
                    sentences.append((chunk_text[left:i+1], st_time, ed_time))
                    st_time = -1
                    ed_time = -1
                    left = -1
    # 将句子写入文件
    with open('sentences.txt', 'w', encoding='utf-8') as f:
        for text, st_time, ed_time in sentences:
            f.write(f"{text}  {st_time}  {ed_time}\n")

if __name__ == '__main__':
    main()
```