
# 相关资源

[Tiktoken Visualization App](https://tiktokenizer.vercel.app/)
[Karpathy's google notebook](https://colab.research.google.com/drive/1y0KnCFZvGVf_odSfcNAws6kcDD7HsI0L?usp=sharing#scrollTo=pkAPaUCXOhvW)
[Karpathy's Youtube tutorial](https://www.youtube.com/watch?v=zduSFxRajkE&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=10)
[MinBPE project](https://github.com/karpathy/minbpe)
[TikToken](https://github.com/openai/tiktoken)
[sentencepiece](https://github.com/google/sentencepiece)
[My google colab notebook](https://colab.research.google.com/drive/1_ItMM-_OhW4VlCc9gBLjOq20CXEOY2tk?authuser=0#scrollTo=quqPZMG3BQPU)
[Byte Pair Encoding and Data Structure](https://guillaume-be.github.io/2021-09-16/byte_pair_encoding)

# 练习

根据Karpathy的练习进行的代码练习： [MInBPE Exercise](https://github.com/karpathy/minbpe/blob/master/exercise.md)

# 构建RegexTokenizer

## 正则表达式

因为BaseTokenizer实现比较简单，所以下面直接跳到实现RegexTokenizer。GPT2论文——《Language Model are Unsupervised MultiTask Learners》中提出了一个观点：

![[GPT2 Tokenizer BPE Regex.png#center]]

这个观点的意思就是单纯的BPE算法会将里面经常出现的一些词语组合进行合并，然而这些词语组合并不是都有意义的，比如论文里面的例子(dog后面加了很多标点符号，无意义的变体！)所以在仓库中，它们的具体做法其实使用了正则表达式进行了一个词语划分：

```python
# Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
```

这个表达式非常复杂，下面是Karpathy的解释（==其实可以发现了，空格是其中的关键开头==）：
- `'s|'t|'re|'ve|'m|'ll|'d`: 这部分匹配常见的英语缩写，如 it's, don't, they're, I've, I'm, I'll, I'd 等。
- ` ?\p{L}+`: 匹配任何Unicode字母字符（可能前面有一个可选的空格）。`\p{L}` 表示任何种类的字母。
- ` ?\p{N}+`: 匹配任何Unicode数字字符（可能前面有一个可选的空格）。`\p{N}` 表示任何种类的数字。
- ` ?[^\s\p{L}\p{N}]+`: 匹配任何不是空白、字母或数字的字符（可能前面有一个可选的空格）。
- ` \s+(?!\S)`: 匹配一个或多个空白字符，**但只有在后面没有非空白字符的情况下**。这通常用于匹配行尾的空白。
- `\s+`: 匹配一个或多个空白字符。

当然，这里需要添加上一个`re.IGNORECASE`参数，否则BPE算法无法合并类似于DON'T,LET'S等这种词元结构。

## BPE算法实现

对于一个长期打算法竞赛的人来说，这个真的是没有难度。具体而言，BPE算法会将字符串里面的出现频率最高的词对进行重新编码合并（所以每一次都是词表大小加一，字符串长度减去这个词对出现的次数，是一种贪婪算法）。那么我们暴力实现即可，每一次都扫描字符串获得其中的出现最多次数的词对，然后再进行一个滑动窗口合并。

```python
def get_stats(self, ids):
    # count the pair <c1,c2> in ids that encode in UTF-8
    count = defaultdict(int)
    for id in ids:
      for p in zip(id, id[1:]):
        count[p] += 1
    return count
  
  def merge(self, ids, pair, index):
    # merge pair in ids
    new_ids = []
    for id in ids:
      new_id = []
      i = 0
      while i < len(id):
        if i+1 < len(id) and id[i] == pair[0] and id[i+1] == pair[1]:
          new_id.append(index)
          i += 2
        else:
          new_id.append(id[i])
          i += 1
      new_ids.append(new_id)
    return new_ids

  def train(self, text, vocab_size, verbose=False):  # 训练，增大词表
    # 使用GPT2一样的正则表达式来将文本进行分组
    pat_str = re.compile(r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""",re.IGNORECASE)
    print(f"The train text is {len(text)} characters long.")
    text_groups = re.findall(pat_str, text)
    ids = []
    for text_group in text_groups:
      ids.append(list(text_group.encode('utf-8')))
    iters = vocab_size - self.vocab_size
    for i in range(iters):
      stats = self.get_stats(ids)
      max_pair = max(stats, key=stats.get)
      idx = self.vocab_size + i
      ids = self.merge(ids, max_pair, idx)
      self.merge_stats[max_pair] = idx
      print(f"merging {max_pair} into a new token {idx}")
```

注意这里的训练有一个很重要的参数：`vocab_size`，这个参数决定了我们的BPE算法要进行多少次合并，是一个超参数。词表大小是一个玄学，并且非常影响模型的表现，Karpathy已经演示很多了很多tokenizer导致的badcase。如果参数较大，那么我们就可以获得更大的词表，并且可以让模型看到更长的上下文（因为我们显著减小了input_ids的长度），下面是几个缺点总结：

1. 计算效率：
   - 更大的词表意味着模型的**输入和输出层会变得更大**，增加了计算复杂度。
   - 这会导致训练和推理时间增加，同时也需要更多的内存和存储空间。

2. 稀疏性问题：
   - ==词表过大会导致许多词很少出现在训练数据中，这会使模型难以学习这些罕见词的有效表示==。
   - 稀疏性会增加过拟合的风险，特别是对于较小的训练集。

3. 泛化能力：
   - 较小的词表可以鼓励模型学习更通用的表示，而不是依赖于特定的、可能很少使用的词。
   - 这可以提高模型在处理未见过的文本时的泛化能力。

4. 子词单元的效果：
   - 现代LLM通常使用子词单元（subword units）而不是完整的词。
   - 较小的子词词表可以有效地表示大量的词，同时保持词表大小的可管理性。

5. 训练数据利用：
   - 较小的词表允许模型更有效地利用有限的训练数据，因为每个词或子词单元会有更多的出现次数。
   - 较小的词表迫使模型更好地处理未知词或罕见词，通常通过使用子词或字符级别的表示。

然而，相对来说，token级别的词表比字符级别的语言模型还是要好得多（可能是还没有出现一个超强的不需要tokenizer的模型~），总的来说，有下面的优点：

较大的词表大小相对于很小的词表大小确实也有一些优势：

1. 表达精度：
   - 更大的词表可以包含更多的完整词，这可以提高模型表达特定概念的精度。
   - 某些专业术语或罕见词汇可以直接表示，而不需要拆分成子词。

2. 减少歧义：
   - 大词表可以区分同音异义词或相似词，减少歧义。
   - 例如，"bank"（银行）和"bank"（河岸）可以有不同的表示。

3. 处理多语言：
   - 对于多语言模型，较大的词表可以更好地覆盖多种语言的词汇。

4. 捕捉细微差别：
   - 大词表可以保留更多词形变化，如时态、数量等，有助于捕捉语言的细微差别。

5. 减少序列长度：
   - 使用完整词而不是子词可以减少输入序列的长度，这在某些情况下可能有利于处理长文本。

6. 提高推理效率：
   - 在某些任务中，使用完整词可能比使用多个子词更高效，尤其是在推理阶段。

7. 特定领域适应性：
   - 在特定领域（如医学、法律）的应用中，大词表可以更好地包含领域特定术语。
   - 保留了更多的单词前后缀信息，以及更好识别命名实体。



