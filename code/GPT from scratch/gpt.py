import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

# hyperparameters
batch_size = 128 # how many independent sequences will we process in parallel?
block_size = 64 # what is the maximum context length for predictions?
max_iters = 60000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 8
n_layer = 6
dropout = 0.2

with open('addition_dataset.txt', 'r', encoding='utf-8') as f:
  text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {s:i for i,s in enumerate(chars)}
itos = {i:s for i,s in enumerate(chars)}
encode = lambda x: [stoi[c] for c in x]
decode = lambda x: ''.join([itos[c] for c in x])
n = int(0.9 * len(text))
data = torch.tensor(encode(text), dtype=torch.long)
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data) - block_size, (batch_size,))
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  x, y = x.to(device), y.to(device)
  return x,y

@torch.no_grad()
def estimate_loss():
  out = {}
  model.eval()
  for split in ['train', 'val']:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      X, Y = get_batch(split)
      logits, loss = model(X, Y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  model.train()
  return out

class Head(nn.Module):

  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(n_embd, head_size, bias=False)  # [B, T, C] @ [C, H] -> [B, T, H]
    self.value = nn.Linear(n_embd, head_size, bias=False) # bias = False, 因为后面要LayerNorm
    self.query = nn.Linear(n_embd, head_size, bias=False) 
    
    # torch的缓冲区，不会进行反向传播
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) 
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, x):
    B, T, C = x.shape
    k = self.key(x) # [B, T, H]
    q = self.query(x) # [B, T, H]
    # self-attention
    dk = k.shape[-1]
    # [B, T, H] @ [B, H, T] -> [B, T, T]
    wei = q @ k.transpose(-2, -1) * (dk**-0.5) 
    # mask, 实际上这里可以广播
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # mask the bool matrix where is true
    wei = F.softmax(wei, dim=-1) # [B, T, T], dim是一个必要的参数，对最后一个维度进行softmax
    wei = self.dropout(wei) # dropout，舍弃掉一些参数
    v = self.value(x) # [B, T, H]
    out = wei @ v # [B, T, T] @ [B, T, H] -> [B, T, H]
    return out

class MultiHeadAttention(nn.Module):

  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.proj = nn.Linear(n_embd, n_embd)
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, x):
    # [B, T, H] * num_heads ---cat---> [B, T, H * num_heads] -> [B, T, C]
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    # dropout 正则化加上了一个残差连接
    out = self.dropout(self.proj(out)) # [B, T, C] @ [C, C] -> [B, T, C]
    return out

class FeedFoward(nn.Module):

  def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(n_embd, 4 * n_embd),   # paper 512 -> 2048 -> 512
        nn.ReLU(),
        nn.Linear(4 * n_embd, n_embd),
        nn.Dropout(dropout)
    )
  
  def forward(self, x):
    return self.net(x)  # 可以直接调用Module的方法

class Block(nn.Module):

  def __init__(self, n_embd, n_head):
    super().__init__()
    head_size = n_embd//n_head
    self.sa = MultiHeadAttention(n_head, head_size)
    self.ffwd = FeedFoward(n_embd)

    # LayerNorm 两次
    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)

  def forward(self, x): 
    # 输入前进行归一化，这和论文有点不一样
    x = x + self.sa(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x


class GPTLanguageModel(nn.Module):

  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # [voc, C]
    self.position_embedding_table = nn.Embedding(block_size, n_embd) # [block, C] 
    self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
    self.ln_f = nn.LayerNorm(n_embd)
    self.lm_head = nn.Linear(n_embd, vocab_size)

  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      torch.nn.init.normal_(module.weight, mean=0.0,std=0.02)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def forward(self, idx, targets=None):
    B, T = idx.shape
    # embedding
    # [B, T] -> [B, T, C], [*] ---embedding---> [*,C]
    tok_emb = self.token_embedding_table(idx)
    # [T] -> [T, C]
    pos_emb = self.position_embedding_table(torch.arange(T, device=device)) 
    x = tok_emb + pos_emb # [B, T, C]
    x = self.blocks(x) # [B, T, C]
    x = self.ln_f(x) # [B, T, C]
    logits = self.lm_head(x) # [B, T, C] @ [B, C, voc] -> [B, T, voc]

    if targets is None:
      loss = None # infer stage
    else:
      B, T, voc = logits.shape
      logits = logits.view(B*T, voc)
      targets = targets.view(B*T) # [B, T]
      loss = F.cross_entropy(logits, targets)

    return logits, loss

  def generate(self, idx, max_new_token):
    for _ in range(max_new_token):
      # 我们这里没有special_token，所以不会自动停止
      idx_second = idx[:, -block_size:] # 截断 [B, T]
      logits, loss = self(idx_second) # [B, T, voc] 因为是推理阶段
      logits = logits[:, -1, :] # [B, voc]
      probs = F.softmax(logits, dim=-1) # [B, voc]
      idx_next = torch.multinomial(probs, num_samples=1) # [B, voc]
      idx = torch.cat([idx, idx_next], dim=1) # [B, T + 1]
    return idx

def eval_generate():
    context = torch.zeros((1,1), dtype=torch.long, device=device)
    res = decode(m.generate(context, max_new_token=500)[0].tolist())
    with open('result.txt', 'a+', encoding='utf-8') as f:
        f.write(res)
        f.write('\n')

model = GPTLanguageModel()
m = model.to(device)

# 使用AdamW优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

import logging

# 设置日志
logging.basicConfig(filename='train.log', level=logging.INFO, 
                    format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

print(sum(p.nelement() for p in m.parameters()) / 10**6, 'M parameters')

for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters-1:
        losses = estimate_loss()
        log_message = f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        logging.info(log_message)
        print(log_message)  # console output
    
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

eval_generate() # 经过训练

# 保存模型
torch.save(m.state_dict(), 'math_nano_gpt.pth')


