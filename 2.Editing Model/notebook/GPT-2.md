# GPT-2 学习笔记

> 这份笔记围绕你前面问过的内容整理：GPT-2 的整体框架、一个简化版实现、各模块张量流动、`generate` 流程、数据集构造，以及实现时最容易踩的坑。

---

## 1. GPT-2 一句话理解

GPT-2 本质上是一个 **纯 Decoder 的自回归语言模型**。

它做的事情可以概括为：

- 输入一串 token
- 通过多层 Transformer Decoder Block 建模上下文
- 在每个位置预测“下一个 token”
- 训练时做 next-token prediction
- 推理时通过 `generate()` 一个一个往后生成

---

## 2. GPT-2 整体结构

一个简化版 GPT-2 通常包含：

1. **Token Embedding**：把 token id 变成向量
2. **Position Embedding**：给每个位置一个位置向量
3. **N 个 Transformer Block**
   - Masked Multi-Head Attention
   - Feed Forward Network
   - Add & Norm
4. **Final LayerNorm**
5. **LM Head（Linear）**：映射到词表维度，输出 logits

可以写成：

```text
输入 token ids
-> token embedding + position embedding
-> 多层 Transformer Block
-> final layer norm
-> lm_head
-> logits
-> softmax（生成时）
```

---

## 3. `@dataclass` 在配置类里的作用

在代码里常见：

```python
from dataclasses import dataclass

@dataclass
class GPTConfig:
    max_seq_len: int = 512
    batch_size: int = 12
    n_layer: int = 6
    n_head: int = 12
    hidden_dim: int = 768
    head_dim: int = hidden_dim // n_head
    dropout: float = 0.1
    vocab_size: int = 50257
```

### 作用

`@dataclass` 适合“主要存数据”的类，它会自动帮你生成：

- `__init__`
- `__repr__`
- `__eq__`

所以配置类不用手写很多样板代码。

### 这里的配置项含义

- `max_seq_len`：模型一次最多处理多少个 token
- `n_layer`：Block 数量
- `n_head`：多头注意力里 head 数
- `hidden_dim`：隐藏维度，也就是 embedding 维度
- `head_dim`：每个 head 的维度，通常 `hidden_dim // n_head`
- `vocab_size`：词表大小

---

## 4. `vocab_size` 到底有什么用

`vocab_size` 可以理解成：

> 模型认识的 token 总数。

它主要影响两个地方：

### 4.1 输入端：Token Embedding

```python
self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
```

它的 weight 形状是：

```python
[vocab_size, hidden_dim]
```

每一行对应一个 token 的向量。

### 4.2 输出端：LM Head

```python
self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
```

它把每个位置的隐藏状态映射到词表空间，输出：

```python
[B, T, vocab_size]
```

即：每个位置都对词表里所有 token 打分。

### 一句话记忆

- 输入时：`vocab_size` 决定“能查多少个 token 的 embedding”
- 输出时：`vocab_size` 决定“每步要在多少个 token 里做分类”

---

## 5. Token Embedding 和 Position Embedding

### 5.1 `token_emb` 是什么

表示：

> 这个 token 本身的语义向量。

例如：

```python
x = [5, 9, 2]
```

假设对应：

- 5 -> `I`
- 9 -> `love`
- 2 -> `you`

那么：

```python
token_emb = self.token_embedding(x)
```

得到的是这三个 token 的语义表示。

### 5.2 `position_emb` 是什么

表示：

> 这个 token 在序列里的位置向量。

如果 `seq_len = 3`，位置编号就是：

```python
[0, 1, 2]
```

所以代码里会写：

```python
pos_emb = self.position_embedding(torch.arange(seq_len, device=x.device))
```

这里 `torch.arange(seq_len)` 的作用是生成：

```python
[0, 1, 2, ..., seq_len-1]
```

即“当前位置编号”。

### 5.3 为什么二者可以直接相加

```python
x = token_emb + pos_emb
```

含义是：

> 最终输入表示 = token 语义信息 + 位置信息

这不是说后面还要减回去。位置没有消失，而是被编码进了最终表示里。

从线性层角度看：

```python
W(token_emb + pos_emb) = W(token_emb) + W(pos_emb)
```

所以后续网络依然能同时利用这两部分信息。

### 一句话记忆

- `token_emb`：这个 token 是谁
- `position_emb`：这个 token 在第几位
- 相加后：这个 token 在当前位置上的最终输入表示

---

## 6. 单头注意力 `SingleHeadAttention`

一个简化实现如下：

```python
class SingleHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head_dim = config.head_dim

        self.q = nn.Linear(config.hidden_dim, config.head_dim)
        self.k = nn.Linear(config.hidden_dim, config.head_dim)
        self.v = nn.Linear(config.hidden_dim, config.head_dim)

        self.register_buffer(
            'attention_mask',
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, _ = x.size()
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        score = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        score = score.masked_fill(self.attention_mask[:T, :T] == 0, float('-inf'))
        weight = F.softmax(score, dim=-1)
        weight = self.dropout(weight)
        out = weight @ v
        return out
```

### 6.1 注意力计算流程

1. 先算 `q/k/v`
2. 再算分数矩阵：

```python
score = q @ k.transpose(-2, -1)
```

3. 再除以 `sqrt(head_dim)`
4. 再加 mask
5. 再做 softmax
6. 最后 `weight @ v`

### 6.2 为什么一定要在 softmax 前加 mask

正确流程是：

```python
score = q @ k^T / sqrt(d)
score = score.masked_fill(mask == 0, -inf)
weight = softmax(score)
out = weight @ v
```

如果你先 softmax 再 mask，会让未来位置已经参与分母计算，不对。

### 6.3 为什么缩放用 `sqrt(head_dim)` 而不是 `sqrt(hidden_dim)`

注意力公式里除的是：

```python
sqrt(d_k)
```

这里 `d_k` 是**每个 head 的维度**，不是总隐藏维度。

所以多头注意力里应该用：

```python
math.sqrt(self.head_dim)
```

这是一个很常见的坑。

---

## 7. `register_buffer` 的作用

代码里常见：

```python
self.register_buffer(
    'attention_mask',
    torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
)
```

### 它的含义

把这个张量注册成 **buffer**。

buffer 的特点：

- 是模型的一部分
- 不是可训练参数
- 不参与优化器更新
- 会随着 `model.to(device)` 一起搬到 GPU/CPU
- 会出现在 `state_dict()` 里

### 为什么适合 `attention_mask`

因为 `attention_mask` 只是一个固定规则模板，不需要训练。

### 和普通写法相比的好处

如果你只写：

```python
self.attention_mask = ...
```

它就只是普通属性，PyTorch 不会正式把它当作模块状态来管理。

---

## 8. 多头注意力 `MultiHeadAttention`

你看过两种写法：

### 写法 A：先写单头，再拼多个 head

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head = nn.ModuleList([
            SingleHeadAttention(config) for _ in range(config.n_head)
        ])
        self.proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        output = torch.cat([h(x) for h in self.head], dim=-1)
        output = self.proj(output)
        output = self.dropout(output)
        return output
```

### 写法 B：一次性算大矩阵，再 reshape 成多个头

这种更接近工程实现，效率通常更高，但对初学者不如写法 A 直观。

### 两种写法的关系

它们本质上都在做：

```text
多头 = 多个单头并行 -> concat -> output projection
```

### `nn.ModuleList` 和 `nn.Sequential` 的区别

- `ModuleList`：只是“把多个模块存起来”，forward 过程你自己写
- `Sequential`：模块按顺序自动依次执行

多头注意力里常用 `ModuleList`，因为你要自己做：

```python
torch.cat([h(x) for h in self.head], dim=-1)
```

---

## 9. Feed Forward Network（FFN）

简化版实现：

```python
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 4),
            nn.GELU(),
            nn.Linear(config.hidden_dim * 4, config.hidden_dim),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        return self.net(x)
```

### 作用

FFN 本质上就是一个逐位置的 MLP：

- 先升维 `hidden_dim -> 4 * hidden_dim`
- 非线性激活
- 再降回 `hidden_dim`

### 为什么有人说 FFN 就是 MLP

因为它本质上就是两层线性层 + 激活，只不过 Transformer 里它是逐 token 独立作用的。

---

## 10. 一个 Transformer Block

```python
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = MultiHeadAttention(config)
        self.ffn = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.hidden_dim)
        self.ln2 = nn.LayerNorm(config.hidden_dim)

    def forward(self, x):
        x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
```

### 这里的结构是 Pre-LN

即：

- 先 LayerNorm
- 再 Attention / FFN
- 再残差相加

### Block 做了什么

1. 通过注意力让每个 token 看上下文
2. 通过 FFN 做更强的非线性变换
3. 通过残差连接和 LayerNorm 保持训练稳定

---

## 11. `*` 解包为什么会出现在 `nn.Sequential` 里

代码里常见：

```python
self.blocks = nn.Sequential(
    *[Block(config) for _ in range(config.n_layer)]
)
```

### 里面先做了什么

```python
[Block(config) for _ in range(config.n_layer)]
```

会得到一个列表，比如：

```python
[Block(config), Block(config), Block(config)]
```

### `*` 的作用

`*` 是 **解包**。

它会把列表拆开，相当于：

```python
nn.Sequential(
    Block(config),
    Block(config),
    Block(config)
)
```

如果不加 `*`，你传进去的是“一个列表”，不是多个模块。

---

## 12. 完整 GPT 模型骨架

一个简化版 GPT 可以写成：

```python
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.hidden_dim)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_final = nn.LayerNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        B, T = x.size()

        token_emb = self.token_embedding(x)  # [B, T, C]
        pos_emb = self.position_embedding(torch.arange(T, device=x.device))  # [T, C]
        x = token_emb + pos_emb

        x = self.blocks(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)  # [B, T, vocab_size]

        loss = None
        if targets is not None:
            B, T, V = logits.size()
            logits = logits.view(B * T, V)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
```

---

## 13. `self.apply(self._init_weights)` 是什么

这句的作用是：

> 递归遍历整个模型，把每个子模块都交给 `_init_weights` 处理。

即：

```python
self.apply(self._init_weights)
```

相当于在说：

> 模型里的所有层，统一按我定义的规则初始化。

### `isinstance` 是什么

`isinstance(obj, Type)` 用来判断：

> 这个对象是不是某种类型的实例。

例如：

```python
isinstance(module, nn.Linear)
```

表示：当前传进来的 `module` 是不是线性层。

### 为什么 `Linear` 和 `Embedding` 分开判断

因为不同模块里参数结构不一样：

- `Linear` 有 `weight`，可能还有 `bias`
- `Embedding` 只有 `weight`

所以通常要分类处理。

### 一个注意点

你之前截图里的初始化代码里，`bias` 部分曾误写成再次初始化 `weight`。更合理写法是：

```python
if module.bias is not None:
    torch.nn.init.zeros_(module.bias)
```

---

## 14. Tie Weight（可选）

GPT-2 类代码里经常会提到：

```python
self.token_embedding.weight = self.lm_head.weight
```

这叫 **Tie Weight**，即输入 embedding 和输出 lm_head 共享同一份权重。

### 好处

- 减少参数量
- 经常对语言模型训练有帮助

### 为什么成立

- 输入端：从词表查向量
- 输出端：把隐藏状态映射回词表空间

这两者本来就在同一个“词表空间”上，可以共享。

---

## 15. `forward` 中 logits 为什么要 reshape

在训练阶段，`logits` 的形状是：

```python
[B, T, vocab_size]
```

而交叉熵通常希望输入是：

```python
[N, vocab_size]
```

所以会写：

```python
logits = logits.view(B * T, vocab_size)
targets = targets.view(B * T)
loss = F.cross_entropy(logits, targets)
```

### 含义

把 batch 中每个位置都当成一个分类样本：

- 一共有 `B*T` 个位置
- 每个位置都要在 `vocab_size` 个 token 里做分类

### 注意事项

你之前笔记里的代码有一个小 bug：

```python
if target is None:
    ...
else:
    ...
    targets = targets.view(...)
```

这里变量名要统一。要么全用 `target`，要么全用 `targets`。

---

## 16. `generate()` 的流程

### 16.1 核心思想

GPT-2 的生成是 **自回归生成**：

> 已知前面的 token，预测下一个 token；
> 再把这个新 token 拼回输入；
> 然后继续预测。

### 16.2 一个简化版实现

```python
def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]

        logits, _ = self(idx_cond)              # [B, T, vocab_size]
        logits = logits[:, -1, :]              # [B, vocab_size]
        probs = F.softmax(logits, dim=-1)      # 概率分布
        idx_next = torch.multinomial(probs, num_samples=1)  # [B, 1]
        idx = torch.cat((idx, idx_next), dim=1)

    return idx
```

### 16.3 为什么只取最后一个位置的 logits

因为输入序列里：

- 位置 0 的输出是在预测位置 1
- 位置 1 的输出是在预测位置 2
- ...
- **最后一个位置的输出** 才是在预测“下一个新 token”

所以：

```python
logits = logits[:, -1, :]
```

是在拿“下一个 token 的预测分数”。

### 16.4 为什么要 softmax

`logits` 只是原始分数，不是概率。

```python
probs = F.softmax(logits, dim=-1)
```

是把分数变成概率分布，方便后面选 token。

### 16.5 为什么是随机采样 `multinomial`

```python
idx_next = torch.multinomial(probs, num_samples=1)
```

这表示：按概率分布随机抽一个 token。

#### 好处

- 文本更自然
- 更有多样性

#### 不足

- 有随机性
- 可能不稳定

### 16.6 如果不想随机，可以用贪心解码

```python
idx_next = torch.argmax(probs, dim=-1, keepdim=True)
```

这表示永远选概率最大的 token。

#### 对比

- `argmax`：稳定，但容易死板、重复
- `multinomial`：有变化，但更随机

### 16.7 生成阶段的本质

你可以把 `generate()` 理解成 6 步：

1. 取当前上下文
2. 如果太长，只保留最后 `max_seq_len` 个
3. 喂给模型得到 logits
4. 只拿最后一个位置的 logits
5. 变成概率分布
6. 选出一个新 token，并拼回去

---

## 17. `MyDataset`：数据集输入部分怎么理解

一个简化版本如下：

```python
class MyDataset(Dataset):
    def __init__(self, path, block_size=512):
        self.enc = tiktoken.get_encoding("gpt2")
        self.block_size = block_size
        self.eos_token = self.enc.encode(
            "<|endoftext|>",
            allowed_special={"<|endoftext|>"}
        )[0]

        self.encoded_data = []
        self.max_lines = 1000
        raw_data = []

        import json
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= self.max_lines:
                    break
                try:
                    text = json.loads(line.strip())['text']
                    raw_data.append(text)
                except Exception:
                    continue

        full_encoded = []
        for text in raw_data:
            encoded_text = self.enc.encode(text)
            full_encoded.extend(encoded_text + [self.eos_token])

        for i in range(0, len(full_encoded), self.block_size):
            chunk = full_encoded[i:i+self.block_size+1]
            if len(chunk) < self.block_size + 1:
                chunk = chunk + [self.eos_token] * (self.block_size + 1 - len(chunk))
            self.encoded_data.append(chunk)

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx):
        chunk = self.encoded_data[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y
```

### 17.1 整体在干什么

它做了 5 件事：

1. 读取原始文本
2. 用 tokenizer 编码成 token ids
3. 在每段文本后加 `eos_token`
4. 把所有 token 拼成长序列，再切块
5. 构造 `(x, y)`，让 `y` 比 `x` 整体右移一位

### 17.2 为什么 `x = chunk[:-1]`，`y = chunk[1:]`

例如：

```python
chunk = [10, 20, 30, 40, 50]
```

则：

```python
x = [10, 20, 30, 40]
y = [20, 30, 40, 50]
```

这正对应 GPT 的训练目标：

> 给前面的 token，预测下一个 token。

### 17.3 为什么要 `block_size + 1`

因为你最后要构造两个长度都是 `block_size` 的序列：

- `x = chunk[:-1]`
- `y = chunk[1:]`

所以 chunk 本身必须有 `block_size + 1` 个 token。

### 17.4 为什么后面补 `eos_token`

如果最后一个 chunk 不够长，就用 `eos_token` 补齐。教学版代码里这样写最简单。

---

## 18. 数据读取里的几个现实问题

你前面碰到的几个问题，基本都很典型。

### 18.1 Windows 下的编码问题

如果你写：

```python
with open(path, 'r') as f:
```

在 Windows 里经常会默认按 `gbk` 读文件，从而报：

```text
UnicodeDecodeError: 'gbk' codec can't decode ...
```

更稳妥的写法是：

```python
with open(path, 'r', encoding='utf-8') as f:
```

如果还有 BOM 问题，可以试：

```python
encoding='utf-8-sig'
```

### 18.2 `.json` 和 `.jsonl` 不是一回事

你的代码里：

```python
for line in f:
    text = json.loads(line.strip())['text']
```

这默认文件格式是 **jsonl**，即：

```json
{"text": "样本文本1"}
{"text": "样本文本2"}
```

每一行都是一个独立 JSON。

如果文件其实是普通 JSON：

```json
[
  {"text": "样本文本1"},
  {"text": "样本文本2"}
]
```

那就不能逐行 `json.loads(line)`，而应该：

```python
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)
```

### 18.3 你之前的数据长度为 0 的原因

`DataLoader` 报：

```text
ValueError: num_samples should be a positive integer value, but got num_samples=0
```

本质是：

```python
len(dataset) == 0
```

也就是 `raw_data` 根本没读到有效样本，通常因为：

- 读文件格式不对（把 `.json` 当 `.jsonl` 读）
- 字段不是 `text`
- 文件内容已经乱码/错码
- 所有异常都被 `continue` 吞掉了

### 18.4 调试建议

数据读取阶段建议临时加打印：

```python
print("raw_data条数:", len(raw_data))
print("full_encoded长度:", len(full_encoded))
print("encoded_data条数:", len(self.encoded_data))
```

这样能快速判断问题卡在哪一步。

---

## 19. 关于 Mobvoi 序列猴子开源数据集的注意事项

你后来确认了：官方开源数据是真实存在的，但你最开始拿到的文件并不是正确可训练格式。

### 正确认识

- 仓库里 `images/` 里的 `png/gif` 不是训练数据
- 训练数据下载信息在 `docs/pretrain_open_corpus.md`
- 真正要下载的是压缩包
- 解压后拿到的才应该是 **JSONL** 数据文件

### 也就是说

如果你本地文件不是：

```json
{"text": "......"}
```

这种一行一个 JSON 的格式，那就要怀疑：

- 文件下错了
- 解压不对
- 拿到的是中间文件
- 或者编码已经坏了

---

## 20. 常见实现坑位总结

### 20.1 注意力缩放别写错

应该是：

```python
/math.sqrt(head_dim)
```

不是：

```python
/math.sqrt(hidden_dim)
```

### 20.2 `masked_fill` 的 mask 方向别反了

如果你的 mask 是下三角 1/0 模板：

```python
self.attention_mask[:T, :T] == 0
```

表示“非法位置填 `-inf`”。

别直接：

```python
masked_fill(self.attention_mask, -inf)
```

否则很可能把合法位置也填掉。

### 20.3 `forward` 里变量名要统一

如果函数签名是：

```python
def forward(self, x, targets=None):
```

那下面就也统一写 `targets`。

### 20.4 位置 embedding 的索引要用 `torch.arange(seq_len)`

因为 position embedding 查的是“位置 id”，不是 token id。

### 20.5 Windows 路径更推荐写 `/`

例如：

```python
'dataset/mobvoi_seq_monkey_general_open_corpus_1000.json'
```

比：

```python
'dataset\mobvoi_seq_monkey_general_open_corpus_1000.json'
```

更省心。

### 20.6 中文数据用 GPT-2 tokenizer 只是教学上能跑通

如果你真要认真做中文模型，通常会选更适合中文的 tokenizer，而不是直接套 GPT-2 的英文分词器。

### 20.7 `random_split([0.9, 0.1])` 之前先确认数据集长度 > 0

不然分完还是空，后面 `DataLoader` 会继续报错。

---

## 21. 你可以直接复习的“最小 GPT-2 心智模型”

### 输入阶段

```text
token ids
-> token embedding
-> position embedding
-> 相加
```

### 建模阶段

```text
多层 Block
每层 = Masked Multi-Head Attention + FFN + 残差 + LayerNorm
```

### 输出阶段

```text
hidden states
-> lm_head
-> logits
```

### 训练阶段

```text
x = chunk[:-1]
y = chunk[1:]
loss = cross_entropy(logits, y)
```

### 生成阶段

```text
当前上下文 -> logits -> 取最后一个位置 -> softmax -> 选一个 token -> 拼回去 -> 循环
```

---

## 22. 一个更稳的参考骨架（简化版）

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class GPTConfig:
    max_seq_len: int = 512
    n_layer: int = 6
    n_head: int = 12
    hidden_dim: int = 768
    dropout: float = 0.1
    vocab_size: int = 50257

    @property
    def head_dim(self):
        return self.hidden_dim // self.n_head


class SingleHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head_dim = config.head_dim
        self.q = nn.Linear(config.hidden_dim, config.head_dim)
        self.k = nn.Linear(config.hidden_dim, config.head_dim)
        self.v = nn.Linear(config.hidden_dim, config.head_dim)
        self.register_buffer(
            'attention_mask',
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, _ = x.shape
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        score = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        score = score.masked_fill(self.attention_mask[:T, :T] == 0, float('-inf'))
        weight = F.softmax(score, dim=-1)
        weight = self.dropout(weight)
        return weight @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([SingleHeadAttention(config) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.hidden_dim, 4 * config.hidden_dim),
            nn.GELU(),
            nn.Linear(4 * config.hidden_dim, config.hidden_dim),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_dim)
        self.ln2 = nn.LayerNorm(config.hidden_dim)
        self.att = MultiHeadAttention(config)
        self.ffn = FeedForward(config)

    def forward(self, x):
        x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.max_seq_len = config.max_seq_len
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.hidden_dim)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_final = nn.LayerNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        B, T = x.shape
        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(torch.arange(T, device=x.device))
        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, V = logits.shape
            logits = logits.view(B * T, V)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
```

---

## 23. 最后一页复习版

### GPT-2 最重要的 10 个点

1. GPT-2 是 **纯 Decoder、自回归语言模型**。  
2. 输入 = `token_emb + position_emb`。  
3. Position embedding 查的是位置编号，所以用 `torch.arange(seq_len)`。  
4. Attention 流程：`qk^T -> /sqrt(head_dim) -> mask -> softmax -> @v`。  
5. `register_buffer` 适合存 `attention_mask` 这种非训练张量。  
6. 多头注意力本质是多个单头并行再 concat。  
7. `self.apply(self._init_weights)` 是统一初始化全模型。  
8. 训练时目标是 `x -> 下一个 token`，所以 `y = x` 右移一位。  
9. 生成时只取最后一个位置的 logits，因为它对应“下一个 token”的预测。  
10. 数据读取阶段一定先确认：**编码、文件格式（json/jsonl）、字段名、数据条数**。

---

## 24. 你当前最适合继续关注的部分

如果后面继续往下学，建议按这个顺序：

1. 把 `MyDataset` 跑通，确认能正常产出 `(x, y)`  
2. 跑通 `forward()`，确认 logits / loss 的形状都对  
3. 用很小的数据跑一次 overfit  
4. 再研究 `generate()` 的采样策略（argmax / multinomial / top-k）  
5. 最后再去看更工程化的实现：KV cache、Flash Attention、tie weight、并行化等

