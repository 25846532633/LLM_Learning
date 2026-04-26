# 03 Transformer Decoder 笔记

## 1. 这一份 Notebook 在学什么

这一份 `03_Transformer_Decoder.ipynb`，是在前面 **Single-Head Self-Attention** 和 **Multi-Head Self-Attention** 的基础上，继续往前走一步：

> 不再只看“一个注意力模块怎么写”，而是开始把它拼成一个 **Decoder Block**，再把多个 Decoder Block 堆起来，形成一个最简化版的 Decoder-only 模型。

也就是说，这一份代码的核心目标，不是再单独解释 attention，而是让你看到：

1. 一个 **Decoder block** 里面到底有哪些子模块
2. 这些模块是按什么顺序连起来的
3. 残差连接、LayerNorm、FFN 在 decoder 里分别起什么作用
4. 多层堆叠以后，整个模型是怎么从 token id 走到最终预测分布的

你可以把这份 notebook 理解成：

- 前两份是“零件级学习”
- 这一份是“开始组装整机”

---

## 2. 先说结论：一个最基本的 Decoder 是由什么组成的

一个最基本的 Transformer Decoder block，通常包含两大部分：

### 第一部分：Masked Multi-Head Self-Attention

作用是：

> 让当前位置的 token 去看前面允许看到的 token，并聚合上下文信息。

这里之所以要 **masked**，是因为在语言模型里，当前位置不能偷看未来位置。

---

### 第二部分：Feed Forward Network（FFN）

作用是：

> 对 attention 输出的每个位置，再做一次更强的非线性变换。

它不负责 token 和 token 之间的信息交互，而是对每个 token 自己的表示做进一步加工。

---

### 再加上两类配套结构

每个子模块旁边通常还会带：

1. **Residual Connection（残差连接）**
2. **LayerNorm（层归一化）**
3. **Dropout（防止过拟合）**

所以，一个 block 的数据流可以写成：

```text
输入 x
  -> Masked Multi-Head Self-Attention
  -> Residual + LayerNorm
  -> FFN
  -> Residual + LayerNorm
  -> 输出
```

这正是你这份代码在做的事情。

---

## 3. 你的代码整体结构

这份 notebook 主要有两个类：

### 3.1 `SimpleDecoder`

它表示 **一个 Decoder block**。

里面包含：

- 多头自注意力
- attention 输出投影
- attention 的 dropout
- attention 的 LayerNorm
- FFN 上升投影
- FFN 激活函数
- FFN 下降投影
- FFN dropout
- FFN 的 LayerNorm

所以它相当于一个完整的：

```text
Attention Block + FFN Block
```

---

### 3.2 `Decoder`

它表示 **整个 decoder-only 模型**。

里面包含：

- `Embedding`：把 token id 变成向量
- `ModuleList`：堆叠多个 `SimpleDecoder`
- `Linear` 输出层：把 hidden state 映射回词表维度
- `softmax`：得到每个位置对词表的概率分布

所以它相当于：

```text
输入 token id
-> embedding
-> 多层 decoder block
-> 输出层
-> 概率分布
```

---

## 4. 从整体角度看，这份代码的数据流

你这个模型的 forward 流程，本质上是下面这样：

```python
X -> Embedding -> 多层 SimpleDecoder -> Linear -> Softmax
```

如果用更完整的话说：

1. 输入是 token id，shape 是 `[B, T]`
2. 先通过 `Embedding` 变成 token 向量，shape 变成 `[B, T, C]`
3. 依次通过多个 Decoder block，每一层都做：
   - masked self-attention
   - 残差 + LayerNorm
   - FFN
   - 残差 + LayerNorm
4. 最后用线性层把 hidden state 映射到词表维度
5. 每个位置得到一个对所有 token 的预测分布

---

## 5. 先看 `SimpleDecoder.__init__`：它在初始化什么

你的初始化代码是：

```python
class SimpleDecoder(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()

        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        self.dropout = dropout

        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.drop_attn = nn.Dropout(self.dropout)
        self.LayerNorm_attn = nn.LayerNorm(hidden_dim, eps=1e-5)

        self.upward_proj = nn.Linear(hidden_dim, hidden_dim * 4)
        self.downward_proj = nn.Linear(hidden_dim * 4, hidden_dim)
        self.LayerNorm_ffn = nn.LayerNorm(hidden_dim, eps=1e-5)
        self.act_fn = nn.GELU()
        self.drop_ffn = nn.Dropout(self.dropout)
```

这一段可以拆成两部分理解。

---

## 6. 第一部分：Attention 相关组件

### 6.1 `self.q / self.k / self.v`

这三个线性层负责把输入 `x` 映射成：

- Query
- Key
- Value

也就是：

```python
Q = xWq
K = xWk
V = xWv
```

输入输出维度都是 `hidden_dim -> hidden_dim`。

这并不是说它们“什么都没变”，而是说：

> 它们在维度大小上没变，但在线性变换后的语义空间已经变了。

---

### 6.2 `self.output_proj`

这是多头 attention 算完之后的输出投影层。

多头 attention 会先把多个头拼接回 `[B, T, hidden_dim]`，然后再通过一个线性层做融合。

它的作用不是改 shape，而是：

> 把各个 head 的信息重新线性混合，得到更合适的输出表示。

---

### 6.3 `self.drop_attn`

这是 attention 权重后的 dropout。

它的作用是：

> 在训练时随机丢掉一部分注意力连接，减少过拟合。

---

### 6.4 `self.LayerNorm_attn`

这是 attention block 结束后的 LayerNorm。

你这里采用的是：

```python
x + attention_output
-> LayerNorm
```

也就是 **Post-LN（后归一化）** 风格。

---

## 7. 第二部分：FFN 相关组件

### 7.1 `self.upward_proj = nn.Linear(hidden_dim, hidden_dim * 4)`

这是 FFN 的第一层。

它会把每个 token 的 hidden 表示从：

```python
hidden_dim
```

扩展到：

```python
4 * hidden_dim
```

这是 Transformer 中非常经典的做法。

为什么先升维？

因为这样可以让模型在更高维空间里做更强的非线性变换。

---

### 7.2 `self.act_fn = nn.GELU()`

这是激活函数。

它负责给 FFN 引入非线性能力。

在现代 Transformer 模型里，`GELU` 非常常见，比起传统的 `ReLU` 更平滑。

---

### 7.3 `self.downward_proj = nn.Linear(hidden_dim * 4, hidden_dim)`

这是 FFN 的第二层。

它会把升维后的表示再压回原来的 hidden_dim。

所以 FFN 整体就是：

```text
hidden_dim -> 4 * hidden_dim -> hidden_dim
```

---

### 7.4 `self.LayerNorm_ffn`

这是 FFN block 结束后的 LayerNorm。

也就是：

```python
x + ffn_output
-> LayerNorm
```

仍然是 **Post-LN**。

---

## 8. `self.head_dim = hidden_dim // num_heads` 的意义

这里是多头注意力的基础设定：

```python
self.head_dim = hidden_dim // num_heads
```

意思是：

> 总 hidden_dim 会平均分给每个 head。

比如你这里：

```python
hidden_dim = 64
num_heads = 8
```

那么：

```python
head_dim = 64 // 8 = 8
```

也就是说：

- 一共有 8 个头
- 每个头负责 8 维特征
- 全部头拼起来还是 64 维

这是后面拆头、做 attention、再拼回去的基础。

---

## 9. `attention_output()`：这是整个 attention 的核心计算

你写的函数是：

```python
def attention_output(self, query, key, value, attention_mask=None):
```

这个函数接收已经拆好头的：

- `query`
- `key`
- `value`

它们的 shape 是：

```python
[B, H, T, D]
```

其中：

- `B`：batch size
- `H`：num_heads
- `T`：seq_len
- `D`：head_dim

然后这个函数完成整个 attention 计算流程。

---

## 10. 第一步：计算 attention score

```python
attention_value = query @ key.transpose(-2, -1)
attention_weights = attention_value / math.sqrt(self.head_dim)
```

### 10.1 这一句在算什么

这是标准的缩放点积注意力：

```python
QK^T / sqrt(d_k)
```

如果 `query` 和 `key` 的 shape 都是：

```python
[B, H, T, D]
```

那么：

```python
key.transpose(-2, -1)
```

就会变成：

```python
[B, H, D, T]
```

于是矩阵乘法结果是：

```python
[B, H, T, T]
```

这表示：

> 对每个 batch、每个 head、每个 query 位置，都算出了它对所有 key 位置的相关性分数。

---

### 10.2 为什么要除以 `sqrt(self.head_dim)`

因为如果不做缩放，点积结果会随着维度变大而变得很大，导致 softmax 过于尖锐，训练不稳定。

所以标准做法就是除以：

```python
sqrt(head_dim)
```

你这里这部分是正确的。

---

## 11. 第二步：加 mask

你的代码：

```python
if attention_mask is not None:
    attention_mask = attention_mask.tril()
    attention_weights = attention_weights.masked_fill(attention_mask == 0, float("-1e20"))
else:
    attention_mask = torch.ones_like(attention_weights).tril()
    attention_weights = attention_weights.masked_fill(attention_mask == 0, float("-1e20"))
```

这一段非常关键，因为它体现了 decoder 的“不能看未来”。

---

### 11.1 `tril()` 在干什么

`tril()` 会把矩阵变成下三角形式。

比如长度为 4 的序列，对应的 causal mask 会像这样：

```python
[[1, 0, 0, 0],
 [1, 1, 0, 0],
 [1, 1, 1, 0],
 [1, 1, 1, 1]]
```

这表示：

- 第 1 个 token 只能看自己
- 第 2 个 token 只能看前两个
- 第 3 个 token 只能看前三个
- 第 4 个 token 可以看前四个

也就是说：

> 当前位置不能看到未来位置。

这就是 **causal mask** 的本质。

---

### 11.2 `masked_fill(attention_mask == 0, -1e20)` 在干什么

这句的意思是：

> 把不允许看的位置，分数强行改成一个极小值。

这样在后面的 softmax 中，这些位置的概率就会非常接近 0。

所以这一步不是直接“删掉”未来位置，而是：

> 在 softmax 之前，把它们的分数打到极小。

---

### 11.3 你这里其实是在做“padding mask + causal mask”的合并

你给的注释是：

```python
# padding mask + 下三角casual mask
```

这里思路是对的，只是有两个细节要注意：

#### 第一，`casual` 应该写作 `causal`

标准术语是：

- **causal mask**：因果掩码

不是 `casual mask`。

---

#### 第二，`attention_mask.tril()` 的含义

你传入的 `attention_mask` 本身已经是一个四维 mask：

```python
[B, H, T, T]
```

然后又做了一次 `.tril()`。

这表示：

> 无论外面传入什么 mask，你都会再额外加上一层下三角约束。

这样做在“decoder only”的教学例子里是可以的，但在工程里通常会更明确地区分：

- padding mask
- causal mask
- 二者合并后的最终 mask

这样逻辑会更清晰。

---

## 12. 如果没有传入 mask，会发生什么

你的代码里写了：

```python
else:
    attention_mask = torch.ones_like(attention_weights).tril()
```

意思是：

> 如果外部没有传 mask，就自动构造一个标准的 causal mask。

这是一种很好的默认行为，因为 decoder 自注意力本来就应该带因果约束。

---

## 13. 第三步：softmax

```python
attention_weights = torch.softmax(attention_weights, dim=-1)
```

这里的 `dim=-1` 非常重要。

因为 `attention_weights` 的 shape 是：

```python
[B, H, T, T]
```

最后一维表示：

> 当前 query 位置，对所有 key 位置的打分。

所以在最后一维做 softmax，表示：

> 让每个 query 位置，对所有可见 key 位置的权重和为 1。

换句话说，softmax 后每一行都是一个注意力分布。

---

## 14. 第四步：dropout

```python
attention_weights = self.drop_attn(attention_weights)
```

这一步是在注意力权重上做 dropout。

训练时，它会随机让一部分连接失效；推理时则不会。

你可以把它理解成：

> 不要让模型过度依赖某几个固定的注意力连接。

---

## 15. 第五步：加权求和得到每个 head 的输出

```python
output_mid = attention_weights @ value
```

如果：

- `attention_weights` 是 `[B, H, T, T]`
- `value` 是 `[B, H, T, D]`

那么结果就是：

```python
[B, H, T, D]
```

这一步的含义是：

> 对于每个 query 位置，用它的注意力分布，对所有 value 做加权平均。

于是每个位置都得到了一份新的上下文表示。

---

## 16. 第六步：把多头结果拼回去

```python
output_mid = output_mid.transpose(1, 2).contiguous()
batch, seq_len, _, _ = output_mid.size()
output_mid = output_mid.view(batch, seq_len, -1)
output = self.output_proj(output_mid)
```

这段就是标准的多头拼接流程。

---

### 16.1 为什么要先 `transpose(1, 2)`

前面 `output_mid` 的 shape 是：

```python
[B, H, T, D]
```

但我们最终希望每个位置聚合所有 head 的结果，所以要变成：

```python
[B, T, H, D]
```

这样才能把最后两个维度拼起来。

---

### 16.2 为什么要 `.contiguous()`

因为 `transpose()` 之后，张量在内存中的存储往往不是连续的。

而 `view()` 要求底层内存通常是连续的。

所以这里写：

```python
.contiguous().view(...)
```

是非常标准的做法。

---

### 16.3 `view(batch, seq_len, -1)` 在做什么

它会把：

```python
[B, T, H, D]
```

合并成：

```python
[B, T, H * D]
```

因为：

```python
H * D = hidden_dim
```

所以结果就是：

```python
[B, T, hidden_dim]
```

这表示：

> 多个头的输出重新拼回了原始 hidden 维度。

---

## 17. `attention_block()`：一个完整的 attention 子模块

你的代码：

```python
def attention_block(self, x, attention_mask=None):
    batch_size, seq_len, _ = x.shape

    query = self.q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    key = self.k(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    value = self.v(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    output = self.attention_output(query, key, value, attention_mask=attention_mask)
    return self.LayerNorm_attn(x + output)
```

这一段的逻辑非常完整：

1. 输入 `x` 是 `[B, T, C]`
2. 线性映射成 Q/K/V
3. 拆成多头，变成 `[B, H, T, D]`
4. 调用 `attention_output()` 完成 attention
5. 输出再和原输入做残差连接
6. 做 LayerNorm

所以这一段就是：

```text
Masked Multi-Head Self-Attention + Residual + LayerNorm
```

---

## 18. 为什么 attention block 里一定要有残差连接

这里的：

```python
x + output
```

就是残差连接。

它的作用主要有两个：

### 18.1 保留原始信息

attention 虽然会生成新的表示，但原始输入信息也很重要。

残差连接能让模型在“原始信息”和“新信息”之间做平衡。

---

### 18.2 让深层网络更容易训练

Transformer 往往会堆很多层，如果没有残差，梯度传播会更难，训练更不稳定。

所以残差几乎是现代深层网络的标配。

---

## 19. `ffn_block()`：第二个核心子模块

你的代码：

```python
def ffn_block(self, x):
    up = self.upward_proj(x)
    up = self.act_fn(up)
    down = self.downward_proj(up)
    down = self.drop_ffn(down)
    return self.LayerNorm_ffn(x + down)
```

它的流程非常标准：

1. 先升维
2. 过激活函数
3. 再降维
4. dropout
5. 残差连接
6. LayerNorm

这就是标准 Transformer FFN block 的简化写法。

---

## 20. FFN 的本质是什么

这是一个很容易被忽略、但很重要的问题。

attention 的核心是：

> token 和 token 之间的信息交互。

FFN 的核心则是：

> 对每个 token 位置自己的表示，做更强的非线性变换。

所以可以这么区分：

- **Attention**：负责“看别人”
- **FFN**：负责“加工自己”

Transformer block 同时需要这两部分，缺一不可。

---

## 21. `forward()`：一个 Decoder block 的完整前向过程

你的代码：

```python
def forward(self, X, attention_mask=None):
    att_output = self.attention_block(X, attention_mask=attention_mask)
    ffn_output = self.ffn_block(att_output)
    return ffn_output
```

这段非常清楚：

```text
输入 X
-> attention_block
-> ffn_block
-> 输出
```

这就是一个最标准、最简化的 decoder block 流程。

---

## 22. `Decoder` 类：开始堆多层 block

你的代码：

```python
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_list = nn.ModuleList([
            SimpleDecoder(64, 8) for i in range(5)
        ])
        self.emb = nn.Embedding(12, 64)
        self.out = nn.Linear(64, 12)
```

这里可以拆成三部分。

---

### 22.1 `self.emb = nn.Embedding(12, 64)`

意思是：

- 词表大小 `vocab_size = 12`
- 每个 token 被映射成 64 维向量

所以输入如果是：

```python
[B, T]
```

那么 embedding 后就会变成：

```python
[B, T, 64]
```

这也是你之前反复问过的一个点：

> embedding 并不是“多加了一维”，而是把每个离散 token id 替换成一个向量。

---

### 22.2 `self.layer_list = nn.ModuleList([...])`

这里堆了 5 层：

```python
SimpleDecoder(64, 8)
```

也就是说整个模型一共有 5 个 decoder block。

这就对应了“深层 Transformer”的思想：

> 不断重复同一种 block，让表示越来越抽象。

---

### 22.3 `self.out = nn.Linear(64, 12)`

这个输出层会把每个位置的 hidden state：

```python
64 维
```

映射回：

```python
12 维
```

这 12 维就对应词表中的 12 个 token。

于是模型就能对每个位置给出“下一个 token 是谁”的打分。

---

## 23. `Decoder.forward()`：整机级别的数据流

你的代码：

```python
def forward(self, X, mask=None):
    x = self.emb(X)
    for i, l in enumerate(self.layer_list):
        x = l(x, mask)
    print(x.shape)
    print('---------')
    output = self.out(x)
    return torch.softmax(output, dim=-1)
```

可以逐步拆开理解。

---

### 23.1 输入 `X`

输入是 token id：

```python
[B, T]
```

你这里测试时是：

```python
x = torch.randint(0, 12, (3, 4))
```

所以：

- `B = 3`
- `T = 4`

输入 shape 为：

```python
[3, 4]
```

---

### 23.2 embedding 后

```python
x = self.emb(X)
```

会得到：

```python
[3, 4, 64]
```

表示 3 个样本，每个样本 4 个 token，每个 token 用 64 维向量表示。

---

### 23.3 经过 5 层 decoder

```python
for i, l in enumerate(self.layer_list):
    x = l(x, mask)
```

每一层输入输出 shape 都还是：

```python
[3, 4, 64]
```

因为 decoder block 不会改变 batch、seq_len、hidden_dim 这三个主要维度。

它改变的是：

> 每个位置 64 维向量里的语义内容。

---

### 23.4 经过输出层

```python
output = self.out(x)
```

shape 会变成：

```python
[3, 4, 12]
```

意思是：

- 3 个样本
- 每个样本 4 个位置
- 每个位置都对 12 个词表 token 给出一个分数

---

### 23.5 最后 softmax

```python
torch.softmax(output, dim=-1)
```

最后一维是词表维度，所以 softmax 后：

> 每个位置都会得到一个长度为 12 的概率分布。

---

## 24. 结合你的测试样例，把所有 shape 串起来

你的测试代码是：

```python
x = torch.randint(0, 12, (3, 4))
net = Decoder()

mask = (
    torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0], [1, 1, 1, 0]])
    .unsqueeze(1)
    .unsqueeze(2)
    .repeat(1, 8, 4, 1)
)
print(mask.shape)
net(x, mask).shape
```

这里可以完整梳理一遍。

---

### 输入 token id

```python
x.shape = [3, 4]
```

---

### embedding 后

```python
[3, 4, 64]
```

---

### 进入每一层 attention 前

Q/K/V 线性层后还是：

```python
[3, 4, 64]
```

---

### 拆成多头后

```python
[3, 8, 4, 8]
```

因为：

- `num_heads = 8`
- `head_dim = 8`

---

### attention score

```python
[3, 8, 4, 4]
```

表示每个 head 内部，长度为 4 的序列两两之间的相关性。

---

### attention output（每个 head）

```python
[3, 8, 4, 8]
```

---

### 拼接回 hidden_dim 后

```python
[3, 4, 64]
```

---

### attention block 输出后

```python
[3, 4, 64]
```

---

### FFN block 输出后

```python
[3, 4, 64]
```

---

### 经过 5 层后

仍然是：

```python
[3, 4, 64]
```

---

### 输出层后

```python
[3, 4, 12]
```

---

## 25. 这份代码里很重要的一个点：mask 的 shape

你的 mask 最终构造成了：

```python
[3, 8, 4, 4]
```

这和 attention score 的 shape 是一致的。

这是因为 attention score 本身是：

```python
[B, H, T, T]
```

所以 mask 也通常要能广播到这个 shape。

你这里是直接显式构造成相同 shape，这在教学代码里很直观。

不过要注意：

> 你在 `forward` 的注释里写的是 `shape 一般是：(batch, nums_head, seq)`，这个并不对应你当前真正使用的 mask 形状。

在你这份代码里，真正参与计算的 mask 是：

```python
[B, H, T, T]
```

这一点最好在笔记里记清楚。

---

## 26. 这份代码是一个“简化版 decoder”，还缺了什么

这份实现已经很好地抓住了 decoder 的核心骨架，但如果和真正的大语言模型相比，它还缺几个常见组件。

---

### 26.1 没有位置编码（Positional Encoding）

你这里是：

```python
x = self.emb(X)
```

只做了 token embedding，没有加入 positional embedding。

但 Transformer 本身没有卷积也没有循环，它天生不懂顺序。

所以在真正模型里，通常还要加入：

- 绝对位置编码
- 或相对位置编码
- 或 RoPE 这类旋转位置编码

否则模型只能看到“有哪些 token”，但不容易精确知道它们的顺序。

---

### 26.2 输出层后直接 softmax，不是训练中最常见的写法

你这里是：

```python
return torch.softmax(output, dim=-1)
```

这在教学展示里是很直观的，因为能直接看到概率。

但在训练语言模型时，更常见的是：

> 直接返回 logits，而不是先 softmax。

因为像 `CrossEntropyLoss` 这类损失函数内部会自己处理 softmax。

如果提前做 softmax，反而会影响数值稳定性。

所以你这里更像是：

> 用来帮助理解和观察输出分布的教学写法。

---

### 26.3 没有 KV Cache

你之前也问过 KV cache。

这份 decoder 代码里，每次前向都会重新计算全部序列的 K 和 V。

但在真实推理里，生成是一个 token 一个 token 往外吐的。如果每一步都重新算全部历史，就很浪费。

所以真实 LLM 推理里通常会缓存：

- 过去 token 的 K
- 过去 token 的 V

这就是 KV cache 的作用。

---

### 26.4 没有 weight tying

真实语言模型里，常常会把：

- 输入 embedding 矩阵
- 输出层投影矩阵

做 **tie weight**。

而你这里：

```python
self.emb = nn.Embedding(12, 64)
self.out = nn.Linear(64, 12)
```

它们是独立参数。

教学代码这样写更清楚，但真实模型常常会共享它们来减少参数量。

---

## 27. 这份代码和你前两份笔记的关系

这三份内容其实是一条主线。

### 第一份：Single-Head Self-Attention

你学到的是：

> 注意力是怎么从 Q/K/V 开始算起来的。

---

### 第二份：Multi-Head Self-Attention

你学到的是：

> 一个 attention 不够，于是拆成多个 head 并行计算，再拼接回来。

---

### 第三份：Transformer Decoder

你学到的是：

> attention 只是 block 里的一个模块，真正的 decoder block 还要加残差、LayerNorm、FFN，再堆多层，才能组成完整的 decoder-only 模型。

所以这份 notebook 是你从“模块理解”迈向“架构理解”的关键一步。

---

## 28. 这份代码最值得记住的 5 个核心点

### 核心点 1：Decoder block = Attention block + FFN block

这是整体框架的骨架。

---

### 核心点 2：Attention 负责 token 间交互，FFN 负责单位置非线性加工

这两个角色不要混。

---

### 核心点 3：每个子模块外面都配了残差和 LayerNorm

这样模型更稳定，也更容易堆深。

---

### 核心点 4：Decoder 的 attention 必须带 causal mask

否则模型就会偷看未来 token，不符合自回归生成。

---

### 核心点 5：多层堆叠时，shape 往往不变，变的是表示语义

这一点非常重要。

从第一层到第五层：

```python
[3, 4, 64] -> [3, 4, 64] -> [3, 4, 64] ...
```

shape 没变，但内部表示不断被加工。

---

## 29. 这份代码里几个容易混淆的点

### 29.1 `SimpleDecoder` 不是“整个大模型”，只是一个 block

很多初学者会把一个 decoder block 和完整 decoder 模型混为一谈。

你这里：

- `SimpleDecoder`：单层 block
- `Decoder`：多层堆叠后的完整模型

要分清。

---

### 29.2 `attention_mask` 的注释和实际 shape 不一致

你注释里写的是：

```python
shape 一般是： (batch, nums_head, seq)
```

但你实际计算时用的是：

```python
[B, H, T, T]
```

笔记里最好记成后者，不然以后容易乱。

---

### 29.3 这里没有 encoder，所以它是 Decoder-only，不是完整 Transformer

原始 Transformer 包括：

- Encoder
- Decoder

而你这份代码只有 masked self-attention + FFN 这一套，没有 cross-attention，也没有 encoder 输出。

所以它更接近：

> GPT 这一类的 Decoder-only 结构

而不是“Encoder-Decoder”结构。

---

### 29.4 没有位置编码，所以它只是结构演示版

这一点非常重要。

如果以后你自己继续补代码，位置编码通常是很自然的下一步。

---

## 30. 面试/复习时可以怎么说

### 30.1 什么是 Transformer Decoder block？

可以这样答：

> 一个标准的 Transformer Decoder block 主要由 masked multi-head self-attention 和 feed-forward network 两部分组成，并且每个子模块外面通常都带有残差连接和 LayerNorm。attention 负责建模 token 间关系，FFN 负责对每个位置的表示做非线性变换。

---

### 30.2 为什么 decoder 里要加 mask？

可以这样答：

> 因为 decoder 通常用于自回归生成任务，当前位置只能看当前和过去的位置，不能看到未来 token，所以要加 causal mask，把未来位置的 attention score 屏蔽掉。

---

### 30.3 FFN 的作用是什么？

可以这样答：

> attention 主要负责不同 token 之间的信息交互，而 FFN 则对每个位置自己的 hidden representation 做进一步的非线性变换，两者作用不同但互补。

---

### 30.4 为什么堆多层 decoder 时 shape 经常不变？

可以这样答：

> 因为每层 block 通常都保持 hidden_dim 不变，这样便于残差连接。虽然 shape 不变，但每层都会不断更新 token 表示，使语义逐层抽象。

---

## 31. 如果你要把这份代码改得更标准，可以怎么改

你后面可以沿着这几个方向继续升级。

### 31.1 加位置编码

比如：

```python
x = self.emb(X) + self.pos_emb(position_ids)
```

---

### 31.2 输出 logits，不在 forward 里 softmax

比如：

```python
return self.out(x)
```

训练时交给 `CrossEntropyLoss`。

---

### 31.3 把 mask 的构造单独封装出来

这样 `attention_output()` 里逻辑会更清楚。

---

### 31.4 明确区分 Post-LN 和 Pre-LN

你现在是：

```python
x + sublayer_output -> LayerNorm
```

如果以后你想更贴近 GPT-2 / GPT-NeoX 一类实现，可以尝试 Pre-LN 写法。

---

### 31.5 去掉 `print(x.shape)`，改成注释或调试模式

因为在真实训练代码里，`forward()` 里通常不会长期保留这种打印。

---

## 32. 复习这份 notebook 时，你应该重点问自己这几个问题

### 问题 1

为什么说 decoder block 不是只有 attention，还必须有 FFN？

### 问题 2

attention block 和 FFN block 分别负责什么？

### 问题 3

为什么 decoder 的 self-attention 必须加 causal mask？

### 问题 4

一个 block 里，残差连接和 LayerNorm 是怎么接的？

### 问题 5

从输入 token id 到最后词表概率分布，中间 shape 是怎么变化的？

### 问题 6

为什么说这份实现更接近“教学版 GPT”，而不是完整工业级 LLM？

---

## 33. 一句话总结

这份 `03_Transformer_Decoder.ipynb` 的核心，不再是单独理解 attention，而是开始理解：

> 一个真正能堆叠起来的 Decoder block，到底是如何由 **Masked Multi-Head Self-Attention + FFN + Residual + LayerNorm** 组合出来的，以及多个 block 又是如何组成一个最简化的 decoder-only 模型的。

