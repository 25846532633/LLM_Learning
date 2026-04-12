# 02 Multi-Head Self-Attention 笔记

## 1. 这一份 Notebook 在学什么

这一份 `02_MultiHead_self_attention.ipynb`，核心是在 **Single-Head Self-Attention** 的基础上，进一步实现 **Multi-Head Self-Attention（多头自注意力）**。

如果说单头注意力是在做：

> “每个 token 看一遍其他 token，然后把重要的信息加权聚合起来。”

那么多头注意力做的事情就是：

> “不要只用一种方式看别人，而是把特征维度拆成多个子空间，让每个头分别学习不同的关注关系，最后再把这些头的结果拼起来。”

所以，这一份代码的重点不是“注意力是什么”，而是：

1. 为什么要从单头变成多头
2. 多头的 shape 是怎么拆开的
3. 多头计算后又是怎么拼回去的
4. `mask` 在多头里是如何广播到每个 head 上的

---

## 2. 先说结论：多头注意力到底比单头多了什么

单头注意力里，输入 `x` 的 shape 通常是：

```python
[B, T, C]
```

- `B`：batch size
- `T`：序列长度（seq_len）
- `C`：hidden_dim

单头做法是直接：

```python
Q, K, V: [B, T, C]
attention score: [B, T, T]
output: [B, T, C]
```

多头做法是先把 `C` 拆成多个头：

```python
C = num_heads * head_dim
```

于是：

```python
Q, K, V: [B, T, C]
      -> [B, T, num_heads, head_dim]
      -> [B, num_heads, T, head_dim]
```

然后每个 head 单独做注意力：

```python
attention score: [B, num_heads, T, T]
output per head: [B, num_heads, T, head_dim]
```

最后把所有头再拼回来：

```python
[B, num_heads, T, head_dim]
-> [B, T, num_heads, head_dim]
-> [B, T, C]
```

所以你可以把多头注意力理解成：

> “并行地做了多次小注意力，每个小注意力只负责 hidden_dim 的一部分特征。”

---

## 3. 为什么要做多头，而不是一个大头就够了

这是多头注意力的一个非常核心的问题。

### 3.1 单头的问题

如果只用一个 head，那么整个 `hidden_dim` 只有一套注意力权重分布。
也就是说，模型只能用一种“相似度视角”去判断 token 之间的关系。

但真实语言中，token 之间的关系可能有很多种：

- 有的头更关注语法关系
- 有的头更关注位置关系
- 有的头更关注主谓宾搭配
- 有的头更关注长距离依赖
- 有的头更关注局部上下文

如果只用一个头，所有这些关系都要被塞进同一套注意力权重里，表达能力会受限。

### 3.2 多头的好处

多头本质上是在说：

> “我把特征切成几份，每一份都独立学习一种关注模式。”

这样不同 head 就可能学到不同的关系模式，最后再把它们拼起来，表达能力更强。

---

## 4. 你的代码整体结构

你这个 notebook 的核心类是：

```python
class MultiHeadAttention(nn.Module):
```

其结构非常标准，主要分成下面几步：

1. 初始化参数：`hidden_dim`、`head_num`、`head_dim`
2. 定义四个线性层：`q`、`k`、`v`、`output_proj`
3. 前向传播里先算 `q/k/v`
4. 把 `q/k/v` reshape 成多头形式
5. 做注意力分数计算
6. 加 mask
7. 做 softmax
8. 用注意力权重对 `v` 做加权求和
9. 把多头结果拼回去
10. 过最后一层输出投影

这就是一个完整的多头自注意力模块。

---

## 5. 初始化部分详解

你的代码：

```python
class MultiHeadAttention(nn.Module):
    def __init__(self,hidden_dim,head_num):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.head_dim = hidden_dim // head_num

        self.q = nn.Linear(hidden_dim,hidden_dim)
        self.k = nn.Linear(hidden_dim,hidden_dim)
        self.v = nn.Linear(hidden_dim,hidden_dim)
        self.output_proj = nn.Linear(hidden_dim,hidden_dim)

        self.dropout = nn.Dropout(0.1)
```

### 5.1 `self.head_dim = hidden_dim // head_num`

这句的意思是：

- 总维度是 `hidden_dim`
- 现在要拆成 `head_num` 个头
- 那么每个头负责的维度就是 `head_dim`

例如你这里测试时：

```python
hidden_dim = 128
head_num = 8
```

那么：

```python
head_dim = 128 // 8 = 16
```

也就是说：

- 总共有 8 个头
- 每个头看 16 维
- 最后 `8 * 16 = 128`

### 5.2 为什么 `q/k/v` 线性层还是 `128 -> 128`

很多初学者这里会疑惑：

> “既然每个头只看 16 维，为啥线性层不是 `128 -> 16`？”

因为这里不是只做一个头，而是一次性把 **所有头需要的表示一起算出来**。

所以先整体做：

```python
[B, T, 128] -> [B, T, 128]
```

然后再把最后这个 `128` 拆成：

```python
8 * 16
```

也就是：

```python
[B, T, 128] -> [B, T, 8, 16]
```

所以线性层本身不负责“拆头”，它只负责生成总的 `Q/K/V` 表示，拆头是在后面的 `view + transpose` 中完成的。

---

## 6. forward 的整体流程

你的代码：

```python
def forward(self,x,attention_mask = None):
    batch_size,seq_len,_ = x.size()
```

这里假设输入：

```python
x.shape = [3, 2, 128]
```

也就是：

- batch_size = 3
- seq_len = 2
- hidden_dim = 128

---

## 7. 第一步：计算 q、k、v

```python
q = self.q(x)
k = self.k(x)
v = self.v(x)
print(q.shape) # [batch_size, seq_len, hidden_dim]
```

输出：

```python
[3, 2, 128]
```

这一步和单头注意力是一样的。

### 7.1 含义

- `q`：当前 token 想去查询什么信息
- `k`：每个 token 能提供什么索引线索
- `v`：每个 token 真正携带的信息内容

### 7.2 为什么输入输出 shape 没变

因为这里做的是线性映射：

```python
nn.Linear(128, 128)
```

线性层只会变最后一维，不会改变 batch 和 seq_len。

所以：

```python
[B, T, 128] -> [B, T, 128]
```

---

## 8. 第二步：拆成多头

你的代码：

```python
q_state = q.view(batch_size,seq_len,self.head_num,self.head_dim).transpose(1,2)
k_state = k.view(batch_size,seq_len,self.head_num,self.head_dim).transpose(1,2)
v_state = v.view(batch_size,seq_len,self.head_num,self.head_dim).transpose(1,2)
print(q_state.shape)
```

输出：

```python
[3, 8, 2, 16]
```

### 8.1 这一步到底做了什么

原来：

```python
q.shape = [3, 2, 128]
```

先 `view`：

```python
[3, 2, 128] -> [3, 2, 8, 16]
```

含义是：

- batch 中有 3 个样本
- 每个样本有 2 个 token
- 每个 token 的 128 维表示被拆成 8 个头
- 每个头 16 维

再 `transpose(1, 2)`：

```python
[3, 2, 8, 16] -> [3, 8, 2, 16]
```

把 `seq_len` 和 `head_num` 交换位置。

### 8.2 为什么要 transpose

因为后面要按“每个 head 内部”去做注意力计算。

换句话说，我们想让 tensor 的结构变成：

```python
[batch, head, seq, head_dim]
```

这样每个 head 的 `Q` 和 `K` 才能方便地做矩阵乘法。

如果不 transpose，head 这个维度夹在中间不方便后续操作。

---

## 9. 第三步：计算注意力分数

你的代码：

```python
attention_value = q_state @ k_state.transpose(-1,-2)
attention_weight = attention_value / math.sqrt(self.hidden_dim)
```

### 9.1 第一句在算什么

`q_state.shape` 是：

```python
[3, 8, 2, 16]
```

`k_state.transpose(-1,-2)` 后是：

```python
[3, 8, 16, 2]
```

两者相乘：

```python
[3, 8, 2, 16] @ [3, 8, 16, 2] -> [3, 8, 2, 2]
```

所以：

```python
attention_value.shape = [batch_size, head_num, seq_len, seq_len]
```

在你这个例子里就是：

```python
[3, 8, 2, 2]
```

### 9.2 这个 `[2, 2]` 是什么意思

因为你的 `seq_len = 2`。

所以对于每一个样本、每一个 head：

- 行表示“当前 query token”
- 列表示“被关注的 key token”

也就是说每个 head 都会生成一个 `2 x 2` 的注意力分数矩阵。

---

## 10. 一个很重要的更正：这里缩放分母应该是 `sqrt(head_dim)`

你现在的代码写的是：

```python
attention_weight = attention_value / math.sqrt(self.hidden_dim)
```

这在标准 Transformer 里通常是不对的。

标准写法应该是：

```python
attention_weight = attention_value / math.sqrt(self.head_dim)
```

### 10.1 为什么是 `head_dim`

因为当前每个 head 里做点积的向量维度是 `head_dim`，不是整个 `hidden_dim`。

你现在每个 head 的向量长度是：

```python
head_dim = 16
```

所以分母应该是：

```python
sqrt(16)
```

而不是：

```python
sqrt(128)
```

### 10.2 为什么要缩放

如果不缩放，点积结果会随着维度增大而变大，进入 softmax 后就容易让分布太尖锐，导致训练不稳定。

### 10.3 你这里如果写成 `sqrt(hidden_dim)` 会怎样

不会报错，也不是完全不能跑，但缩放力度变大了，数值会比标准做法更小。

这样会导致：

- softmax 分布可能变得更平
- 与标准 Transformer 实现不一致
- 学习效果可能受影响

所以这里建议你立刻改成：

```python
attention_weight = attention_value / math.sqrt(self.head_dim)
```

---

## 11. 第四步：mask 的作用

你的代码：

```python
print(attention_mask.shape)
if attention_mask is not None:
    attention_weight = attention_weight.masked_fill(
        attention_mask == 0,float("-inf")
    )
```

你的 `attention_mask.shape` 是：

```python
[3, 8, 2, 2]
```

### 11.1 为什么 mask 也要是四维

因为 `attention_weight` 本身就是：

```python
[batch_size, head_num, seq_len, seq_len]
```

所以 mask 也必须能和它对齐，才能逐元素屏蔽。

### 11.2 `masked_fill(attention_mask == 0, -inf)` 的含义

意思是：

- mask 中等于 0 的位置，不允许被关注
- 这些位置对应的 score 被改成 `-inf`
- 后面 softmax 以后，这些位置的概率就会变成 0

也就是：

> “不该看的位置，直接让它永远选不到。”

---

## 12. 你的 mask 是怎么构造出来的

你的代码：

```python
attention_mask = (
    torch.tensor(
        [
            [0, 1],
            [0, 0],
            [1, 0],
        ]
    )
    .unsqueeze(1)
    .unsqueeze(2)
    .expand(3, 8, 2, 2)
)
```

我们一步一步看。

### 12.1 原始 tensor

```python
[
    [0, 1],
    [0, 0],
    [1, 0],
]
```

shape 是：

```python
[3, 2]
```

可以理解为：

- batch 里有 3 个样本
- 每个样本有长度为 2 的 key 可见性标记

### 12.2 `unsqueeze(1).unsqueeze(2)`

先变成：

```python
[3, 1, 1, 2]
```

### 12.3 `expand(3, 8, 2, 2)`

再广播成：

```python
[3, 8, 2, 2]
```

含义是：

- 对每个 batch 样本
- 复制到 8 个 head
- 对每个 query 位置都用同一份 key 可见性规则

所以这更接近一种：

> “key padding mask 被广播到所有 head、所有 query 位置上”

而不是标准的 causal mask。

---

## 13. 一个非常关键的问题：你这个 mask 里第二个样本会导致 NaN

你的原始 mask 中第二行是：

```python
[0, 0]
```

这意味着该样本的两个 key 位置都被屏蔽掉了。

扩展后，这个样本的某个 head 上的注意力矩阵会全部被填成：

```python
[-inf, -inf]
```

接着做：

```python
softmax([-inf, -inf])
```

结果会是：

```python
[nan, nan]
```

也就是说，这个样本会直接传播出 `NaN`。

### 13.1 为什么会这样

因为 softmax 的本质是：

```python
exp(x_i) / sum(exp(x_j))
```

如果所有位置都是 `-inf`，那分子分母都会变成 0，最后数值不合法。

### 13.2 所以 mask 设计时要注意什么

**至少要保证每个 query 对应的 key 里，有一个位置是可见的。**

也就是说，不要让一整行全被 mask 掉。

### 13.3 你可以怎么改

例如改成：

```python
attention_mask = (
    torch.tensor(
        [
            [1, 1],
            [1, 0],
            [1, 0],
        ]
    )
    .unsqueeze(1)
    .unsqueeze(2)
    .expand(3, 8, 2, 2)
)
```

这样每个样本至少还有一个 key 可看。

---

## 14. 第五步：softmax

你的代码：

```python
attention_weight = torch.softmax(attention_weight,dim=-1)
print(attention_weight.shape)
```

输出：

```python
[3, 8, 2, 2]
```

### 14.1 为什么是 `dim=-1`

因为最后一维表示“当前 query 对所有 key 的打分”。

softmax 必须沿着 key 这一维做，才能把“对所有 key 的分数”变成“对所有 key 的概率分布”。

也就是说：

- 每个 query token
- 在每个 head 内
- 都会得到一组对所有 key 的权重

### 14.2 softmax 后每一行代表什么

例如某个 head 中：

```python
[[0.2, 0.8],
 [0.6, 0.4]]
```

意思是：

- 第 1 个 query token 对两个 key 的关注权重分别是 0.2 和 0.8
- 第 2 个 query token 对两个 key 的关注权重分别是 0.6 和 0.4

每一行加起来等于 1。

---

## 15. 第六步：dropout

```python
attention_weight = self.dropout(attention_weight)
```

这一步是在注意力权重上做 dropout。

作用是：

- 防止模型过度依赖某些固定连接
- 提升泛化能力
- 是 Transformer 中的常见操作

注意：

- 训练时生效
- `eval()` 模式下不生效

---

## 16. 第七步：用注意力权重加权求和

你的代码：

```python
output_mid = attention_weight @ v_state
print(output_mid.shape)
```

输出：

```python
[3, 8, 2, 16]
```

### 16.1 这一步在算什么

`attention_weight.shape`：

```python
[3, 8, 2, 2]
```

`v_state.shape`：

```python
[3, 8, 2, 16]
```

矩阵乘法后：

```python
[3, 8, 2, 2] @ [3, 8, 2, 16] -> [3, 8, 2, 16]
```

意思是：

- 对每个 query token
- 按注意力权重对所有 value 向量做加权求和
- 得到这个 query 的新表示

所以这里其实就是：

> “根据关注程度，把别人的信息融合进自己。”

---

## 17. 第八步：把多头结果拼回去

你的代码：

```python
output_mid = output_mid.transpose(1, 2).contiguous()
print(output_mid.shape)

output = output_mid.view(batch_size, seq_len, -1)
print(output.shape)
```

输出分别是：

```python
[3, 2, 8, 16]
[3, 2, 128]
```

### 17.1 为什么先 transpose

当前：

```python
output_mid.shape = [3, 8, 2, 16]
```

这是：

```python
[batch, head, seq, head_dim]
```

我们想把它拼回标准输出形式：

```python
[batch, seq, hidden_dim]
```

所以先把 head 放回 seq 后面：

```python
[3, 8, 2, 16] -> [3, 2, 8, 16]
```

### 17.2 为什么要 `.contiguous()`

因为 `transpose()` 之后，tensor 在内存里通常不是连续存放的。

而 `view()` 要求底层内存连续，所以常见写法是：

```python
tensor.transpose(...).contiguous().view(...)
```

你这句注释写得很好：

- `view` 和 `transpose` 大多只是改“怎么看数据”的方式
- 并没有真正重排底层存储
- 所以如果后面要 `view`，通常先 `contiguous()` 更安全

### 17.3 `view(batch_size, seq_len, -1)` 在干嘛

此时：

```python
[3, 2, 8, 16] -> [3, 2, 128]
```

也就是把：

```python
8 个头 * 每个头 16 维
```

重新拼成：

```python
128 维
```

于是多头结果又回到了标准 hidden representation 的形式。

---

## 18. 第九步：输出投影

你的代码：

```python
return self.output_proj(output)
```

这里 `output.shape` 是：

```python
[3, 2, 128]
```

再经过：

```python
nn.Linear(128, 128)
```

输出仍然是：

```python
[3, 2, 128]
```

### 18.1 这个 `output_proj` 的作用

它的作用不是“改 shape”，而是：

- 让不同 head 拼接后的结果重新做一次融合
- 给模型多一层可学习变换
- 让多头的信息可以混合起来

所以这个输出层非常常见，也通常是 Transformer 标准实现的一部分。

---

## 19. 结合你这个例子，完整 shape 流程梳理一遍

你这里测试代码是：

```python
x = torch.rand(3, 2, 128)
net = MultiHeadAttention(128, 8)
net(x, attention_mask).shape
```

所以整个流程的 shape 变化如下：

### 输入

```python
x: [3, 2, 128]
```

### 线性映射后

```python
q, k, v: [3, 2, 128]
```

### 拆头后

```python
q_state, k_state, v_state: [3, 8, 2, 16]
```

### attention score

```python
attention_value: [3, 8, 2, 2]
```

### softmax 后权重

```python
attention_weight: [3, 8, 2, 2]
```

### 加权求和后

```python
output_mid: [3, 8, 2, 16]
```

### 交换维度后

```python
[3, 2, 8, 16]
```

### 拼接 heads 后

```python
output: [3, 2, 128]
```

### 最后输出投影后

```python
final output: [3, 2, 128]
```

这一整条链路你一定要能自己默写出来。

---

## 20. 你这份代码里最值得记住的 3 个核心点

### 核心点 1：多头不是多套输入，而是把最后一维拆开

不是说把输入复制 8 份再做。
而是把：

```python
hidden_dim = 128
```

拆成：

```python
8 个头，每个头 16 维
```

### 核心点 2：attention 是在每个 head 内部单独算的

所以 score shape 才是：

```python
[B, num_heads, T, T]
```

每个头都有自己的一套注意力矩阵。

### 核心点 3：最后一定要拼回 `[B, T, C]`

因为后面的 Transformer 模块（残差连接、LayerNorm、FFN）通常都要求输入输出维度一致。

---

## 21. 你这份 notebook 和上一份 Single Head 的关系

可以这样理解：

### Single Head

流程是：

```python
[B, T, C]
-> Q, K, V
-> [B, T, T]
-> [B, T, C]
```

### Multi Head

流程是：

```python
[B, T, C]
-> Q, K, V
-> [B, num_heads, T, head_dim]
-> [B, num_heads, T, T]
-> [B, num_heads, T, head_dim]
-> [B, T, C]
```

所以多头注意力其实不是新的大逻辑，而是在单头基础上多了一层：

> “拆头并行计算，再拼接回来。”

---

## 22. 面试/复习时可以怎么说

如果别人问你：

### 22.1 什么是 Multi-Head Attention？

你可以答：

> Multi-Head Attention 就是把 hidden_dim 拆成多个 head，每个 head 在自己的子空间里独立计算注意力，这样模型可以从不同角度学习 token 间关系。最后再把所有 head 的输出拼接起来，并通过输出投影层融合。

### 22.2 为什么要多头？

你可以答：

> 因为单头只能学习一种注意力模式，而多头可以并行学习多种关系，比如局部依赖、长距离依赖、语法关系等，从而提升表示能力。

### 22.3 多头的 shape 怎么变？

你可以答：

> 输入通常是 `[B, T, C]`，先映射得到 `Q/K/V`，再 reshape 成 `[B, T, num_heads, head_dim]`，再 transpose 成 `[B, num_heads, T, head_dim]`。算完注意力输出后，再 transpose 回去并 reshape 成 `[B, T, C]`。

---

## 23. 这份代码里你最容易混淆的点

### 23.1 `hidden_dim` 和 `head_dim` 不要混

- `hidden_dim`：总维度
- `head_dim`：每个头内部的维度

标准缩放用的是：

```python
sqrt(head_dim)
```

不是 `sqrt(hidden_dim)`。

### 23.2 mask 不一定就是 causal mask

你这里这个 mask 更像是 **key padding mask 广播后** 的形式。

标准 causal mask 一般是下三角矩阵，比如：

```python
[[1, 0, 0],
 [1, 1, 0],
 [1, 1, 1]]
```

表示当前位置不能看未来。

而你这里每个样本最开始只是：

```python
[0, 1]
[0, 0]
[1, 0]
```

更像是在描述“哪些 key 位置可见，哪些不可见”。

### 23.3 全 0 mask 会产生 NaN

这是非常重要的数值稳定性问题，后面自己写 decoder 时一定要注意。

---

## 24. 推荐你把这份代码改成更标准的版本

下面是更推荐的写法，只保留核心逻辑：

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, head_num, dropout=0.1):
        super().__init__()
        assert hidden_dim % head_num == 0

        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.head_dim = hidden_dim // head_num

        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        B, T, C = x.shape

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        q = q.view(B, T, self.head_num, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.head_num, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.head_num, self.head_dim).transpose(1, 2)

        scores = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.output_proj(out)
        return out
```

---

## 25. 复习这份 notebook 时，你应该重点问自己这几个问题

### 问题 1
为什么 `q/k/v` 一开始还是 `[B, T, hidden_dim]`，而不是 `[B, T, head_dim]`？

### 问题 2
`view(...).transpose(1, 2)` 这两步分别在干什么？

### 问题 3
为什么 attention score 是 `[B, num_heads, T, T]`？

### 问题 4
为什么最后一定要把多头重新拼回 `[B, T, hidden_dim]`？

### 问题 5
为什么标准缩放因子是 `sqrt(head_dim)`？

### 问题 6
如果某一行全被 mask 掉，会发生什么？

只要这些问题你都能回答清楚，这份 Multi-Head Attention 就算真正掌握了。

---

## 26. 一句话总结

**Multi-Head Self-Attention 的本质，就是先把 hidden_dim 拆成多个头，让每个头在各自的子空间里独立计算注意力，最后再把这些头的结果拼接回来，从而让模型能并行学习多种 token 关系。**

---

## 27. 这一份笔记对应你当前 notebook 的两个立刻可改点

### 可改点 1：缩放因子

把：

```python
attention_weight = attention_value / math.sqrt(self.hidden_dim)
```

改成：

```python
attention_weight = attention_value / math.sqrt(self.head_dim)
```

### 可改点 2：mask 示例

不要给某个样本设置全 0 的 key mask，否则 softmax 后会出 NaN。

---

## 28. 下一步该怎么衔接

这一份学完后，最自然的下一步就是：

1. 把 `MultiHeadAttention` 放进一个完整的 `Decoder Block`
2. 理解 residual、layer norm、FFN
3. 理解为什么 decoder 要用 causal mask

也就是说，下一份 `Transformer_Decoder.ipynb` 会正好承接这份内容。

