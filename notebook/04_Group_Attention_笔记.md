# 04 Group Attention 笔记

## 1. 这一份 Notebook 在学什么

这一份 `04_GroupAttention.ipynb`，是在前面 **Multi-Head Self-Attention** 的基础上，继续往前走一步：

> 不再让每一个 Query Head 都各自拥有一套独立的 K、V，而是让多个 Query Head 共享同一组 K、V。

这就是这一份代码要表达的核心思想，也就是常说的：

- **GQA：Grouped Query Attention**
- 中文可以理解成：**分组查询注意力**

它最重要的意义不是“让公式变得更复杂”，而是：

> **在尽量保持多头注意力效果的同时，减少 K/V 的参数量与 KV cache 开销。**

所以这份 notebook 的重点，不是重新讲一遍普通注意力，而是回答下面这个问题：

**为什么要从 MHA 走向 GQA？**

---

## 2. 先说结论：GQA 到底在改什么

先回忆普通的 Multi-Head Attention。

如果有：

- `num_heads = 8`

那么在标准 MHA 里，通常会有：

- 8 个 Query Head
- 8 个 Key Head
- 8 个 Value Head

也就是：

```text
Q 的头数 = K 的头数 = V 的头数 = 8
```

而在你这份代码里：

```python
net = GroupQueryAttention(128, 8, 4)
```

意思是：

- `hidden_dim = 128`
- `num_heads = 8`
- `nums_key_value_heads = 4`

也就是：

```text
Q 有 8 个头
K 只有 4 个头
V 只有 4 个头
```

然后让这 4 组 K/V 去服务 8 个 Query Head。

换句话说：

> **多个 Q 头共享同一组 K/V 头。**

---

## 3. 为什么要这样做

原因主要有两个。

### 3.1 减少参数量

看你的定义：

```python
self.q = nn.Linear(hidden_dim, num_heads * self.head_dim)
self.k = nn.Linear(hidden_dim, nums_key_value_heads * self.head_dim)
self.v = nn.Linear(hidden_dim, nums_key_value_heads * self.head_dim)
```

这里有一个关键区别：

- `q` 的输出维度仍然是 `num_heads * head_dim`
- 但 `k` 和 `v` 的输出维度变成了 `nums_key_value_heads * head_dim`

因为 `nums_key_value_heads < num_heads`，所以：

- K 的参数更少
- V 的参数更少

---

### 3.2 减少 KV Cache 开销

在自回归推理里，每一步都会缓存历史 token 的 K 和 V。

如果是普通 MHA，要缓存：

```text
[batch, num_heads, seq_len, head_dim]
```

如果是 GQA，只需要缓存：

```text
[batch, nums_key_value_heads, seq_len, head_dim]
```

假设：

- `num_heads = 8`
- `nums_key_value_heads = 4`

那么 K/V cache 的头数直接减半。

这也是很多大模型喜欢 GQA 的重要原因：

> **它特别适合优化推理阶段的显存占用和吞吐。**

---

## 4. 它和 MHA、MQA 的关系

其实这三者可以放在一条线上看。

### 4.1 MHA（Multi-Head Attention）

```text
Q 头数 = K 头数 = V 头数
```

比如：

```text
8 : 8 : 8
```

最标准，但 K/V 成本最高。

---

### 4.2 MQA（Multi-Query Attention）

```text
Q 有很多头
K/V 只有 1 头
```

比如：

```text
8 : 1 : 1
```

K/V 成本最低，但共享得太狠，有时表达能力会受影响。

---

### 4.3 GQA（Grouped Query Attention）

它是中间路线。

比如：

```text
8 : 4 : 4
```

既比 MHA 更省，又比 MQA 更灵活。

所以可以把 GQA 理解成：

> **MHA 和 MQA 之间的折中方案。**

---

## 5. 你的代码整体结构

这份 notebook 的核心类是：

```python
class GroupQueryAttention(nn.Module):
```

它的结构非常清晰，主要包含：

1. 定义 `q / k / v / o` 四个线性层
2. 在 `forward` 中计算 Q、K、V
3. 把 Q/K/V reshape 成多头形式
4. 对 K/V 做分组共享
5. 计算 attention score
6. 做 mask、softmax、dropout
7. 用权重加权 V
8. 拼回原维度并输出

所以整体流程可以概括成：

```text
x
-> q, k, v 线性映射
-> reshape 成 head 形式
-> K/V 复制到和 Q 同样的 head 数
-> attention(Q, K, V)
-> 拼接各头
-> output projection
```

---

## 6. 初始化部分详解

你的初始化代码是：

```python
class GroupQueryAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, nums_key_value_heads):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.nums_key_value_heads = nums_key_value_heads
        self.head_dim = hidden_dim // num_heads

        self.q = nn.Linear(hidden_dim, num_heads * self.head_dim)
        self.k = nn.Linear(hidden_dim, nums_key_value_heads * self.head_dim)
        self.v = nn.Linear(hidden_dim, nums_key_value_heads * self.head_dim)
        self.o = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(0.1)
```

下面逐个理解。

---

## 7. `head_dim` 是怎么来的

你写的是：

```python
self.head_dim = hidden_dim // num_heads
```

这和普通多头注意力是一样的。

如果：

- `hidden_dim = 128`
- `num_heads = 8`

那么：

```text
head_dim = 128 // 8 = 16
```

意思是：

> 每个 Query Head 的通道维度是 16。

这里要注意：

**K/V 头虽然数量减少了，但每个头的 `head_dim` 仍然是 16。**

所以：

- Q 的总维度：`8 * 16 = 128`
- K 的总维度：`4 * 16 = 64`
- V 的总维度：`4 * 16 = 64`

---

## 8. `q / k / v / o` 这四个线性层分别干什么

### 8.1 `self.q`

```python
self.q = nn.Linear(hidden_dim, num_heads * self.head_dim)
```

因为 `num_heads * head_dim = hidden_dim`，所以这里本质上是：

```python
nn.Linear(128, 128)
```

Q 仍然保留完整的多头数量。

---

### 8.2 `self.k`

```python
self.k = nn.Linear(hidden_dim, nums_key_value_heads * self.head_dim)
```

这里变成：

```python
nn.Linear(128, 64)
```

说明 K 不再为每一个 Q 头单独准备，而是压缩成更少的 KV 头。

---

### 8.3 `self.v`

和 `self.k` 一样：

```python
nn.Linear(128, 64)
```

V 也只保留较少的头数。

---

### 8.4 `self.o`

```python
self.o = nn.Linear(hidden_dim, hidden_dim)
```

这是输出投影层。

在各头拼回 `[B, T, hidden_dim]` 之后，再通过这个线性层混合各头信息。

---

## 9. 这一份实现隐含了一个重要前提

注意这一句：

```python
k = k.repeat_interleave(self.num_heads // self.nums_key_value_heads, dim=1)
```

它要求：

```text
num_heads % nums_key_value_heads == 0
```

也就是 Query Head 数必须能整除 KV Head 数。

例如：

- `8 // 4 = 2`，可以
- `8 // 2 = 4`，可以
- `8 // 1 = 8`，可以
- `8 // 3`，就不合适了

所以这份代码默认的是：

> **每一组 KV Head 被等量地分配给若干个 Query Head。**

这个约束非常重要，建议你在代码里加一个断言：

```python
assert num_heads % nums_key_value_heads == 0
```

---

## 10. `forward` 的整体流程

你的 `forward` 是：

```python
def forward(self, x, attention_mask=None):
    batch_size, seq_len, _ = x.size()

    q = self.q(x)
    k = self.k(x)
    v = self.v(x)

    q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
    k = k.view(batch_size, seq_len, self.nums_key_value_heads, self.head_dim)
    v = v.view(batch_size, seq_len, self.nums_key_value_heads, self.head_dim)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    k = k.repeat_interleave(self.num_heads // self.nums_key_value_heads, dim=1)
    v = v.repeat_interleave(self.num_heads // self.nums_key_value_heads, dim=1)

    attention_weight = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)

    if attention_mask is not None:
        attention_weight = attention_weight.masked_fill(
            attention_mask == 0,
            float("-1e20")
        )

    attention_weight = torch.softmax(attention_weight, dim=-1)
    attention_weight = self.dropout(attention_weight)

    output = attention_weight @ v
    output = output.transpose(1, 2).contiguous()
    output = output.view(batch_size, seq_len, -1)

    return self.o(output)
```

这段代码可以分成 8 步来看。

---

## 11. 第一步：取出 batch 和 seq_len

```python
batch_size, seq_len, _ = x.size()
```

如果你的输入是：

```python
x = torch.rand(3, 2, 128)
```

那么：

- `batch_size = 3`
- `seq_len = 2`
- `hidden_dim = 128`

也就是：

```text
x.shape = [3, 2, 128]
```

含义是：

- batch 里有 3 个样本
- 每个样本有 2 个 token
- 每个 token 的向量维度是 128

---

## 12. 第二步：先算出原始 q、k、v

```python
q = self.q(x)
k = self.k(x)
v = self.v(x)
```

shape 会变成：

- `q.shape = [3, 2, 128]`
- `k.shape = [3, 2, 64]`
- `v.shape = [3, 2, 64]`

这里最值得记住的一点是：

> 在 GQA 里，Q 仍然保持完整头数，所以总维度还是 128；K 和 V 因为头数减少，所以总维度只有 64。

---

## 13. 第三步：reshape 成多头形式

```python
q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
k = k.view(batch_size, seq_len, self.nums_key_value_heads, self.head_dim)
v = v.view(batch_size, seq_len, self.nums_key_value_heads, self.head_dim)
```

代入数值后：

- Q：`[3, 2, 8, 16]`
- K：`[3, 2, 4, 16]`
- V：`[3, 2, 4, 16]`

此时最后两个维度分别表示：

```text
[num_heads, head_dim]
```

或

```text
[num_kv_heads, head_dim]
```

---

## 14. 第四步：transpose，把 head 维提前

```python
q = q.transpose(1, 2)
k = k.transpose(1, 2)
v = v.transpose(1, 2)
```

shape 变成：

- Q：`[3, 8, 2, 16]`
- K：`[3, 4, 2, 16]`
- V：`[3, 4, 2, 16]`

这样做是为了后面方便做矩阵乘法。

此时维度顺序是：

```text
[batch, heads, seq_len, head_dim]
```

这是注意力实现里最常见的布局之一。

---

## 15. 第五步：GQA 的关键——复制 K/V 头

这一段是全篇最核心的地方：

```python
k = k.repeat_interleave(self.num_heads // self.nums_key_value_heads, dim=1)
v = v.repeat_interleave(self.num_heads // self.nums_key_value_heads, dim=1)
```

因为：

- `num_heads = 8`
- `nums_key_value_heads = 4`

所以：

```python
self.num_heads // self.nums_key_value_heads = 2
```

也就是每个 K/V 头重复两次。

于是：

- 原来的 K：`[3, 4, 2, 16]`
- 复制后的 K：`[3, 8, 2, 16]`

同理：

- 原来的 V：`[3, 4, 2, 16]`
- 复制后的 V：`[3, 8, 2, 16]`

这样它们就和 Q 的头数对齐了。

---

## 16. “复制”到底是什么意思

这里很容易误解。

你可以把它理解成：

- Q 有 8 个头
- K/V 只有 4 个头
- 所以把每个 K/V 头分给两个 Q 头共用

比如可以想象成：

```text
Q0, Q1 共享 KV0
Q2, Q3 共享 KV1
Q4, Q5 共享 KV2
Q6, Q7 共享 KV3
```

而 `repeat_interleave` 在代码层面做的事，就是把这种共享关系展开成可直接计算的张量形状。

---

## 17. 这里有一个容易忽略的工程点

虽然逻辑上 GQA 是“共享 K/V”，但你现在这份实现用了：

```python
repeat_interleave(...)
```

这意味着在当前前向计算里，K/V 被**显式复制**出来了。

所以这份代码更适合：

- 教学理解
- 验证 shape
- 理解 GQA 的思路

但如果追求真正的高效实现，很多框架不会简单粗暴地物理复制整个张量，而是会用更底层、更高效的 kernel 或广播方式来避免额外开销。

也就是说：

> **这份实现把“概念讲清楚了”，但不一定是最省显存、最快的生产写法。**

---

## 18. 第六步：计算注意力分数

```python
attention_weight = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
```

这一步和普通多头注意力完全一致。

其中：

- `q.shape = [3, 8, 2, 16]`
- `k.transpose(-1, -2).shape = [3, 8, 16, 2]`

矩阵乘法后得到：

```text
attention_weight.shape = [3, 8, 2, 2]
```

含义是：

- 对 batch 中每个样本
- 对每个 query head
- 对序列中每个位置
- 去计算它和所有 key 位置之间的相关性

最后一个 `2` 表示“当前 token 能看向哪些 token”。

---

## 19. 为什么分母仍然是 `sqrt(head_dim)`

你这里写的是：

```python
/math.sqrt(self.head_dim)
```

这是对的。

因为注意力分数是单个 head 内部做的点积，所以缩放要按每个 head 的维度来，而不是按总 hidden_dim。

这一点和普通 Multi-Head Attention 没有区别。

---

## 20. 第七步：mask 的作用

```python
if attention_mask is not None:
    attention_weight = attention_weight.masked_fill(
        attention_mask == 0,
        float("-1e20")
    )
```

这里的意思是：

- mask 为 0 的位置，不允许关注
- 给这些位置填一个极小值
- 这样经过 softmax 之后，这些位置的权重就几乎为 0

这通常用来做两类事情：

### 20.1 Padding Mask
屏蔽无效的 padding token。

### 20.2 Causal Mask
在 decoder 中屏蔽未来位置，防止偷看答案。

---

## 21. mask 的 shape 要能广播

你的代码没有显示构造 `attention_mask`，但如果要传入，它的 shape 必须和：

```text
attention_weight.shape = [B, H, T, T]
```

兼容。

常见可行写法有：

- `[B, 1, 1, T]`
- `[B, 1, T, T]`
- `[B, H, T, T]`

只要能够广播到 `[B, H, T, T]` 就可以。

---

## 22. 第八步：softmax + dropout

```python
attention_weight = torch.softmax(attention_weight, dim=-1)
attention_weight = self.dropout(attention_weight)
```

### 22.1 `softmax(dim=-1)`

表示对最后一个维度做归一化。

在这里最后一个维度表示：

> 当前 query 位置对所有 key 位置的注意力分布。

所以每一行权重和为 1。

---

### 22.2 `dropout`

训练时会随机把部分注意力权重置零，用来防止过拟合。

注意：

> dropout 作用在 attention 权重上，不会改变张量 shape。

---

## 23. 第九步：用注意力权重加权 V

```python
output = attention_weight @ v
```

此时：

- `attention_weight.shape = [3, 8, 2, 2]`
- `v.shape = [3, 8, 2, 16]`

相乘后得到：

```text
output.shape = [3, 8, 2, 16]
```

含义是：

- 每个 head
- 每个位置
- 都从所有 value 位置里做加权求和
- 得到这个 head 的输出表示

---

## 24. 第十步：拼回原来的 hidden 维度

```python
output = output.transpose(1, 2).contiguous()
output = output.view(batch_size, seq_len, -1)
```

先做 transpose：

- `[3, 8, 2, 16] -> [3, 2, 8, 16]`

再把多头拼接起来：

- `[3, 2, 8, 16] -> [3, 2, 128]`

也就是回到：

```text
[batch_size, seq_len, hidden_dim]
```

这一步说明：

> 尽管 K/V 头数减少了，但最终输出仍然和普通多头注意力一样，回到了完整的 hidden_dim。

---

## 25. 第十一步：输出投影

```python
return self.o(output)
```

这里的 `self.o` 是：

```python
nn.Linear(hidden_dim, hidden_dim)
```

所以输出 shape 仍然是：

```text
[3, 2, 128]
```

这也正好和你 notebook 的输出一致：

```python
torch.Size([3, 2, 128])
```

---

## 26. 结合你的例子，把完整 shape 流程梳理一遍

你的示例是：

```python
x = torch.rand(3, 2, 128)
net = GroupQueryAttention(128, 8, 4)
net(x).shape
```

完整流程如下。

### 输入
```text
x = [3, 2, 128]
```

### 线性映射后
```text
q = [3, 2, 128]
k = [3, 2, 64]
v = [3, 2, 64]
```

### reshape 后
```text
q = [3, 2, 8, 16]
k = [3, 2, 4, 16]
v = [3, 2, 4, 16]
```

### transpose 后
```text
q = [3, 8, 2, 16]
k = [3, 4, 2, 16]
v = [3, 4, 2, 16]
```

### K/V 复制后
```text
k = [3, 8, 2, 16]
v = [3, 8, 2, 16]
```

### 注意力分数
```text
attention_weight = [3, 8, 2, 2]
```

### 加权求和后
```text
output = [3, 8, 2, 16]
```

### 拼回去后
```text
output = [3, 2, 128]
```

### 最终输出
```text
self.o(output) = [3, 2, 128]
```

---

## 27. 你这份 notebook 里很值得记住的 4 个核心点

### 核心点 1
GQA 不是减少 Query Head，而是减少 **Key/Value Head**。

---

### 核心点 2
Q 头数和 K/V 头数不一样时，需要建立“多个 Q 头共享一个 KV 头”的关系。

---

### 核心点 3
你这份代码通过 `repeat_interleave` 把共享关系显式展开，从而让矩阵乘法可以直接进行。

---

### 核心点 4
GQA 的最终输出 shape 和普通多头注意力一样，仍然是：

```text
[B, T, hidden_dim]
```

所以它是内部结构优化，不会改外部接口。

---

## 28. 这一份代码和上一份 Multi-Head 的关系

你可以这样理解两者关系：

### Multi-Head Self-Attention
每个头都有自己独立的 Q/K/V。

### Group Query Attention
Q 头依然很多，但多个 Q 头共享同一组 K/V。

所以 GQA 可以看成：

> **在 Multi-Head Attention 基础上，对 K/V 做参数共享的改造版。**

这就是为什么它不是一个全新的范式，而更像是多头注意力的工程优化变体。

---

## 29. 这份实现里有几个可补强的小地方

### 29.1 建议加断言

```python
assert hidden_dim % num_heads == 0
assert num_heads % nums_key_value_heads == 0
```

这样可以避免 shape 不合法。

---

### 29.2 变量名建议统一

你现在写的是：

```python
nums_key_value_heads
```

更常见的写法是：

```python
num_key_value_heads
```

即 `num` 而不是 `nums`。

虽然不影响运行，但标准命名会更清晰。

---

### 29.3 注释里关于 KV cache 的表述要更谨慎

你写了：

```python
# 此时的k、v都是持久化存储在显存中，作为KV Cache的一部分
```

这句话更准确地说应该是：

> 在自回归推理场景下，历史步的 K/V 通常会被缓存成 KV cache；  
> 但在你这个普通前向示例里，只是当前一次 forward 里临时产生了 K/V 张量，并不等于已经实现了完整的 KV cache 机制。

也就是说：

- 这句注释表达的是“GQA 为什么适合 KV cache”
- 但不是说这份 notebook 已经实现了真正的 cache 管理逻辑

---

### 29.4 `print` 更适合调试，不适合正式 forward

你现在在 `forward` 里写了：

```python
print(k.shape)
print(q.shape)
```

这很适合理解 shape。

但如果以后作为正式模块使用，建议删掉或者改成可选调试开关，不然每次前向都会打印。

---

## 30. 面试/复习时可以怎么说

你可以这样描述 GQA：

> 标准 Multi-Head Attention 中，Q/K/V 的头数通常相同；而 Grouped Query Attention 保留较多的 Query Head，但减少 Key/Value Head，让多个 Query Head 共享同一组 K/V。这样做可以减少 K/V 参数量和自回归推理时的 KV cache 开销，同时又比单一 K/V 头的 MQA 保留更多表达能力。

这个说法已经比较完整了。

---

## 31. 你复习这一份时，最应该问自己的问题

### 问题 1
为什么 GQA 只减少 K/V 头，而不减少 Q 头？

### 问题 2
为什么减少 K/V 头能节省 KV cache？

### 问题 3
为什么最终输出 shape 还是 `[B, T, hidden_dim]`？

### 问题 4
`repeat_interleave` 在这里到底是在做“数学共享”，还是“实现上的显式复制”？

### 问题 5
GQA 和 MHA、MQA 三者各自的权衡是什么？

如果这 5 个问题你都能顺下来讲清楚，这份 notebook 就算真正掌握了。

---

## 32. 一句话总结

> **GQA 的本质，就是保留多 Query 头的表达能力，同时让多个 Query Head 共享较少的 K/V Head，以换取更低的参数与 KV cache 成本。**

---

## 33. 这一份 notebook 对你当前学习路径的意义

你现在的路径大概是：

- Single Head：理解注意力最基本的计算
- Multi Head：理解为什么要拆头
- Decoder：理解这些模块怎么堆成一个 block / 模型
- Group Attention：理解工业界为什么还要继续优化注意力结构

所以这一份 notebook 非常重要的一点是：

> 它让你从“会实现 Transformer”开始转向“会理解 Transformer 为什么会被工程化改造”。

这是继续往大模型工程、推理优化方向走时很关键的一步。

---

## 34. 下一步最自然的衔接

学完这一份之后，你最适合接着思考两件事：

### 第一件事：KV Cache 到底是什么
因为 GQA 的价值和 KV cache 强相关。

### 第二件事：LoRA 为什么不是改注意力结构，而是改参数更新方式
这样你会更清楚：
- GQA 是 **推理/结构优化**
- LoRA 是 **参数高效微调**

这两类方法解决的问题并不一样。
