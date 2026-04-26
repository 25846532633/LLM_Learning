# 01 Single-Head Self-Attention 笔记

> 对应文件：`SingleHead_self_attention.ipynb`  
> 笔记目标：把这个 notebook 里的 **单头自注意力** 从“能跑代码”整理成“能讲原理、能解释 shape、能回答面试问题”。

---

# 一、这个文件整体在讲什么？

这个 notebook 其实不是只写了一个版本的单头自注意力，而是分成了三层：

1. **第一重：简单操作**  
   先写出最基础、最直观的单头自注意力，让你看懂核心流程。

2. **第二重：效率优化**  
   把原来分开的 `q / k / v` 三个线性层，合并成一个投影层，提高实现效率。

3. **第三重：面试 / 工程写法**  
   加入 `attention_mask`、`dropout`、`output_proj` 等，更接近真实工程代码，也更适合面试回答。

所以，这个文件的本质不是“重复写三遍”，而是让你看到：

**最朴素的注意力实现 → 更高效的实现 → 更接近工程与面试的实现**

---

# 二、先用一句话理解 Self-Attention

Self-Attention 的核心思想是：

**序列中的每个 token，都去看一遍序列里的其他 token，然后根据相关性，把别人的信息加权汇总到自己身上。**

换句话说：

- `Q`（Query）：我想找什么信息
- `K`（Key）：我能提供什么信息
- `V`（Value）：我真正携带的信息内容

计算过程就是：

1. 用 `Q` 和 `K` 计算“谁和谁相关”
2. 对相关性做 `softmax`，得到注意力权重
3. 用权重对 `V` 做加权求和，得到新的表示

公式是：

\[
Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

---

# 三、第一重：简单操作版 SelfAttention

## 3.1 原始代码（带解释）

```python
class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, *args, **kwargs):
        super(SelfAttention, self).__init__()
        self.hidden_dim = hidden_dim

        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # 1. 获取 q, k, v
        q = self.q(x)  # [B, L, D]
        k = self.k(x)  # [B, L, D]
        v = self.v(x)  # [B, L, D]

        # 2. 计算注意力分数
        attention_weights = (q @ k.transpose(-2, -1)) / math.sqrt(self.hidden_dim)  # [B, L, L]

        # 3. softmax 归一化
        attention_weights = torch.softmax(attention_weights, dim=-1)  # [B, L, L]

        # 4. 加权求和
        output = attention_weights @ v  # [B, L, D]

        return output
```

---

## 3.2 每一步到底在干什么？

### 第一步：输入 `x`

假设输入：

```python
X = torch.randn(3, 4, 2)
```

那么：

- `B = 3`：batch size
- `L = 4`：序列长度
- `D = 2`：hidden_dim

所以输入 shape 是：

\[
x \in \mathbb{R}^{[B, L, D]} = [3, 4, 2]
\]

这里可以这样理解：

- 一共有 3 个样本
- 每个样本有 4 个 token
- 每个 token 用 2 维向量表示

---

### 第二步：通过线性层得到 `q / k / v`

```python
q = self.q(x)  # [B, L, D]
k = self.k(x)  # [B, L, D]
v = self.v(x)  # [B, L, D]
```

这里的三个 `nn.Linear(hidden_dim, hidden_dim)` 本质上就是三个不同的可学习矩阵：

- `W_q`
- `W_k`
- `W_v`

所以：

\[
Q = XW_q,\quad K = XW_k,\quad V = XW_v
\]

输入输出维度都还是 `D -> D`，因此 shape 不变，仍然是 `[B, L, D]`。

---

### 第三步：计算注意力分数

```python
attention_weights = (q @ k.transpose(-2, -1)) / math.sqrt(self.hidden_dim)
```

先看 `k.transpose(-2, -1)`：

- 原来 `k` 是 `[B, L, D]`
- 转置后变成 `[B, D, L]`

于是：

- `q` 是 `[B, L, D]`
- `k.transpose(-2, -1)` 是 `[B, D, L]`

相乘以后得到：

\[
[B, L, D] @ [B, D, L] = [B, L, L]
\]

这意味着：  
对于每个 batch 中的样本，都会得到一个 `L x L` 的打分矩阵。

### 这个 `L x L` 矩阵表示什么？

它表示：

- 行：当前这个 query token
- 列：它要去看的 key token

也就是说，第 `i` 行第 `j` 列表示：

**第 i 个 token 对第 j 个 token 的关注程度（原始打分）**

---

### 第四步：为什么要除以 `sqrt(hidden_dim)`？

```python
/ math.sqrt(self.hidden_dim)
```

这是 **缩放点积注意力（Scaled Dot-Product Attention）** 的关键步骤。

原因是：

如果向量维度 `D` 比较大，`q @ k^T` 的值会变得很大。  
一旦分数太大，softmax 之后会变得特别尖锐，导致：

- 梯度不稳定
- 训练困难
- 模型容易过早“极端自信”

所以要除以：

\[
\sqrt{d_k}
\]

在这个单头实现里，`d_k = hidden_dim`。

---

### 第五步：为什么 softmax 要写 `dim=-1`？

```python
attention_weights = torch.softmax(attention_weights, dim=-1)
```

`attention_weights` 的 shape 是 `[B, L, L]`。

最后一个维度 `-1` 表示：  
**固定一个 query，看它对所有 key 的分配概率。**

也就是说，softmax 后每一行加起来等于 1。

这才符合注意力的语义：

> 对于当前 token，我应该把多少注意力分给序列里的每个位置？

这部分很容易说反。  
更准确的说法是：

- **固定 query**
- **对所有 key 做归一化**

而不是“固定 key 对所有 query 归一化”。

---

### 第六步：加权求和得到输出

```python
output = attention_weights @ v
```

shape 变化：

- `attention_weights` 是 `[B, L, L]`
- `v` 是 `[B, L, D]`

所以输出是：

\[
[B, L, L] @ [B, L, D] = [B, L, D]
\]

这说明：

- 每个 query 位置都会拿到一个新的向量表示
- 这个表示是对所有 value 的加权和
- 最终输出 shape 和输入 shape 一致，仍然是 `[B, L, D]`

---

## 3.3 第一版的优点和不足

### 优点
- 写法直观
- 非常适合理解核心公式
- 能清晰看出 `Q、K、V` 分别是怎么来的

### 不足
- 需要三个线性层，计算上不够紧凑
- 没有 mask
- 没有 dropout
- 没有输出投影层
- 更像“教学代码”，不太像完整工程代码

---

# 四、第一版的 shape 总结

假设：

```python
X.shape = [3, 4, 2]
```

那么完整过程是：

1. 输入：
   - `x`: `[3, 4, 2]`

2. 线性映射：
   - `q`: `[3, 4, 2]`
   - `k`: `[3, 4, 2]`
   - `v`: `[3, 4, 2]`

3. 打分：
   - `k.transpose(-2, -1)`: `[3, 2, 4]`
   - `q @ k^T`: `[3, 4, 4]`

4. softmax：
   - `attention_weights`: `[3, 4, 4]`

5. 加权求和：
   - `output = attention_weights @ v`: `[3, 4, 2]`

---

# 五、第二重：效率优化版 SelfAttention2

## 5.1 原始代码（带解释）

```python
class SelfAttention2(nn.Module):
    def __init__(self, hidden_dim, *args, **kwargs):
        super(SelfAttention2, self).__init__()

        self.hidden_dim = hidden_dim

        self.proj = nn.Linear(hidden_dim, hidden_dim * 3)

        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # 1. 获取 q, k, v
        qkv = self.proj(x)
        q, k, v = torch.split(qkv, self.hidden_dim, dim=-1)

        # 2. 计算注意力权重
        attention_weights = (q @ k.transpose(-2, -1)) / math.sqrt(self.hidden_dim)

        # 3. softmax
        attention_weights = torch.softmax(attention_weights, dim=-1)

        # 4. 加权求和
        output = attention_weights @ v

        return self.output_proj(output)
```

---

## 5.2 它优化了什么？

第一版里，`q / k / v` 是这样来的：

```python
self.q = nn.Linear(D, D)
self.k = nn.Linear(D, D)
self.v = nn.Linear(D, D)
```

第二版把它改成：

```python
self.proj = nn.Linear(D, 3D)
```

然后：

```python
qkv = self.proj(x)
q, k, v = torch.split(qkv, self.hidden_dim, dim=-1)
```

也就是说：

原本做三次线性变换，现在合成一次。

---

## 5.3 为什么这样做更高效？

因为底层实现中，大矩阵乘法通常更容易充分利用硬件并行能力。

原本是：

- 做一次 `xW_q`
- 做一次 `xW_k`
- 做一次 `xW_v`

现在变成：

- 一次性做 `xW_{qkv}`，然后再切开

这种写法在真实 Transformer 实现里非常常见。

---

## 5.4 `torch.split(..., dim=-1)` 是什么意思？

```python
q, k, v = torch.split(qkv, self.hidden_dim, dim=-1)
```

假设：

- `qkv.shape = [B, L, 3D]`

那么沿着最后一个维度，每 `D` 个切一块，就会得到：

- `q`: `[B, L, D]`
- `k`: `[B, L, D]`
- `v`: `[B, L, D]`

---

## 5.5 为什么多了 `output_proj`？

```python
self.output_proj = nn.Linear(hidden_dim, hidden_dim)
```

在单头版本里，这一层不是绝对必须。  
但它有两个现实意义：

### 第一，和 Multi-Head Attention 的接口保持一致
多头注意力会把多个头拼接起来，再过一个输出投影层。  
提前保留 `output_proj`，以后从单头扩展到多头会更自然。

### 第二，增强表示能力
即便单头输出还是 `[B, L, D]`，再做一次线性映射，也可以进一步调整输出特征。

---

## 5.6 第二版的本质

第二版并没有改变 Self-Attention 的数学本质。  
它只是把：

- **Q / K / V 的生成方式**
- **输出的后处理方式**

写得更像工程代码。

注意力核心公式依旧没变：

\[
softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

---

# 六、第三重：面试 / 工程版 SelfAttention3

## 6.1 原始代码（带解释）

```python
class SelfAttention3(nn.Module):
    def __init__(self, hidden_dim, *args, **kwargs):
        super(SelfAttention3, self).__init__()
        self.hidden_dim = hidden_dim

        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(0.1)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, attention_mask=None):
        # 1. 得到 q, k, v
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # 2. 计算注意力权重
        attention_value = q @ k.transpose(-2, -1)
        attention_weights = attention_value / math.sqrt(self.hidden_dim)

        if attention_mask is not None:
            attention_weights = attention_weights.masked_fill(
                attention_mask == 0,
                float("-1e20")
            )

        # 3. softmax
        attention_weights = torch.softmax(attention_weights, dim=-1)

        # 4. dropout
        attention_weights = self.dropout(attention_weights)

        # 5. 输出
        output = attention_weights @ v
        return self.output_proj(output)
```

---

## 6.2 这一版比前面多了什么？

### 1）`attention_mask`
用于屏蔽不该看到的位置。

### 2）`dropout`
用于防止过拟合。

### 3）`output_proj`
更接近完整注意力模块的写法。

所以第三版最重要的价值不是“更复杂”，而是：

**它更接近真实模型中的注意力模块。**

---

# 七、`attention_mask` 到底在干什么？

## 7.1 mask 的作用

在实际任务里，并不是所有位置都应该被关注。

常见有两种 mask：

### （1）Padding Mask
屏蔽补齐位置 `<pad>`  
因为 pad 是为了凑齐长度，不是真实内容，不应该参与注意力计算。

### （2）Causal Mask / Future Mask
在 Decoder 中屏蔽未来信息  
因为第 `t` 个位置不能偷看第 `t+1` 之后的 token。

---

## 7.2 你这个 notebook 里的 mask 属于哪一种？

你文件里的例子：

```python
b = torch.tensor(
    [
        [1, 1, 1, 0],
        [1, 1, 0, 0],
        [1, 0, 0, 0],
    ]
)

mask = b.unsqueeze(dim=1).repeat(1, 4, 1)
```

这里构造出来的是一个 **padding mask 风格** 的 mask，表示：

- `1` 的位置可以看
- `0` 的位置不能看

它本质上是在屏蔽无效位置，而不是严格意义上的“下三角 future mask”。

所以如果面试时你要说准确一些，可以这样表述：

> 这个例子展示的是 attention mask 的基本用法，当前这个具体 mask 更接近 padding mask，而不是 decoder 里的 causal mask。

---

## 7.3 为什么要 `masked_fill(..., -1e20)`？

```python
attention_weights = attention_weights.masked_fill(
    attention_mask == 0,
    float("-1e20")
)
```

这是在 softmax 之前进行的。

因为 softmax 有这样一个性质：

- 很小很小的数，经过 softmax 后，概率会接近 0

所以被 mask 的位置先填成一个极小值，例如 `-1e20`，再做 softmax，这些位置的注意力权重就几乎变成 0 了。

这就实现了“看不见这些位置”。

---

# 八、为什么 dropout 加在 attention 权重上？

```python
attention_weights = self.dropout(attention_weights)
```

这是一种常见写法。

含义是：

- 在训练时，随机丢弃一部分注意力连接
- 防止模型过分依赖某几个固定位置
- 提高泛化能力

这和普通全连接层后面加 dropout 的思想一致，只不过这里丢弃的是“注意力概率图”中的一部分连接。

---

# 九、第三版里几个容易说混的点

## 9.1 `attention_value` 和 `attention_weights` 的区别

```python
attention_value = q @ k.transpose(-2, -1)
attention_weights = attention_value / math.sqrt(self.hidden_dim)
```

严格来说：

- `attention_value`：原始打分（未缩放）
- `attention_weights`：缩放后、还没 softmax 的分数

再往后经过 softmax 后，才更接近真正意义上的“权重概率分布”。

所以这几个名字有时在不同代码里会写得不一样，但逻辑是一致的。

---

## 9.2 `hidden_dim` 在这里扮演的角色

在这份单头代码中：

- `Q、K、V` 的维度都等于 `hidden_dim`
- 所以缩放时直接用 `sqrt(hidden_dim)`

但如果以后扩展到多头注意力：

- 总 hidden size 可能是 `hidden_dim`
- 每个头的维度会变成 `head_dim = hidden_dim / num_heads`

那时就应该除以：

\[
\sqrt{head\_dim}
\]

不是再除以总的 `hidden_dim`。

---

# 十、三个版本怎么对比？

| 版本 | 特点 | 适合场景 |
|---|---|---|
| `SelfAttention` | 最基础，三套 `q/k/v` 线性层 | 入门理解原理 |
| `SelfAttention2` | 合并成一个 `qkv` 投影，更高效 | 理解工程优化 |
| `SelfAttention3` | 加了 `mask`、`dropout`、`output_proj` | 面试回答 / 工程实现 |

你可以把它们理解成同一个知识点的三个阶段：

- **会写**
- **会优化**
- **会讲工程细节**

---

# 十一、这份文件最值得你掌握的 8 个点

## 1. Self-Attention 的公式
\[
softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

## 2. `Q / K / V` 的来源
都是输入 `x` 经过不同线性层得到的。

## 3. 为什么 `k` 要转置
为了让 `[B, L, D] @ [B, D, L]` 得到 `[B, L, L]` 的相关性矩阵。

## 4. 为什么除以 `sqrt(d_k)`
防止数值过大，避免 softmax 过尖锐。

## 5. 为什么 `softmax(dim=-1)`
固定 query，对所有 key 归一化。

## 6. 为什么输出 shape 还是 `[B, L, D]`
因为本质上是“每个位置重新聚合一次信息”，不是改变序列长度，也不是改变隐藏维度。

## 7. mask 加在哪里
加在 softmax 之前的打分矩阵上。

## 8. `output_proj` 的作用
增强表达能力，并和多头注意力接口保持一致。

---

# 十二、结合这个 notebook，面试时你可以怎么讲？

你可以这样回答：

> 单头自注意力的核心是先把输入通过线性层映射成 Q、K、V，然后用 Q 和 K 做点积计算相关性分数，除以根号下的维度做缩放，再经过 softmax 得到注意力权重，最后对 V 做加权求和，得到每个位置新的表示。  
> 如果从实现角度看，最基础的写法是分别定义三个线性层生成 Q、K、V；更高效的写法会把它们合并成一个 `Linear(D, 3D)` 再切分；在工程里通常还会加入 `attention_mask`、`dropout` 和 `output_proj`。  
> 其中 `attention_mask` 可以用来屏蔽 pad 位置或者未来信息，softmax 一般沿最后一维做，这表示固定 query 对所有 key 的分配概率。

---

# 十三、你这个文件里容易踩坑的地方

## 1. `softmax(dim=-1)` 的语义容易说反
正确理解是：

**固定 query，对所有 key 做归一化。**

---

## 2. notebook 里的 mask 示例更像 padding mask
不是标准 decoder 的 future mask。

---

## 3. `output_proj` 在单头里不是“必须的数学步骤”
但在工程实现里很常见。

---

## 4. `hidden_dim` 和 `head_dim` 不要混
单头时二者相同；多头时不同。

---

# 十四、你下一步复习建议

学完这个文件后，建议立刻接着整理：

1. `MultiHead_self_attention.ipynb`
2. `Transformer_Decoder.ipynb`

因为它们和当前这一份的关系最紧密：

- **Single Head**：先学会一个头怎么算
- **Multi Head**：再学会多个头为什么更强
- **Decoder**：最后看注意力模块如何嵌入完整 Transformer Block

这样知识链就完整了。

---

# 十五、最后给你一个一句话总结

这个 notebook 的真正重点不是“把注意力写出来”，而是让你逐步理解：

**单头自注意力的核心公式、shape 变化、工程优化方式，以及 mask / dropout / output projection 这些真实模型中的关键细节。**

如果你后面复习时能做到下面这三点，就说明这份内容已经掌握了：

1. 不看代码，能把公式和流程讲出来  
2. 看着 `[B, L, D]`，能说出每一步 shape 怎么变  
3. 能解释 `mask`、`dropout`、`output_proj` 为什么存在

