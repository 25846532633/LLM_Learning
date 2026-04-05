# 注意力机制（Self-Attention）问题整理

本文整理自学习与实现注意力时讨论过的要点，对应例如 `build_gpt.ipynb` 里的缩放点积注意力。**全文主体与附录问答均为简体中文（zh-cn）。**

---

## 目录

| 区块 | 内容 |
|------|------|
| **§1–§10** | 点积注意力、`masked_fill`、缩放、softmax 维度的、QKV、bias、单头流程速查 |
| **§11–§17** | 多头形状、`contiguous`、`unpack` 错误、shape 解读、mask 构造、形状总览、`k.T` 误区 |
| **§18** | Decoder 实现（decoder-only、与论文差异、mask、Embedding、Dropout、Post/Pre-LN、`Transformer_Decoder.ipynb` 对应概念） |
| **附录一** | 单头 Q&A（Q1–Q4） |
| **附录二** | 多头 Q&A（Q5–Q12） |
| **附录三** | `Transformer_Decoder.ipynb` 代码相关 Q&A（Q1–Q18） |

---

## 1. `q @ k.transpose(-2, -1)` 在做什么？

- **不是**“先做点积再单独做一次维度变换”，而是一次 **批量矩阵乘法**。
- 形状（单头、忽略 batch 细节时）：  
  `q`: `[B, T, d]`，`k`: `[B, T, d]`，`k.transpose(-2,-1)`: `[B, d, T]`。  
  则 `[B, T, d] @ [B, d, T] → [B, T, T]`。
- **语义**：对每个 batch，位置 `i` 的 query 向量与位置 `j` 的 key 向量做 **点积**，结果放在 `weight[b, i, j]`，得到 **所有 (query, key) 对的未归一化分数**。
- **公式**：`(Q K^T)_{i,j} = q_i · k_j`（也可看成某行与某列的矩阵乘法点积）。

---

## 2. 为什么用 `masked_fill`？为什么要除以 `√head_size`？

### `masked_fill`

- 在 softmax **之前**，把“不允许 attend 的位置”对应的 logit 设成 `-∞`，softmax 后该位置权重为 **0**。
- **因果遮罩**（如 GPT）：第 `i` 个位置不能看 `j > i` 的未来 token。
- **与 scale 的顺序**：若遮罩处为 `-inf`，先 mask 再除 `√d` 或先除再 mask，对合法位置数学等价；`-inf` 除常数仍为 `-inf`。

### 除以 `√d`（scaled dot-product）

- `d = head_size`。点积随维度增大，数值尺度变大，**softmax 易饱和**（分布过尖、梯度不稳）。
- 除以 `√d` 可压缩 logits 尺度，使训练更稳定。缩放应在 **softmax 之前** 对 logits 做。

---

## 3. Softmax、`dropout`、`weight @ v` 的顺序与含义

1. `weight`（logits）→ **softmax（通常 `dim=-1`）** → 每个 query 对所有 key 的 **概率分布**（非负、列和为 1）。
2. **attention dropout**：在训练时随机将部分权重置零并对剩余部分缩放，**正则化**、减轻对固定连接的过拟合；`eval()` 时关闭。
3. `out = weight @ v`：用权重对 **value** 加权求和，得到单头输出，形状通常 `[B, T, d]`。

---

## 4. Softmax 作用在 `[B, T, T]` 的哪一维？

- 常见写法：`F.softmax(weight, dim=-1)`，即 **最后一维**（**key 维**，`dim=2`）。
- **语义**：固定 batch、固定 query 位置 `i`，对所有 key 下标 `j` 归一化——得到“第 `i` 个位置对每个 `j` 的注意力占比”。

---

## 5. 为什么 Q、K 要用这种乘法（点积）造分数？

- 注意力需要先得到 **`weight[i, j]`**：位置 `i` 对位置 `j` 有多“相关”。
- **点积** \(q_i · k_j\) 是常用的可微、易并行的相似度；`Q @ K^T` 一次算出全部 `(i,j)`。
- 与“相加/拼接”不同：点积直接把两个向量压成一个 **标量分数**，刚好对应 `[T, T]` 打分表。

---

## 6. `attention_mask`（如 `register_buffer` + `torch.tril`）是什么？

- **`torch.tril(torch.ones(block_size, block_size))`**：下三角为 1、上三角为 0，配合 `masked_fill(... == 0, -inf)` 实现 **因果注意力**（不能看到未来）。
- **`register_buffer`**：不是可训练参数，但会随模型 **迁移设备**、并可进 `state_dict`；不计梯度，省内存。

### 与“padding”的关系

- **Padding**：batch 内序列长度不一，末尾补 `<pad>` 等到同一长度 `T`，才能组成 `[B, T, ...]`。
- **Padding mask**：避免 query 对 **填充位置** 的 key 分到非零权重；需在 softmax 前把非法 key 对应的 logit 遮掉（常与因果 mask **合并**使用）。
- 因果 mask 挡 **未来**；padding mask 挡 **假 token**——两种用途不同，实务上常同时需要。

---

## 7. `masked_fill` 怎么作用？

- `tensor.masked_fill(mask, value)`：在 **mask 为 True** 的位置，把 `tensor` 该元素替换成 `value`，其余不变；`mask` 须能与 `tensor` **broadcast**。
- 注意力里常用 `value=float('-inf')`，再 softmax，使被遮位置权重为 0。

---

## 8. 将 `[B, T]` mask 扩成 `[B, T, T]`：`repeat(1, T, 1)` vs `repeat(1, 1, T)`

- `mark` 形状 `[B, T]`（例如每个 **key** 位置是否有效）。
- **`unsqueeze(1)` → `[B, 1, T]`**，再 **`repeat(1, T, 1)` → `[B, T, T]`**：  
  **`mask[b, i, j] = orig[b, j]`**——每一 **列 j** 在所有 query 行 **i** 上相同，适合 **按 key / padding** 遮罩。
- **`unsqueeze(2)` → `[B, T, 1]`**，再 **`repeat(1, 1, T)` → `[B, T, T]`**：  
  **`mask[b, i, j] = orig[b, i]`**——由 **query 下标 i** 决定，语义不同；若要做“遮 invalid key”，应采用前一种扩展方式，避免与 `softmax(..., dim=-1)` 对 key 的归一化方向不一致。

---

## 9. Q、K、V 线性层里的 `bias`

- `nn.Linear`：`y = xW^T + b`。**bias** 为每个输出维度提供可学的 **常数偏移**，增加仿射自由度。
- 许多 **GPT / Transformer** 实现对 Q、K、V **一律 `bias=False`**（参数更少，且常与 LayerNorm 等搭配）；若 Q 与 K/V bias 设定不一致，多为教学或个例，实践中常保持 **对称**。

---

## 10. 速查：单头注意力一条龙（与常见实现对齐）

1. `Q, K, V = xW_q, xW_k, xW_v`
2. `scores = Q @ K^T / √d_k`
3. `scores = masked_fill(scores, ~allow, -inf)`（因果 + padding 等）
4. `attn = softmax(scores, dim=-1)`
5. `attn = dropout(attn)`（训练时）
6. `out = attn @ V`

---

## 11. 多头注意力为什么要把 `(b, s, h)` 变成 `(b, head_num, s, head_dim)`？

**问题**：为什么在 MHA 里要先把隐藏维拆开、再 `transpose`，而不是一直用 `(b, s, h)`？

**核心结论**：为了同时满足两件事——

1. **每个头独立算注意力**：每个 head 有自己的 `head_dim`，在子空间里做 scaled dot-product。
2. **多头并行**：把 `head_num` 放到 batch 维旁边，让 `QK^T` 一次对 **所有头** 做批量矩阵乘。

**关系**：`h = head_num × head_dim`。

**常见两步**：

1. `view`：` (b, s, h) → (b, s, head_num, head_dim)` —— 在 **最后一维**上把 `h` 拆成“头数 × 每头维度”。
2. `transpose(1, 2)`：` (b, s, head_num, head_dim) → (b, head_num, s, head_dim)` —— 让 **head** 与 **序列** 的顺序对齐，便于后面 `Q @ K^T` 得到 **`(b, head_num, s, s)`**（每个 batch、每个头一张 `s×s` 的分数矩阵）。

---

## 12. 为什么 `transpose` / `permute` 后常常要接 `contiguous()`？

**问题**：为什么常看到 `x.transpose(1, 2).contiguous().view(...)` 这种组合？

**核心结论**：

- `transpose`、`permute` 多半只改 **步长（stride）与索引方式**，**不一定**把资料在内存里重排；张量可能变成 **非连续（non-contiguous）**。
- **`view` 要求底层储存连续**（或满足特定条件），否则可能报错或行为不如预期。
- **`contiguous()`** 会在必要时 **拷贝成连续内存**，之后再 `view` / `reshape` 才安全。

**常见组合**：先换维 → `contiguous()` → `view(...)`，把 `(b, head_num, s, head_dim)` 合回 `(b, s, h)` 等。

---

## 13. 为什么会 `ValueError: too many values to unpack (expected 2)`？

**问题**：写了 `batch_size, seq_len = x.size()`，`MultiHeadAttention` 却报“要拆的值太多”？

**出错原因**：输入 `x` 是 **三维** `(batch, seq_len, hidden_dim)`，例如 `torch.rand(3, 2, 128)` 时 `x.size()` 是 **`(3, 2, 128)` 三个数**，左边只有两个变量去接，Python 就会报错。

**正确写法示例**：

```python
batch_size, seq_len, hidden_dim = x.size()
# 或暫時不用最後一維時：
batch_size, seq_len, _ = x.size()
```

**本质**：MHA 的输入典型是 **`(batch, seq_len, hidden_dim)`**，不是二维 `(batch, seq_len)`。

---

## 14. 打印结果怎么对应到 shape？`[3, 8, 2, 2]`、`[1, 3, 4, 2]` 怎么读？

**问题**：每次看到 `torch.Size([3, 8, 2, 2])` 或萤幕上一大坨嵌套括号，觉得很抽象，不知道对应到哪一层。

**读法（由外往内）**：

| `torch.Size` 示例 | 维度语义（常见约定） |
|-------------------|----------------------|
| `[3, 8, 2, 2]` | **batch=3** → 每个样本 **8 个 head** → 每头一张 **2×2** 矩阵（**query 长度 × key 长度**，常为 `s×s` 的注意力分数或 mask） |
| `[1, 3, 4, 2]` | **batch=1** → **3 个 head** → **4 个 token** → **每头特征维 2**（多头里的 Q/K/V 一类中间张量，即 `(b, head_num, s, head_dim)`） |

**从打印括号层数对齐**：最外一层有几个“大块”，通常就对应 **第 0 维**；往内一层一层对 **第 1、2、3 维**。因此 **四层方括号** 一般表示 **4D**，和 `torch.Size` 长度为 4 一致。

**和 `[B, T, C]` 的区别**：最终输出若是 **`[B, T, C]`**，只有 **三维**（无“头”这一维）；若你看到的是 **`[1, 3, 4, 2]`**，语义是 **多头中间形状**（`B=1` 时打印会多一层最外括号），不要和 **`[B, T, C]`** 混成同一种约定。

---

## 15. 这段 `attention_mask` 构造在做什么？

**问题**：例如：

```python
torch.tensor([
    [0, 1],
    [0, 0],
    [1, 0],
]).unsqueeze(1).unsqueeze(2).expand(3, 8, 2, 2)
```

**形状变化**：

- 起始 **`(3, 2)`**：3 个样本，序列长度 2；每个元素是 **每个 key 位置**（或与 key 对齐）的 mask 值。
- `unsqueeze(1)` → **`(3, 1, 2)`**：为后面插入“头”或“query”维度预留位置。
- `unsqueeze(2)` → **`(3, 1, 1, 2)`**。
- `expand(3, 8, 2, 2)` → **`(3, 8, 2, 2)`**：在 **不复制大块内存**的前提下，把同一套 mask **广播**成 **`(batch, head_num, query_len, key_len)`** 可与注意力 logits 对齐的形状。

**语义**：把 **每个样本的一条 key 方向 mask**，复制到 **8 个头**、并让 **每个 query 列** 都看到 **同一行 key mask**（与 `softmax(..., dim=-1)` 沿 key 维归一化一致）。实际项目里 `8`、`2` 会换成你的 `head_num` 与 `seq_len`。

---

## 16. 这几次问题的总主线：多头注意力的形状流转

把常见 MHA 流程串成一条线（与实现对齐时可再细调变量名）：

`(b, s, h)`  
→ `view` → `(b, s, head_num, head_dim)`  
→ `transpose(1, 2)` → `(b, head_num, s, head_dim)`  
→ `Q @ K^T`（对最后两维）→ `(b, head_num, s, s)`  
→ `softmax(dim=-1)`、`mask`、`dropout` 等  
→ `@ V` → `(b, head_num, s, head_dim)`  
→ `transpose(1, 2).contiguous().view(b, s, h)` → 再回到 **`(b, s, h)`**，最后常接 `output_proj`。

**速记**：

| 疑问 | 简答 |
|------|------|
| 为什么要 `transpose`？ | 为了 **按 head 并行** 做 `QK^T` 与后续注意力。 |
| 为什么要 `contiguous()`？ | 为了后面能 **安全使用 `view`**（连续内存）。 |
| 为什么 `batch_size, seq_len = x.size()` 报错？ | 因为输入是 **三维**，`size()` 返回 **三个数**。 |
| `[3, 8, 2, 2]` 怎么理解？ | **batch、head、query 长度、key 长度** 四层嵌套（常见为 mask 或分数张量）。 |

---

## 17. 补充：三维张量做注意力时不要用 `k.T`

**问题**：写 `q @ k.T` 出现 `RuntimeError`（例如第 0 维 3 与 2 对不上）。

**原因**：对维度大于 2 的张量，PyTorch 的 **`.T` 会翻转“全部”维度顺序**，不是只交换最后两维。`q` 为 `[B, L, D]` 时，应使用 **`k.transpose(-2, -1)`** 得到 `[B, D, L]`，再与 `q` 做批量矩阵乘，得到 **`[B, L, L]`**。

（与本文第 1 节 `q @ k.transpose(-2, -1)` 一致。）

---

## 18. Decoder 实现相关整理（SimpleDecoder、遮罩、Embedding、规范）

以下汇总 **Decoder 专用** 的常见问题，以及与 **原论文 Encoder–Decoder Transformer** 的对照；内容涵盖教学用 **多层堆叠 SimpleDecoder**、**padding／因果遮罩**、**Embedding**、**Post-LN／Pre-LN**、**FFN 内 Dropout** 等。

### 18.1 整体资料流与架构定位（decoder-only）

典型 **简化 Decoder 堆叠**（例如 5 层 `SimpleDecoder`）流程：

1. **输入**：token ID（整数索引）。
2. **`nn.Embedding`**：ID → 向量。
3. **多层模块**：每层含 **Masked Self-Attention** → **Add & Norm** → **FFN** → **Add & Norm**。
4. **输出**：线性层映到词表维度，再 **softmax**（或训练时只取 logits）。

这类结构本质上是 **“纯 Decoder／decoder-only”**，语义上较接近 **GPT 系** 的语言模型，而不是原论文里 **带 Encoder、且 Decoder 每层有三个子块** 的完整翻译架构。

### 18.2 为什么和论文里的 Decoder 图不完全一样？

| 项目 | 原论文 Decoder 每层 | 常见简化 decoder-only 每层 |
|------|---------------------|---------------------------|
| 子模块 | ① Masked Multi-Head Attention<br>② **Encoder–Decoder Attention（Cross-Attention）**<br>③ FFN | ① Masked Self-Attention<br>② FFN |
| 依赖 | Decoder 需读 Encoder 输出，故要有 **cross-attention** | 无独立 Encoder 时 **不需要** cross-attention |

**结论**：若实现里只有 **masked self-attention + FFN**，缺少的是 **Encoder–Decoder（交叉）注意力**；这是 **seq2seq 场景** 与 **自回归语言模型** 的结构差异，不是“做错图”而是 **任务与架构选型不同**。

### 18.3 单层 Decoder block 内部（教学版对齐）

**Attention 子层**（概念顺序）：

1. 由输入算出 **Q、K、V**（常经线性投影）。
2. **Masked Self-Attention**（因果遮罩：第 \(t\) 步不能看 \(>t\) 的位置）。
3. **残差 + LayerNorm**（见 18.8：Post-LN 时为“先加再 LN”）。

**FFN 子层**（常见配置）：

1. `hidden_dim` → **`4 × hidden_dim`**（上投影）。
2. 启用函数（例如 **GELU**）。
3. 再投影回 **`hidden_dim`**（下投影）。
4. **Dropout**（见 18.7：须传 **概率 float**，不可把另一个 `nn.Dropout` 模块当参数）。
5. **残差 + LayerNorm**。

整体可视为 **标准教学用“单层 Decoder block”** 的原型；与工业级 LLM 的细节（RMSNorm、SwiGLU、并行化等）仍可再扩充。

### 18.4 `attention_mask`（padding）与 `tril()`（因果）是两件事

- **Padding mask（有效位 mask）**：标出 **哪些位置是真 token、哪些是填充**；避免对 **pad 位置** 的 key 分到注意力权重。处理的是 **可变长度 batch** 对齐后的“假位置”。
- **因果 mask（causal / look-ahead）**：常用 **下三角**（例如 `torch.tril` 造出的形状），保证第 \(t\) 个位置 **只能 attend 到 \(\le t\)**，不能 **偷看未来**。

Decoder 的自注意力 **常见需求是同时具备两者**：一边挡 pad，一边挡未来。实践中应 **在语义上分开思考**，再在 logits 上 **合并成最终 mask**（见 18.9），而不要让“padding 张量”与“直接 `.tril()`”混在同一变量语义里难以阅读。

### 18.5 把 `(batch, seq)` padding mask 扩成 `(batch, heads, seq, seq)`

多头注意力里，分数／mask 常为 **4D**：**`(batch, num_heads, query_len, key_len)`**。若起手是 **每个 key 位置是否有效** 的 **`(B, T)`**（例如 1=有效、0=需遮），可透过 **插入维度 + `repeat`／`expand`** 对齐到与 attention logits 相同形状。

**范例**（与教学推导一致；数字仅示意）：

```python
# 起始 (3, 4)：3 個樣本，序列長 4；每列為 key 方向有效位
mask = (
    torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0], [1, 1, 1, 0]])
    .unsqueeze(1)   # (3, 1, 4)
    .unsqueeze(2)   # (3, 1, 1, 4)
    .repeat(1, 8, 4, 1)  # (3, 8, 4, 4)：batch、8 頭、query 列、key 列
)
```

**语义**：把 **`(batch, seq)`** 的 padding 规则，**广播**成 **`(batch, heads, seq, seq)`**，以便与 **`softmax(dim=-1)`** 沿 **key 维** 的打分表逐元素对齐（与本文第 8、15 节一脉相承）。

### 18.6 `nn.Embedding`：形状规则与输入类型

**形状**：若 `emb = nn.Embedding(num_embeddings=12, embedding_dim=64)`，输入 token ID 张量为 **`(3, 4)`**（3 条序列、每条 4 个 ID），则输出为 **`(3, 4, 64)`**。可记：

**Embedding 输出形状 ≈ 输入形状 再接上 `[embedding_dim]`**。

**类型**：`nn.Embedding` 的输入必须是 **整数索引**（如 **`LongTensor`**），不能是已经 embed 好的 **浮点向量**。错误示例：先用 `x = torch.rand(3, 4, 64)` 再 `self.emb(x)` 会不符合 API 语义。

**正确示例**（词表大小 12，合法下标 `0 … 11`）：

```python
x = torch.randint(low=0, high=12, size=(3, 4))
```

### 18.7 FFN 里的 Dropout：`nn.Dropout(self.dropout)` 不成立

`nn.Dropout(p)` 的 **`p`** 必须是 **丢弃概率（float）**，不能把另一个 **`nn.Dropout` 模块实例** 当成 `p` 传入——那不是支持的“嵌套”，行为也不对。

**正确做法**：`self.drop_ffn = nn.Dropout(dropout)`（与前面同一个 **float 变量**）；或 **直接重用** 同一个模块：`self.drop_ffn = self.dropout`。

### 18.8 Post-LN vs Pre-LN；Attention 与 FFN 不要共用同一个 `LayerNorm`

**Post-LN（原论文常见）**：子层输出先与残差相加，再 **LayerNorm**，例如：

```python
return self.LayerNorm(x + sublayer_output)
```

**Pre-LN（GPT-2 等常见）**：先对输入做 LN，再进子层，残差在另一种顺序下接回；深层训练时往往更稳，但与 **2017 Transformer 图** 的画法不同。

**规范建议**：**Attention 子层**与 **FFN 子层**应各用 **独立的 LayerNorm**（例如 `ln_1`、`ln_2`），因两处的 **可学仿射参数** 语义不同；共用同一个 `nn.LayerNorm` 实例虽可能能跑，但 **不够规范**。

### 18.9 遮罩合并：避免对“外部传入的 padding mask”直接 `.tril()`

若外部传入的是 **padding 语义** 的 `attention_mask`，再写 **`attention_mask = attention_mask.tril()`** 容易让 **“pad 遮罩”与“因果下三角”** 混在同一变量名里，可读性与维护性差。

**较清晰做法**：

1. 明确构造 **`padding_mask`**（对齐 key／需 softmax 前挡掉的无效位）。
2. 明确构造 **`causal_mask`**（下三角，挡未来）。
3. 在 **加到 logits 前** 合并为 **`final_mask`**（例如两者都要挡的位置一并设为 \(-\infty\)），与本文第 6 节“两种 mask 常合并使用”呼应。

### 18.10 Decoder 要点一览（对照速查）

| 层面 | 要点 |
|------|------|
| **输入** | Embedding 前为 **token ID**；之后为 **`(batch, seq, hidden_dim)`** |
| **Mask** | **Padding** 管有效长度；**因果（tril）** 管自回归方向；Decoder self-attn **常两者都要** |
| **结构** | 简化版为 **decoder-only**；原论文 Decoder 另含 **cross-attention** |
| **LayerNorm** | 教学实现常为 **Post-LN**；GPT-2 系多为 **Pre-LN**；**子层各用独立 LN** 较规范 |
| **FFN Dropout** | 只传 **float 概率** 或 **重用同一 Dropout 模块**，勿 `nn.Dropout(另一个Dropout)` |

---

## 附录：常见问题精练（简体中文 zh-cn）

以下附录中 **问／答／解析** 均为 **简体中文**；与正文 §1–§18 用语一致，便于检索与复习。

---

### 一、单头（Scaled Dot-Product）注意力

#### Q1：Softmax 应该作用在哪一维？为什么不是 `dim=0`（batch）？

**答：** 在分数矩阵形状为 `[B, T_q, T_k]`（自注意力时常为 `[B, T, T]`）时，应对 **key 所在的那一维** 做 softmax，PyTorch 里通常写 **`dim=-1`**（即最后一维）。

**解析：** 语义是：对每个 query 位置 \(i\)，要在所有 key 位置 \(j\) 上得到一组和为 1 的权重。key 维是“被归一化的那一组候选”。若对 batch 维做 softmax，会把不同样本混在一起，破坏“每条序列内部”的注意力含义。

---

#### Q2：`q @ k.T` 在三维输入 `[B, L, D]` 下为什么经常报错？正确写法是什么？

**答：** 不要用 **`k.T`**；应使用 **`k.transpose(-2, -1)`**（或等价地只交换序列维与特征维），再写 **`q @ k.transpose(-2, -1)`**。

**解析：** 对维度大于 2 的张量，`.T` 会 **整体反转所有维度顺序**（例如 `[B,L,D]` 变成 `[D,L,B]`），与“只在最后两维上做 \(QK^\top\)”不符，容易导致 matmul 广播维度对不齐。`transpose(-2,-1)` 只在 **每个 batch 切片内** 把 `[L,D]` 变成 `[D,L]`，得到 `[B, L, L]` 的分数矩阵。

---

#### Q3：缩放因子该除以 \(\sqrt{d_k}\)，这里的 \(d_k\) 是 `hidden_dim` 还是别的？

**答：** 理论上应除以 **每个注意力里用于点积的向量维度**，即 **key/query 的特征维** \(d_k\)。单头且 Q、K 维度等于 `hidden_dim` 时，就是 **`sqrt(hidden_dim)`**。

**解析：** 点积方差随 \(d_k\) 增大而增大，logits 过大使 softmax 饱和、梯度不稳定。若实现里把维度拆错（例如多头里误用总 `hidden_dim` 而非 `head_dim`），缩放会不匹配，训练可能变慢或不稳。许多实现里 **单头用 `hidden_dim`，多头每头用 `head_dim`**（见下一节）。

---

#### Q4：因果掩码（不能看未来）应该在 softmax 之前还是之后做？

**答：** 在 **softmax 之前**，把不允许 attend 的位置的 logit 设为 **\(-\infty\)**（或足够小的负数），再 softmax。

**解析：** softmax 之后权重已归一化；若之后再 mask，很难保持“每行和为 1”且语义清晰。先 mask 再 softmax，被遮位置的权重自然为 0。

---

### 二、多头注意力（MHA）

#### Q5：为什么要先把 `(b, s, h)` reshape 成 `(b, s, num_heads, head_dim)` 再 `transpose(1,2)`？

**答：** 为了把 **head 维** 放到与 batch 相邻的位置，得到 **`(b, num_heads, s, head_dim)`**，从而对 **每个 head** 并行计算 \(QK^\top\)（得到 **`(b, num_heads, s, s)`**）。

**解析：** 同一套序列在每个头里用不同的线性子空间（更细的投影）去算相似度，最后再拼回 `h`。若不 transpose，最后一维仍是混合后的 `h`，无法按 head 批量做 `[s, d] @ [d, s]`。

---

#### Q6：`transpose(1, 2)` 之后为什么要 `.contiguous()` 才能 `view`？

**答：** `transpose`/`permute` 往往只改步长、数据在内存中可能 **不连续**；**`view` 要求底层存储连续**（或满足特定布局）。因此常见写法是 **`x.transpose(1, 2).contiguous().view(b, s, h)`**。

**解析：** 若跳过 `contiguous()`，可能报错或产生非预期视图。`reshape` 有时会隐式处理，但显式 `contiguous()` 更清楚地表达“为合并 head 维先拷贝成连续块”。

---

#### Q7：`batch_size, seq_len = x.size()` 为什么报 `too many values to unpack`？

**答：** 因为 MHA 输入 **`x`** 一般是 **`[batch, seq_len, hidden_dim]`**，`x.size()` 返回 **三个数**，不能只用两个变量接收。

**解析：** 应写成 `batch_size, seq_len, hidden_dim = x.size()`，或 `batch_size, seq_len, _ = x.size()`。本质是：**别把三维输入当成二维。**

---

#### Q8：`hidden_dim` 必须能被 `num_heads` 整除吗？

**答：** 在标准“把最后一维均分为 num_heads 份”的实现里，**必须整除**，且 **`head_dim = hidden_dim // num_heads`**。

**解析：** 否则 `view(..., num_heads, head_dim)` 无法整分，`head_dim` 不是整数。选型时常见如 `d_model=512`、`heads=8`、`head_dim=64`。

---

#### Q9：多头里缩放用 \(\sqrt{\text{head\_dim}}\) 还是 \(\sqrt{\text{hidden\_dim}}\)？

**答：** 与 **每个 head 内点积的维度** 一致，通常用 **`sqrt(head_dim)`**（与论文中 \(d_k\)  per head 一致）。若错误地用总 `hidden_dim`，缩放过强，可能影响训练动态。

**解析：** 每个头的 Q、K 向量长度是 `head_dim`，点积方差随 `head_dim` 增长；缩放应对齐这个维度。部分代码为省事仍写 `sqrt(hidden_dim)`，与严格论文形式不完全一致，需注意与论文/预训练权重对齐。

---

#### Q10：打印出来是 `torch.Size([3, 8, 2, 2])`，四个数分别常代表什么？

**答：** 在“batch + 多头 + 序列”的常见布局下，常表示 **`batch=3`，`num_heads=8`，后两个 `2, 2` 多为序列维**（例如 **`T_q, T_k`** 的注意力分数或 mask，自注意力时即 **`T, T`**）。

**解析：** 从外到内：第 0 维样本、第 1 维 head、第 2/3 维往往是“query 位置 × key 位置”。若形状是 **`[3, 8, 2, 16]`**，则常为 **`(batch, heads, seq_len, head_dim)`**（Q/K/V 拆头后）。要以你代码里 **最近一次 `transpose/view` 的结果** 为准。

---

#### Q11：padding mask 扩成四维 `(B, H, T, T)` 时，`expand` 起什么作用？

**答：** 把 **较小的 mask**（例如与 key 对齐的 `(B, T)` 或 `(B, 1, 1, T)`）在 **head 维** 和 **query 行** 上 **广播**成与 `attention_weight` 相同形状，使每个头、每个 query 位置使用 **同一套 key 是否有效** 的规则。

**解析：** `expand` 不立刻复制整块数据（视图广播），节省内存；与 `masked_fill` 结合时，mask 为 0 的位置 logits 变 \(-\infty\)，softmax 后权重为 0。

---

#### Q12：多头输出合并时，把 `(b, num_heads, s, head_dim)` 变回 `(b, s, h)` 丢失了“头”的信息吗？

**答：** 没有丢失：**每个头的输出仍作为向量的一段拼接在最后一维**，`h = num_heads * head_dim`；之后往往还有 **`output_proj`（线性层）** 再混合各维。

**解析：** “合并”是 **按维度拼接**（concat），不是求和丢维。最后的线性层负责再整合多子空间的信息；这与“单头一个大矩阵”在表达力上有联系也有实现差异，但标准 Transformer 里 **concat + Linear** 是完整流程的一环。

---

### 三、`Transformer_Decoder.ipynb` 代码相关问答

以下针对笔记本中的 `SimpleDecoder`、`Decoder`、mask 与训练习惯等，采用 **问 / 答** 形式，编号独立（与附录一、二无关）。

#### Q1：这个 `Decoder` 整体在做什么？和论文里的「完整 Transformer Decoder」一样吗？

**答：** 不一样。这是 **decoder-only**：`Embedding → 5 层 SimpleDecoder（每层：Masked 自注意力 + FFN）→ Linear → softmax`。论文中带 Encoder 的 Decoder 每层还有 **Encoder–Decoder（交叉）注意力**，这里没有。

#### Q2：`Decoder.forward` 里张量形状怎么变？

**答：** `X` 为 `(batch, seq)` 的 token id；`self.emb(X)` 得到 `(batch, seq, 64)`；每层 `SimpleDecoder` 保持 `(batch, seq, 64)`；最后 `self.out(x)` 得到 `(batch, seq, 12)`，再对最后一维做 softmax。

#### Q3：为什么要把 `hidden_dim` 拆成 `num_heads` 和 `head_dim`？

**答：** 多头要在 **每个 head** 上各自算 \(QK^\top\)，故把 `(B, T, H)` 变为 `(B, num_heads, T, head_dim)`，且 **`H = num_heads * head_dim`**。代码里 `head_dim = hidden_dim // num_heads`，若不能整除会出错。

#### Q4：`attention_output` 里 `query @ key.transpose(-2,-1)` 得到什么？

**答：** 对每个 batch、每个 head，得到 **(seq, seq)** 的未缩放相似度；再除以 \(\sqrt{\text{head\_dim}}\) 做 scaled dot-product。

#### Q5：代码注释里 `attention_weights @ value` 后形状写成类似 `[3,8,4,8]` 是否合理？

**答：** 按维度应为 **`(..., head_dim)`**（本配置下 `head_dim=8`），不应与 `num_heads` 混淆；注释易误导，宜与 **`head_dim`** 一致书写。

#### Q6：为什么 `transpose(1,2).contiguous()` 后要 `view(batch, seq_len, -1)`？

**答：** 把 `(B, T, num_heads, head_dim)` 合并回 **`(B, T, hidden_dim)`**，再交给 `output_proj`。

#### Q7：`attention_mask` 为 `None` 时 `else` 分支在做什么？

**答：** 用 `torch.ones_like(attention_weights).tril()` 构造 **纯因果下三角**，禁止看到未来 token，相当于只有 causal、没有 padding。

#### Q8：有 mask 时写 `attention_mask = attention_mask.tril()` 意图是什么？有何注意点？

**答：** 意图是在 **padding 规则上再施加因果约束**。但把两种语义写在同一变量上可读性差；更清晰的是分别构造 **padding_mask** 与 **causal_mask** 再合并。

#### Q9：`forward` 注释写 mask 为 `(batch, nums_head, seq)`，与示例一致吗？

**答：** **不一致。** 示例里构造的是 **`(3, 8, 4, 4)`**，即 **`(batch, num_heads, seq, seq)`**，与 attention logits 对齐；注释应改成真实形状，避免接 API 时维数搞错。

#### Q10：`masked_fill(..., -1e20)` 和 `float('-inf')` 有何区别？

**答：** 常用 **`float('-inf')`**，softmax 后对应位置为 0。`-1e20` 在 float32 里多半可用，但 **不如 `-inf` 语义干净**，极端情况下数值行为可能略有差别。

#### Q11：`self.drop_ffn = nn.Dropout(self.dropout)` 是否正确？（与「不能把 Dropout 嵌进 Dropout」是否矛盾？）

**答：** 在该笔记本里 **正确**：`__init__` 中 **`self.dropout = dropout`** 保存的是 **float 概率**，不是 `nn.Dropout` 模块。若写成 `self.dropout = nn.Dropout(dropout)` 再 `nn.Dropout(self.dropout)` 就错了。

#### Q12：注释里「先残差后 LayerNorm = 标准，先 LN 后残差 = GPT-2」指什么？

**答：** 当前实现是 **Post-LN**：`LayerNorm(x + sublayer_output)`。GPT-2 等常用 **Pre-LN**，深层训练往往更稳，属于结构选择。

#### Q13：Attention 与 FFN 各用一个 `LayerNorm` 是否有必要？

**答：** **有必要且更规范。** 笔记本使用 `LayerNorm_attn` 与 `LayerNorm_ffn`，两套参数独立，符合常见写法。

#### Q14：`Decoder.forward` 里的 `print` 是否应保留？

**答：** 调试可以；正式训练/推理建议去掉，减少日志与开销，分布式下更不友好。

#### Q15：在模块内 `return torch.softmax(output, dim=-1)`，训练语言模型是否合适？

**答：** 若用 **`nn.CrossEntropyLoss`**，通常返回 **logits**，由损失内部处理；**不必在 forward 里先 softmax**。当前写法便于看概率或演示；真要训练多返回 logits。

#### Q16：词表 12、`Embedding(12,64)`、`Linear(64,12)` 是否要一致？输入 id 范围？

**答：** 要一致；token 须在 **`[0, 11]`**。`torch.randint(0, 12, (3, 4))` 正确。

#### Q17：若用 `x = torch.rand(3, 4, 64)` 当 `Decoder` 输入会怎样？

**答：** `nn.Embedding` 需要 **整型 token id**，浮点会报错或语义错误。

#### Q18：`mask` 的 `repeat(1, 8, 4, 1)` 中各数字含义？

**答：** 在 `(3,1,1,4)` 上 repeat 成 `(3,8,4,4)`：**8 = num_heads**，**4** 扩展到 **query 长度**，末维 **4** 为 **key 长度**（与 `seq_len` 一致），与 **`(B, H, T, T)`** 的注意力分数对齐。

---

*文件说明：`Attention.md` — 注意力与 Decoder 实现要点（含 §18 与附录）；**全文简体中文（zh-cn）**。*
