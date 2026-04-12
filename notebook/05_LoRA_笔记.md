# 05 LoRA 笔记

## 1. 这一份 Notebook 在学什么

这一份 `05_LoRA_test.ipynb`，重点已经不再是 Attention 结构本身，而是：

> **当一个大模型已经训练好了以后，如何只训练很少一部分新增参数，就让模型适应新任务。**

这就是 **LoRA（Low-Rank Adaptation）** 的核心思想。

你这份 notebook 做的事情，可以概括成一句话：

> 把原来的 `nn.Linear` 保留下来不动，只额外加一个低秩增量分支 `ΔW`，训练时只更新这个增量分支。

所以这份代码，本质上是在回答下面这个问题：

- 为什么微调大模型不一定要把原权重全部更新？
- 为什么只加两个小矩阵，也能完成适配？
- merge 和 unmerge 到底是什么意思？
- 为什么合并前后输出应该一致？

---

## 2. 先说结论：LoRA 到底改了什么

你在 notebook 一开始就写了这一条公式：

```text
output = (W+ΔW)*x+b
       = W*x + ΔW*x + b
       = W*x + α/r * (AB)*x + b
```

这已经把 LoRA 的本质说出来了。

普通线性层是：

```text
y = Wx + b
```

LoRA 不直接去改原始权重 `W`，而是改成：

```text
y = (W + ΔW)x + b
```

其中：

```text
ΔW = α/r * (A @ B)
```

也就是说：

- `W`：原模型已经学好的大矩阵
- `ΔW`：新加的、可训练的增量
- `A @ B`：用两个小矩阵相乘，来近似一个大的更新矩阵
- `α / r`：缩放系数

所以 LoRA 的核心思想就是：

> **不直接训练大矩阵 W，而是训练一个低秩的小更新 ΔW。**

---

## 3. 为什么要这么做

因为直接微调大模型，成本很高。

比如一个线性层：

```text
W.shape = [out_features, in_features]
```

如果直接全量训练，就要更新：

```text
out_features × in_features
```

这么多参数。

但 LoRA 只训练：

- 一个 `A` 矩阵
- 一个 `B` 矩阵

如果 rank 远小于输入输出维度，那么新增参数量会小很多。

所以 LoRA 的优势在于：

### 3.1 参数更少
只训练低秩分支，不动原始大权重。

### 3.2 显存更省
需要保存和更新的梯度更少，优化器状态也更少。

### 3.3 更适合大模型微调
尤其适合：

- 预训练模型已经很大
- 没有足够算力做 full fine-tuning
- 想为不同任务保存多套轻量适配器

---

## 4. 这份代码到底实现了什么

你的核心类是：

```python
class LinearLoRALayer(nn.Module):
```

它实现的是：

> **把 LoRA 加在一个线性层 `nn.Linear` 上。**

所以更准确地说，这份 notebook 实现的是：

- **LoRA for Linear Layer**
- 或者说：**给线性层加低秩适配分支**

这也是大模型里最常见的 LoRA 写法之一，因为 Transformer 里大量模块本来就是线性层，比如：

- attention 里的 `q_proj / k_proj / v_proj / o_proj`
- FFN 里的上投影、下投影
- 其他 MLP 投影层

---

## 5. 初始化部分逐行理解

### 5.1 类定义

```python
class LinearLoRALayer(nn.Module):
```

这表示你封装了一个“带 LoRA 能力的线性层”。

它既能像普通 `Linear` 一样前向传播，又能在需要的时候把 LoRA 权重合并进原始权重。

---

### 5.2 构造参数

```python
def __init__(self,
    in_features,
    out_features,
    merge = False,
    rank = 8,
    lora_alpha = 16,
    dropout = 0.1,
):
```

这些参数含义如下：

- `in_features`：输入维度
- `out_features`：输出维度
- `merge`：是否把 LoRA 权重直接合并进原始线性层权重
- `rank`：低秩分解的秩，通常记作 `r`
- `lora_alpha`：LoRA 缩放系数
- `dropout`：dropout 概率

其中最关键的是两个：

- `rank`
- `lora_alpha`

---

### 5.3 保存基础属性

```python
self.in_features = in_features
self.out_features = out_features
self.merge = merge
self.rank = rank
```

这些只是把传入参数存下来，方便后续使用。

---

### 5.4 定义原始线性层

```python
self.Linear = nn.Linear(in_features,out_features)
```

这就是原始的线性层，也就是公式里的：

```text
W*x + b
```

你还写了一句很重要的注释：

```python
# W的shape (out_featrues,in_features)
# Linear内部的乘法是：x@W^T + b
```

这个理解非常关键。

因为在 PyTorch 里：

```python
nn.Linear(in_features, out_features)
```

内部权重的 shape 是：

```text
[out_features, in_features]
```

而前向传播时，实际计算是：

```text
x @ W^T + b
```

所以你后面写：

```python
x @ (self.lora_a @ self.lora_b).T
```

在维度上才能和 `self.Linear(x)` 对齐。

---

## 6. LoRA 分支是怎么定义的

### 6.1 只有当 rank > 0 时才启用 LoRA

```python
if rank > 0:
```

这表示：

- `rank > 0`：启用 LoRA
- `rank = 0`：退化成普通线性层

这是一种很常见的写法，方便做消融实验。

---

### 6.2 定义 LoRA A 矩阵

```python
self.lora_a = nn.Parameter(
    torch.zeros(out_features,rank)
)
nn.init.kaiming_normal_(self.lora_a, a=0.01)
```

这里的 shape 是：

```text
lora_a.shape = [out_features, rank]
```

也就是说，它负责把低秩空间再映射回输出维度。

虽然变量名叫 `a`，但你这里的矩阵摆放方向是：

```text
ΔW = lora_a @ lora_b
```

所以：

- `lora_a`：`[out_features, rank]`
- `lora_b`：`[rank, in_features]`

两者相乘得到：

```text
[out_features, in_features]
```

正好和原始线性层权重 `W` 对齐。

---

### 6.3 定义 LoRA B 矩阵

```python
self.lora_b = nn.Parameter(
    torch.zeros(rank,in_features)
)
```

它的 shape 是：

```text
[rank, in_features]
```

这一步相当于：

> 先把原输入投影到一个更小的 rank 空间里。

然后再用 `lora_a` 把它映射回输出空间。

---

### 6.4 缩放系数

```python
self.scale = lora_alpha/rank
```

也就是公式里的：

```text
α / r
```

这个缩放项的作用是：

> 控制 LoRA 分支对最终输出的影响强弱。

如果没有这个缩放，`A @ B` 的数值幅度可能不方便控制。

---

### 6.5 冻结原始线性层

```python
self.Linear.weight.requires_grad = False
self.Linear.bias.requires_grad = False
```

这两句非常关键。

这表示：

- 原始线性层的 `weight` 不训练
- 原始线性层的 `bias` 也不训练
- 只训练 `lora_a` 和 `lora_b`

这正是 LoRA 的核心思想：

> **冻结原模型参数，只训练增量分支。**

---

## 7. Dropout 是怎么写的

```python
self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
```

这里的意思是：

- 如果 `dropout > 0`，就用 Dropout
- 如果 `dropout = 0`，就直接恒等映射

你还写了这两句注释：

```python
# dropout = 0 - x = x
# dropout ≠ 0 - x = self.dropout(x)
```

这个意思本身没问题。

不过要注意一点：

> 你这份 notebook 里，dropout 是加在最终输出上的。

因为你在 `forward` 最后写的是：

```python
return self.dropout(output)
```

这意味着：

```text
dropout( base_output + lora_output )
```

而很多标准 LoRA 实现里，更常见的是：

> 只对 LoRA 分支的输入或者 LoRA 分支本身做 dropout，而不是对整个输出做 dropout。

所以这里要记住：

- **你的实现是教学版、可运行版**
- **但和很多库里的标准 LoRA 写法并不完全一样**

这一点在做工程时要分清。

---

## 8. merge 和 unmerge 是什么

这是这份 notebook 的重点之一。

### 8.1 merge_weight

```python
def merge_weight(self,):
    if self.merge and self.rank > 0:
        self.Linear.weight.data += self.scale*(self.lora_a @ self.lora_b)
```

这一步表示：

> 直接把 LoRA 学到的增量加到原始线性层权重里。

也就是把：

```text
W
```

变成：

```text
W + ΔW
```

这样做的好处是：

- 推理时不需要额外再算一条 LoRA 分支
- 直接把它当成一个普通线性层来用
- 部署更方便

---

### 8.2 unmerge_weight

```python
def unmerge_weight(self, ):
    if self.rank > 0:
        self.Linear.weight.data -= self.scale * (self.lora_a @ self.lora_b)
```

这一步表示：

> 再把之前加进去的增量减回来。

也就是把：

```text
W + ΔW
```

恢复成：

```text
W
```

所以：

- `merge`：把增量折叠进原始权重
- `unmerge`：再把增量拆出来

---

### 8.3 merge 的直觉理解

你可以把它理解成两种推理方式：

#### 方式一：不合并
前向时显式算两项：

```text
W*x + ΔW*x
```

#### 方式二：先合并
先把权重合成：

```text
W' = W + ΔW
```

然后只算：

```text
W'*x
```

只要实现正确，这两种结果应该一致。

---

## 9. forward 逐行理解

你的前向传播是：

```python
def forward(self,x):
    if self.rank > 0 and not self.merge:
        output = self.Linear(x) + self.scale*(x @ (self.lora_a @ self.lora_b).T)
    elif self.rank > 0 and self.merge:
        output = self.Linear(x)
    else:
        output = self.Linear(x)

    return self.dropout(output)
```

可以拆成三种情况。

---

### 9.1 情况一：启用 LoRA，但不 merge

```python
if self.rank > 0 and not self.merge:
    output = self.Linear(x) + self.scale*(x @ (self.lora_a @ self.lora_b).T)
```

这是最核心的一种情况，也最适合训练阶段。

它的逻辑就是：

```text
base_output + lora_output
```

也就是：

```text
W*x + α/r * (AB)x + b
```

这里维度关系非常重要。

假设：

- `x.shape = [B, T, in_features]`
- `lora_a.shape = [out_features, rank]`
- `lora_b.shape = [rank, in_features]`

那么：

```text
lora_a @ lora_b
```

的 shape 是：

```text
[out_features, in_features]
```

转置以后：

```text
[in_features, out_features]
```

于是：

```text
x @ (ΔW)^T
```

输出就是：

```text
[B, T, out_features]
```

刚好能和 `self.Linear(x)` 相加。

---

### 9.2 情况二：启用 LoRA，并且已经 merge

```python
elif self.rank > 0 and self.merge:
    output = self.Linear(x)
```

因为这个时候你已经把：

```text
ΔW
```

加到 `self.Linear.weight` 里了，所以前向时不需要再单独算 LoRA 分支。

直接：

```python
self.Linear(x)
```

就等价于：

```text
(W + ΔW)x + b
```

---

### 9.3 情况三：rank = 0

```python
else:
    output = self.Linear(x)
```

这种情况就是普通线性层。

---

## 10. 这份实现里的核心 shape 变化

假设你测试代码里：

```python
batch_size = 32
seq_len = 128
in_features = 768
out_features = 512
rank = 8
```

并且：

```python
x.shape = [32, 128, 768]
```

那么：

### 原始线性层输出

```text
self.Linear(x).shape = [32, 128, 512]
```

### LoRA 增量矩阵

```text
(lora_a @ lora_b).shape = [512, 768]
```

### 转置后

```text
(lora_a @ lora_b).T.shape = [768, 512]
```

### LoRA 分支输出

```text
x @ (lora_a @ lora_b).T
```

shape 为：

```text
[32, 128, 512]
```

### 最终输出

```text
[32, 128, 512]
```

这也对应了你 notebook 里的打印结果：

```python
Output shape (no merge): torch.Size([32, 128, 512])
Output shape (merged): torch.Size([32, 128, 512])
```

---

## 11. 为什么初始化时通常让增量一开始等于 0

你现在的初始化方式是：

- `lora_a`：随机初始化
- `lora_b`：全 0 初始化

这样一来：

```text
lora_a @ lora_b = 0
```

于是初始时：

```text
ΔW = 0
```

因此模型刚开始的行为就和原始预训练模型完全一样。

这很重要，因为：

> 我们通常希望 LoRA 一开始不要破坏原模型，而是在训练过程中慢慢学出一个有效增量。

这也是为什么你后面的测试里会看到：

```python
Max difference after merge/unmerge cycle: 0.0
Max difference: 0.0
```

至少在初始化阶段，这很正常。

---

## 12. 测试代码 1 在验证什么

### 12.1 先构造输入

```python
x = torch.randn(batch_size, seq_len, in_features)
```

也就是：

```text
x.shape = [32, 128, 768]
```

---

### 12.2 测试不合并模式

```python
lora_layer = LinearLoRALayer(
    ...
    merge=False
)
output = lora_layer(x)
```

这时走的是：

```text
self.Linear(x) + LoRA_branch(x)
```

---

### 12.3 测试合并模式

```python
lora_layer_merged = LinearLoRALayer(
    ...
    merge=True
)
output_merged = lora_layer_merged(x)
```

这里要注意：

> 这个 `lora_layer_merged` 是重新 new 出来的一个新层，并不是前一个层的拷贝。

所以这一段代码主要只能说明：

- merged 模式也能跑通
- 输出 shape 没问题

但**不能严格说明 merged 和 no-merge 的数值一定等价**，因为它们不是同一组底层权重。

这一点你在后面的 `deepcopy` 测试里才真正做对了。

---

### 12.4 merge/unmerge cycle 测试

```python
lora_layer.merge_weight()
output_after_merge = lora_layer(x)
lora_layer.unmerge_weight()
output_after_unmerge = lora_layer(x)
```

这段代码的本意是：

- 先 merge
- 再 unmerge
- 看恢复后是否和原来一样

不过你要注意一个细节：

```python
merge_weight()
```

函数内部写的是：

```python
if self.merge and self.rank > 0:
```

而你这里的 `lora_layer` 一开始是：

```python
merge=False
```

所以这里的 `merge_weight()` 实际上并不会执行合并。

但由于当前初始化下 `ΔW = 0`，所以最终打印出来仍然是：

```python
0.0
```

这份 notebook 可以正常理解 LoRA 思路，但如果以后你要把它改成更严格的工程测试，这一段值得再检查一下。

---

## 13. 测试代码 2 在验证什么

第二段测试更严谨一些。

### 13.1 先建一个基础层

```python
lora_layer = LinearLoRALayer(..., merge=False)
lora_layer.eval()
```

这里切到 `eval()` 的原因，你的注释写得很好：

- Dropout 在 train 和 eval 下行为不同
- 为了公平比较 merge 和 no-merge，要去掉随机性干扰

---

### 13.2 用 deepcopy 拷贝一份完全相同的层

```python
lora_layer_merged = copy.deepcopy(lora_layer)
lora_layer_merged.merge = True
lora_layer_merged.merge_weight()
lora_layer_merged.eval()
```

这一段才是真正合理的比较方式。

因为：

- 两个层来自同一份权重
- 一个走 no-merge 路线
- 一个先 merge 再前向

理论上它们输出应该一致。

---

### 13.3 比较输出差异

```python
with torch.no_grad():
    output_no_merge = lora_layer(x)
    output_merge = lora_layer_merged(x)

print("Max difference:", torch.max(torch.abs(output_no_merge - output_merge)).item())
```

这里得到：

```python
Max difference: 0.0
```

这说明在当前设置下：

- no-merge 版本
- merge 版本

输出完全一致。

这正是 merge 的目的：

> **训练时可以保留 LoRA 分支，推理时可以把它折叠进原权重，但数值结果不变。**

---

## 14. 这份 notebook 里最值得你掌握的 5 个点

### 14.1 LoRA 不是替换原层，而是在原层旁边加一条低秩分支
也就是：

```text
原输出 + 增量输出
```

---

### 14.2 LoRA 训练的是 A、B，不是原始大权重 W
这就是参数高效微调的关键。

---

### 14.3 `ΔW = A @ B` 的 shape 必须和原始 `W` 对齐
否则无法和原始线性层的输出相加。

---

### 14.4 merge 只是推理优化，不改变数学本质
不 merge 是：

```text
W*x + ΔW*x
```

merge 后是：

```text
(W+ΔW)x
```

理论上应当等价。

---

### 14.5 初始化成 `ΔW = 0` 是为了不破坏预训练模型
先保留原模型行为，再让 LoRA 慢慢学出有效偏移。

---

## 15. 这份实现和更标准 LoRA 写法的区别

这一部分很重要，因为它能体现你不是“只会背公式”，而是能区分教学实现和工程实现。

### 15.1 你的 dropout 加在最终输出上
你这里是：

```python
return self.dropout(output)
```

很多更标准的实现会把 dropout 放在 LoRA 支路上，而不是整个输出上。

所以你这份代码更适合学习 LoRA 核心思想，而不是直接当作工业级实现。

---

### 15.2 第一段 merged 测试不是同一组底层权重
重新创建的新层和原层不是同一个权重副本，所以那一段主要只能验证 shape，不适合直接拿来做严格数值对比。

---

### 15.3 当前打印出的 0 差异，和初始 `ΔW = 0` 也有关系
因为 `lora_b` 一开始全 0，所以 LoRA 增量一开始就是 0。

这使得很多“等价性测试”在初始状态下天然更容易得到 0 差异。

---

## 16. 你可以怎么把它和前面的内容串起来

你前面学的是：

- Single-Head Attention
- Multi-Head Attention
- Decoder
- GQA

这些都属于：

> **模型结构本身怎么设计**

而 LoRA 属于另一个层面：

> **模型已经有了以后，怎么高效微调**

所以要把这两条线分开：

### 结构线
- Attention 怎么做
- Decoder 怎么堆
- GQA 怎么省 K/V

### 训练线
- full fine-tuning 怎么做
- LoRA 为什么能省参数
- merge/unmerge 怎么服务训练与推理

也就是说：

> **LoRA 不是新的注意力结构，而是一种参数高效微调方法。**

---

## 17. 面试 / 复述时可以怎么讲

你可以这样说：

> LoRA 的核心思想是冻结预训练模型中的原始权重，只为某些线性层增加一个低秩可训练增量 `ΔW`。  
> 在前向时，输出由原始线性变换和 LoRA 增量两部分组成。  
> 由于 `ΔW` 被分解成两个小矩阵的乘积，因此需要训练的参数量远小于直接微调整个大矩阵。  
> 训练时通常采用 no-merge 形式，推理时可以把 LoRA 增量 merge 到原始权重里，这样既保留了效果，也方便部署。

---

## 18. 复习时你必须能回答的 8 个问题

### 问题 1：LoRA 为什么叫 Low-Rank Adaptation？
因为它把增量矩阵 `ΔW` 分解成两个低秩小矩阵的乘积。

### 问题 2：LoRA 真正训练的是谁？
训练的是 `lora_a` 和 `lora_b`，不是原始 `Linear.weight`。

### 问题 3：为什么要冻结原始权重？
因为 LoRA 的目标就是低成本适配，而不是全量微调。

### 问题 4：为什么 `lora_a @ lora_b` 的 shape 要和原始权重对齐？
因为它要作为 `ΔW` 加到原始权重上，或者在前向时形成一个同维度的增量输出。

### 问题 5：merge 和 no-merge 的数学关系是什么？
`W*x + ΔW*x` 与 `(W+ΔW)x` 是等价的。

### 问题 6：为什么初始时常让 LoRA 分支等于 0？
为了不破坏原始预训练模型的行为。

### 问题 7：为什么比较 merged 和 no-merge 时要切到 `eval()`？
为了避免 Dropout 带来的随机性。

### 问题 8：LoRA 和前面学的 Attention/GQA 是一类东西吗？
不是。Attention/GQA 是模型结构设计，LoRA 是微调方法。

---

## 19. 这一份笔记你最终要记住的一句话

> **LoRA 的本质，不是重训整个大模型，而是在冻结原模型的前提下，用一个低秩增量去学习“该往哪个方向微调”。**

---

## 20. 下一步怎么继续复习

你现在前面的学习链条已经比较完整了：

- `01`：单头注意力
- `02`：多头注意力
- `03`：Decoder
- `04`：GQA
- `05`：LoRA

接下来你最适合做两件事：

### 第一件事：做一份总复习笔记
比如：

```text
Attention / Decoder / GQA / LoRA 总结.md
```

把这五份内容串成一个完整体系。

### 第二件事：补一份“常见疑问汇总”
把你之前问过的这些都整理进去：

- Q/K/V 到底是什么
- 为什么要缩放
- 为什么要 mask
- decoder-only 和 encoder-decoder 的区别
- GQA 和 MHA / MQA 的关系
- LoRA merge/unmerge 的含义
- KV cache 是什么

这样以后你复习时，不只是看“老师讲了什么”，而是能直接看到：

> **我当时哪里没懂，我后来是怎么想通的。**