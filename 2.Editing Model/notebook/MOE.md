# MOE 学习笔记

> 这份笔记整理了你前面问过的关于 MOE（Mixture of Experts）的核心问题、三类常见/教学版 MOE 写法，以及代码里最容易混淆的 shape、索引、路由、加权融合等问题。  
> 目标：**复习时能快速回忆整体框架，写代码时能定位 shape，面试/汇报时能讲清楚原理。**

---

# 1. MOE 是什么

MOE（Mixture of Experts，专家混合）可以理解成：

- 模型里不是只有一个前馈层（FFN）
- 而是有很多个 expert（专家子网络）
- 再加一个 gate/router（门控/路由器）
- gate/router 负责决定：**当前输入更适合交给哪些 expert 处理**
- 多个 expert 的结果再进行融合

一句话概括：

> **MOE = 多个 expert + 一个 gate/router + 加权融合机制。**

---

# 2. MOE 的几个核心部件

## 2.1 Expert 是什么

Expert 本质上就是一个子网络。

最简单时可以只是一个线性层：

```python
class BasicExpert(nn.Module):
    def __init__(self, feature_in, feature_out):
        super().__init__()
        self.linear = nn.Linear(feature_in, feature_out)

    def forward(self, x):
        return self.linear(x)
```

更复杂时也可以是两层 MLP：

```python
class BasicExpert(nn.Module):
    def __init__(self, feature_in, hidden_dim, feature_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_out)
        )

    def forward(self, x):
        return self.net(x)
```

也可以写成更贴近现代 LLM 的 **SwiGLU expert**：

```python
class BasicExpert(nn.Module):
    def __init__(self, feature_in, feature_out, hidden_dim):
        super().__init__()
        self.w_gate = nn.Linear(feature_in, hidden_dim, bias=False)
        self.w_up = nn.Linear(feature_in, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, feature_out, bias=False)

    def forward(self, x):
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        return self.w_down(gate * up)
```

### 记忆点

- `Linear expert`：最简单，适合先学清楚 MOE 流程。
- `MLP expert`：比单层线性更合理。
- `SwiGLU expert`：更贴近现代大模型里的 FFN / expert 写法。

---

## 2.2 Gate / Router 是什么

Gate / Router 的作用是：

> 输入一个 token/sample 的表示，给所有 expert 打分，决定它该走哪些 expert。

例如：

```python
self.gate = nn.Linear(hidden_dim, expert_number)
```

那么：

```python
router_logits = self.gate(hidden_states)
```

就会得到：

- 每个 token 对每个 expert 的原始打分（logits / scores）
- shape 一般是 `(token_num, expert_number)`

---

## 2.3 Logits / Scores 是什么

如果 gate 是 `Linear`：

```python
self.gate = nn.Linear(feature_in, expert_number)
```

那么：

```python
self.gate(x)
```

本质就是：

\[
Wx + b
\]

这个输出就可以叫：

- logits
- scores
- raw routing scores

注意这里的“logits / scores”里的 `/` **不是除法**，而是“也叫做”的意思。

### 特点

这些原始分数：

- 可能为负
- 可能很大很小
- 不受约束
- 总和不一定是 1

所以如果想让它们更像“路由概率/权重”，通常会再接：

```python
routing_probs = F.softmax(router_logits, dim=-1)
```

---

# 3. 三种 MOE 类型总览

这部分重点整理你前面反复讨论过的三类结构：

1. **基础版 BasicMOE（Dense/Full MoE）**
2. **SparseMOE（Token-level top-k 路由）**
3. **ShareExpertMOE（DeepSeek 风格 Shared Experts）**

---

# 4. 第一类：基础版 BasicMOE（Dense / Full MoE）

## 4.1 核心思想

最基础的教学版 MoE 不是 sparse 的，而是：

> 每个输入都经过所有 expert，gate 给每个 expert 一个权重，最后做加权求和。

也就是说，这里没有 top-k，没有 dispatch，没有 token 分组。

---

## 4.2 典型代码

```python
class BasicMOE(nn.Module):
    def __init__(self, feature_in, feature_out, expert_number):
        super().__init__()
        self.experts = nn.ModuleList(
            [BasicExpert(feature_in, feature_out) for _ in range(expert_number)]
        )
        self.gate = nn.Linear(feature_in, expert_number)

    def forward(self, x):
        # x: [batch, feature_in]
        expert_weight = F.softmax(self.gate(x), dim=-1)   # [batch, expert_number]

        expert_out_list = [
            expert(x).unsqueeze(1) for expert in self.experts
        ]
        # 每个元素: [batch, 1, feature_out]

        expert_output = torch.cat(expert_out_list, dim=1)
        # [batch, expert_number, feature_out]

        expert_weight = expert_weight.unsqueeze(1)
        # [batch, 1, expert_number]

        output = expert_weight @ expert_output
        # [batch, 1, feature_out]

        return output.squeeze(1)
        # [batch, feature_out]
```

---

## 4.3 Shape 主线

假设：

- `batch = 2`
- `feature_in = 4`
- `feature_out = 3`
- `expert_number = 2`

那么：

### 输入

```python
x.shape = (2, 4)
```

### gate 输出

```python
expert_weight.shape = (2, 2)
```

### 每个 expert 输出

```python
expert(x).shape = (2, 3)
expert(x).unsqueeze(1).shape = (2, 1, 3)
```

### 拼接后

```python
expert_output.shape = (2, 2, 3)
```

### gate 加一维

```python
expert_weight.unsqueeze(1).shape = (2, 1, 2)
```

### 批量矩阵乘法

```python
(2, 1, 2) @ (2, 2, 3) -> (2, 1, 3)
```

### squeeze 后

```python
output.shape = (2, 3)
```

---

## 4.4 这类 MOE 的特点

### 优点

- 代码简单
- shape 容易看懂
- 适合教学

### 缺点

- 所有 expert 都要算，计算量大
- 没有真正体现 sparse 路由的优势

---

## 4.5 关于 softmax 的常见误区

很多时候会误以为：

> 既然用了 softmax，最后输出 `out` 每一行是不是和为 1？

不是。

真正和为 1 的是：

```python
expert_weight = F.softmax(self.gate(x), dim=-1)
```

也就是：

- 对每个样本
- 在 `expert_number` 这一维上
- 分给所有 expert 的权重和为 1

而 `out` 是：

> 多个 expert 输出向量的加权和

所以：

- `out` 可以有负数
- `out` 的各维不需要和为 1
- `out` 只是一个融合后的特征向量，不是概率分布

---

# 5. 第二类：SparseMOE（Token-level top-k 路由）

这是你前面重点问得最多的一类，也是当前大模型里更关键的一类。

## 5.1 核心思想

> 对每个 token，不再让它经过所有 expert，而是只经过 top-k 个最合适的 expert。

这就把 Dense MoE 变成了 Sparse MoE。

---

## 5.2 为什么 `hidden_states` 会是 `(b*s, hidden_dim)`

你前面反复问过：

> 为什么路由器输入不是 `(b, s, hidden_dim)`，而是 `(b*s, hidden_dim)`？

原因是：

> **Sparse MoE 一般是按 token 路由。**

也就是说，router 关心的是：

- 第 0 个 token 该去哪个 expert
- 第 1 个 token 该去哪个 expert
- ……

而不是“一整句话该去哪个 expert”。

所以通常先把：

```python
(b, s, hidden_dim)
```

展平为：

```python
(b*s, hidden_dim)
```

意思是：

- 当前 batch 一共有 `b*s` 个 token
- 每个 token 都是一个独立的路由对象

---

## 5.3 Router 的典型实现

```python
class MOERouter(nn.Module):
    def __init__(self, hidden_dim, expert_number, top_k):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, expert_number)
        self.expert_number = expert_number
        self.top_k = top_k

    def forward(self, hidden_states):
        # [b*s, expert_number]
        router_logits = self.gate(hidden_states)

        # [b*s, expert_number]
        routing_probs = F.softmax(router_logits, dim=-1, dtype=torch.float)

        # [b*s, top_k], [b*s, top_k]
        router_weights, selected_experts = torch.topk(
            routing_probs, self.top_k, dim=-1
        )

        # 在 top-k 内重新归一化
        router_weights = router_weights / router_weights.sum(dim=-1, keepdim=True)
        router_weights = router_weights.to(hidden_states.dtype)

        # [b*s, top_k, expert_number]
        expert_mask = F.one_hot(
            selected_experts,
            num_classes=self.expert_number
        )

        # [expert_number, top_k, b*s]
        expert_mask = expert_mask.permute(2, 1, 0)

        return router_logits, router_weights, selected_experts, expert_mask
```

---

## 5.4 Router 四个输出的含义

### 1）`router_logits`

shape:

```python
(b*s, expert_number)
```

含义：

- 每个 token 对所有 expert 的原始打分

---

### 2）`router_weights`

shape:

```python
(b*s, top_k)
```

含义：

- 每个 token 选中的 top-k expert 的最终归一化权重

---

### 3）`selected_experts`

shape:

```python
(b*s, top_k)
```

含义：

- 每个 token 选中了哪些 expert

例如：

```python
selected_experts =
[
  [3, 0],
  [1, 2]
]
```

表示：

- token0 选了 expert3、expert0
- token1 选了 expert1、expert2

---

### 4）`expert_mask`

shape:

```python
(expert_number, top_k, b*s)
```

含义：

- 方便后续从 expert 角度查看：
  - 哪些 token 选中了当前 expert
  - 当前 expert 是这些 token 的 top-1 还是 top-2

---

## 5.5 为什么要 one-hot 再 permute

你前面问得最多的点之一就是这个。

### 先 one-hot

```python
selected_experts.shape = (b*s, top_k)
```

例如：

```python
selected_experts =
[
  [2, 1],
  [0, 3]
]
```

one-hot 后：

```python
(b*s, top_k, expert_number)
```

意思是：

- 第 0 维：第几个 token
- 第 1 维：这个 token 的第几个 top-k 位置
- 第 2 维：是哪一个 expert

这里 **必须保留 top_k 这一维**，因为：

> `selected_experts = [2, 1]` 不只是“选中了 2 和 1”，还表示：
> - top-1 是 expert2
> - top-2 是 expert1

这和 `router_weights = [w1, w2]` 是逐位置对应的。

---

### 再 permute

```python
(b*s, top_k, expert_number) -> (expert_number, top_k, b*s)
```

原因是：

后面要按 expert 遍历：

```python
for expert_idx in range(expert_number):
    ...
```

所以更希望变成：

- 固定某个 expert
- 看哪些 token 在哪个 top-k 位置选中了它

---

## 5.6 SparseMOE 的主流程代码（带关键解释）

```python
class SparseMOE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.expert_number = config.expert_number
        self.top_k = config.top_k

        self.experts = nn.ModuleList(
            [
                BasicExpert(self.hidden_dim, self.hidden_dim)
                for _ in range(self.expert_number)
            ]
        )

        self.router = MOERouter(self.hidden_dim, self.expert_number, self.top_k)

    def forward(self, x):
        # x: [b, s, hidden_dim]
        batch_size, seq_len, hidden_dim = x.size()

        # 展平为 token 级别
        hidden_states = x.view(-1, hidden_dim)
        # [b*s, hidden_dim]

        router_logits, router_weights, selected_experts_indices, expert_mask = self.router(hidden_states)

        final_hidden_states = torch.zeros(
            (batch_size * seq_len, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )
        # [b*s, hidden_dim]

        for expert_idx in range(self.expert_number):
            expert_layer = self.experts[expert_idx]

            # 找出当前 expert 对应的 token
            idx, top_x = torch.where(expert_mask[expert_idx])
            # idx: 当前 expert 对这些 token 来说是 top1/top2/...
            # top_x: token 编号

            current_state = hidden_states[top_x]
            # [selected_token_number, hidden_dim]

            current_hidden_states = expert_layer(current_state)
            # [selected_token_number, hidden_dim]

            current_output = current_hidden_states * router_weights[top_x, idx].unsqueeze(-1)
            # [selected_token_number, hidden_dim]

            final_hidden_states.index_add_(0, top_x, current_output.to(hidden_states.dtype))

        final_hidden_states = final_hidden_states.reshape(batch_size, seq_len, hidden_dim)
        # [b, s, hidden_dim]

        return final_hidden_states, router_logits
```

---

## 5.7 `torch.where(expert_mask[expert_idx])` 到底返回什么

这是你前面反复卡住的点。

如果：

```python
expert_mask[expert_idx].shape = (top_k, b*s)
```

例如：

```python
expert_mask[2] =
tensor([
    [1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 1]
])
```

那么：

```python
idx, top_x = torch.where(expert_mask[2])
```

会返回：

```python
idx   = tensor([0, 0, 1, 1])
top_x = tensor([0, 3, 1, 5])
```

含义：

- token0 在 top-1 选了当前 expert
- token3 在 top-1 选了当前 expert
- token1 在 top-2 选了当前 expert
- token5 在 top-2 选了当前 expert

### 关键记忆

- `idx`：top-k 位置编号（0 表示 top-1，1 表示 top-2）
- `top_x`：token 编号

它们都是：

- **一维 tensor**
- 可以理解为“索引列表”

---

## 5.8 `current_state = hidden_states[top_x]` 在做什么

这句等价于你老师课堂里那种稍绕的写法：

```python
current_state = hidden_states.unsqueeze(0)[:, top_x, :].reshape(-1, hidden_dim)
```

本质上就是：

> 从所有 token 的 hidden_states 里，取出编号为 `top_x` 的那些 token 向量。

shape：

```python
(selected_token_number, hidden_dim)
```

也就是：

> 当前 expert 要处理的所有 token 的输入。

---

## 5.9 `current_output = current_hidden_states * router_weights[top_x, idx].unsqueeze(-1)` 在做什么

这里是 SparseMOE 里最容易写错的一句。

### 正确逻辑

- `current_hidden_states.shape = (selected_token_number, hidden_dim)`
- `router_weights[top_x, idx].shape = (selected_token_number,)`

为了能广播相乘，需要：

```python
router_weights[top_x, idx].unsqueeze(-1)
```

把它变成：

```python
(selected_token_number, 1)
```

然后和：

```python
(selected_token_number, hidden_dim)
```

做广播相乘。

### 常见 bug

如果忘了 `unsqueeze(-1)`，就会出现你前面遇到的那种 shape 错误。

### 另一个常见 bug

有时会误写成：

```python
router_weights[top_x, expert_idx]
```

这是错的。

因为：

- `router_weights` 的第二维是 `top_k`
- 不是 `expert_number`

所以这里第二维必须用的是：

```python
idx
```

也就是“当前 expert 对该 token 来说是第几个 top-k 位置”。

---

## 5.10 `index_add_` 的原理

你前面也专门问过这个。

```python
final_hidden_states.index_add_(0, top_x, current_output)
```

等价于：

```python
for i in range(len(top_x)):
    final_hidden_states[top_x[i]] += current_output[i]
```

它表示：

> 把 `current_output` 的每一行，加到 `final_hidden_states` 中 `top_x` 指定的那些 token 位置上。

为什么必须是“加”？

因为一个 token 可能会经过多个 expert，例如：

\[
y_t = w_1 E_1(x_t) + w_2 E_2(x_t)
\]

所以：

- expert1 的贡献先加一次
- expert2 的贡献再加一次

最后才得到 token 的最终输出。

### 为什么比 `final_hidden_states[top_x] += current_output` 更稳

因为 `top_x` 里可能有重复索引，`index_add_` 在这种 scatter-add 场景下更安全，也更高效。

---

## 5.11 SparseMOE 最终输出是什么

最终：

```python
final_hidden_states.shape = (b, s, hidden_dim)
```

这表示：

> 每个 token 经过 top-k expert 加权融合后的新 hidden representation。

此外还会返回：

```python
router_logits.shape = (b*s, expert_number)
```

这个常用来算辅助损失、做分析或者调试路由。

---

# 6. 第三类：ShareExpertMOE（DeepSeek 风格 Shared Experts）

## 6.1 核心思想

这是在 SparseMOE 基础上的进一步设计。

除了 sparse experts 外，再额外引入一组：

> **所有 token 都会经过的 shared experts。**

所以最终输出是：

\[
output = SparseMOE(x) + \sum_i SharedExpert_i(x)
\]

---

## 6.2 为什么要加 shared experts

如果完全只靠 sparse experts，会有一些问题：

- 某些通用知识不容易稳定覆盖
- router 选错时，信息可能损失
- 专门化太强，通用能力不足

加入 shared experts 后，相当于多了一条：

> 不依赖路由、所有 token 都会通过的通用处理分支。

所以可以把它理解成：

- sparse experts：专科医生
- shared experts：全科医生

最终输出就是：

> 专科结果 + 全科结果

---

## 6.3 典型代码

```python
class ShareExpertMOE(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.moe_model = SparseMOE(config)
        self.shared_experts = nn.ModuleList(
            [
                BasicExpert(config.hidden_dim, config.hidden_dim)
                for _ in range(config.shared_experts_number)
            ]
        )

    def forward(self, x):
        # x: (b, s, hidden_dim)
        sparse_moe_out, router_logits = self.moe_model(x)

        shared_experts_out = [
            expert(x) for expert in self.shared_experts
        ]
        # 每个 expert(x): [b, s, hidden_dim]

        shared_experts_out = torch.stack(shared_experts_out, dim=0).sum(dim=0)
        # [b, s, hidden_dim]

        return sparse_moe_out + shared_experts_out, router_logits
```

---

## 6.4 Shape 解释

假设：

- `x.shape = (b, s, hidden_dim)`
- 共享专家数量 = `shared_experts_number`

那么：

### 稀疏分支输出

```python
sparse_moe_out.shape = (b, s, hidden_dim)
```

---

### 每个 shared expert 输出

```python
expert(x).shape = (b, s, hidden_dim)
```

---

### stack 后

```python
torch.stack(shared_experts_out, dim=0).shape
= (shared_experts_number, b, s, hidden_dim)
```

---

### sum 后

```python
shared_experts_out.shape = (b, s, hidden_dim)
```

---

### 最终输出

```python
sparse_moe_out + shared_experts_out
```

shape 仍然是：

```python
(b, s, hidden_dim)
```

---

## 6.5 一个非常重要的实现提醒

如果你已经把 `BasicExpert` 改成了 **三参数版**，例如：

```python
BasicExpert(feature_in, feature_out, hidden_dim)
```

那么在 `SparseMOE` 和 `ShareExpertMOE` 里实例化 expert 时都要同步改。

例如：

```python
BasicExpert(config.hidden_dim, config.hidden_dim, config.hidden_dim)
```

或者更规范地写成：

```python
BasicExpert(config.hidden_dim, config.hidden_dim, config.expert_hidden_dim)
```

不要一边用三参数版 `BasicExpert`，一边还按双参数去实例化，不然一定会报参数不匹配错误。

---

# 7. 你前面问过的核心问题整理（FAQ）

## Q1：为什么有时候输入是 `(batch, feature_in)`，有时候是 `(batch, seq, hidden_dim)`？

### 基础版 BasicMOE

通常为了教学简化，输入写成：

```python
(batch, feature_in)
```

表示每一行就是一个样本。

### LLM / token-level SparseMOE

更常见的输入是：

```python
(batch, seq, hidden_dim)
```

表示：

- 一个 batch 里有多条序列
- 每条序列有多个 token
- 每个 token 有 hidden representation

再把前两维展平为 `(b*s, hidden_dim)` 来做 token 路由。

---

## Q2：`squeeze` / `unsqueeze` 是干嘛的？

### `unsqueeze(dim)`

在指定位置插入一个大小为 1 的维度。

例如：

```python
x.shape = (batch, feature_out)
x.unsqueeze(1).shape = (batch, 1, feature_out)
```

### `squeeze(dim)`

去掉指定位置大小为 1 的维度。

例如：

```python
output.shape = (batch, 1, feature_out)
output.squeeze(1).shape = (batch, feature_out)
```

### 为什么更推荐 `squeeze(1)` 而不是 `squeeze()`

因为 `squeeze()` 会把所有 size=1 的维度都挤掉。  
如果 `batch=1`，可能把 batch 维也挤没。

---

## Q3：top-k 之后为什么还要归一化？

因为：

- `softmax` 是在所有 expert 上归一化
- 选完 top-k 后，其余 expert 被丢掉了
- 剩下的 top-k 权重和一般小于 1

所以通常会再做：

```python
router_weights = router_weights / router_weights.sum(dim=-1, keepdim=True)
```

这样 top-k 内部的权重和又恢复为 1。

---

## Q4：top-k 后能不能再 softmax 一次？

可以，但通常不这么做。

### 除以和

表示：

> 保留原来 softmax 概率之间的相对比例，只在选中的 expert 内部重新缩放。

### 再 softmax

表示：

> 把已经是概率的数再当作 logits 做一次竞争，会重新改变权重关系。

教学和实现里更常见的是“除以和”。

---

## Q5：`torch.topk` 返回什么？

例如：

```python
routing_probs = [0.1, 0.3, 0.4, 0.2]
```

如果：

```python
top_k = 2
```

那么：

```python
router_weights = [0.4, 0.3]
selected_experts = [2, 1]
```

含义：

- 选中了 expert2 和 expert1
- 它们的权重分别是 0.4 和 0.3

注意：

- `selected_experts` 保留顺序（top-1 / top-2）
- `router_weights` 和 `selected_experts` 是逐位置对应的

---

## Q6：为什么 `expert_mask` 还要保留 top_k 这一维？

因为：

```python
selected_experts = [2, 1]
router_weights = [0.7, 0.3]
```

这里不仅表示“选中了 2 和 1”，还表示：

- top-1 是 expert2，对应权重 0.7
- top-2 是 expert1，对应权重 0.3

如果只写成：

```python
[0, 1, 1, 0]
```

虽然知道选中了 expert1 和 expert2，但不知道谁对应哪个权重。

所以必须保留 top-k 的位置维度。

---

## Q7：`(b*s, top_k, expert_number)` 和 `(expert_number, top_k, b*s)` 分别什么意思？

### `(b*s, top_k, expert_number)`

从 **token 视角** 看：

- 第几个 token
- 这个 token 的第几个 top-k 位置
- 是哪个 expert（one-hot）

### `(expert_number, top_k, b*s)`

从 **expert 视角** 看：

- 第几个 expert
- 这个 expert 是 token 的 top-1 还是 top-2
- 是哪个 token

一句话：

- 前者：token → expert
- 后者：expert → token

---

## Q8：`current_state = hidden_states[top_x]` 到底在取什么？

取的是：

> 当前 expert 负责处理的那些 token 的 hidden states。

也就是：

- `top_x` 里有哪些 token 编号
- 就把这些 token 从 `hidden_states` 里提出来

---

## Q9：`index_add_` 到底在做什么？

等价于：

```python
for i in range(len(top_x)):
    final_hidden_states[top_x[i]] += current_output[i]
```

它表示：

> 按照 token 编号，把当前 expert 对这些 token 的贡献，加回总输出表对应的位置。

关键词：

- 是“加”，不是“覆盖”
- 因为一个 token 可能经过多个 expert

---

## Q10：`self.gate(x)` 的输出到底是权重还是分数？

更准确地说，它是：

- logits
- scores
- raw routing scores

因为它本质是：

\[
Wx+b
\]

还没 softmax 前：

- 可以有负数
- 和不一定是 1
- 不是真正概率

softmax 之后，才更适合叫：

- routing probabilities
- expert weights

---

## Q11：BasicExpert 用 SwiGLU 合不合理？

合理，而且更贴近现代 LLM。

你老师说：

> BasicExpert 也可以用 SwiGLU

这个说法是对的。

因为 expert 本质只是一个子网络：

- 最简版：Linear
- 进阶版：MLP
- 更现代版：SwiGLU

所以如果你已经把 `BasicExpert` 写成 SwiGLU 版，是合理的。  
要注意的只是：

- 构造函数参数变了
- 所有实例化它的地方都要同步改

---

## Q12：为什么有时 `BasicExpert` 只传两个参数，有时要三个参数？

### 两个参数

说明 expert 是最简单单层线性层：

```python
BasicExpert(feature_in, feature_out)
```

### 三个参数

说明 expert 内部有一个中间扩展层，例如：

```python
BasicExpert(feature_in, feature_out, hidden_dim)
```

这里第三个参数一般表示：

- expert 内部的扩展维度
- 更准确可以叫 `ffn_hidden_dim`

如果你改成了三参数版 expert，那么：

- `BasicMOE`
- `SparseMOE`
- `ShareExpertMOE`

里实例化它的地方都要改。

---

# 8. 三类 MOE 的对比总结

## 8.1 BasicMOE

### 思路

- 每个输入都过所有 expert
- gate 给权重
- 做加权求和

### 特点

- 简单
- 易讲清 shape
- 计算量大

### 适合用途

- 入门学习
- 验证 gate + expert_output 的形状

---

## 8.2 SparseMOE

### 思路

- 每个 token 只过 top-k 个 expert
- router 负责稀疏路由
- expert 只处理属于自己的 token
- 最后按权重把结果加回 token 位置

### 特点

- 计算更省
- 更符合大模型实际实现
- shape 和索引更复杂

### 关键词

- flatten tokens
- router
- top-k
- expert_mask
- torch.where
- index_add_

---

## 8.3 ShareExpertMOE

### 思路

- 一条 SparseMOE 路由分支
- 一条 Shared Experts 通用分支
- 两者相加

### 特点

- sparse 负责专门化
- shared 负责通用能力
- 更贴近 DeepSeek 风格设计思路

### 关键词

- sparse experts = 专科
- shared experts = 全科
- output = sparse + shared

---

# 9. 代码书写时最容易出错的地方

## 9.1 `router_weights` 取错列

错误写法：

```python
router_weights[top_x, expert_idx]
```

因为 `router_weights.shape = (b*s, top_k)`，第二维是 top-k 位置，不是 expert 编号。

正确写法：

```python
router_weights[top_x, idx]
```

---

## 9.2 忘了 `unsqueeze(-1)`

错误写法：

```python
current_hidden_states * router_weights[top_x, idx]
```

如果前者是：

```python
[selected_token_number, hidden_dim]
```

后者是：

```python
[selected_token_number]
```

会广播失败。

正确写法：

```python
current_hidden_states * router_weights[top_x, idx].unsqueeze(-1)
```

---

## 9.3 `BasicExpert` 参数数目不匹配

如果你已经把 `BasicExpert` 改成三参数版：

```python
BasicExpert(feature_in, feature_out, hidden_dim)
```

那所有地方都得改成 3 个参数。

---

## 9.4 `squeeze()` 用太猛

建议：

```python
output.squeeze(1)
```

而不是：

```python
output.squeeze()
```

否则 `batch=1` 时容易把 batch 维一起挤掉。

---

## 9.5 `view()` 前最好确认内存连续

如果不是从简单连续 tensor 来的，有时更稳妥会写：

```python
hidden_states = x.reshape(-1, hidden_dim)
```

不过你当前这些教学版代码里用 `view()` 通常没问题。

---

# 10. 复习速记版（考前 / 看代码前先扫一遍）

## 10.1 一句话记住三类 MOE

### BasicMOE

> 所有 expert 全算，最后加权求和。

### SparseMOE

> 每个 token 只去 top-k 个 expert，再把贡献加回来。

### ShareExpertMOE

> Sparse 分支做专门化，Shared 分支做通用处理，最后两者相加。

---

## 10.2 一句话记住 Router

> Router 的任务就是：给每个 token 选 expert，并给选中的 expert 分配权重。

---

## 10.3 一句话记住 `expert_mask`

> `expert_mask` 是把“token 选了哪些 expert”的信息，变成“每个 expert 被哪些 token 选中了”的可索引结构。

---

## 10.4 一句话记住 `torch.where`

> `torch.where(expert_mask[expert_idx])` 返回当前 expert 对应的所有非零位置：
> - `idx` 表示 top-k 位置
> - `top_x` 表示 token 编号

---

## 10.5 一句话记住 `index_add_`

> 把当前 expert 对 token 的贡献，按 token 编号加回总输出表。

---

# 11. 推荐的学习顺序

如果你后面还要继续复习/手写，建议按这个顺序：

## 第一步：先完全吃透 BasicMOE

重点看：

- `expert_output` 为什么是 `(batch, expert_number, feature_out)`
- `expert_weight @ expert_output` 为什么能做加权求和

---

## 第二步：再吃透 Router

重点看：

- `router_logits`
- `routing_probs`
- `topk`
- `router_weights`
- `expert_mask`

---

## 第三步：再吃透 SparseMOE 的 dispatch

重点看：

- `torch.where`
- `current_state = hidden_states[top_x]`
- `router_weights[top_x, idx].unsqueeze(-1)`
- `index_add_`

---

## 第四步：最后再理解 Shared Experts

重点看：

- 为什么 shared experts 不走 router
- 为什么所有 token 都过 shared experts
- 为什么最后是 `sparse + shared`

---

# 12. 最后总结

你前面关于 MOE 问的核心，实际上都可以收束成三句话：

1. **BasicMOE** 让你先看懂“多个 expert 的输出如何按 gate 权重做加权和”。  
2. **SparseMOE** 让你真正理解“每个 token 如何只走 top-k experts，以及 expert 如何只处理属于自己的 token”。  
3. **ShareExpertMOE** 让你看到“专门化 sparse experts 和通用 shared experts 如何并行补充”。

如果你后面复习时只记住一句总纲，可以记这个：

> **MOE 的本质不是 expert 本身有多神秘，而是：输入先被路由，再由不同 expert 分工处理，最后把这些 expert 的输出融合回来。**

