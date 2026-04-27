# LlamaIndex 笔记

## 1. LlamaIndex 是什么

LlamaIndex 是一个围绕**“你的数据 + LLM + agent + workflow”** 来构建系统的框架。  
它不只是一个简单的“调模型库”，而是想把下面这些能力串起来：

- 你的数据接入
- 文档解析与检索
- 工具调用
- Agent 决策
- Workflow 编排

在课程里，可以把它整体理解成一条主线：

**LlamaHub → Components → Tools → Agents → Workflows**

---

## 2. 整体地图

### 2.1 LlamaHub
作用：找组件、找集成、找工具。

它相当于 LlamaIndex 生态中的一个注册表。  
以后如果你想接：

- 新的 LLM
- 新的 embedding 模型
- 新的向量库
- 新的工具或服务

第一反应应该是先去看 LlamaHub 有没有现成集成。

### 2.2 Components
作用：搭建 RAG 的底层管线。

这是 LlamaIndex 很核心的一层，尤其是：

- Document
- Node
- Embedding
- Vector Store
- VectorStoreIndex
- QueryEngine

### 2.3 Tools
作用：把“能力”暴露给 Agent。

LlamaIndex 不仅能把普通函数包装成工具，也能把 QueryEngine 这种 RAG 能力变成工具。

### 2.4 Agents
作用：让模型自己决定什么时候调用工具。

重点是：

- Function Calling Agent
- ReAct Agent
- Multi-agent 协作

### 2.5 Workflows
作用：组织复杂流程、状态、循环、多 agent 协作。

这是 LlamaIndex 和很多纯工具调用框架相比，特别有特点的地方。

---

## 3. LlamaHub

### 3.1 它是什么
LlamaHub 是一个组件注册表，里面有很多：

- integrations
- agents
- tools

### 3.2 安装命名规律
很多扩展包遵循这种格式：

```bash
pip install llama-index-{component-type}-{framework-name}
```

例如：

```bash
pip install llama-index-llms-huggingface-api
pip install llama-index-embeddings-huggingface
pip install llama-index-vector-stores-chroma
```

### 3.3 这一节真正要记住什么
不是去背每个包名，而是建立一种意识：

> 想接新能力时，优先去 LlamaHub 看有没有现成集成。

---

## 4. Components：RAG 主线

这一节最核心的其实是 **QueryEngine**，因为它是后面做 RAG tool 和 agent 的桥梁。

LlamaIndex 课程里把整个过程拆成 5 步：

1. Loading
2. Indexing
3. Storing
4. Querying
5. Evaluation

你可以把它概括成：

**读数据 → 组织数据 → 存起来 → 查询 → 评估**

---

## 5. Loading：先把数据读进来

常见方式有：

- `SimpleDirectoryReader`
- `LlamaParse`
- 各类 LlamaHub 数据集成

这一阶段得到的通常是 **Document**。

### 5.1 Document 是什么
Document 表示“整份原始资料”。

比如：

- 一篇 PDF
- 一份 txt
- 一篇网页内容

都可以先表示成一个 Document。

---

## 6. Document 和 Node 的区别

### 6.1 Document
原始大文档。

### 6.2 Node
从 Document 中切分出来的小块。

为什么要切分？

因为整份文档通常太长：

- 不适合直接检索
- 不适合直接给模型
- 太粗粒度，不容易精确命中相关内容

所以需要把 Document 切成更适合检索的小块 Node。

### 6.3 一句话记忆
- **Document = 原始资料**
- **Node = 可检索的小知识块**

---

## 7. IngestionPipeline：切块 + 向量化

常见步骤包括：

- `SentenceSplitter`
- `HuggingFaceEmbedding`

### 7.1 SentenceSplitter
把文档切成更小的块。

### 7.2 HuggingFaceEmbedding
把每个文本块转成 embedding 向量。

### 7.3 embedding 的真正作用
embedding 主要是为了**语义检索**，不是直接负责生成回答。

也就是说：

- embedding：负责“找相关内容”
- LLM：负责“把内容说成人话”

---

## 8. Storing：为什么要存入向量库

如果每次提问都重新：

- 读文档
- 切块
- 算 embedding

那会非常低效。

所以通常会把这些东西存进向量库，比如课程示例用的是 **Chroma**。

### 8.1 向量库存的是什么
主要是：

- Node 对应的向量
- 元数据
- 与文档的关联信息

### 8.2 意义
- 避免重复建索引
- 提高查询效率
- 可以持久化保存知识库

---

## 9. VectorStoreIndex：把向量库组织成索引

LlamaIndex 会用 `VectorStoreIndex` 把底层向量库封装成一个可查询的索引对象。

它的作用不是“再存一次”，而是：

> 把原始 vector store 包装成 LlamaIndex 可以统一查询的索引结构。

### 9.1 关键点
构建 `VectorStoreIndex` 时，通常要使用和 ingestion 阶段一致的 embedding 模型，否则查询向量和文档向量可能不在同一个语义空间里。

---

## 10. QueryEngine：最核心的查询对象

`QueryEngine` 是 LlamaIndex 中非常关键的对象。

它做的事情是：

1. 接收用户问题
2. 基于索引检索相关 Node
3. 把结果交给 LLM
4. 生成自然语言回答

### 10.1 常见接口
- `as_retriever`
- `as_query_engine`
- `as_chat_engine`

### 10.2 三者区别
#### `as_retriever`
只负责检索。

#### `as_query_engine`
负责单轮问答：检索 + LLM 回答。

#### `as_chat_engine`
适合多轮对话。

### 10.3 一句话理解
**QueryEngine = RAG 的可问答接口**

---

## 11. ResponseSynthesizer：回答是怎么组织出来的

QueryEngine 内部还会用到 `ResponseSynthesizer` 来决定怎么整合检索结果。

常见模式：

- `refine`
- `compact`
- `tree_summarize`

### 11.1 直观理解
- `refine`：一块一块逐步修正答案
- `compact`：尽量合并上下文，减少调用
- `tree_summarize`：更适合综合总结多块信息

---

## 12. Evaluation：怎么评估回答

LlamaIndex 提供 evaluator，例如：

- `FaithfulnessEvaluator`
- `AnswerRelevancyEvaluator`
- `CorrectnessEvaluator`

### 12.1 三者分别是什么
#### Faithfulness
回答是否忠实于检索到的上下文。

#### Relevancy
回答是否和问题相关。

#### Correctness
回答是否正确。

### 12.2 核心意义
系统不能只会答，还要能评估“答得好不好”。

---

## 13. Observability：为什么还要看内部过程

除了评估结果，还要看系统内部执行流程。

可观测性能帮你看到：

- 检索到了什么
- 哪一步调用了什么
- 哪一步可能出错
- LLM 到底拿到了什么上下文

可以理解成：

**给 RAG / agent 系统配一个调试仪表盘**

---

## 14. Components 一句话总结

整个 Components 主线可以总结成：

**数据 → Document → Node → embedding → 向量库 → VectorStoreIndex → QueryEngine → LLM 回答 → evaluator / observability**

---

## 15. Tools：把能力暴露给 Agent

LlamaIndex 中工具主要分为四类：

1. `FunctionTool`
2. `QueryEngineTool`
3. `ToolSpecs`
4. `Utility Tools`

---

## 16. FunctionTool

### 16.1 是什么
把普通 Python 函数包装成 agent 能调用的工具。

### 16.2 适合场景
- 本地函数
- 简单 API 包装
- 数学工具
- 文件工具

### 16.3 关键点
工具的：

- name
- description
- 参数类型
- docstring

都很重要，因为 LLM 会基于这些信息理解工具用途。

### 16.4 一句话理解
**FunctionTool = 普通函数 → Agent 工具**

---

## 17. QueryEngineTool

### 17.1 是什么
把 `QueryEngine` 包装成 tool。

### 17.2 意义
这意味着：

> 知识库检索能力，也可以成为 agent 的一个工具。

也就是说，agent 的工具箱里不只是：

- 天气工具
- 计算器工具
- 邮件工具

还可以有：

- **知识库问答工具**

### 17.3 一句话理解
**QueryEngineTool = RAG 能力 → Agent 工具**

---

## 18. ToolSpecs

### 18.1 是什么
成套工具集合，不是单个函数。

例如：

- GmailToolSpec
- Google 相关工具集
- MCP 工具集

### 18.2 什么时候适合用
当你接入的是某个服务的一整套能力，而不是一个单独函数时。

### 18.3 一句话理解
**ToolSpec = 某个生态服务的一整套工具包**

---

## 19. Utility Tools

### 19.1 为什么存在
因为有些工具会返回很大一坨数据，直接交给 LLM 会：

- 浪费 token
- 撑爆上下文
- 增加噪音

### 19.2 它们在做什么
核心思想是：

> 先加载 / 索引，再搜索，而不是把全部原始输出一股脑给模型。

### 19.3 一句话理解
**Utility Tools = 让大输出更适合被 LLM 使用**

---

## 20. Tools 一句话总结

LlamaIndex 的工具体系不只是支持简单函数，还支持：

- 普通函数
- RAG 查询能力
- 第三方服务集成
- 对大规模输出的二次处理

统一暴露给 agent 使用。

---

## 21. Agents：让模型决定“做什么”

Agent 的本质可以理解成：

**LLM + tools + 决策机制**

课程里主要提到三类 agent：

1. Function Calling Agent
2. ReAct Agent
3. Advanced Custom Agent

---

## 22. Function Calling Agent

### 22.1 特点
适合支持函数调用 API 的模型。

### 22.2 优点
- 工具调用更结构化
- 参数更规范
- 通常更稳定

### 22.3 适合什么情况
你使用的模型本身支持 tool calling / function calling。

---

## 23. ReAct Agent

### 23.1 特点
适合更广泛的 chat / text 模型。

### 23.2 思路
遵循：

- Thought
- Action
- Observation

这种思路。

### 23.3 直观理解
模型先思考，再决定要不要调用工具，用完工具再继续推理。

---

## 24. Agent 默认是无状态的

默认情况下，agent 是 **stateless** 的。

也就是说：

```python
response = await agent.run(user_msg="...")
```

只处理当前这次消息。

### 24.1 如果想保留状态
就要传入同一个 `Context`。

这样多轮对话中，agent 才能“记住之前说过什么”。

### 24.2 一句话理解
- 不传 `Context`：每次独立
- 传同一个 `Context`：多轮记忆

---

## 25. Agentic RAG

普通 RAG 通常是：

- 先检索
- 再回答

而 Agentic RAG 是：

- agent 先判断要不要查知识库
- 如果要，再调用 QueryEngineTool
- 如果不需要，也可以用别的工具或逻辑

### 25.1 核心区别
**普通 RAG 是固定流程；Agentic RAG 是自主决策流程。**

---

## 26. Multi-agent

LlamaIndex 支持多个 agent 协作。

关键点：

- 有一个 `root_agent`
- 当前 active speaker 一次只有一个
- agent 可以 `handoff` 给其他 agent

### 26.1 为什么要这样做
因为让每个 agent 只负责一个更小的职责范围，通常更稳定。

### 26.2 例子
- 一个数学 agent
- 一个知识库查询 agent

用户消息先进 root_agent，必要时再 handoff。

### 26.3 一句话理解
**multi-agent 的重点不是“多”，而是“分工”**

---

## 27. AgentWorkflow

这是把多个 agent 组织起来的工作流对象。

典型形式：

```python
workflow = AgentWorkflow(
    agents=[agent_a, agent_b],
    root_agent="agent_a",
)
```

然后：

```python
response = await workflow.run(user_msg="...")
```

### 27.1 它解决什么问题
- 多 agent 路由
- handoff
- 状态共享
- 多步骤协作

---

## 28. Workflows：LlamaIndex 很重要的部分

LlamaIndex 的 Workflow 是：

- event-driven
- async-first
- step-based

它的核心思想是：

> Step 通过 Event 连接，流程由事件驱动，而不是你手写一条死顺序。

---

## 29. 最小 Workflow 骨架

最基础的骨架是：

```python
class MyWorkflow(Workflow):
    @step
    async def my_step(self, ev: StartEvent) -> StopEvent:
        return StopEvent(result="...")
```

### 29.1 这里的角色
- `Workflow`：工作流类
- `@step`：声明一个步骤
- `StartEvent`：流程开始时的事件
- `StopEvent`：流程结束时的事件

### 29.2 一句话理解
**最小 workflow = 一个 step，接收 StartEvent，返回 StopEvent**

---

## 30. 自定义 Event

如果是多步 workflow，就要自己定义事件，例如：

```python
class ProcessingEvent(Event):
    intermediate_result: str
```

这样前一步可以返回 `ProcessingEvent`，后一步接收它。

### 30.1 Event 的作用
把一步的输出传给下一步。

### 30.2 一句话理解
**Event = Workflow 里的“数据包”**

---

## 31. Type hinting 为什么特别重要

在 Workflow 里，类型标注不是装饰，而是调度逻辑的一部分。

系统会根据：

- 这个 step 接收什么 event
- 那个 step 返回什么 event

自动把步骤连接起来。

### 31.1 所以 workflow 的顺序不是你手写死的
而是由事件类型驱动出来的。

---

## 32. Loops 和 Branches

Workflow 里可以通过联合类型实现循环和分支，比如：

- `StartEvent | LoopEvent`
- 返回 `ProcessingEvent | LoopEvent`

### 32.1 直观理解
如果某个 step 返回 `LoopEvent`，系统就会再次把流程路由到接收 `LoopEvent` 的 step。

所以 workflow 的循环不是普通 `while`，而是：

**“再来一次”的事件不断触发自己。**

---

## 33. draw_all_possible_flows

Workflow 可以被可视化画出来。

意义：

- 复杂流程更容易看清
- 能直观看到分支与循环
- 有利于调试

---

## 34. Context：共享状态

在 Workflow 里，如果你希望多个 step 共享状态，就可以在 step 中加入：

```python
ctx: Context
```

然后通过：

```python
await ctx.store.set(...)
await ctx.store.get(...)
```

来存取状态。

### 34.1 核心意义
让整个 workflow 的不同步骤能共享公共信息。

---

## 35. Event 和 Context 的区别

这是非常重要的一个点。

### 35.1 Event
用于**局部传值**。

也就是：

> 这一步产生的结果，交给后面某一步。

### 35.2 Context / state
用于**全局协作**。

也就是：

> 整个 workflow 里多个 step / tool / agent 都可能要访问和修改的信息。

### 35.3 记忆口诀
- **局部传递：Event**
- **共享状态：Context**

---

## 36. AgentWorkflow + state

AgentWorkflow 不仅支持多个 agent，还支持共享状态。

例如：

```python
workflow = AgentWorkflow(
    agents=[...],
    root_agent="...",
    initial_state={"num_fn_calls": 0},
    state_prompt="Current state: {state}. User message: {msg}",
)
```

### 36.1 `initial_state`
给整次 workflow 运行一份初始共享状态。

### 36.2 `state_prompt`
把 state 注入到 agent 的上下文里。

### 36.3 一句话理解
**AgentWorkflow 可以让多个 agent 围绕一份共享状态协作。**

---

## 37. 为什么 state 像“全局变量”，但又不完全是

你的直觉是对的：  
state 确实很像一份所有参与者都能读写的共享变量。

但它和普通 Python 全局变量不一样。

### 37.1 普通全局变量的问题
- 容易污染整个程序
- 多次运行容易串状态
- 不好管理
- 异步情况下更危险

### 37.2 Workflow 中的 state 更像什么
它更像：

**“本次 workflow 运行期间的一份共享状态容器”**

它绑定在这次 `Context` 上，而不是整个 Python 进程全局共享。

### 37.3 所以更准确的理解
它不是“全局变量”，而是：

> 本次 workflow 运行内的共享状态。

---

## 38. 为什么要设计共享状态

因为真实场景里，很多信息不是某个 step 独有的，而是整个流程都可能要看。

例如：

- 调用了多少次工具
- 当前任务做到第几步
- 用户偏好
- 中间决策
- 已经查过哪些资源

这些信息如果都靠 Event 传来传去，会非常乱。  
所以 Context/state 的存在，是为了让协作更自然。

---

## 39. 你学习时最容易混淆的点

### 39.1 embedding 不是回答的主体
embedding 是为了检索，LLM 才是最终回答的主体。

### 39.2 Document 和 Node 不一样
- Document：原始大文档
- Node：切分的小块

### 39.3 QueryEngine 和 QueryEngineTool 不一样
- QueryEngine：知识库查询接口
- QueryEngineTool：把查询能力变成 agent 工具

### 39.4 Event 和 Context 不一样
- Event：传中间结果
- Context：共享状态

### 39.5 Agent 和 Workflow 不一样
- Agent：做决策、调工具
- Workflow：组织步骤、事件、状态和流程

---

## 40. 最核心的主线总结

如果把 LlamaIndex 压缩成一条最值得复习的主线，就是：

**LlamaIndex 是一个围绕“你的数据 + 检索 + 工具 + agent + workflow”来构建系统的框架。它先通过 LlamaHub 接入外部组件，再通过 Components 把数据加工成可检索结构；典型流程是将 Document 切成 Node、做 embedding、存入向量库、构建 VectorStoreIndex，并通过 QueryEngine 提供问答接口。接着，这些能力可以被包装成 Tools，尤其是 QueryEngineTool，使 RAG 能力成为 agent 的可调用工具。再往上，Agents 负责决定何时调用工具、是否 handoff 给其他 agent；最后，Workflows 负责把步骤、事件、共享状态和多 agent 协作组织成一个完整系统。**

---

## 41. 高频速记版

### 41.1 五层地图
- LlamaHub
- Components
- Tools
- Agents
- Workflows

### 41.2 RAG 主线
Document → Node → embedding → vector store → VectorStoreIndex → QueryEngine

### 41.3 四类工具
- FunctionTool
- QueryEngineTool
- ToolSpecs
- Utility Tools

### 41.4 三类 agent
- Function Calling Agent
- ReAct Agent
- Advanced Custom Agent

### 41.5 Workflow 的两个关键词
- Event
- Context

### 41.6 Multi-agent 的两个关键词
- root_agent
- handoff

---

## 42. 对你当前阶段的学习建议

如果你现在时间有限，不需要把所有 API 都背下来。  
更重要的是抓住三件事：

### 42.1 建立地图感
知道 LlamaIndex 分成哪几层，各自做什么。

### 42.2 理解共性思想
比如：

- RAG 管线
- QueryEngine
- Tool calling
- Agent 决策
- handoff
- workflow
- shared state

这些思想是跨框架通用的。

### 42.3 保留迁移能力
以后即使不用 LlamaIndex，用别的框架，你也能迁移过去。

---

## 43. 最后一句总结

**LlamaIndex 的本质，不只是“做问答”，而是“围绕你的数据，把检索、工具、agent 和 workflow 组织成一个完整系统”。**
