# Smolagents 笔记

## 一、这一章在讲什么

`smolagents` 是 Hugging Face 的轻量级 Agent 框架，主打 **简单、抽象少、代码优先（code-first）**。Unit 2.1 整章的目标，不是让你死记很多 API，而是带你建立一套完整的 Agent 框架认识：

- 为什么选 `smolagents`
- Agent 的动作可以怎么表达
- Tool 是什么、怎么定义
- 怎么给 Agent 接入检索能力
- 怎么做多 Agent 协作
- 怎么让 Agent 处理图像和浏览网页

你可以把整章压成一句话：

> **smolagents = 用一个轻量级的多步 Agent 框架，把模型、工具、检索、多 Agent、视觉能力拼起来。**

---

## 二、smolagents 的核心定位

### 1. 它是什么

`smolagents` 是 Hugging Face 提供的一个轻量级 Agent 框架，用来构建能够：

- 搜索信息
- 执行代码
- 调用工具
- 访问网页
- 组合多个 Agent
- 处理图像

的智能体。

### 2. 它的核心优势

这一章强调了 4 个关键词：

#### （1）Simplicity
- 代码复杂度低
- 抽象层少
- 容易理解、上手、扩展

#### （2）Flexible LLM Support
- 不绑定某个固定大模型
- 可以接入不同来源的模型/服务
- 框架本身不是模型，而是 Agent 外壳

#### （3）Code-First Approach
- 更强调让 Agent **直接用代码表达动作**
- 少一层“JSON 解析 → 再转代码”的过程

#### （4）HF Hub Integration
- 能和 Hugging Face 生态很好地结合
- 工具、Spaces、Hub 资源都可以复用

### 3. 什么时候适合用

适合：
- 想快速实验
- 想搭轻量级 Agent
- 应用逻辑比较直接
- 想少写复杂配置

不一定最适合：
- 特别重型、企业级、复杂编排系统
- 需要大量工程治理和复杂工作流的平台型项目

---

## 三、整章最重要的总公式

这一章最值得记住的公式是：

> **Agent = 模型 + 多步推理循环 + Tools + 记忆/日志 +（可选）检索、多 Agent、视觉能力**

其中：

- **模型**：负责思考和生成下一步动作
- **多步推理循环**：让 Agent 一步一步完成任务
- **Tools**：让 Agent 真正“做事”
- **记忆/日志**：保存历史动作和观察结果
- **检索**：让 Agent 去查外部信息
- **多 Agent**：让复杂任务拆分协作
- **视觉能力**：让 Agent 不只看文字，还能看图

---

## 四、底层运行骨架：MultiStepAgent

`smolagents` 背后的核心思想是 **多步 Agent**。

可以把它理解成这样的循环：

1. 读任务
2. 思考当前该做什么
3. 调用一个工具 / 执行动作
4. 得到观察结果
5. 把结果记到日志里
6. 再继续下一步
7. 直到得到最终答案

这一点非常重要，因为后面学到的：

- `CodeAgent`
- `ToolCallingAgent`
- Retrieval Agent
- Multi-Agent
- Vision Agent

本质上都是在这个多步循环上“加能力”。

---

## 五、CodeAgent：smolagents 的默认主角

### 1. 它是什么

`CodeAgent` 是 `smolagents` 里默认、也是最核心的 Agent 类型。

它的特点是：

> **不是输出 JSON 动作说明，而是直接输出 Python 代码动作。**

例如，它更像这样：

```python
result = web_search("best party music")
final_answer(result)
```

而不是这样：

```json
{
  "tool": "web_search",
  "arguments": {"query": "best party music"}
}
```

### 2. 为什么强调 CodeAgent

课程里强调了 4 个优势：

#### （1）Composability
代码更容易组合多个步骤。

#### （2）Object Management
代码更容易处理复杂对象，比如图片、列表、字典等。

#### （3）Generality
理论上任何可计算任务都能表达。

#### （4）Natural for LLMs
高质量代码本来就在大模型训练数据中很多，所以模型更熟悉这种表达方式。

### 3. CodeAgent 的运行流程

`CodeAgent.run()` 可以理解成：

1. 把 system prompt 和任务写入日志
2. 把日志整理成模型能读的消息
3. 模型生成一段代码
4. 系统执行这段代码
5. 把执行结果写回日志
6. 继续下一轮，直到得到 `final_answer`

### 4. 你要记住的本质

> **CodeAgent = 用代码来表达动作的多步 Agent。**

---

## 六、ToolCallingAgent：另一种动作表达方式

### 1. 它是什么

`ToolCallingAgent` 是 `smolagents` 支持的第二类 Agent。

它不是直接生成可执行 Python 代码，而是生成 **结构化工具调用**，通常表现为 JSON 风格的数据结构。

### 2. 和 CodeAgent 的区别

一句话概括：

- `CodeAgent`：**写代码**
- `ToolCallingAgent`：**写 JSON / 结构化工具调用**

举例：

#### CodeAgent
```python
for query in ["best catering", "party ideas"]:
    print(web_search(query))
```

#### ToolCallingAgent
```json
[
  {"name": "web_search", "arguments": "best catering"},
  {"name": "web_search", "arguments": "party ideas"}
]
```

### 3. 什么时候更适合它

更适合：
- 简单系统
- 步骤不多
- 不太需要复杂变量传递
- 不太需要复杂工具组合

### 4. arguments 是什么

`arguments` 就是：

> **调用工具时传进去的参数**

例如：

```python
get_weather(city="Chongqing", day="tomorrow")
```

对应结构化调用时，`arguments` 可以理解成：

```json
{
  "city": "Chongqing",
  "day": "tomorrow"
}
```

### 5. 你要记住的本质

> **ToolCallingAgent = 用结构化工具调用来表达动作的多步 Agent。**

---

## 七、Tools：Agent 真正“做事”的手

### 1. Tool 是什么

在 `smolagents` 里，Tool 可以理解成：

> **一个可被 LLM 调用的函数接口**

Tool 至少要说明 4 件事：

- `name`：工具名
- `description`：做什么
- `inputs / arguments`：输入参数说明
- `output_type`：输出是什么

没有 Tool，Agent 只能“说”；有了 Tool，Agent 才能“做”。

---

### 2. 定义 Tool 的两种方式

#### 方式一：`@tool`
适合简单工具，也是推荐方式。

你写一个普通 Python 函数，再用 `@tool` 包装即可。

关键点：
- 函数名要清楚
- 参数和返回值要写类型标注
- docstring 尤其 `Args:` 要写好

#### 方式二：继承 `Tool`
适合复杂工具。

需要写：
- `name`
- `description`
- `inputs`
- `output_type`
- `forward`

这里的 `forward`，本质上就是：

> **这个工具真正干活的地方**

可以记成一句：

- `@tool` 写法里，函数体就是工具逻辑
- `Tool` 子类写法里，`forward` 就是工具逻辑

---

### 3. 默认工具箱

`smolagents` 自带一些常见工具，例如：

- `PythonInterpreterTool`
- `FinalAnswerTool`
- `UserInputTool`
- `DuckDuckGoSearchTool`
- `GoogleSearchTool`
- `VisitWebpageTool`

也就是说，很多常见能力并不需要你从零写。

---

### 4. 工具的复用生态

这部分很重要，说明 Tool 不只是自己写。

#### （1）`push_to_hub()`
把自定义工具上传到 Hugging Face Hub。

#### （2）`load_tool()`
加载别人上传到 Hub 的工具。

#### （3）`Tool.from_space()`
把一个 Hugging Face Space 变成 Tool。

它的本质是：

> **把一个已经部署好的 Gradio 应用，当成 Agent 可调用的远程工具。**

你原来是自己打开网页点按钮；现在 Agent 可以像调用函数一样去调用它。

#### （4）`Tool.from_langchain()`
复用 LangChain 工具。

#### （5）`ToolCollection.from_mcp()`
从 MCP server 批量导入工具。

---

## 八、MCP 是什么

MCP = **Model Context Protocol**。

你可以把它理解成：

> **让 Agent 和外部工具/数据源“说同一种语言”的协议。**

它不是某个具体工具，而是一套标准。

### MCP 的直觉理解

- `@tool`：你自己本地手写一个工具
- `Tool.from_space()`：把一个在线 Space 接进来
- `ToolCollection.from_mcp()`：去连接某个 MCP server，把它提供的一整批工具接进来

MCP 的意义在于：
- 一次接入协议
- 就能复用大量已有的服务端工具

但也要注意安全，尤其是信任远程代码和本地执行相关设置。

---

## 九、Retrieval Agents：让 Agent 学会“查”

### 1. 为什么需要 Retrieval

如果 Agent 只靠模型参数里的知识，就会遇到问题：

- 知识可能过时
- 专业知识不够全
- 单靠记忆不够稳

所以要给 Agent 接入检索能力。

### 2. 什么是传统 RAG

传统 RAG 的基本流程：

1. 用户提问
2. 检索相关内容
3. 把检索结果和问题一起给模型
4. 模型生成回答

问题是：
- 往往只检索一次
- 过于依赖原问题的直接相似度
- 可能漏掉真正重要的信息

### 3. 什么是 Agentic RAG

Agentic RAG 更进一步：

> **Agent 不只是被动接收一次检索结果，而是能主动控制检索过程。**

它可以：
- 自己改写查询
- 评价当前结果好不好
- 再检索一轮
- 综合多个来源
- 决定何时停止

### 4. 这一节的重要策略

课程里提到一些增强检索能力的方法：

- Query Reformulation：改写查询
- Query Decomposition：拆解子问题
- Query Expansion：扩展多种表达
- Reranking：重排序
- Multi-Step Retrieval：多步检索
- Source Integration：多源整合
- Result Validation：结果验证

### 5. 自定义知识库

不仅能搜索网页，也可以接自己的知识库。

常见思路：

1. 准备文档
2. 文档切块
3. 建 retriever
4. 封装成 tool
5. 交给 agent 使用

页面里举了 `BM25Retriever` 和文档切块的例子。

### 6. 你要记住的本质

> **Retrieval Agent = Agent + 检索工具 + 外部知识入口**

---

## 十、Multi-Agent Systems：让复杂任务学会分工

### 1. 为什么要多 Agent

不是为了炫技，而是因为复杂任务往往会出现：

- 上下文太长
- 单个 Agent 负担过重
- 任务类型差异大
- token、延迟、成本上涨
- 系统更容易乱

所以复杂任务不一定要靠“一个超级 Agent”全做完。

### 2. 典型结构

多 Agent 最经典的结构是：

- **Manager / Orchestrator Agent**：负责规划、调度、整合
- **Specialized Agents**：负责某类子任务

例如：
- 搜索 Agent
- 代码执行 Agent
- 检索 Agent
- 画图 Agent

### 3. 课程例子里的核心思想

课程展示了：
- `web_agent`：负责查网页、访问网页、做相关计算
- `manager_agent`：负责任务总控、整合、绘图、最终输出

二者的协作，不一定是你手写 `manager_agent.run(web_agent.run(...))`。
更常见的是：

> **manager 在运行过程中，把合适的子任务委托给 `managed_agents` 中的子 Agent。**

### 4. `managed_agents` 是什么

`managed_agents=[web_agent]` 的含义就是：

> 当前这个 manager agent 可以调用这些子 agent。

### 5. `planning_interval` 是什么

`planning_interval=n` 表示：

> **Agent 每隔 n 步插入一次“规划步骤”。**

也就是中途停下来想一想：
- 当前做到哪了
- 后面该先做什么
- 是否要调整计划

### 6. 多 Agent 的收益

- 每个 Agent 更专注
- 降低单个 Agent 的上下文压力
- 减少 token 消耗
- 降低延迟和成本
- 系统更容易维护和扩展

### 7. 你要记住的本质

> **Multi-Agent = manager + specialists + 分工协作 +（可选）结果验证**

---

## 十一、Vision Agents：让 Agent 不只看文字

### 1. Vision Agent 是什么

Vision Agent 就是：

> **把图像也放进 Agent 的推理循环里。**

也就是说，Agent 的输入不再只有文字，还能有图片。

### 2. 两种图像输入方式

#### （1）静态输入
在 `run()` 一开始就把图片传进去。

这时图片会作为任务输入的一部分。

#### （2）动态输入
在执行过程中不断获取图像，比如：
- 浏览网页截图
- 把截图写入日志
- 下一步继续基于这些截图推理

### 3. 关键日志概念

动态视觉检索里，一个重要点是：

- 任务开始前的图片：可以作为 `task_images`
- 执行过程中得到的截图：可以保存为 `observation_images`

所以图像也可以进入 Agent 的 memory。

### 4. Browser Agent

这一节还把 Vision Agent 和浏览器自动化结合起来，形成了 Browser Agent。

典型思路：

- 搜索网页
- 打开网页
- 在网页中查找文本
- 操作浏览器
- 截图
- 把截图交给 VLM 继续推理

这本质上是：

> **浏览工具 + 视觉模型 + 多步推理**

### 5. step callback 的作用

课程里的截图函数会被作为 `step_callbacks` 传给 agent。

作用是：

> **在每一步结束时做额外操作，比如自动截图，并把截图写入 memory。**

### 6. 你要记住的本质

> **Vision Agent = 文本推理 + 图像输入 + 动态视觉记忆**

---

## 十二、几个常见易混点

### 1. `provider` 是什么

这个词要看上下文。

#### 在模型里
如：

```python
InferenceClientModel("Qwen/...", provider="together")
```

这里的 `provider` 指：

> **模型推理服务提供方**

也就是同一个模型，可以通过不同服务商去跑。

#### 在工具里
如搜索工具里的 `provider`

这里更像：

> **搜索/工具的后端服务来源**

所以同名参数，语义可能不同。

---

### 2. `forward` 为什么存在

当你用 `Tool` 子类方式定义工具时，`forward` 就是工具真正执行逻辑的地方。

一句话：

> **`forward` = 复杂工具写法中的“函数体”**

---

### 3. `Tool.from_space()` 到底是啥

它不是把代码复制到本地重写，而是：

> **把一个已经部署好的 Hugging Face Space，当成 Agent 可调用的远程工具来用。**

---

### 4. `visualize()` 是干嘛的

它一般是用来：

> **可视化 agent 的结构树**

例如：
- manager 自己
- 它有哪些 tools
- 它有哪些 managed agents

它不是任务执行逻辑本身，而是调试/展示结构用。

---

## 十三、这一章真正要掌握什么

### 必须掌握

1. `smolagents` 是轻量级、代码优先的 Agent 框架
2. `CodeAgent` 和 `ToolCallingAgent` 的区别
3. Tool 是 Agent 能力扩展的基本单位
4. Retrieval 是给 Agent 接外部知识
5. Multi-Agent 是复杂任务分工
6. Vision Agent 是把图像接入推理循环
7. 整体都建立在多步 Agent 工作流上

### 看懂即可

这些现在不需要死背：
- `planning_interval`
- `managed_agents`
- `final_answer_checks`
- `visualize()`
- `provider`
- `Tool.from_space()`
- `ToolCollection.from_mcp()`
- 具体模型封装类名

知道它们大概干什么，后面用到再查即可。

---

## 十四、建议记忆的简单代码模板

这一块是适合你单独记忆的“最小代码骨架”。

### 1. 最简单的 CodeAgent

```python
from smolagents import CodeAgent, DuckDuckGoSearchTool, InferenceClientModel

agent = CodeAgent(
    tools=[DuckDuckGoSearchTool()],
    model=InferenceClientModel()
)

agent.run("Search for the best music recommendations for a party.")
```

### 2. 最简单的 ToolCallingAgent

```python
from smolagents import ToolCallingAgent, WebSearchTool, InferenceClientModel

agent = ToolCallingAgent(
    tools=[WebSearchTool()],
    model=InferenceClientModel()
)

agent.run("Search for party ideas.")
```

### 3. 用 `@tool` 定义简单工具

```python
from smolagents import tool

@tool
def catering_service_tool(query: str) -> str:
    """
    Return the highest-rated catering service.

    Args:
        query: Search term for finding catering services.
    """
    services = {
        "Gotham Catering Co.": 4.9,
        "Wayne Manor Catering": 4.8,
        "Gotham City Events": 4.7,
    }
    return max(services, key=services.get)
```

### 4. 用 `Tool` 子类定义复杂工具

```python
from smolagents import Tool

class SuperheroPartyThemeTool(Tool):
    name = "superhero_party_theme_generator"
    description = "Suggest superhero-themed party ideas."
    inputs = {
        "category": {
            "type": "string",
            "description": "Type of superhero party theme"
        }
    }
    output_type = "string"

    def forward(self, category: str):
        themes = {
            "classic heroes": "Justice League Gala",
            "villain masquerade": "Gotham Rogues' Ball",
            "futuristic gotham": "Neo-Gotham Night"
        }
        return themes.get(category.lower(), "Theme not found")
```

### 5. 把 HF Space 变成 Tool

```python
from smolagents import Tool

image_generation_tool = Tool.from_space(
    "black-forest-labs/FLUX.1-schnell",
    name="image_generator",
    description="Generate an image from a prompt"
)
```

### 6. Retrieval 风格的最小例子

```python
from smolagents import CodeAgent, DuckDuckGoSearchTool, InferenceClientModel

search_tool = DuckDuckGoSearchTool()
model = InferenceClientModel()

agent = CodeAgent(model=model, tools=[search_tool])
agent.run("Search for luxury superhero-themed party ideas.")
```

### 7. Multi-Agent 最小骨架

```python
from smolagents import CodeAgent, GoogleSearchTool, VisitWebpageTool, InferenceClientModel

web_agent = CodeAgent(
    tools=[GoogleSearchTool(), VisitWebpageTool()],
    model=InferenceClientModel(),
    name="web_agent",
    description="Browses the web to find information"
)

manager_agent = CodeAgent(
    tools=[],
    model=InferenceClientModel(),
    managed_agents=[web_agent],
    planning_interval=5
)
```

### 8. Vision Agent 的最小骨架

```python
from smolagents import CodeAgent, OpenAIServerModel

model = OpenAIServerModel(model_id="gpt-4o")
agent = CodeAgent(tools=[], model=model)

agent.run("Describe the person in the image.", images=images)
```

---

## 十五、整章一句话总总结

> **smolagents 是 Hugging Face 的轻量级、代码优先 Agent 框架。它用多步工作流组织模型思考和工具执行；用 CodeAgent 或 ToolCallingAgent 表达动作；用 Tools 扩展能力；用 Retrieval 接外部知识；用 Multi-Agent 处理复杂任务分工；再进一步用 Vision/Browser agents 把图像和网页交互纳入推理循环。**

---

## 十六、最后的复习建议

如果你接下来要进入新篇章，这一章请至少确保你能脱口而出下面 6 句：

1. `smolagents` 是轻量级、代码优先的 Agent 框架
2. `CodeAgent` 写代码，`ToolCallingAgent` 写结构化工具调用
3. Tool 是 Agent 真正“做事”的手
4. Retrieval 是让 Agent 学会“查”
5. Multi-Agent 是让复杂任务学会“分工”
6. Vision Agent 是让 Agent 不只看文字，也能看图

如果这 6 句你已经能讲顺，说明这一章你已经掌握住主干了。
