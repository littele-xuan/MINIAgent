# MINIAgent

一个最小化、自包含的通用 Agent 循环 —— 以实现为导向的参考项目，精确展示各组件如何组装，不依赖任何图框架作为关键路径。

项目名 **MINIAgent** 有意体现其定位：实现了完整的 GenericAgent 模式（LLM → 工具调用 → 观察 → 重复），配备真实的生产级配套组件，同时保持每一层在一次阅读内都可完全理解。

---

## 设计理念

大多数 Agent 框架把循环隐藏在抽象层之后。MINIAgent 把它暴露出来。核心是 `src/miniagent/core/loop.py` 中的一个 `while` 循环（约 120 行）。每个子系统 —— LLM 客户端、工具注册表、上下文管理器、记忆存储 —— 都是一个普通 Python dataclass，可以独立替换、扩展或测试。

---

## 快速启动

```bash
pip install -e .
export MCP_API_BASE="https://your-openai-compatible-endpoint/v1"
export MCP_API_KEY="..."
export MCP_MODEL="..."
python agent.py "hello"
```

或在项目根目录创建 `.env` 文件：

```env
MCP_API_BASE=https://your-openai-compatible-endpoint/v1
MCP_API_KEY=...
MCP_MODEL=...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

### 入口命令

```bash
python agent.py "描述当前的 MINIAgent 架构"    # 单次任务
python agent.py                                # 交互 REPL
printf "hello" | python agent.py              # 管道输入
python agent.py --doctor                      # 仅做配置诊断，不调用 LLM
python run_all_tests.py                       # 离线 + 真实 LLM 测试（有环境变量时）
python run_all_tests.py --require-real        # 强制真实 LLM 测试
python run_real_tests.py                      # 只跑真实 LLM 测试
```

---

## 架构

### 目录结构

```text
MINIAgent/
├── src/miniagent/          # 可安装包（src layout）
│   ├── core/               # agent 循环、状态、结果对象、错误类型
│   ├── llm/                # OpenAI SDK 客户端，Chat Completions / Responses API
│   ├── tools/              # 文件、shell/python、上下文、记忆工具 + 注册表
│   ├── context/            # working checkpoint、轮次摘要、压缩
│   ├── memory/             # L0–L2 基于文件的长期记忆
│   └── runtime/            # 配置加载、.env、工作区安全、JSONL 日志
├── config/agent.yaml       # 运行时配置（模型、API 模式、上下文限制）
├── memory/                 # 持久化记忆文件（markdown）
├── workspace/              # 每次运行的工作目录 + JSONL 运行日志
├── tests_offline/          # 快速离线测试，覆盖每个子系统
├── tests_real/             # 真实 LLM Agent 测试（需要 MCP_API_KEY）
├── v0/                     # 原型代码（保留作为参考）
└── v1/                     # LangGraph 实验（保留作为对比）
```

### 核心 Agent 循环（`src/miniagent/core/loop.py`）

循环是一个普通的 `while`，没有图、没有节点回调、没有编译后的状态机：

```
start_packet()
  ↓
[第 N 轮]
  llm.create_response(system, messages, tools)
    ↓ 只有文本 → final_answer，结束
    ↓ 工具调用 → 执行工具 → 追加结果 → after_turn()
    ↓ 空响应  → 注入包含任务上下文 + working_checkpoint 的重试提示
  重复
```

关键设计点：
- `AgentState`（dataclass，`slots=True`）携带单次运行的 `input_items`、`tool_events`、`usage`、`exit_reason`，每次 `agent.run()` 调用时全新创建，循环外不可变更。
- `AgentResult`（frozen dataclass）是不可变的返回值：`final_text`、`turns`、`exit_reason`、`session_id`、`usage`、`tool_events`。
- 空响应恢复：最多 `max_empty_response_retries=5` 次恢复轮次，每次注入包含当前任务和最后一个 `working_checkpoint` 的 user 消息。

### LLM 客户端（`src/miniagent/llm/openai_client.py`）

**运行时：** `openai` Python SDK（≥1.0）—— 与 OpenAI、Azure、Groq、Together、任何 OpenAI 兼容网关兼容的同一 SDK。

**两种 API 模式**（在 `config/agent.yaml` 中选择）：

| 模式 | Endpoint | 使用场景 |
|---|---|---|
| `chat_completions` | `/v1/chat/completions` | 默认，兼容所有 MCP 网关 |
| `responses` | `/v1/responses` | 原生 OpenAI Responses API 实验 |
| `auto` | 根据 hostname 自动检测 | `api.openai.com` → `responses`；自定义 → `chat_completions` |

**消息转换：** `_input_items_to_chat_messages()` 将内部 `input_items` 列表（Responses API 格式的 `function_call` / `function_call_output` typed dict）转换为标准 Chat Completions 消息格式，使系统其他部分与 API 模式无关。

**可观测性：** 当 `LANGFUSE_PUBLIC_KEY` 和 `LANGFUSE_SECRET_KEY` 已设置时，`langfuse.openai` 直接替换 `OpenAI` 客户端，每次 LLM 调用自动追踪，无需修改代码。未安装 Langfuse 时静默回退到原生 `openai.OpenAI`。

**回退链：** `MCP_API_KEY` → `OPENAI_API_KEY`；`MCP_MODEL` → `OPENAI_MODEL`；`MCP_API_BASE` → `OPENAI_BASE_URL`。

### 工具系统（`src/miniagent/tools/`）

每个工具继承 `BaseTool`：

```python
class BaseTool:
    name: str           # LLM 可见的函数名
    description: str    # LLM 端描述
    parameters: dict    # JSON Schema（OpenAI function-calling 格式）

    def run(self, args: dict, ctx: ToolContext) -> ToolResult: ...
```

`ToolRegistry` 持有 `dict[str, BaseTool]`，通过 `openai_schemas()` 生成 OpenAI function-calling schema，通过 `dispatch()` 路由调用。无装饰器魔法，无元类，只有 `register()` 和 `dispatch()`。

**内置工具：**

| 工具 | 功能 |
|---|---|
| `list_dir` | 带深度控制的目录列表 |
| `search_files` | Glob 文件搜索 |
| `file_read` | 读取文件（可选行范围） |
| `read_many_files` | 单次批量读取多个文件 |
| `file_write` | 写入或覆盖文件 |
| `file_patch` | 精确补丁：`old_str` → `new_str`，不唯一则失败 |
| `grep_text` | 在目录中进行正则 grep |
| `shell_run` | 在工作区执行 shell 命令 |
| `python_run` | 通过 `subprocess` 执行 Python 片段 |
| `update_working_checkpoint` | 向 `metadata` 写入持久化任务检查点 |
| `memory_recall` | 查询长期记忆 |
| `memory_propose_update` | 提议一次记忆写入（暂存） |
| `memory_commit_update` | 将暂存的记忆写入提交到磁盘 |
| `ask_user` | 阻塞并向用户提问 |

`ToolContext` 携带 `workspace`、`memory`、`session_id` 和 `metadata`，注入每次工具调用。工具从不导入循环或 agent，只能访问自己的 context。

### 上下文管理（`src/miniagent/context/`）

上下文管理在轮次间有状态，但每次运行重置：

- **`ContextPacket`**：运行开始时组装的快照，包含编译后的系统提示、面向用户的 packet（任务 + 长期记忆召回 + working checkpoint + 最近轮次摘要）及元数据。
- **`HeuristicTurnSummarizer`**：每轮后生成紧凑摘要 dict：`{turn, user_intent, assistant_action, tools, next_goal}`，文本截断到 500 字符。
- **`ContextCompactor`**：保留最近 `keep_recent_summaries=8` 轮摘要，工具输出截断到 `max_tool_output_chars=12000` 字符，防止大文件读取时上下文爆炸。
- **`working_checkpoint`**：Agent 通过 `update_working_checkpoint` 写入的自由文本字符串，作为 `### Working checkpoint` 注入后续每轮的 user packet，为模型在长多轮任务中提供稳定的任务锚点。

### 记忆（`src/miniagent/memory/`）

记忆完全基于文件，无向量数据库、无 embedding、无外部服务。

**三层结构：**

| 层 | 文件 | 内容 |
|---|---|---|
| L0 | `memory/l0_policy.md` | 静态策略：记什么、不记什么 |
| L1 | `memory/l1_index.md` | 记忆条目运行索引（ID、标签、摘要） |
| L2 | `memory/l2_facts.md` | 已提交事实的完整内容 |

**技能记忆：** `memory/skills/*.md` —— 过程性知识（如何重构代码、如何编辑文件等），Agent 在开始相关任务时可召回。

**会话事件：** `memory/sessions/<session_id>.jsonl` —— 每会话事件日志，用于调试。

**暂存写入：** `memory/pending/` —— 尚未提交的暂存记忆提案。

**召回**（`FileMemoryStore.recall()`）：扫描 L1 索引进行关键词匹配，加载匹配的 L2 条目，裁剪到 token 预算。不使用 embedding 相似度 —— 有意保持简单和可审计。

### 运行时（`src/miniagent/runtime/`）

- **`AgentConfig`**（`config.py`）：通过 PyYAML 加载 `config/agent.yaml`（有纯 Python 回退解析器）。所有设置都有文档化的默认值。
- **`Workspace`**（`workspace.py`）：带路径安全检查的工作目录封装，工具的文件操作被约束在工作区根目录内。
- **`JsonlRunLogger`**（`logging.py`）：将结构化 JSONL 运行日志写入 `workspace/logs/<session_id>.jsonl`，每条记录格式为 `{event, timestamp, ...payload}`。
- **`load_dotenv_if_present`**（`env.py`）：从项目根目录加载 `.env`，不硬依赖 `python-dotenv`，不存在时静默跳过。
- **诊断**（`diagnostics.py`）：`python agent.py --doctor` 打印已解析的模型、API key 存在性、base URL 和可观测性状态。

---

## 配置参考（`config/agent.yaml`）

```yaml
name: MINIAgent
max_turns: 40

llm:
  api_mode: chat_completions   # chat_completions | responses | auto
  max_output_tokens: 4096
  request_timeout_seconds: 120
  max_retries: 2
  temperature: null             # null = 模型默认值

context:
  keep_recent_summaries: 8
  max_tool_output_chars: 12000

observability:
  provider: langfuse
  enabled: true
```

---

## 技术栈详情

| 组件 | 库 / 方案 |
|---|---|
| Python | 3.11+，全局使用 `from __future__ import annotations` |
| 数据模型 | `dataclasses` + `slots=True`（零依赖，高性能） |
| LLM API | `openai` ≥1.0（Chat Completions + Responses API） |
| 可观测性 | `langfuse`（可选，`openai.OpenAI` 的无侵入式 wrapper） |
| 配置 | `PyYAML` + 纯 Python 回退解析器 |
| 环境变量 | `python-dotenv`（可选；不存在时静默跳过） |
| 记忆存储 | 纯 Markdown 文件（无数据库、无 embedding） |
| 运行日志 | 每会话 JSONL 文件 |
| 离线测试 | 普通 Python 脚本，无需 pytest |
| 真实 LLM 测试 | 通过 `MCP_API_KEY` / `MCP_MODEL` 发起真实调用 |
| 打包 | `pyproject.toml`，src layout，`pip install -e .` |

---

## 测试

### 离线测试（不需要 LLM）

```bash
python run_all_tests.py
```

覆盖：配置加载、工具注册表、文件/shell 工具、记忆、上下文管理器、使用假 LLM stub 的 agent 循环。

### 真实 LLM 测试

```bash
python run_real_tests.py
```

需要 `MCP_API_KEY` 和 `MCP_MODEL`。覆盖：

| 测试 | 测试内容 |
|---|---|
| `01_llm_smoke.py` | 单次 LLM 调用、usage 元数据、Langfuse 集成 |
| `02_tool_file_patch.py` | 多轮：读取 → 补丁 → 验证 |
| `03_code_run.py` | `python_run` 工具，结果验证 |
| `04_memory_recall.py` | 跨两个会话写入和召回长期记忆 |
| `05_context_long_task.py` | 多文件扫描、`update_working_checkpoint`、文件写入、完整上下文持久化 |
| `06_agent_end_to_end.py` | 端到端 bug 修复：定位 → 补丁 → 验证 |

---

## 环境变量

| 变量 | 用途 | 是否必须 |
|---|---|---|
| `MCP_API_KEY` | LLM API 密钥 | 是 |
| `MCP_MODEL` | 模型标识符（如 `gpt-4o`） | 是 |
| `MCP_API_BASE` | OpenAI 兼容 base URL | 强烈推荐 |
| `OPENAI_API_KEY` | 回退 API 密钥 | 否 |
| `OPENAI_MODEL` | 回退模型 | 否 |
| `OPENAI_BASE_URL` | 回退 base URL | 否 |
| `LANGFUSE_PUBLIC_KEY` | Langfuse 项目公钥 | 否 |
| `LANGFUSE_SECRET_KEY` | Langfuse 项目密钥 | 否 |
| `LANGFUSE_BASE_URL` | Langfuse 主机（默认 cloud.langfuse.com） | 否 |

---

## 版本目录

- `v0/` —— 原型实现，原样保留，用于迁移参考。
- `v1/` —— LangGraph 实验，保留用于对比。展示同一架构的 LangGraph 版本：`StateGraph`、`ToolNode`、`langgraph.store` 用于记忆。

两者都有意保留，方便你 diff 对比两种方案。当前默认运行时（`src/miniagent/`）不使用其中任何一个。

---

## 参考项目

MINIAgent 是一种在多个 Agent 框架中都能找到的模式的专注实现。如需深入了解：

| 项目 | 关注点 |
|---|---|
| **[LangGraph](https://github.com/langchain-ai/langgraph)** | `v1/` 实验所使用的基于图的 Agent 运行时。`StateGraph`、`ToolNode`、`langgraph.store` 和 `MemorySaver` 是 MINIAgent 循环、工具注册表和记忆存储的 LangGraph 对应物。 |
| **[GenericAgent](https://github.com/lsdefine/GenericAgent)** | MINIAgent 命名参考的底层设计模式：带可插拔工具和记忆的通用 观察–规划–行动 循环。 |
| **[smolagents](https://github.com/huggingface/smolagents)** | HuggingFace 的最小 Agent 库，与 MINIAgent 理念相近：保持循环小、工具接口简单、模型无关。使用 `CodeAgent` / `ToolCallingAgent` 分离。 |
| **[Hermes](https://github.com/nousresearch/hermes-agent)** | NousResearch 的 function-calling 微调和提示规范，用于理解工具调用提示和 JSON Schema 函数定义在开源权重模型中的演进。 |
