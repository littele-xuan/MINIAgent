# MINIAgent v2 Architecture

## 1. 设计定位

MINIAgent v2 是一个轻量 GenericAgent Core。它不是 LangGraph 包装器，也不是多模型兼容层，而是一个便于继续研究和开发的底层 agent 平台。

```text
GenericAgentCore
├── OpenAIResponsesClient
├── ToolRegistry
├── ContextManager
├── FileMemoryStore
└── Workspace
```

## 2. Agent loop

`AgentLoop` 使用 OpenAI Responses API 的 tool calling 流程：

```text
user packet
  ↓
responses.create(... tools=...)
  ↓
function_call items
  ↓
ToolRegistry.dispatch
  ↓
function_call_output items
  ↓
responses.create(...)
  ↓
final message
```

每轮写入：

- `workspace/logs/<session_id>.jsonl`
- `memory/sessions/<session_id>.jsonl`
- tool events
- usage 信息

## 3. 工具系统

默认工具：

- 文件：`list_dir`, `search_files`, `file_read`, `read_many_files`, `file_write`, `file_patch`, `grep_text`
- 执行：`shell_run`, `python_run`
- 上下文：`update_working_checkpoint`, `ask_user`
- 记忆：`memory_recall`, `memory_propose_update`, `memory_commit_update`

所有工具实现 `BaseTool`，由 `ToolRegistry.openai_schemas()` 转成 OpenAI function tools。

## 4. 上下文管理

`ContextManager` 组织：

- 用户当前任务
- 长期记忆召回结果
- working checkpoint
- 最近 turn summaries
- 执行指令

长任务中旧 turn 进入 summaries；大工具输出由 `ContextCompactor` 截断。

## 5. 记忆系统

```text
memory/
├── l0_policy.md
├── l1_index.md
├── l2_facts.md
├── skills/
├── sessions/
└── pending/
```

写入流程：

```text
memory_propose_update
  ↓
MemoryPolicy.validate
  ↓
pending/memory_proposals.jsonl
  ↓
memory_commit_update
  ↓
l2_facts.md or skills/*.md + l1_index.md
```

强规则：没有 evidence，不写 durable memory。

## 6. Langfuse

配置好以下变量后，`OpenAIResponsesClient` 优先使用 `langfuse.openai` 作为 OpenAI SDK drop-in：

```bash
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

其他 agent step 同时写本地 JSONL 日志。

## 7. Root entry and tests

The root `agent.py` now supports three safe modes:

```bash
python agent.py "hello"       # one-shot run
python agent.py               # interactive REPL in a TTY
printf "hello" | python agent.py
python agent.py --doctor      # local diagnostics, no LLM call
```

The old blocking behavior came from calling `sys.stdin.read()` in a TTY; the new entrypoint only reads stdin when stdin is not a TTY.

`run_all_tests.py` is the main root test runner. It runs compile checks, diagnostics, offline tool/context/memory/loop checks, and then real OpenAI API tests if `OPENAI_API_KEY` is configured.
