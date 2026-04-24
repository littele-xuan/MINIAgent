# MINIAgent v5：MCP API + Langfuse + GenericAgent Core

本项目已重构为一个模块化 Agent 底层平台：`v0/` 保留原始实现，`v1/` 保留 LangGraph 实验实现，新核心位于 `src/miniagent/`。

这一版默认读取你的真实 LLM 环境变量：

```bash
MCP_API_BASE
MCP_API_KEY
MCP_MODEL
```

`OPENAI_API_KEY / OPENAI_MODEL / OPENAI_BASE_URL` 仍作为兼容回退，但默认入口和测试都会优先使用 `MCP_*`。

## 快速运行

```bash
pip install -e .
export MCP_API_BASE="https://your-openai-compatible-endpoint/v1"
export MCP_API_KEY="..."
export MCP_MODEL="..."
python agent.py "hello"
```

也可以把 key 放到项目根目录 `.env`：

```env
MCP_API_BASE=https://your-openai-compatible-endpoint/v1
MCP_API_KEY=...
MCP_MODEL=...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

## 根目录入口

```bash
python agent.py "请介绍当前 MINIAgent 的结构"   # 单次运行
python agent.py                                 # 交互模式
printf "hello" | python agent.py                # 管道输入
python agent.py --doctor                        # 配置检查，不调用 LLM
python run_all_tests.py                         # 一键测试
python run_all_tests.py --require-real           # 强制真实 LLM Agent 测试
python run_real_tests.py                         # 只跑真实 LLM Agent 测试
```

## 一键测试

```bash
python run_all_tests.py
```

- 没有 `MCP_API_KEY / MCP_MODEL`：运行离线测试，跳过真实 API 测试。
- 有 `MCP_API_KEY / MCP_MODEL`：离线测试 + `tests_real/` 真实 Agent 测试。
- 建议同时设置 `MCP_API_BASE`，否则 OpenAI SDK 会使用默认 endpoint。
- 强制真实测试：`python run_all_tests.py --require-real`
- 只跑真实测试：`python run_real_tests.py` 或 `python run_all_tests.py --real-only --require-real`
- 跳过真实测试：`python run_all_tests.py --skip-real`

## LLM API 模式

`config/agent.yaml` 中默认：

```yaml
llm:
  api_mode: chat_completions
```

这是为了兼容大多数 `MCP_API_BASE` 指向的 OpenAI-compatible `/v1/chat/completions` 网关。可选值：

- `chat_completions`：默认，适合 MCP/OpenAI-compatible 网关。
- `responses`：使用 OpenAI Responses API。
- `auto`：OpenAI 官方 endpoint 优先 Responses，自定义 base url 优先 Chat Completions。

## 新核心

- `core/`：GenericAgentCore、AgentLoop、状态与结果对象
- `llm/`：OpenAI SDK 兼容客户端，支持 Chat Completions / Responses，默认 Langfuse tracing
- `tools/`：文件、执行、上下文、记忆工具
- `context/`：working checkpoint、turn summary、上下文压缩
- `memory/`：L0/L1/L2/skills/sessions/pending 文件记忆
- `runtime/`：配置、.env、诊断、工作区路径安全、JSONL 日志

如果没有 Langfuse key，会自动退回官方 OpenAI SDK。
