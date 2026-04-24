# MINIAgent v5: MCP API + Langfuse + GenericAgent Core

This project keeps the original code in `v0/`, the LangGraph experiment in
`v1/`, and the new modular GenericAgentCore runtime in `src/miniagent/`.

The new default real LLM configuration uses your MCP/OpenAI-compatible variables:

```bash
MCP_API_BASE
MCP_API_KEY
MCP_MODEL
```

`OPENAI_API_KEY`, `OPENAI_MODEL`, and `OPENAI_BASE_URL` remain compatibility
fallbacks, but `MCP_*` is preferred by default.

## Quick start

```bash
pip install -e .
export MCP_API_BASE="https://your-openai-compatible-endpoint/v1"
export MCP_API_KEY="..."
export MCP_MODEL="..."
python agent.py "hello"
```

Or create a project `.env` file:

```env
MCP_API_BASE=https://your-openai-compatible-endpoint/v1
MCP_API_KEY=...
MCP_MODEL=...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

## Root commands

```bash
python agent.py "describe the current architecture"  # single task
python agent.py                                      # interactive REPL
printf "hello" | python agent.py                     # stdin pipe
python agent.py --doctor                             # diagnostics only
python run_all_tests.py                              # offline + real if env exists
python run_all_tests.py --require-real                # force real agent tests
python run_real_tests.py                              # real agent tests only
```

## Test behavior

`run_all_tests.py` always runs compile + offline tests.  It runs `tests_real/`
only when `MCP_API_KEY` and `MCP_MODEL` are present.  `MCP_API_BASE` is strongly
recommended for your endpoint.

The real tests are executable scripts, not pytest mocks.  They exercise real LLM
calls, tool calling, file patching, code execution, memory recall, context
management, and an end-to-end bug fix.

## LLM API mode

`config/agent.yaml` defaults to:

```yaml
llm:
  api_mode: chat_completions
```

This maximizes compatibility with MCP/OpenAI-compatible `/v1/chat/completions`
gateways.  Supported values:

- `chat_completions`: default for MCP gateways.
- `responses`: native OpenAI Responses API.
- `auto`: OpenAI official hosts use Responses first; custom base URLs use Chat Completions first.

## Core structure

```text
src/miniagent/
├── core/       # GenericAgentCore, AgentLoop, state, results
├── llm/        # OpenAI SDK compatible client, Langfuse tracing
├── tools/      # files, shell/python, context, memory tools
├── context/    # working checkpoint, turn summaries, compaction
├── memory/     # L0/L1/L2/skills/sessions/pending file memory
└── runtime/    # config, .env, diagnostics, workspace safety, JSONL logs
```
