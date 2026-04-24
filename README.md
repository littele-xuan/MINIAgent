# MINIAgent

A minimal, self-contained generic agent loop — an implementation-focused reference showing exactly how the pieces fit together, without a graph framework in the critical path.

The project is intentionally named **MINIAgent** to reflect its scope: it implements the essential GenericAgent pattern (LLM → tool call → observe → repeat) with real, production-grade plumbing, but keeps every layer inspectable in a single reading.

---

## Philosophy

Most agent frameworks hide the loop inside abstractions. MINIAgent exposes it. The core is a `while` loop in `src/miniagent/core/loop.py` (~120 lines). Every subsystem — LLM client, tool registry, context manager, memory store — is a plain Python dataclass that you can swap, extend, or test in isolation.

---

## Quick start

```bash
pip install -e .
export MCP_API_BASE="https://your-openai-compatible-endpoint/v1"
export MCP_API_KEY="..."
export MCP_MODEL="..."
python agent.py "hello"
```

Or use a `.env` file at the project root:

```env
MCP_API_BASE=https://your-openai-compatible-endpoint/v1
MCP_API_KEY=...
MCP_MODEL=...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

### Entry points

```bash
python agent.py "describe the current architecture"   # single task
python agent.py                                       # interactive REPL
printf "hello" | python agent.py                      # stdin pipe
python agent.py --doctor                              # config diagnostics only
python run_all_tests.py                               # offline + real tests if env is set
python run_all_tests.py --require-real                # force real LLM tests
python run_real_tests.py                              # real agent tests only
```

---

## Architecture

### Directory layout

```text
MINIAgent/
├── src/miniagent/          # installable package (src layout)
│   ├── core/               # agent loop, state, results, error types
│   ├── llm/                # OpenAI SDK client, Chat Completions / Responses API
│   ├── tools/              # file, shell/python, context, memory tools + registry
│   ├── context/            # working checkpoint, turn summaries, compaction
│   ├── memory/             # L0–L2 file-based long-term memory
│   └── runtime/            # config loader, .env, workspace safety, JSONL logs
├── config/agent.yaml       # runtime settings (model, API mode, context limits)
├── memory/                 # persistent memory files (markdown)
├── workspace/              # per-run working directory + JSONL run logs
├── tests_offline/          # fast, no-LLM tests for every subsystem
├── tests_real/             # real LLM agent tests (require MCP_API_KEY)
├── v0/                     # original prototype (preserved for reference)
└── v1/                     # LangGraph experiment (preserved for reference)
```

### Core agent loop (`src/miniagent/core/loop.py`)

The loop is a plain `while` with no graph, no node callbacks, no compiled state machine:

```
start_packet()
  ↓
[turn N]
  llm.create_response(system, messages, tools)
    ↓ text only → final_answer, break
    ↓ tool calls → execute tools → append results → after_turn()
    ↓ empty → inject retry prompt with task context + working checkpoint
  repeat
```

Key design points:
- `AgentState` (dataclass, `slots=True`) carries `input_items`, `tool_events`, `usage`, `exit_reason` for one run. It is created fresh per `agent.run()` call and never mutated outside the loop.
- `AgentResult` (frozen dataclass) is the immutable return value: `final_text`, `turns`, `exit_reason`, `session_id`, `usage`, `tool_events`.
- Empty-response recovery: up to `max_empty_response_retries=5` recovery turns, each injecting a user message that includes the current task and the last `working_checkpoint`.

### LLM client (`src/miniagent/llm/openai_client.py`)

**Runtime:** `openai` Python SDK (≥1.0) — the same SDK that works with OpenAI, Azure, Groq, Together, any OpenAI-compatible gateway.

**Two API modes** (selectable in `config/agent.yaml`):

| Mode | Endpoint | When to use |
|---|---|---|
| `chat_completions` | `/v1/chat/completions` | Default. Works with all MCP gateways. |
| `responses` | `/v1/responses` | Native OpenAI Responses API experiments. |
| `auto` | Detected by hostname | `api.openai.com` → `responses`; custom → `chat_completions`. |

**Message translation:** `_input_items_to_chat_messages()` converts the internal `input_items` list (Responses-API format with `function_call` / `function_call_output` typed dicts) into standard Chat Completions message format. This translation layer is what makes the rest of the system API-mode-agnostic.

**Observability:** When `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set, the `langfuse.openai` drop-in wrapper replaces the `OpenAI` client. Every LLM call is traced automatically, zero code change required. Falls back to plain `openai.OpenAI` silently if Langfuse is not installed.

**Fallback chain:** `MCP_API_KEY` → `OPENAI_API_KEY`; `MCP_MODEL` → `OPENAI_MODEL`; `MCP_API_BASE` → `OPENAI_BASE_URL`. This lets the project work with the standard `OPENAI_*` variables as a compatibility layer for existing setups.

### Tool system (`src/miniagent/tools/`)

Every tool is a subclass of `BaseTool`:

```python
class BaseTool:
    name: str           # function name visible to the LLM
    description: str    # LLM-facing description
    parameters: dict    # JSON Schema (OpenAI function-calling format)

    def run(self, args: dict, ctx: ToolContext) -> ToolResult: ...
```

`ToolRegistry` holds a `dict[str, BaseTool]`, generates OpenAI function-calling schemas via `openai_schemas()`, and dispatches calls. There is no decorator magic or metaclass — just `register()` and `dispatch()`.

**Bundled tools:**

| Tool | What it does |
|---|---|
| `list_dir` | Directory listing with depth control |
| `search_files` | Glob-based file search |
| `file_read` | Read a file (optionally a line range) |
| `read_many_files` | Batch-read up to N files in one call |
| `file_write` | Write or overwrite a file |
| `file_patch` | Surgical patch: `old_str` → `new_str`, fails if not unique |
| `grep_text` | Regex grep across a directory |
| `shell_run` | Execute a shell command in the workspace |
| `python_run` | Execute a Python snippet via `subprocess` |
| `update_working_checkpoint` | Write a durable task checkpoint to `metadata` |
| `memory_recall` | Query long-term memory |
| `memory_propose_update` | Propose a memory write (staged) |
| `memory_commit_update` | Commit a staged memory write to disk |
| `ask_user` | Block and ask the user a clarifying question |

`ToolContext` carries `workspace`, `memory`, `session_id`, and `metadata` — the runtime context injected into every tool call. Tools never import the loop or the agent; they only see their context.

### Context management (`src/miniagent/context/`)

Context management is stateful across turns but reset per run:

- **`ContextPacket`**: A snapshot assembled at run start. Contains the compiled system prompt, a user-facing packet (task + long-term memory recall + working checkpoint + recent turn summaries), and metadata.
- **`HeuristicTurnSummarizer`**: After each turn, produces a compact summary dict: `{turn, user_intent, assistant_action, tools, next_goal}`. Text is clipped to 500 chars.
- **`ContextCompactor`**: Keeps the last `keep_recent_summaries=8` turn summaries. Clips tool output to `max_tool_output_chars=12000` chars to prevent context explosion on large file reads.
- **`working_checkpoint`**: A free-text string that the agent writes via `update_working_checkpoint`. It is injected into every subsequent turn's user packet as `### Working checkpoint`. This gives the model a stable task anchor across long multi-turn runs.

The context layer produces no tokens itself. It assembles `input_items` lists for the LLM client and keeps the window bounded.

### Memory (`src/miniagent/memory/`)

Memory is entirely file-based. No vector DB, no embeddings, no external service required.

**Three layers:**

| Layer | File | Content |
|---|---|---|
| L0 | `memory/l0_policy.md` | Static policy: what to remember, what not to |
| L1 | `memory/l1_index.md` | Running index of memory entries (ID, tags, summary) |
| L2 | `memory/l2_facts.md` | Full content of committed facts |

**Skills memory:** `memory/skills/*.md` — procedural knowledge (how to refactor code, how to edit files, etc.) that the agent can recall when starting a relevant task.

**Session events:** `memory/sessions/<session_id>.jsonl` — per-session event log for debugging.

**Pending writes:** `memory/pending/` — staged memory proposals not yet committed.

**Recall** (`FileMemoryStore.recall()`): scans L1 index for keyword matches, loads matching L2 entries, clips to a token budget. No embedding similarity — intentionally simple and auditable.

### Runtime (`src/miniagent/runtime/`)

- **`AgentConfig`** (`config.py`): loads `config/agent.yaml` via PyYAML (with a fallback pure-Python YAML parser for environments without PyYAML). All settings have documented defaults.
- **`Workspace`** (`workspace.py`): wraps the working directory with path safety checks — tool file operations are constrained to the workspace root.
- **`JsonlRunLogger`** (`logging.py`): writes structured JSONL run logs to `workspace/logs/<session_id>.jsonl`. Each entry is `{event, timestamp, ...payload}`.
- **`load_dotenv_if_present`** (`env.py`): loads `.env` from the project root. No hard dependency on `python-dotenv`; graceful no-op if absent.
- **Diagnostics** (`diagnostics.py`): `python agent.py --doctor` prints the resolved model, API key presence, base URL, and observability status.

---

## Configuration reference (`config/agent.yaml`)

```yaml
name: MINIAgent
max_turns: 40

llm:
  api_mode: chat_completions   # chat_completions | responses | auto
  max_output_tokens: 4096
  request_timeout_seconds: 120
  max_retries: 2
  temperature: null             # null = model default

context:
  keep_recent_summaries: 8
  max_tool_output_chars: 12000

observability:
  provider: langfuse
  enabled: true
```

---

## Tech stack

| Component | Library / approach |
|---|---|
| Python | 3.11+, `from __future__ import annotations` throughout |
| Data models | `dataclasses` with `slots=True` (zero dependencies, fast) |
| LLM API | `openai` ≥ 1.0 (Chat Completions + Responses API) |
| Observability | `langfuse` (optional drop-in wrapper around `openai.OpenAI`) |
| Config | `PyYAML` with a pure-Python fallback parser |
| Env management | `python-dotenv` (optional; graceful no-op if absent) |
| Memory storage | Plain Markdown files (no DB, no embeddings) |
| Run logs | JSONL files per session |
| Tests (offline) | Plain Python scripts, no pytest required |
| Tests (real) | Real LLM calls via `MCP_API_KEY` / `MCP_MODEL` |
| Packaging | `pyproject.toml`, src layout, `pip install -e .` |

---

## Tests

### Offline (no LLM required)

```bash
python run_all_tests.py   # or: cd tests_offline && python 01_import_and_config.py
```

Covers: config loading, tool registry, file/shell tools, memory, context manager, agent loop with a fake LLM stub.

### Real LLM tests

```bash
python run_real_tests.py
```

Requires `MCP_API_KEY` and `MCP_MODEL`. Covers:

| Test | What it exercises |
|---|---|
| `01_llm_smoke.py` | Single LLM call, usage metadata, Langfuse integration |
| `02_tool_file_patch.py` | Multi-turn: read → patch → verify |
| `03_code_run.py` | `python_run` tool, result verification |
| `04_memory_recall.py` | Write and recall long-term memory across two sessions |
| `05_context_long_task.py` | Multi-file scan, `update_working_checkpoint`, file write, full context persistence |
| `06_agent_end_to_end.py` | End-to-end bug fix: find → patch → verify |

---

## Environment variables

| Variable | Purpose | Required |
|---|---|---|
| `MCP_API_KEY` | LLM API key | Yes |
| `MCP_MODEL` | Model identifier (e.g. `gpt-4o`) | Yes |
| `MCP_API_BASE` | OpenAI-compatible base URL | Recommended |
| `OPENAI_API_KEY` | Fallback API key | No |
| `OPENAI_MODEL` | Fallback model | No |
| `OPENAI_BASE_URL` | Fallback base URL | No |
| `LANGFUSE_PUBLIC_KEY` | Langfuse project public key | No |
| `LANGFUSE_SECRET_KEY` | Langfuse project secret key | No |
| `LANGFUSE_BASE_URL` | Langfuse host (default: cloud.langfuse.com) | No |

---

## Versioned directories

- `v0/` — original prototype. Preserved as-is for migration reference.
- `v1/` — LangGraph experiment. Preserved for comparison. Shows what the LangGraph version of the same architecture looks like: `StateGraph`, `ToolNode`, `langgraph.store` for memory.

Both are kept intentionally so you can diff the approaches. The current default runtime (`src/miniagent/`) makes no use of either.

---

## Related projects

MINIAgent is a focused implementation of patterns you'll find across several agent frameworks. If you want to go deeper:

| Project | What to look at |
|---|---|
| **[LangGraph](https://github.com/langchain-ai/langgraph)** | The graph-based agent runtime that `v1/` experiments with. `StateGraph`, `ToolNode`, `langgraph.store`, and `MemorySaver` are the LangGraph equivalents of MINIAgent's loop, tool registry, and memory store. |
| **[GenericAgent](https://github.com/langchain-ai/langgraph/tree/main/libs/langgraph)** | The underlying design pattern MINIAgent is named after: a generic observe–plan–act loop with pluggable tools and memory. |
| **[smolagents](https://github.com/huggingface/smolagents)** | HuggingFace's minimal agent library. Similar philosophy to MINIAgent: keep the loop small, tool interface simple, model-agnostic. Uses a `CodeAgent` / `ToolCallingAgent` split. |
| **[Hermes](https://github.com/NousResearch/Hermes-Function-Calling)** | NousResearch's function-calling fine-tune and prompting specification. Relevant for understanding how tool-call prompting and JSON schema function definitions evolved in open-weight models. |
