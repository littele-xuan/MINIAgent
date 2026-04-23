# Context & Memory Architecture

This implementation upgrades the original agent from a prompt-only context builder into a LLM-driven memory engine with these layers:

1. **Immutable store** (`memory.sqlite3`)
   - append-only turn log for every user / assistant / tool message
   - never overwrites raw history
   - source of truth for lossless expansion

2. **Active context** (`active_context_items` + summary DAG)
   - raw recent messages stay verbatim
   - older blocks are replaced by summary nodes
   - summaries keep leaf message ids so any compacted span can be expanded losslessly
   - soft threshold schedules asynchronous compaction
   - hard threshold blocks the next turn until compaction finishes

3. **Structured memory** (`facts`)
   - engine-managed extraction from turns
   - bi-temporal style updates: old facts are superseded instead of silently overwritten
   - cross-session namespace support

4. **Filesystem memory repository** (`repository/`)
   - human-inspectable mirror of logs / summaries / profile facts / failures / artifacts
   - optional git backing for versioned memory operations

5. **Failure log** (`memory_events`)
   - F1 data/state errors
   - F2 tool/runtime errors
   - F3 workflow/process errors

6. **Stale-context protection**
   - tracked file references carry mtimes/checksums
   - when a file changes after capture, retrieved memory emits a warning to re-read before trusting cached state

## Why this architecture

The design is intentionally **engine-managed**, not **model-managed**:

- the model is not trusted to decide whether memory should be saved before compaction
- every turn is persisted immediately
- compaction only changes the active prompt view, never the source of truth
- structured memory promotion is model-driven and configurable via the shared LLM client

## Main code locations

- `agent_core/memory/engine.py` — orchestration / lifecycle
- `agent_core/memory/store.py` — SQLite backend
- `agent_core/memory/summarizers.py` — model-driven compaction
- `agent_core/memory/extractors.py` — fact promotion + failure classification
- `agent_core/memory/retriever.py` — model-driven retrieval
- `agent_core/memory/repository.py` — filesystem mirror / git-friendly memory repo
- `agent_core/context.py` — prompt injection of active/retrieved memory
- `agent_core/agent.py` — hooks from the agent main loop
