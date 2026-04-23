# Context Runtime Design

This project now separates context and memory from `agent_core` into a standalone package:

- `context_runtime/context.py`: prompt-layer assembly
- `context_runtime/memory/engine.py`: lifecycle and compaction orchestration
- `context_runtime/memory/store.py`: SQLite durable store with append-only records and active-context state
- `context_runtime/memory/repository.py`: filesystem mirror for logs, session transcripts, summaries, artifacts, and profile facts
- `context_runtime/memory/extractors.py`: model-driven fact extraction and failure classification
- `context_runtime/memory/retriever.py`: multi-signal retrieval over facts, summaries, messages, events, and artifacts
- `context_runtime/memory/query_resolver.py`: model-driven answer resolution for memory/context questions
- `context_runtime/memory/summarizers.py`: model-driven compaction summaries with lossless expansion via leaf message ids
- `context_runtime/memory/api.py`: direct API independent of agent orchestration

## Storage model

1. **SQLite** (`memory.sqlite3`)
   - immutable messages
   - summary DAG nodes
   - active context items
   - durable/session facts
   - failure events
   - file/artifact references

2. **Filesystem repository** (`repository/`)
   - append-only day logs
   - session markdown transcripts
   - profile facts grouped by namespace/category
   - summary markdown files
   - artifact index + artifact bodies
   - session state snapshots

## Retrieval model

- task-local and cross-session facts are stored separately by scope
- task queries prefer session-scoped facts
- cross-task queries ignore task-local facts
- retrieval uses a broad SQLite-backed candidate pool plus model-driven selection and ranking for messages, summaries, facts, failures, and artifacts

## Compaction model

- soft threshold: async compaction
- hard threshold: blocking compaction before planning
- summaries replace older active items in the active-context list
- summaries retain `leaf_message_ids` so original messages can be expanded losslessly

## Tests

Use `agent_context_memory_test.py` for end-to-end agent-entry tests.
Use `context_runtime_api_demo.py` for direct runtime API usage.
