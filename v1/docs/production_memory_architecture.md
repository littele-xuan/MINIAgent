# Production-grade Memory Runtime Structure

## Goals

- Separate prompts from code for maintainability.
- Use abstract base classes for future evolution.
- Keep compatibility with the current `ContextMemoryEngine` and `ContextRuntimeAPI`.
- Support long-term memory, task-local memory, summary compaction, artifact offloading, and memory QA.

## Current layout

- `prompts/`：all agent and memory prompts.
- `prompt_runtime/`：prompt provider abstraction and file-based implementation.
- `llm_runtime/base.py`：common LLM base interface.
- `context_runtime/memory/base/`：memory contracts.
- `context_runtime/memory/factory.py`：default production composition.
- `context_runtime/memory/engine.py`：runtime orchestration.
- `context_runtime/memory/store.py`：SQLite execution store.
- `context_runtime/memory/repository.py`：filesystem mirror and audit trail.
- `tests/`：deterministic API and integration tests.

## Memory tiers

1. **Session memory**: current task state, temporary constraints, project-specific context.
2. **Cross-session memory**: durable profile facts and long-term preferences.
3. **Summary DAG**: compacted history that remains expandable.
4. **Artifacts**: large tool outputs and tracked file references.
5. **Failure events**: normalized runtime failures for future recovery.

## Extension points

You can replace any of the following via custom components:

- store
- repository
- fact extractor
- failure classifier
- summary generator
- retriever
- query resolver
- llm

Use `MemoryRuntimeFactory` as the default builder or instantiate `ContextMemoryEngine(..., components=...)`.
