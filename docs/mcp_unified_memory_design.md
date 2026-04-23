# MCP-unified memory design

## Why the previous design was weak

The previous memory implementation treated memory extraction, retrieval selection, memory QA, and summary generation as separate mini-protocols:

- fact extraction expected `{"facts": [...]}`
- retrieval expected `{"picks": [...]}`
- summary expected `{"summary": "..."}`
- memory QA expected `{"answer": "..."}`

That design works in demos, but it drifts away from an MCP-native agent architecture because the main planner speaks one JSON contract while the memory subsystem speaks several unrelated ones.

## New design principle

All model-mediated runtime decisions now use the same envelope shape:

```json
{
  "thought": "brief action summary",
  "mode": "mcp" | "final",
  "final": {"output_mode": "application/json", "data": {...}} | null,
  "mcp_calls": [
    {"tool_name": "...", "arguments": {...}, "reason": "..."}
  ]
}
```

This is the same structural contract used by the top-level planner.

## Memory architecture after refactor

### 1. Long-term memory

Durable cross-session facts, for example:

- user profile
- stable preferences
- enduring project-independent constraints

### 2. Task context memory

Session-scoped context, for example:

- current task
- project name
- temporary constraints
- recent tool failures
- summary DAG nodes
- large output artifacts

### 3. Prompt injection policy

Prompt context now separates:

- `memory-long-term`
- `memory-task-context`
- `memory-active-summaries`
- `memory-recent-raw-messages`

This prevents durable preferences from being mixed conceptually with task-local execution state.

## Internal MCP tools

The memory runtime now uses local MCP-like tools for internal orchestration:

- `memory_fact_upsert`
- `memory_fact_revoke`
- `memory_failure_record`
- `memory_candidate_select`
- `memory_answer_emit`
- `memory_summary_emit`

The memory subsystem is still runtime-owned, but it no longer invents a separate non-MCP output language.

## Compatibility strategy

`run_internal_mcp_turn()` first requests JSON broadly, then normalizes:

- standard MCP envelope
- legacy memory payloads such as `{"memories": [...]}`
- older direct payloads such as `{"summary": "..."}` or `{"answer": ...}`

This prevents schema drift from breaking the whole runtime.
