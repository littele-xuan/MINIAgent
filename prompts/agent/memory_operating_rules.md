Context is engine-managed, not model-managed.
Treat long-term memory and task-context memory as different layers.
Prefer exact task-local context first, then retrieved structured long-term memory, then expandable summaries.
Treat summaries as derived views over immutable history.
When retrieved context references mutable files or warns about staleness, re-read before acting.
Memory management is runtime-owned; the planner should continue to speak only the standard MCP-style JSON contract.
