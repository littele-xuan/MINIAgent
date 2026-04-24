---
name: a2a-collaboration
description: Delegate work to a peer agent through the standard A2A protocol when a specialist agent is a better executor than the current agent.
when_to_use: Use when the best execution path is peer delegation, agent-to-agent collaboration, another agent card, 多智能体协作, 委派给其他 agent, A2A routing, or specialist handoff.
argument-hint: [delegation-target-or-task]
allowed-tools: []
output-modes:
  - text/plain
  - application/json
accepted-output-modes:
  - text/plain
  - application/json
examples:
  - 把这个数学问题委派给 math-agent，并要求返回 JSON
  - 让 registry-agent 检查当前 MCP 工具目录并返回结构化结果
  - 通过标准 A2A 把任务交给更合适的 agent，而不是在 prompt 里逐个挑工具
mcp:
  protocol: live-catalog-only
  selection: do not replace A2A delegation with ad hoc local tool picking
  notes:
    - local MCP tools remain available for this agent's own execution path
    - peer delegation must use A2A metadata, not guessed tool names
    - this skill should not become an LLM output mode; it is a runtime routing capability
a2a:
  enabled: true
  accepted_output_modes:
    - text/plain
    - application/json
  strategy: choose peers from agent cards, capabilities, tags, and output negotiation
  transport: standard A2A HTTP+JSON / JSON-RPC interfaces
---

# A2A Collaboration

Use this skill when another peer agent is the correct executor.

## Workflow
1. Inspect the available peer agent cards, tags, and descriptions from runtime context.
2. Delegate only when the peer has a clearer specialization or owns the relevant capability boundary.
3. Preserve `accepted_output_modes` when sending the task so downstream JSON contracts remain valid.
4. Forward or summarize the peer result faithfully after delegation completes.

## Rules
- Do not treat this skill as an LLM-facing output mode.
- Do not simulate A2A by manually calling unrelated local MCP tools one by one.
- Do not invent peer names or capabilities that are not present in the live A2A peer catalog.
- If the current agent can solve the task directly with its own live MCP catalog and there is no delegation advantage, do not force A2A.
- Runtime routing should decide whether to invoke this capability before spending prompt budget on MCP planning.

## Output
Return plain text when the caller accepts text only.
Return JSON when the caller requires `application/json` and the downstream peer can satisfy it.
