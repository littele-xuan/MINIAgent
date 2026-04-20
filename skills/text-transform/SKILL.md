---
name: text-transform
description: Transform text with bundled MCP tools such as slugify and reverse-text.
when_to_use: Use when the user asks to normalize text, generate slugs, reverse text, or perform lightweight text transformation.
allowed-tools:
  - slugify_text
  - reverse_text
argument-hint: [text]
output-modes:
  - text/plain
accepted-output-modes:
  - text/plain
  - application/json
examples:
  - 请把 "Hello World 2026" 转成 slug
  - 请把 `release-candidate-2026` 反转
  - 先 slugify 再 reverse，测试 agent 的多步 MCP 执行能力
mcp:
  protocol: bundled-local-mcp
  selection: prefer bundled tools over generic utilities
  batch_calls: false
a2a:
  enabled: false
---

# Text Transform

Use the bundled MCP tools when the user asks for small deterministic text transformations.

## Workflow
1. Prefer `slugify_text` for URL/path-safe text.
2. Prefer `reverse_text` for reverse-order demonstrations or debugging.
3. When a request requires multiple sequential transforms, call the MCP tools step by step and feed the prior tool result into the next step.
4. Keep the output deterministic.
