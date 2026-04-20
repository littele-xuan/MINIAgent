---
name: registry-maintainer
description: Manage, inspect, and evolve the MCP tool registry.
when_to_use: Use when the task is about adding tools, updating tools, enabling or disabling tools, checking versions, inspecting the tool registry, 工具注册表, 工具管理, 添加工具, 更新工具, 启用工具, 禁用工具.
allowed-tools:
  - tool_list
  - tool_search
  - tool_get
  - tool_info
  - tool_stats
  - registry_stats
  - tool_add
  - tool_update
  - tool_enable
  - tool_disable
  - tool_deprecate
  - tool_versions
  - tool_alias
  - tool_merge
  - tool_remove
output-modes:
  - text/plain
  - application/json
accepted-output-modes:
  - text/plain
  - application/json
examples:
  - 列出当前 MCP 注册表里所有外部工具，并标注状态
  - 查一下 `weather` 这个工具的 schema 和版本信息
  - 把 weather_tool 升级到 1.2.0，并检查版本历史
  - 根据用户给出的 JSON 描述新增一个工具，而不是在调用端硬编码 runtime.call_tool(...)
  - 禁用一个不再使用的 external tool，再验证它是否已不可调用
mcp:
  protocol: live-registry
  selection: tools must come from the current MCP catalog only
  batch_calls: true
a2a:
  enabled: false
---

# Registry Maintainer

Use this skill when the user wants to manage the tool registry itself.

## Workflow
1. Start with `tool_search` / `tool_list`, `tool_info` / `tool_get`, or `registry_stats` / `tool_stats` to inspect live state.
2. Mutate external tools only through governance tools.
3. Derive the governance JSON arguments from the user query; do not expect the caller to hardcode `runtime.call_tool(...)` outside the agent loop.
4. After each mutation, re-check the live registry before reporting success when the workflow requires verification.
5. Prefer structured summaries when the user is auditing registry state.

## Safety
Never claim a tool exists until you have checked the live registry.
Never mutate protected internal tools.
