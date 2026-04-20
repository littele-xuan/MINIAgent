# MCP Tool Manager

[中文说明](./README.zh-CN.md)

An English-first reference implementation for full-lifecycle MCP tool management.

This project combines a live tool registry, governance operations, an MCP server adapter, and an API-first MCP/A2A agent that plans in strict JSON, executes against the live MCP catalog, and can collaborate with peer agents through the standard A2A protocol. The result is not just "an MCP server with some tools", but a small control plane for tool lifecycle management.

## What This Repository Implements

- A `ToolRegistry` as the single source of truth for tool metadata, aliases, enablement state, versions, and dispatch.
- A three-tier tool model:
  - `INTERNAL_SYSTEM`: protected governance tools.
  - `INTERNAL_UTILITY`: protected built-in tools.
  - `EXTERNAL`: mutable tools managed at runtime.
- Governance operations exposed as tools themselves, so an MCP client or LLM agent can add, update, disable, deprecate, alias, merge, inspect, and remove external tools.
- An MCP server that reads the live registry instead of hardcoding tool declarations.
- A structured-output agent using Pydantic schemas instead of fragile string parsing.
- Tiered tests covering registry logic, MCP integration, and an optional LLM-driven demo.

## Design Philosophy

### 1. Registry-first, not server-first

The MCP server is an adapter, not the center of the system. The registry owns tool state and behavior; the MCP layer simply exposes that state through the protocol.

### 2. Protected core, mutable edge

Core utility and governance tools are intentionally immutable from governance APIs. Only external tools are lifecycle-managed. This keeps the management plane stable while still allowing runtime extensibility.

### 3. Lifecycle metadata is part of the model

Enable/disable state, deprecation, aliases, version history, tags, and audit hooks are first-class data, not ad hoc conventions. That makes the system easier to reason about, test, and export.

### 4. LLM autonomy with bounded control

The agent can manage tools through governance APIs, but only within explicit category boundaries. This gives the agent useful operational freedom without allowing it to rewrite the protected control layer.

### 5. Dynamic discovery over hardcoded wiring

Clients call `list_tools()` against the live registry. Adding or updating tools should not require duplicating tool definitions inside the agent.

## Architecture

```text
User / LLM
    |
    v
Structured-output Agent (ReAct + Pydantic)
    |
    v
MCP Client Session
    |
    v
MCP Server Adapter
    |
    v
ToolRegistry  <---->  GovernanceManager / registry_ops
    |
    +-- INTERNAL_SYSTEM tools
    +-- INTERNAL_UTILITY tools
    +-- EXTERNAL tools
```

## Lifecycle Flow

1. Bootstrap internal utility tools, governance tools, and seed external tools into the registry.
2. Expose the current enabled tool set through the MCP server.
3. Let an MCP client or agent call tools through registry dispatch.
4. Let governance tools mutate only `EXTERNAL` entries.
5. Refresh the agent's tool view after governance changes.
6. Preserve version history on destructive updates and expose stats, search, and export APIs.

## Repository Layout

| Path | Purpose |
|------|---------|
| `agent.py` | API-first MCP/A2A agent with strict JSON planning output |
| `agent_test.py` | Tiered tests and optional LLM demo |
| `mcp_lib/registry/` | Registry models and dispatch logic |
| `mcp_lib/governance/` | High-level lifecycle management wrapper |
| `mcp_lib/server/` | MCP protocol adapter |
| `mcp_lib/tools/internal/` | Built-in utility and governance tools |
| `mcp_lib/tools/external/` | Seed external tools |
| `v0/` | Earlier prototype retained for comparison |

## Built-in Capabilities

- Utility tools: calculator, Python runner, web search, weather, file operations.
- Governance tools: list, inspect, add, update, enable, disable, deprecate, alias, merge, search, version history, and registry stats.
- Seed external tools: random joke, UUID generation, timestamp, Base64, hash.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run tests

```bash
python agent_test.py
```

This runs Tier 1 and Tier 2 by default.

### 3. Run the optional LLM demo

Set environment variables first:

```bash
export MCP_API_BASE="https://api.openai.com/v1"
export MCP_API_KEY="your_api_key"
export MCP_MODEL="gpt-4o-mini"
python agent_test.py --llm
```

## Security Notes

- Hardcoded API credentials have been removed from the main test entrypoint.
- The repository now expects secrets to be injected through environment variables.
- `.gitignore` excludes common local environments and secret-bearing files.
- Before publishing derived tools created through `tool_add` or `tool_update`, review injected code carefully.

## Why This Matters

Many MCP examples stop at "tool calling works". This repository goes further and treats tools as managed assets with status, governance, and evolution. That is the useful boundary when MCP moves from demo infrastructure to an operational tool platform.

## License

This project is released under the [MIT License](./LICENSE).
