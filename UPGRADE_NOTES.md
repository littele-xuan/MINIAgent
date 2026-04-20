# MiniAgent framework upgrade notes

This version keeps the original project layout and Agent orchestration flow, but upgrades the runtime layers toward more standard, production-oriented implementations.

## What changed

### MCP
- Keeps the official Python MCP SDK as the protocol layer.
- Client runtime now supports both:
  - local `stdio` subprocess MCP servers
  - remote `streamable-http` MCP endpoints
- Registry governance tools (`tool_add`, `tool_update`, `tool_remove`, `tool_merge`, etc.) now prefer:
  - declarative JSON Schema for tool inputs
  - module-based handlers (`handler_mode=python_module`)
  - inline Python only as a compatibility path

### Skills
- Loader renamed conceptually to `AgentSkillsLoader` and keeps `AnthropicSkillLoader` as a backward-compatible alias.
- If `skills-ref` is installed, skill validation prefers the reference implementation.
- Existing `SKILL.md` structure and local `mcp_tools/` layout remain compatible.

### A2A
- `build_a2a_app()` now prefers the official `a2a-sdk` server runtime.
- If the SDK is unavailable or changes shape, the framework falls back to the existing FastAPI-compatible implementation.
- Agent card discovery prefers the official resolver when available.

## Install

```bash
pip install -r requirements.txt
```

## MCP connect modes

Local stdio server:

```python
await agent.connect("./mcp_lib/server/mcp_server.py")
```

Remote streamable-http endpoint:

```python
runtime = MCPClientToolRuntime()
await runtime.connect("https://your-mcp-host.example.com/mcp")
await agent.attach_tool_runtime(runtime)
```

## Safer dynamic tool registration

Prefer module handlers in production:

```json
{
  "name": "normalize_customer_id",
  "description": "Normalize customer ids",
  "handler_mode": "python_module",
  "module_path": "company_tools.customer",
  "callable_name": "normalize_customer_id",
  "input_schema_json": {
    "type": "object",
    "properties": {
      "customer_id": {"type": "string"}
    },
    "required": ["customer_id"],
    "additionalProperties": false
  }
}
```

## Compatibility promises
- Existing Agent framework structure is preserved.
- Existing imports for `AnthropicSkillLoader`, `LocalRegistryToolRuntime`, and the original A2A models remain available.
- Existing skill folders and registry bootstrap flow remain compatible.
