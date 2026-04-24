"""
mcplib/server/mcp_server.py
────────────────────────
Registry-backed MCP server using the official Python MCP SDK.

Why this shape:
  - keeps dynamic tool catalogs (add/update/remove/merge) working, which is
    essential when the LLM manages the MCP registry itself
  - stays on top of the official SDK rather than a custom wire protocol
  - supports both local stdio and a cleaner path for remote clients via the
    client runtime's streamable-http support
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server


def _build_registry():
    from mcp_lib.registry.registry import ToolRegistry
    from mcp_lib.tools import bootstrap_all_tools
    from skill_engine import AgentSkillsLoader, SkillToolRegistrar

    registry = ToolRegistry()
    bootstrap_all_tools(registry)

    skills_root = os.getenv('AGENT_SKILLS_ROOT') or os.getenv('MCP_SKILLS_ROOT')
    if skills_root:
        loader = AgentSkillsLoader(skills_root)
        registrar = SkillToolRegistrar()
        for bundle in loader.discover_bundles():
            for entry in registrar.build_tool_entries(bundle):
                try:
                    registry.register(entry)
                except Exception:
                    pass
    return registry


_server = Server('mcp-tool-registry')
_registry = None


def _schema_to_mcp(entry) -> types.Tool:
    desc = entry.description
    if entry.deprecated:
        repl = f' (替代: {entry.replaced_by})' if entry.replaced_by else ''
        desc = f'[已废弃{repl}] ' + desc
    return types.Tool(name=entry.name, description=desc, inputSchema=entry.input_schema)


def _content_from_result(name: str, result: Any) -> list[types.ContentBlock]:
    if isinstance(result, str):
        return [types.TextContent(type='text', text=result)]
    if isinstance(result, (int, float, bool)) or result is None:
        return [types.TextContent(type='text', text=json.dumps(result, ensure_ascii=False))]
    if isinstance(result, dict):
        primary = result.get('message') or result.get('summary') or result.get('text')
        payload = json.dumps(result, ensure_ascii=False, default=str)
        if primary:
            return [
                types.TextContent(type='text', text=str(primary)),
                types.TextContent(type='text', text=payload),
            ]
        return [types.TextContent(type='text', text=payload)]
    if isinstance(result, list):
        return [types.TextContent(type='text', text=json.dumps(result, ensure_ascii=False, default=str))]
    return [types.TextContent(type='text', text=f'[{name}] {result!r}')]


@_server.list_tools()
async def list_tools() -> list[types.Tool]:
    tools = _registry.list_tools(enabled_only=True, include_deprecated=False, include_system=True)
    return [_schema_to_mcp(t) for t in tools]


@_server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.ContentBlock]:
    try:
        result = await _registry.call(name, arguments)
        return _content_from_result(name, result)
    except Exception as exc:
        return [types.TextContent(type='text', text=f'[{name} ERROR] {type(exc).__name__}: {exc}')]


async def main() -> None:
    global _registry
    _registry = _build_registry()
    stats = _registry.stats()
    print(
        f"[MCP Server] Started — {stats['total']} tools registered ({stats['enabled']} enabled)",
        file=sys.stderr,
    )
    async with stdio_server() as (read_stream, write_stream):
        await _server.run(read_stream, write_stream, _server.create_initialization_options())


if __name__ == '__main__':
    asyncio.run(main())
