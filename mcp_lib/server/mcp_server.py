"""
mcp/server/mcp_server.py
──────────────────────────
MCP Server — adapter between the ToolRegistry and the MCP protocol.
"""

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
    from skill_engine import AnthropicSkillLoader, SkillToolRegistrar

    registry = ToolRegistry()
    bootstrap_all_tools(registry)

    skills_root = os.getenv('AGENT_SKILLS_ROOT') or os.getenv('MCP_SKILLS_ROOT')
    if skills_root:
        loader = AnthropicSkillLoader(skills_root)
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


@_server.list_tools()
async def list_tools() -> list[types.Tool]:
    tools = _registry.list_tools(enabled_only=True, include_deprecated=False, include_system=True)
    return [_schema_to_mcp(t) for t in tools]


@_server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    try:
        result = await _registry.call(name, arguments)
        text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
        return [types.TextContent(type='text', text=text)]
    except Exception as e:
        return [types.TextContent(type='text', text=f'[{name} ERROR] {type(e).__name__}: {e}')]


async def main():
    global _registry
    _registry = _build_registry()
    stats = _registry.stats()
    print(f"[MCP Server] Started — {stats['total']} tools registered ({stats['enabled']} enabled)", file=sys.stderr)
    async with stdio_server() as (read_stream, write_stream):
        await _server.run(read_stream, write_stream, _server.create_initialization_options())


if __name__ == '__main__':
    asyncio.run(main())
