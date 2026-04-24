from __future__ import annotations

import json
from typing import Any

from ...config.models import MCPServerConfig
from ...schemas import ToolDescriptor
from ..base import BaseToolProvider


class LangGraphMCPToolProvider(BaseToolProvider):
    def __init__(self, servers: list[MCPServerConfig]) -> None:
        self.servers = servers
        self.client = None
        self.tools: dict[str, Any] = {}
        self._session_contexts: dict[str, Any] = {}

    def _config_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for server in self.servers:
            if server.transport == 'stdio':
                payload[server.name] = {
                    'transport': 'stdio',
                    'command': server.command,
                    'args': list(server.args),
                    'env': dict(server.env),
                }
            else:
                if not server.url:
                    raise ValueError(f'MCP server {server.name} requires url for HTTP transport')
                payload[server.name] = {
                    'transport': 'http',
                    'url': server.url,
                    'headers': dict(server.headers),
                }
        return payload

    async def connect(self) -> None:
        if not self.servers:
            self.tools = {}
            return
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
        except Exception as exc:  # pragma: no cover
            raise RuntimeError('langchain-mcp-adapters is required for MCP tool support.') from exc

        self.client = MultiServerMCPClient(self._config_payload())
        loaded_tools: dict[str, Any] = {}

        stateless = [server for server in self.servers if not server.stateful_session]
        if stateless:
            for tool in await self.client.get_tools():
                loaded_tools[tool.name] = tool

        stateful = [server for server in self.servers if server.stateful_session]
        if stateful:
            try:
                from langchain_mcp_adapters.tools import load_mcp_tools
            except Exception as exc:  # pragma: no cover
                raise RuntimeError('stateful MCP sessions require langchain-mcp-adapters.tools') from exc
            for server in stateful:
                ctx = self.client.session(server.name)
                session = await ctx.__aenter__()
                self._session_contexts[server.name] = ctx
                for tool in await load_mcp_tools(session):
                    loaded_tools[tool.name] = tool

        self.tools = loaded_tools

    async def list_tools(self) -> list[ToolDescriptor]:
        tools: list[ToolDescriptor] = []
        for tool in self.tools.values():
            schema = {}
            args_schema = getattr(tool, 'args_schema', None)
            if args_schema is not None and hasattr(args_schema, 'model_json_schema'):
                schema = args_schema.model_json_schema()
            elif hasattr(tool, 'get_input_schema'):
                try:
                    schema_obj = tool.get_input_schema()
                    if hasattr(schema_obj, 'model_json_schema'):
                        schema = schema_obj.model_json_schema()
                except Exception:
                    schema = {}
            tools.append(
                ToolDescriptor(
                    name=tool.name,
                    description=getattr(tool, 'description', '') or '',
                    input_schema=schema,
                    metadata={'provider': 'langgraph-mcp'},
                )
            )
        return tools

    async def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name not in self.tools:
            raise KeyError(f'Tool not found: {name}')
        tool = self.tools[name]
        result = await tool.ainvoke(arguments)
        if isinstance(result, str):
            text = result
            payload = None
        elif isinstance(result, dict):
            text = json.dumps(result, ensure_ascii=False, default=str)
            payload = result
        else:
            text = str(result)
            payload = getattr(result, 'artifact', None)
        return {
            'tool_name': name,
            'arguments': arguments,
            'text': text,
            'payload': payload,
        }

    async def close(self) -> None:
        for ctx in list(self._session_contexts.values()):
            try:
                await ctx.__aexit__(None, None, None)
            except Exception:
                pass
        self._session_contexts.clear()
        client = self.client
        self.client = None
        self.tools = {}
        if client is None:
            return
        aclose = getattr(client, 'aclose', None)
        if callable(aclose):
            await aclose()
