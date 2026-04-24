from __future__ import annotations

import importlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, Field


class ToolDescriptor(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)

    def catalog_entry(self) -> dict[str, Any]:
        return {
            'name': self.name,
            'description': self.description,
            'input_schema': self.input_schema,
            'metadata': self.metadata,
        }


class ToolCallResult(BaseModel):
    tool_name: str
    arguments: dict[str, Any]
    content: list[dict[str, Any]]

    @property
    def primary_text(self) -> str:
        parts: list[str] = []
        for item in self.content:
            if item.get('type') == 'text':
                text = item.get('text')
                if isinstance(text, str) and text:
                    parts.append(text)
        return '\n'.join(parts).strip()

    def to_observation(self) -> dict[str, Any]:
        return {
            'tool_name': self.tool_name,
            'arguments': self.arguments,
            'content': self.content,
            'primary_text': self.primary_text,
        }


class BaseToolRuntime(ABC):
    @abstractmethod
    async def list_tools(self) -> list[ToolDescriptor]: ...

    @abstractmethod
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolCallResult: ...

    async def call_tools_batch(self, calls: Iterable[tuple[str, dict[str, Any]]]) -> list[ToolCallResult]:
        results: list[ToolCallResult] = []
        for name, arguments in calls:
            results.append(await self.call_tool(name, arguments))
        return results

    async def close(self) -> None:
        return None

    def supports_dynamic_registration(self) -> bool:
        return False


class MCPClientToolRuntime(BaseToolRuntime):
    """Official MCP client runtime.

    Supported connection modes:
      - local stdio subprocess (default; best for embedded/local MCP servers)
      - remote streamable-http endpoint (recommended for deployed MCP services)
    """

    def __init__(self) -> None:
        self._stack = None
        self._session = None
        self._registry = None
        self._mode: str | None = None

    async def connect(self, server_script_or_url: str, *, env: dict[str, str] | None = None, headers: dict[str, str] | None = None) -> None:
        target = str(server_script_or_url)
        if target.startswith('http://') or target.startswith('https://'):
            await self.connect_streamable_http(target, headers=headers)
        else:
            await self.connect_stdio(target, env=env)

    async def connect_stdio(self, server_script: str, *, env: dict[str, str] | None = None) -> None:
        from contextlib import AsyncExitStack
        from mcp import StdioServerParameters
        from mcp.client.session import ClientSession
        from mcp.client.stdio import stdio_client

        self._stack = AsyncExitStack()
        cmd = 'python'
        script_path = Path(server_script)
        server_params = StdioServerParameters(command=cmd, args=[str(script_path)], env=env)
        read_stream, write_stream = await self._stack.enter_async_context(stdio_client(server_params))
        self._session = await self._stack.enter_async_context(ClientSession(read_stream, write_stream))
        await self._session.initialize()
        self._mode = 'stdio'

    async def connect_streamable_http(self, url: str, *, headers: dict[str, str] | None = None) -> None:
        from contextlib import AsyncExitStack
        from mcp.client.session import ClientSession

        streamable_mod = importlib.import_module('mcp.client.streamable_http')
        streamablehttp_client = getattr(streamable_mod, 'streamablehttp_client')

        self._stack = AsyncExitStack()
        read_stream, write_stream, *_ = await self._stack.enter_async_context(
            streamablehttp_client(url, headers=headers or {})
        )
        self._session = await self._stack.enter_async_context(ClientSession(read_stream, write_stream))
        await self._session.initialize()
        self._mode = 'streamable-http'

    async def list_tools(self) -> list[ToolDescriptor]:
        if self._registry is not None:
            return [
                ToolDescriptor(name=t.name, description=t.description, input_schema=t.input_schema, metadata=dict(getattr(t, 'metadata', {}) or {}))
                for t in self._registry.list_tools(enabled_only=True, include_deprecated=False, include_system=True)
            ]
        if self._session is None:
            return []
        response = await self._session.list_tools()
        return [
            ToolDescriptor(
                name=tool.name,
                description=tool.description or '',
                input_schema=getattr(tool, 'inputSchema', None) or {},
                metadata={},
            )
            for tool in response.tools
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolCallResult:
        if self._registry is not None:
            result = await self._registry.call(name, arguments)
            content = self._normalize_local_result(result)
            return ToolCallResult(tool_name=name, arguments=arguments, content=content)

        if self._session is None:
            raise RuntimeError('MCP session not initialized')

        result = await self._session.call_tool(name, arguments)
        content: list[dict[str, Any]] = []
        for item in result.content:
            text = getattr(item, 'text', None)
            if isinstance(text, str):
                content.append({'type': 'text', 'text': text})
                continue
            data = getattr(item, 'data', None)
            if data is not None:
                content.append({'type': 'data', 'data': data})
                continue
            model_dump = getattr(item, 'model_dump', None)
            if callable(model_dump):
                payload = model_dump(by_alias=True, exclude_none=True)
                content.append(payload if isinstance(payload, dict) else {'type': 'data', 'data': payload})
        return ToolCallResult(tool_name=name, arguments=arguments, content=content)

    def _normalize_local_result(self, result: Any) -> list[dict[str, Any]]:
        if isinstance(result, str):
            return [{'type': 'text', 'text': result}]
        if isinstance(result, dict):
            return [{'type': 'text', 'text': json.dumps(result, ensure_ascii=False, default=str)}]
        return [{'type': 'text', 'text': json.dumps(result, ensure_ascii=False, default=str)}]

    async def close(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
        self._stack = None
        self._session = None
        self._mode = None

    def register_tool_entry(self, entry: Any) -> None:
        if self._registry is not None:
            self._registry.register(entry, overwrite=True)

    def supports_dynamic_registration(self) -> bool:
        return self._registry is not None


class LocalRegistryToolRuntime(BaseToolRuntime):
    def __init__(self, registry: Any) -> None:
        self._registry = registry

    async def list_tools(self) -> list[ToolDescriptor]:
        return [
            ToolDescriptor(name=t.name, description=t.description, input_schema=t.input_schema, metadata=dict(getattr(t, 'metadata', {}) or {}))
            for t in self._registry.list_tools(enabled_only=True, include_deprecated=False, include_system=True)
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolCallResult:
        result = await self._registry.call(name, arguments)
        if isinstance(result, str):
            content = [{'type': 'text', 'text': result}]
        else:
            content = [{'type': 'text', 'text': json.dumps(result, ensure_ascii=False, default=str)}]
        return ToolCallResult(tool_name=name, arguments=arguments, content=content)

    def register_tool_entry(self, entry: Any) -> None:
        self._registry.register(entry, overwrite=True)

    def supports_dynamic_registration(self) -> bool:
        return True
