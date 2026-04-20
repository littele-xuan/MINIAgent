from __future__ import annotations

import json
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class ToolDescriptor:
    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def catalog_entry(self) -> dict[str, Any]:
        return {
            'name': self.name,
            'description': self.description,
            'input_schema': self.input_schema,
            'metadata': self.metadata,
        }


@dataclass(slots=True)
class ToolCallResult:
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    content: list[dict[str, Any]] = field(default_factory=list)
    is_error: bool = False
    raw: Any = None

    @property
    def primary_text(self) -> str:
        chunks: list[str] = []
        for item in self.content:
            text = item.get('text')
            if text:
                chunks.append(str(text))
            elif 'data' in item and item['data'] is not None:
                chunks.append(json.dumps(item['data'], ensure_ascii=False, default=str))
            elif 'url' in item and item['url']:
                chunks.append(str(item['url']))
        return '\n'.join(chunks).strip()

    def to_observation(self) -> dict[str, Any]:
        return {
            'tool_name': self.tool_name,
            'arguments': self.arguments,
            'content': self.content,
            'is_error': self.is_error,
            'primary_text': self.primary_text,
        }


class BaseToolRuntime(ABC):
    @abstractmethod
    async def list_tools(self) -> list[ToolDescriptor]:
        raise NotImplementedError

    @abstractmethod
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolCallResult:
        raise NotImplementedError

    async def call_tools_batch(self, calls: list[tuple[str, dict[str, Any]]]) -> list[ToolCallResult]:
        results: list[ToolCallResult] = []
        for name, arguments in calls:
            results.append(await self.call_tool(name, arguments))
        return results

    async def close(self) -> None:
        return None

    def supports_dynamic_registration(self) -> bool:
        return False


class LocalRegistryToolRuntime(BaseToolRuntime):
    def __init__(self, registry: Any):
        self.registry = registry

    async def list_tools(self) -> list[ToolDescriptor]:
        tools = self.registry.list_tools(enabled_only=True, include_deprecated=False, include_system=True)
        return [
            ToolDescriptor(
                name=entry.name,
                description=entry.description,
                input_schema=entry.input_schema,
                metadata=dict(entry.metadata or {}),
            )
            for entry in tools
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolCallResult:
        raw = await self.registry.call(name, arguments)
        return ToolCallResult(
            tool_name=name,
            arguments=dict(arguments or {}),
            content=[_normalize_content_item(raw)],
            is_error=_looks_like_error(raw),
            raw=raw,
        )

    def supports_dynamic_registration(self) -> bool:
        return True

    def register_tool_entry(self, entry: Any) -> None:
        try:
            self.registry.get(entry.name)
            return
        except Exception:
            self.registry.register(entry)


class MCPClientToolRuntime(BaseToolRuntime):
    def __init__(self) -> None:
        self._transport_cm = None
        self._session_cm = None
        self.session = None

    async def connect(self, server_script: str, env: dict[str, str] | None = None) -> None:
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError("MCP client runtime requires the 'mcp' package") from exc

        params = StdioServerParameters(
            command=sys.executable,
            args=[server_script],
            env={**os.environ, **(env or {})},
        )
        self._transport_cm = stdio_client(params)
        read, write = await self._transport_cm.__aenter__()
        self._session_cm = ClientSession(read, write)
        self.session = await self._session_cm.__aenter__()
        await self.session.initialize()

    async def list_tools(self) -> list[ToolDescriptor]:
        if self.session is None:
            return []
        result = await self.session.list_tools()
        descriptors: list[ToolDescriptor] = []
        for tool in result.tools:
            descriptors.append(
                ToolDescriptor(
                    name=tool.name,
                    description=getattr(tool, 'description', '') or '',
                    input_schema=getattr(tool, 'inputSchema', {}) or {},
                    metadata=_extract_tool_metadata(tool),
                )
            )
        return descriptors

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolCallResult:
        if self.session is None:
            raise RuntimeError('MCP runtime is not connected')
        result = await self.session.call_tool(name, arguments)
        content = [_normalize_mcp_content_item(item) for item in (getattr(result, 'content', None) or [])]
        return ToolCallResult(
            tool_name=name,
            arguments=dict(arguments or {}),
            content=content or [_normalize_content_item(None)],
            is_error=bool(getattr(result, 'isError', False)),
            raw=result,
        )

    async def close(self) -> None:
        if self._session_cm is not None:
            await self._session_cm.__aexit__(None, None, None)
            self._session_cm = None
            self.session = None
        if self._transport_cm is not None:
            await self._transport_cm.__aexit__(None, None, None)
            self._transport_cm = None


def _looks_like_error(value: Any) -> bool:
    return isinstance(value, str) and ' ERROR]' in value


def _normalize_content_item(value: Any) -> dict[str, Any]:
    if value is None:
        return {'type': 'text', 'text': ''}
    if isinstance(value, str):
        return {'type': 'text', 'text': value}
    if isinstance(value, (dict, list, tuple, int, float, bool)):
        return {'type': 'data', 'data': value}
    return {'type': 'text', 'text': str(value)}


def _normalize_mcp_content_item(item: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {'type': getattr(item, 'type', item.__class__.__name__.lower())}
    for field_name in ('text', 'data', 'url', 'mimeType', 'mime_type', 'annotations', 'metadata'):
        value = getattr(item, field_name, None)
        if value is not None:
            payload[field_name] = value
    if 'text' not in payload and 'data' not in payload and 'url' not in payload:
        payload['text'] = str(item)
    return payload


def _extract_tool_metadata(tool: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for field_name in ('annotations', 'metadata', 'tags'):
        value = getattr(tool, field_name, None)
        if value:
            metadata[field_name] = value
    return metadata
