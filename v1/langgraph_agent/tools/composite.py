from __future__ import annotations

import inspect
import json
from typing import Any

from ..schemas import ToolDescriptor
from ..utils import json_safe
from .base import BaseToolProvider
from .normalizer import ToolResultNormalizer


class CompositeToolExecutor:
    def __init__(self, provider: BaseToolProvider, *, local_tools: dict[str, Any] | None = None) -> None:
        self.provider = provider
        self.local_tools = local_tools or {}
        self.normalizer = ToolResultNormalizer()

    async def list_tools(self) -> list[ToolDescriptor]:
        remote = await self.provider.list_tools()
        local = [
            ToolDescriptor(
                name=name,
                description=getattr(tool, 'description', '') or '',
                input_schema=getattr(tool, 'input_schema', {}) or {},
                metadata=getattr(tool, 'metadata', {}) or {},
                risk=getattr(tool, 'risk', 'safe_read') or 'safe_read',
            )
            for name, tool in self.local_tools.items()
        ]
        seen = {tool.name for tool in local}
        return local + [tool for tool in remote if tool.name not in seen]

    async def _execute_local(self, name: str, arguments: dict[str, Any]) -> Any:
        tool = self.local_tools[name]
        if hasattr(tool, 'ainvoke'):
            return await tool.ainvoke(arguments)
        if callable(tool):
            value = tool(arguments)
            return await value if inspect.isawaitable(value) else value
        raise TypeError(f'Unsupported local tool type: {type(tool)!r}')

    async def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            if name in self.local_tools:
                result = await self._execute_local(name, arguments)
            else:
                result = await self.provider.execute(name, arguments)
            return self.normalizer.normalize(tool_name=name, arguments=arguments, result=result)
        except Exception as exc:
            return self.normalizer.normalize(
                tool_name=name,
                arguments=arguments,
                result={'tool_name': name, 'arguments': json_safe(arguments), 'text': '', 'payload': None, 'status': 'error', 'error': str(exc)},
            )

    async def execute_batch(self, calls: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for name, arguments in calls:
            results.append(await self.execute(name, arguments))
        return results
