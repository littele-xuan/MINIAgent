from __future__ import annotations

from typing import Any

from ...schemas import ToolDescriptor
from ..base import BaseToolProvider


class LegacyToolRuntimeProvider(BaseToolProvider):
    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    async def connect(self) -> None:
        return None

    async def list_tools(self) -> list[ToolDescriptor]:
        descriptors = await self.runtime.list_tools()
        tools: list[ToolDescriptor] = []
        for item in descriptors:
            if hasattr(item, 'model_dump'):
                tools.append(ToolDescriptor.model_validate(item.model_dump()))
            else:
                tools.append(
                    ToolDescriptor(
                        name=item.name,
                        description=item.description,
                        input_schema=item.input_schema,
                        metadata=getattr(item, 'metadata', {}),
                    )
                )
        return tools

    async def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        result = await self.runtime.call_tool(name, arguments)
        return {
            'tool_name': name,
            'arguments': arguments,
            'text': result.primary_text,
            'payload': [dict(item) for item in result.content],
        }
