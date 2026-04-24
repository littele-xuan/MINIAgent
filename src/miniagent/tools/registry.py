from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.errors import ToolNotFoundError
from ..core.outcome import ToolResult
from .base import BaseTool, ToolContext


@dataclass(slots=True)
class ToolRegistry:
    tools: dict[str, BaseTool] = field(default_factory=dict)

    def register(self, tool: BaseTool) -> "ToolRegistry":
        if tool.name in self.tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self.tools[tool.name] = tool
        return self

    def register_many(self, tools: list[BaseTool]) -> "ToolRegistry":
        for tool in tools:
            self.register(tool)
        return self

    def openai_schemas(self) -> list[dict[str, Any]]:
        return [tool.spec.as_openai_tool() for tool in self.tools.values()]

    def names(self) -> list[str]:
        return sorted(self.tools)

    def dispatch(self, name: str, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        tool = self.tools.get(name)
        if tool is None:
            raise ToolNotFoundError(f"Unknown tool: {name}")
        return tool.run(args or {}, ctx)
