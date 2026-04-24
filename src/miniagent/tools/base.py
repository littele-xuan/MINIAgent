from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ..core.outcome import ToolResult
from ..memory.store import FileMemoryStore
from ..runtime.workspace import Workspace


@dataclass(slots=True)
class ToolContext:
    workspace: Workspace
    memory: FileMemoryStore
    session_id: str
    metadata: dict[str, Any]


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]

    def as_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class BaseTool(ABC):
    name: str
    description: str
    parameters: dict[str, Any]

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.parameters)

    @abstractmethod
    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        raise NotImplementedError
