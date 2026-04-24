from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..schemas import ToolDescriptor


class BaseToolProvider(ABC):
    @abstractmethod
    async def connect(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def list_tools(self) -> list[ToolDescriptor]:
        raise NotImplementedError

    @abstractmethod
    async def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    async def execute_batch(self, calls: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for name, arguments in calls:
            results.append(await self.execute(name, arguments))
        return results

    async def close(self) -> None:
        return None
