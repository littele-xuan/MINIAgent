from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseSkillManager(ABC):
    @abstractmethod
    async def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def select(self, query: str) -> dict[str, Any] | None:
        raise NotImplementedError

    @abstractmethod
    def local_tools_for(self, selected_skill: dict[str, Any] | None) -> list[Any]:
        raise NotImplementedError

    @abstractmethod
    def filter_tool_catalog(self, selected_skill: dict[str, Any] | None, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        raise NotImplementedError
