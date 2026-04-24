from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseMemoryManager(ABC):
    @abstractmethod
    async def load_context(self, *, user_id: str, thread_id: str, query: str) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    async def record_user_turn(self, *, user_id: str, thread_id: str, text: str) -> None:
        raise NotImplementedError

    @abstractmethod
    async def record_tool_result(self, *, user_id: str, thread_id: str, tool_name: str, content: str) -> None:
        raise NotImplementedError

    @abstractmethod
    async def record_assistant_turn(self, *, user_id: str, thread_id: str, text: str) -> None:
        raise NotImplementedError

    async def consolidate(self, *, user_id: str, thread_id: str) -> None:
        return None
