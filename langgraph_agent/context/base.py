from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .packet import ContextPacket


class BaseContextAssembler(ABC):
    @abstractmethod
    def build_packet(
        self,
        *,
        agent: Any,
        state: dict[str, Any],
        query: str,
        visible_tools: list[dict[str, Any]],
        memory_context: dict[str, Any] | None,
        selected_skill: dict[str, Any] | None,
    ) -> ContextPacket:
        raise NotImplementedError

    @abstractmethod
    def build_messages(
        self,
        *,
        agent: Any,
        state: dict[str, Any],
        query: str,
        visible_tools: list[dict[str, Any]],
        memory_context: dict[str, Any] | None,
        selected_skill: dict[str, Any] | None,
    ) -> list[dict[str, str]]:
        raise NotImplementedError
