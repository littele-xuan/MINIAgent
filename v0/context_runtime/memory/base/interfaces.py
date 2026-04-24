from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from context_runtime.memory.models import ContextPacket, MemoryEvent, MessageRecord, RetrievedMemory, SummaryNode


class BaseMemoryStore(ABC):
    @abstractmethod
    def ensure_session(self, session_id: str, *, namespace: str, created_at: str, metadata: dict | None = None) -> None:
        raise NotImplementedError


class BaseMemoryRepository(ABC):
    @abstractmethod
    def append_message(self, record: MessageRecord) -> None:
        raise NotImplementedError


class BaseFactExtractor(ABC):
    @abstractmethod
    async def materialize(self, *, namespace: str, session_id: str, message: MessageRecord, created_at: str) -> list[Any]:
        raise NotImplementedError


class BaseFailureClassifier(ABC):
    @abstractmethod
    async def from_observation(self, *, session_id: str, content: str, created_at: str, metadata: dict | None = None) -> MemoryEvent | None:
        raise NotImplementedError


class BaseSummaryGenerator(ABC):
    @abstractmethod
    async def summarize_block(self, *, session_id: str, items: list[MessageRecord | SummaryNode], plans: list[Any], created_at: str) -> SummaryNode:
        raise NotImplementedError


class BaseMemoryRetriever(ABC):
    @abstractmethod
    async def retrieve(self, *, namespace: str, session_id: str, query: str, limit: int) -> list[RetrievedMemory]:
        raise NotImplementedError


class BaseMemoryQueryResolver(ABC):
    @abstractmethod
    async def answer(self, *, namespace: str, session_id: str, query: str, warnings: list[str] | None = None) -> str | None:
        raise NotImplementedError


class BaseMemoryEngine(ABC):
    @abstractmethod
    async def begin_turn(self, query: str) -> None:
        raise NotImplementedError

    @abstractmethod
    async def build_context_packet(self, *, query: str) -> ContextPacket:
        raise NotImplementedError

    @abstractmethod
    async def record_observation(self, event: dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    async def finalize_turn(self, *, answer: str, output_mode: str, payload: Any | None = None) -> None:
        raise NotImplementedError

    @abstractmethod
    async def close(self) -> None:
        raise NotImplementedError
