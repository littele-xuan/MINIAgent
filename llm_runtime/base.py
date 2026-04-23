from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class BaseLLM(ABC):
    """Common LLM interface for planners, context runtime, and tests."""

    @abstractmethod
    async def close(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def chat_text(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    async def chat_json(
        self,
        *,
        messages: list[dict[str, Any]],
        schema: dict[str, Any],
        schema_name: str,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        repair_attempts: int = 2,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    async def chat_json_model(
        self,
        *,
        messages: list[dict[str, Any]],
        model_type: type[BaseModel],
        schema_name: str,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        repair_attempts: int = 2,
    ) -> BaseModel:
        raise NotImplementedError
