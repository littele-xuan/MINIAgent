from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypeVar

from pydantic import BaseModel

ModelT = TypeVar('ModelT', bound=BaseModel)


class BaseStructuredLLM(ABC):
    @abstractmethod
    async def ainvoke_text(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int | None = None,
        invoke_config: dict[str, Any] | None = None,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    async def ainvoke_json_model(
        self,
        messages: list[dict[str, str]],
        *,
        model_type: type[ModelT],
        schema_name: str,
        max_output_tokens: int | None = None,
        invoke_config: dict[str, Any] | None = None,
    ) -> ModelT:
        raise NotImplementedError

    @abstractmethod
    async def close(self) -> None:
        raise NotImplementedError
