from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping


class BasePromptProvider(ABC):
    """Abstract prompt source used by planners, memory modules, and routing."""

    @abstractmethod
    def get(self, name: str, *, variables: Mapping[str, Any] | None = None) -> str:
        raise NotImplementedError
