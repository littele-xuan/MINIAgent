from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..utils import compact_text


@dataclass(slots=True)
class ContextBudget:
    max_chars: int = 24000
    memory_chars: int = 6000
    tools_chars: int = 9000
    history_chars: int = 5000
    observation_chars: int = 4000

    def trim_text(self, text: Any, *, bucket: str = 'default') -> str:
        limits = {
            'memory': self.memory_chars,
            'tools': self.tools_chars,
            'history': self.history_chars,
            'observation': self.observation_chars,
        }
        return compact_text(text, limit=limits.get(bucket, self.max_chars))

    def report(self) -> dict[str, Any]:
        return {
            'max_chars': self.max_chars,
            'memory_chars': self.memory_chars,
            'tools_chars': self.tools_chars,
            'history_chars': self.history_chars,
            'observation_chars': self.observation_chars,
        }
