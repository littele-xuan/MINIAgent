from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

EventKind = Literal["llm_start", "llm_end", "tool_start", "tool_end", "message", "error"]


@dataclass(slots=True)
class AgentEvent:
    kind: EventKind
    payload: dict[str, Any]
