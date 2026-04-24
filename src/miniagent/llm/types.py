from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.outcome import ToolCall


@dataclass(slots=True)
class LLMResponse:
    text: str
    output_items: list[dict[str, Any]]
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    raw_id: str | None = None
