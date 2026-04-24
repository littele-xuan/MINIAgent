from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


@dataclass(slots=True)
class AgentState:
    """Mutable state for a single agent run."""

    user_input: str
    session_id: str = field(default_factory=lambda: uuid4().hex)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    turn: int = 0
    input_items: list[dict[str, Any]] = field(default_factory=list)
    final_text: str = ""
    exit_reason: str = "running"
    tool_events: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_usage(self, usage: dict[str, Any] | None) -> None:
        if not usage:
            return
        for key, value in usage.items():
            if isinstance(value, (int, float)):
                self.usage[key] = self.usage.get(key, 0) + value
            elif key not in self.usage:
                self.usage[key] = value
