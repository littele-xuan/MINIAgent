from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..utils import json_safe


@dataclass(slots=True)
class ToolAuditLog:
    events: list[dict[str, Any]] = field(default_factory=list)

    def append(self, *, event_type: str, tool_name: str, payload: dict[str, Any] | None = None) -> None:
        self.events.append({'event_type': event_type, 'tool_name': tool_name, 'payload': json_safe(payload or {})})

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        return self.events[-limit:]
