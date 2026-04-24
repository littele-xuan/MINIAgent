from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class GraphRunResult:
    answer: str
    output_mode: str
    payload: Any = None
    selected_skill: str | None = None
    thread_id: str | None = None
    trace: list[dict[str, Any]] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
