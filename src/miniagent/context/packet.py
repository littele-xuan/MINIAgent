from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ContextPacket:
    system_prompt: str
    user_packet: str
    memory_context: str = ""
    working_checkpoint: str = ""
    summaries: list[dict[str, Any]] = field(default_factory=list)
