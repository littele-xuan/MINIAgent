from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class MemoryPolicy:
    """Rules for durable memory writes.

    Core rule inherited from GenericAgent's memory SOP:
    no durable memory update without explicit evidence from a user message or tool result.
    """

    require_evidence: bool = True
    min_content_chars: int = 12

    def validate(self, content: str, evidence: str | None) -> tuple[bool, str]:
        if len((content or "").strip()) < self.min_content_chars:
            return False, "content too short for durable memory"
        if self.require_evidence and not (evidence or "").strip():
            return False, "durable memory requires evidence"
        volatile_markers = ["just now", "temporary", "for this turn", "本轮", "临时"]
        lowered = content.lower()
        if any(m in lowered for m in volatile_markers):
            return False, "volatile or short-lived information should remain in working context"
        return True, "ok"
