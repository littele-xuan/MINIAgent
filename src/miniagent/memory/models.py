from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


@dataclass(slots=True)
class MemoryItem:
    layer: str
    content: str
    source: str = ""
    evidence: str = ""
    tags: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    id: str = field(default_factory=lambda: uuid4().hex[:12])

    def to_markdown(self) -> str:
        tags = ", ".join(self.tags) if self.tags else "general"
        lines = [
            f"- id: {self.id}",
            f"  layer: {self.layer}",
            f"  tags: {tags}",
            f"  created_at: {self.created_at}",
            f"  content: {self.content.strip()}",
        ]
        if self.source:
            lines.append(f"  source: {self.source}")
        if self.evidence:
            lines.append(f"  evidence: {self.evidence.strip()}")
        return "\n".join(lines)


@dataclass(slots=True)
class MemoryRecallResult:
    query: str
    items: list[dict[str, Any]]

    def format_for_prompt(self) -> str:
        if not self.items:
            return "No relevant long-term memory found."
        chunks = []
        for item in self.items:
            chunks.append(f"[{item.get('source')}] score={item.get('score')}\n{item.get('text')}")
        return "\n\n".join(chunks)
