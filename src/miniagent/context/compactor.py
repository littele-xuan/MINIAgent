from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ContextCompactor:
    max_tool_output_chars: int = 12000
    keep_recent_summaries: int = 8

    def compact_tool_output(self, content: str) -> str:
        if len(content) <= self.max_tool_output_chars:
            return content
        head = content[: self.max_tool_output_chars // 2]
        tail = content[-self.max_tool_output_chars // 2 :]
        omitted = len(content) - len(head) - len(tail)
        return f"{head}\n\n... <compacted {omitted} chars> ...\n\n{tail}"

    def compact_summaries(self, summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return summaries[-self.keep_recent_summaries :]
