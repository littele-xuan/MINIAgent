from __future__ import annotations

from typing import Any

from ..utils import compact_text, json_safe


class ObservationCompressor:
    def __init__(self, *, max_text_chars: int = 4000) -> None:
        self.max_text_chars = max_text_chars

    def compress_tool_result(self, result: dict[str, Any]) -> dict[str, Any]:
        safe = json_safe(result)
        text = compact_text(safe.get('text', ''), limit=self.max_text_chars) if isinstance(safe, dict) else compact_text(safe, limit=self.max_text_chars)
        if isinstance(safe, dict):
            safe['text'] = text
            return safe
        return {'text': text}

    def compress_history(self, history: list[dict[str, Any]], *, window: int) -> list[dict[str, Any]]:
        items = history[-window:] if window > 0 else []
        out: list[dict[str, Any]] = []
        for item in items:
            safe = json_safe(item)
            if isinstance(safe, dict) and 'content' in safe:
                safe['content'] = compact_text(safe['content'], limit=self.max_text_chars)
            out.append(safe if isinstance(safe, dict) else {'content': str(safe)})
        return out
