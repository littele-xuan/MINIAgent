from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class HeuristicTurnSummarizer:
    max_text_chars: int = 500

    def summarize(self, *, turn: int, user_input: str, assistant_text: str, tool_events: list[dict[str, Any]]) -> dict[str, Any]:
        tools = [event.get("name") for event in tool_events if event.get("event") == "tool_end"]
        return {
            "turn": turn,
            "user_intent": _clip(user_input, self.max_text_chars),
            "assistant_action": _clip(assistant_text, self.max_text_chars),
            "tools": tools[-12:],
            "next_goal": "Continue from latest tool result or provide final answer.",
        }


def _clip(text: str, n: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else text[:n] + f"...<{len(text)-n} chars truncated>"
