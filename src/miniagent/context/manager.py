from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .compactor import ContextCompactor
from .packet import ContextPacket
from .summarizer import HeuristicTurnSummarizer


@dataclass(slots=True)
class ContextManager:
    """Builds model context from user input, working memory, summaries, and durable memory."""

    compactor: ContextCompactor = field(default_factory=ContextCompactor)
    summarizer: HeuristicTurnSummarizer = field(default_factory=HeuristicTurnSummarizer)
    summaries: list[dict[str, Any]] = field(default_factory=list)
    working_checkpoint: str = ""

    def start_packet(self, *, system_prompt: str, user_input: str, memory_context: str, metadata: dict[str, Any]) -> ContextPacket:
        self.working_checkpoint = metadata.get("working_checkpoint") or self.working_checkpoint
        parts = [
            "### User task",
            user_input.strip(),
            "",
            "### Long-term memory recall",
            memory_context.strip() or "No relevant long-term memory found.",
            "",
            "### Working checkpoint",
            self.working_checkpoint.strip() or "No working checkpoint yet.",
        ]
        compacted = self.compactor.compact_summaries(self.summaries)
        if compacted:
            parts.extend(["", "### Recent turn summaries"])
            for s in compacted:
                parts.append(f"- turn {s.get('turn')}: user={s.get('user_intent')} | action={s.get('assistant_action')} | tools={s.get('tools')}")
        parts.extend([
            "",
            "### Execution instruction",
            "Use the available tools when useful. Continue autonomously until the task is complete, then answer with a concise implementation summary and any verification results.",
        ])
        return ContextPacket(system_prompt=system_prompt, user_packet="\n".join(parts), memory_context=memory_context, working_checkpoint=self.working_checkpoint, summaries=compacted)

    def initial_input_items(self, packet: ContextPacket) -> list[dict[str, Any]]:
        return [{"role": "user", "content": packet.user_packet}]

    def after_turn(self, *, turn: int, user_input: str, assistant_text: str, tool_events: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
        if metadata.get("working_checkpoint"):
            self.working_checkpoint = metadata["working_checkpoint"]
        summary = self.summarizer.summarize(turn=turn, user_input=user_input, assistant_text=assistant_text, tool_events=tool_events)
        self.summaries.append(summary)
        self.summaries = self.compactor.compact_summaries(self.summaries)
