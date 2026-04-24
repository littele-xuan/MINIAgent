from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ToolCall:
    """A normalized function/tool call returned by the model."""

    id: str
    call_id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str = ""


@dataclass(slots=True)
class ToolResult:
    """Standard tool result passed back to the LLM."""

    ok: bool
    content: str
    data: dict[str, Any] | list[Any] | str | int | float | bool | None = None
    error: str | None = None
    should_continue: bool = True

    def to_model_output(self) -> str:
        payload: dict[str, Any] = {
            "ok": self.ok,
            "content": self.content,
        }
        if self.data is not None:
            payload["data"] = self.data
        if self.error:
            payload["error"] = self.error
        return _json_dumps(payload)


@dataclass(slots=True)
class StepOutcome:
    """One GenericAgentCore loop step outcome."""

    assistant_text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    should_exit: bool = False
    reason: str = ""


@dataclass(slots=True)
class AgentResult:
    """Final result returned by an agent run."""

    final_text: str
    turns: int
    exit_reason: str
    tool_events: list[dict[str, Any]] = field(default_factory=list)
    session_id: str = ""
    usage: dict[str, Any] = field(default_factory=dict)


def _json_dumps(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, default=str)
