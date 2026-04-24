from __future__ import annotations

from typing import Any

from ..core.outcome import ToolResult
from .base import BaseTool, ToolContext
from .files import _schema


class UpdateWorkingCheckpointTool(BaseTool):
    name = "update_working_checkpoint"
    description = "Update the run's working checkpoint: durable task constraints, current plan, key paths, and next action."
    parameters = _schema(
        {
            "key_info": {"type": "string", "description": "Concise checkpoint text that should be preserved in context."},
            "reason": {"type": "string", "description": "Why this checkpoint update is needed.", "default": ""},
        },
        ["key_info"],
    )

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        key_info = args.get("key_info") or ""
        ctx.metadata["working_checkpoint"] = key_info
        return ToolResult(True, "Working checkpoint updated.", data={"working_checkpoint": key_info, "reason": args.get("reason", "")})


class AskUserTool(BaseTool):
    name = "ask_user"
    description = "Ask the user a blocking clarification question only when continuing safely or correctly requires it."
    parameters = _schema(
        {
            "question": {"type": "string"},
            "why_needed": {"type": "string", "default": ""},
        },
        ["question"],
    )

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        question = args.get("question") or "Please provide more information."
        ctx.metadata["awaiting_user"] = question
        return ToolResult(True, f"Need user input: {question}", data={"question": question, "why_needed": args.get("why_needed", "")}, should_continue=False)
