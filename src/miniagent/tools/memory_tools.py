from __future__ import annotations

from typing import Any
from dataclasses import asdict

from ..core.outcome import ToolResult
from ..memory.models import MemoryItem
from .base import BaseTool, ToolContext
from .files import _schema


class MemoryRecallTool(BaseTool):
    name = "memory_recall"
    description = "Recall relevant long-term memory for the current task."
    parameters = _schema(
        {
            "query": {"type": "string"},
            "max_items": {"type": "integer", "minimum": 1, "maximum": 12, "default": 6},
        },
        ["query"],
    )

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        result = ctx.memory.recall(args.get("query") or "", max_items=int(args.get("max_items") or 6))
        return ToolResult(True, result.format_for_prompt(), data={"items": result.items})


class MemoryProposeUpdateTool(BaseTool):
    name = "memory_propose_update"
    description = "Propose a durable memory update. Requires evidence from the user or tool output; proposals are stored pending review/commit."
    parameters = _schema(
        {
            "content": {"type": "string"},
            "evidence": {"type": "string"},
            "layer": {"type": "string", "enum": ["facts", "skills"], "default": "facts"},
            "source": {"type": "string", "default": "agent"},
            "tags": {"type": "array", "items": {"type": "string"}, "default": []},
        },
        ["content", "evidence"],
    )

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        item = MemoryItem(
            layer=args.get("layer") or "facts",
            content=args.get("content") or "",
            evidence=args.get("evidence") or "",
            source=args.get("source") or "agent",
            tags=list(args.get("tags") or []),
        )
        ok, msg, proposal_id = ctx.memory.propose_update(item)
        return ToolResult(ok, msg, data={"proposal_id": proposal_id, "item": asdict(item)}, error=None if ok else msg)


class MemoryCommitUpdateTool(BaseTool):
    name = "memory_commit_update"
    description = "Commit a pending memory proposal by id, or directly commit content with evidence. Use only for stable cross-session facts or reusable skills."
    parameters = _schema(
        {
            "proposal_id": {"type": "string", "default": ""},
            "content": {"type": "string", "default": ""},
            "evidence": {"type": "string", "default": ""},
            "layer": {"type": "string", "enum": ["facts", "skills"], "default": "facts"},
            "tags": {"type": "array", "items": {"type": "string"}, "default": []},
        }
    )

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        proposal_id = args.get("proposal_id") or ""
        if proposal_id:
            ok, msg = ctx.memory.commit_pending(proposal_id)
            return ToolResult(ok, msg, data={"proposal_id": proposal_id}, error=None if ok else msg)
        item = MemoryItem(layer=args.get("layer") or "facts", content=args.get("content") or "", evidence=args.get("evidence") or "", source="direct_commit", tags=list(args.get("tags") or []))
        ok, msg = ctx.memory.commit_item(item)
        return ToolResult(ok, msg, data={"item_id": item.id}, error=None if ok else msg)
