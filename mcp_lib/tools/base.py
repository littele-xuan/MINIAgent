"""
mcp_lib/tools/base.py
─────────────────
Helpers for defining tools cleanly.

Usage:
    from mcp.tools.base import tool_def
    from mcp.registry import ToolEntry, ToolCategory

    entry = tool_def(
        name="calculator",
        version="1.0.0",
        description="Evaluate math expressions",
        category=ToolCategory.INTERNAL_UTILITY,
        tags=["math"],
        handler=my_fn,
        properties={
            "expression": {"type": "string", "description": "Math expression"}
        },
        required=["expression"],
    )
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from mcp_lib.registry.models import ToolCategory, ToolEntry, ToolStatus


def tool_def(
    name:        str,
    description: str,
    handler:     Callable,
    category:    ToolCategory,
    properties:  dict[str, Any],
    required:    list[str]       = None,
    version:     str             = "1.0.0",
    tags:        list[str]       = None,
    aliases:     list[str]       = None,
    metadata:    dict[str, Any]  = None,
    created_by:  str             = "system",
) -> ToolEntry:
    """
    Factory function for creating a ToolEntry with a standard JSON Schema.

    Extension point: add new keyword arguments here as new ToolEntry fields
    are introduced, keeping backward compatibility via defaults.
    """
    return ToolEntry(
        name=name,
        version=version,
        description=description,
        input_schema={
            "type": "object",
            "properties": properties,
            "required": required or [],
        },
        handler=handler,
        category=category,
        tags=tags or [],
        aliases=aliases or [],
        metadata=metadata or {},
        created_by=created_by,
    )


def make_error_result(tool_name: str, error: Exception) -> str:
    """Standardized error string returned by tool handlers."""
    return f"[{tool_name} ERROR] {type(error).__name__}: {error}"


def register_all(registry, entries: list[ToolEntry]) -> None:
    """Batch-register a list of ToolEntry objects."""
    for entry in entries:
        registry.register(entry, overwrite=False)
