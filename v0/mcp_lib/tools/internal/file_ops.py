"""
mcp/tools/internal/file_ops.py
────────────────────────────────
File read / write / list operations.
Category: INTERNAL_UTILITY — immutable via governance API.
"""

import json
import os
from pathlib import Path

from mcp_lib.registry.models import ToolCategory
from mcp_lib.tools.base import tool_def, make_error_result


def _read(path: str, encoding: str = "utf-8") -> str:
    try:
        content = Path(path).read_text(encoding=encoding)
        size = len(content)
        if size > 50_000:
            content = content[:50_000] + f"\n\n… (truncated, total {size} chars)"
        return content
    except Exception as e:
        return make_error_result("read_file", e)


def _write(path: str, content: str, encoding: str = "utf-8") -> str:
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding=encoding)
        return f"✓ Written {len(content)} chars to {path}"
    except Exception as e:
        return make_error_result("write_file", e)


def _list_dir(path: str = ".", pattern: str = "*") -> str:
    try:
        p = Path(path)
        entries = sorted(p.glob(pattern))
        if not entries:
            return f"(empty directory: {path})"
        lines = []
        for e in entries[:200]:
            kind = "📁" if e.is_dir() else "📄"
            size = e.stat().st_size if e.is_file() else 0
            lines.append(f"{kind} {e.name}  ({size} bytes)" if e.is_file() else f"{kind} {e.name}/")
        return "\n".join(lines)
    except Exception as e:
        return make_error_result("list_dir", e)


def _append(path: str, content: str, encoding: str = "utf-8") -> str:
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding=encoding) as f:
            f.write(content)
        return f"✓ Appended {len(content)} chars to {path}"
    except Exception as e:
        return make_error_result("append_file", e)


def make_entries():
    return [
        tool_def(
            name="read_file",
            description="Read a file and return its text content (up to 50 000 chars).",
            handler=_read,
            category=ToolCategory.INTERNAL_UTILITY,
            properties={
                "path":     {"type": "string", "description": "File path to read"},
                "encoding": {"type": "string", "description": "File encoding (default utf-8)"},
            },
            required=["path"],
            tags=["file", "io", "read"],
        ),
        tool_def(
            name="write_file",
            description="Write text content to a file (overwrites if exists).",
            handler=_write,
            category=ToolCategory.INTERNAL_UTILITY,
            properties={
                "path":     {"type": "string", "description": "File path to write"},
                "content":  {"type": "string", "description": "Text content to write"},
                "encoding": {"type": "string", "description": "File encoding (default utf-8)"},
            },
            required=["path", "content"],
            tags=["file", "io", "write"],
        ),
        tool_def(
            name="append_file",
            description="Append text content to a file.",
            handler=_append,
            category=ToolCategory.INTERNAL_UTILITY,
            properties={
                "path":     {"type": "string", "description": "File path"},
                "content":  {"type": "string", "description": "Text to append"},
            },
            required=["path", "content"],
            tags=["file", "io", "append"],
        ),
        tool_def(
            name="list_dir",
            description="List files and directories at the given path.",
            handler=_list_dir,
            category=ToolCategory.INTERNAL_UTILITY,
            properties={
                "path":    {"type": "string", "description": "Directory path (default '.')"},
                "pattern": {"type": "string", "description": "Glob pattern (default '*')"},
            },
            required=[],
            tags=["file", "directory", "ls"],
            aliases=["ls", "dir"],
        ),
    ]
