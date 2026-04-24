from __future__ import annotations

import fnmatch
import re
from pathlib import Path
from typing import Any

from ..core.outcome import ToolResult
from .base import BaseTool, ToolContext


def _schema(props: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": props,
        "required": required or [],
        "additionalProperties": False,
    }


class ListDirTool(BaseTool):
    name = "list_dir"
    description = "List files and directories under a workspace-relative path. Use before reading unfamiliar projects."
    parameters = _schema(
        {
            "path": {"type": "string", "description": "Workspace-relative directory path.", "default": "."},
            "max_entries": {"type": "integer", "minimum": 1, "maximum": 500, "default": 120},
        }
    )

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = ctx.workspace.resolve(args.get("path") or ".")
        if not path.exists():
            return ToolResult(False, f"Directory does not exist: {path}", error="not_found")
        if not path.is_dir():
            return ToolResult(False, f"Not a directory: {path}", error="not_directory")
        max_entries = int(args.get("max_entries") or 120)
        entries = []
        for child in sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))[:max_entries]:
            rel = child.relative_to(ctx.workspace.root)
            entries.append({"path": str(rel), "type": "dir" if child.is_dir() else "file", "size": child.stat().st_size if child.is_file() else None})
        lines = [f"{e['type']:>4}  {e['path']}" + (f"  ({e['size']} bytes)" if e["size"] is not None else "") for e in entries]
        return ToolResult(True, "\n".join(lines) if lines else "<empty directory>", data={"entries": entries})


class SearchFilesTool(BaseTool):
    name = "search_files"
    description = "Find files by glob pattern under a workspace-relative root."
    parameters = _schema(
        {
            "root": {"type": "string", "default": "."},
            "pattern": {"type": "string", "description": "Glob pattern, e.g. **/*.py or README*.", "default": "**/*"},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 100},
        }
    )

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        root = ctx.workspace.resolve(args.get("root") or ".")
        pattern = args.get("pattern") or "**/*"
        max_results = int(args.get("max_results") or 100)
        if not root.exists():
            return ToolResult(False, f"Root does not exist: {root}", error="not_found")
        files = []
        for p in root.rglob("*"):
            if len(files) >= max_results:
                break
            if p.is_file() and fnmatch.fnmatch(str(p.relative_to(root)), pattern):
                files.append(str(p.relative_to(ctx.workspace.root)))
        return ToolResult(True, "\n".join(files) if files else "<no matching files>", data={"files": files})


class FileReadTool(BaseTool):
    name = "file_read"
    description = "Read a text file with line numbers. Supports pagination using start_line and max_lines."
    parameters = _schema(
        {
            "path": {"type": "string", "description": "Workspace-relative file path."},
            "start_line": {"type": "integer", "minimum": 1, "default": 1},
            "max_lines": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 220},
        },
        ["path"],
    )

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = ctx.workspace.resolve(args["path"])
        if not path.exists() or not path.is_file():
            return ToolResult(False, f"File not found: {args['path']}", error="not_found")
        start = max(1, int(args.get("start_line") or 1))
        max_lines = max(1, min(1000, int(args.get("max_lines") or 220)))
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        selected = lines[start - 1 : start - 1 + max_lines]
        numbered = [f"{idx:>5}: {line}" for idx, line in enumerate(selected, start=start)]
        content = "\n".join(numbered)
        if start - 1 + max_lines < len(lines):
            content += f"\n... ({len(lines) - (start - 1 + max_lines)} more lines)"
        return ToolResult(True, content, data={"path": args["path"], "total_lines": len(lines), "start_line": start, "returned_lines": len(selected)})


class ReadManyFilesTool(BaseTool):
    name = "read_many_files"
    description = "Read several small text files at once. Avoid this for large files; use file_read pagination instead."
    parameters = _schema(
        {
            "paths": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 20},
            "max_chars_per_file": {"type": "integer", "minimum": 100, "maximum": 20000, "default": 4000},
        },
        ["paths"],
    )

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        max_chars = int(args.get("max_chars_per_file") or 4000)
        chunks: list[str] = []
        data: list[dict[str, Any]] = []
        for raw in args["paths"]:
            path = ctx.workspace.resolve(raw)
            if not path.exists() or not path.is_file():
                chunks.append(f"## {raw}\n<not found>")
                data.append({"path": raw, "ok": False})
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            truncated = len(text) > max_chars
            body = text[:max_chars]
            chunks.append(f"## {raw}\n{body}" + (f"\n... truncated {len(text)-max_chars} chars" if truncated else ""))
            data.append({"path": raw, "ok": True, "chars": len(text), "truncated": truncated})
        return ToolResult(True, "\n\n".join(chunks), data={"files": data})


class FileWriteTool(BaseTool):
    name = "file_write"
    description = "Write UTF-8 text to a file under the workspace. Supports overwrite, append, and prepend."
    parameters = _schema(
        {
            "path": {"type": "string"},
            "content": {"type": "string"},
            "mode": {"type": "string", "enum": ["overwrite", "append", "prepend"], "default": "overwrite"},
        },
        ["path", "content"],
    )

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = ctx.workspace.ensure_parent(args["path"])
        mode = args.get("mode") or "overwrite"
        content = args.get("content") or ""
        if mode == "overwrite":
            path.write_text(content, encoding="utf-8")
        elif mode == "append":
            old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
            path.write_text(old + content, encoding="utf-8")
        elif mode == "prepend":
            old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
            path.write_text(content + old, encoding="utf-8")
        else:
            return ToolResult(False, f"Unsupported mode: {mode}", error="bad_mode")
        return ToolResult(True, f"Wrote {len(content)} chars to {args['path']} using mode={mode}.", data={"path": args["path"], "mode": mode, "chars": len(content)})


class FilePatchTool(BaseTool):
    name = "file_patch"
    description = "Patch a file by replacing exact text. This is safer than rewriting whole files."
    parameters = _schema(
        {
            "path": {"type": "string"},
            "old_text": {"type": "string", "description": "Exact text to replace."},
            "new_text": {"type": "string", "description": "Replacement text."},
            "occurrence": {"type": "integer", "minimum": 0, "default": 1, "description": "1-based occurrence. Use 0 to replace all occurrences."},
        },
        ["path", "old_text", "new_text"],
    )

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = ctx.workspace.resolve(args["path"])
        if not path.exists() or not path.is_file():
            return ToolResult(False, f"File not found: {args['path']}", error="not_found")
        old_text = args.get("old_text") or ""
        new_text = args.get("new_text") or ""
        if not old_text:
            return ToolResult(False, "old_text must not be empty", error="empty_old_text")
        text = path.read_text(encoding="utf-8", errors="replace")
        count = text.count(old_text)
        if count == 0:
            return ToolResult(False, "old_text not found; read the file again and use an exact snippet.", data={"matches": 0}, error="not_found")
        occurrence = int(args.get("occurrence") or 1)
        if occurrence == 0:
            updated = text.replace(old_text, new_text)
            replaced = count
        else:
            if occurrence > count:
                return ToolResult(False, f"Requested occurrence {occurrence}, but only {count} matches found.", data={"matches": count}, error="bad_occurrence")
            parts = text.split(old_text)
            updated = old_text.join(parts[:occurrence]) + new_text + old_text.join(parts[occurrence:])
            replaced = 1
        path.write_text(updated, encoding="utf-8")
        return ToolResult(True, f"Patched {args['path']}: replaced {replaced} occurrence(s).", data={"path": args["path"], "matches_before": count, "replaced": replaced})


class GrepTextTool(BaseTool):
    name = "grep_text"
    description = "Search text files for a regex pattern under the workspace. Returns path, line number and preview."
    parameters = _schema(
        {
            "pattern": {"type": "string"},
            "root": {"type": "string", "default": "."},
            "glob": {"type": "string", "default": "**/*"},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 500, "default": 80},
            "ignore_case": {"type": "boolean", "default": False},
        },
        ["pattern"],
    )

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        root = ctx.workspace.resolve(args.get("root") or ".")
        flags = re.IGNORECASE if args.get("ignore_case") else 0
        rx = re.compile(args["pattern"], flags)
        glob_pat = args.get("glob") or "**/*"
        max_results = int(args.get("max_results") or 80)
        results = []
        for path in root.rglob("*"):
            if len(results) >= max_results:
                break
            if not path.is_file() or not fnmatch.fnmatch(str(path.relative_to(root)), glob_pat):
                continue
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except Exception:
                continue
            for i, line in enumerate(lines, start=1):
                if rx.search(line):
                    results.append({"path": str(path.relative_to(ctx.workspace.root)), "line": i, "text": line[:240]})
                    if len(results) >= max_results:
                        break
        content = "\n".join(f"{r['path']}:{r['line']}: {r['text']}" for r in results) if results else "<no matches>"
        return ToolResult(True, content, data={"matches": results})
