from __future__ import annotations

import os
import subprocess
import sys
import shlex
import tempfile
from typing import Any

from ..core.outcome import ToolResult
from .base import BaseTool, ToolContext
from .files import _schema


class ShellRunTool(BaseTool):
    name = "shell_run"
    description = "Run a shell command inside the workspace. Use for inspection, installing is discouraged, and long-running commands must use a timeout."
    parameters = _schema(
        {
            "command": {"type": "string"},
            "cwd": {"type": "string", "default": "."},
            "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 120, "default": 30},
            "max_output_chars": {"type": "integer", "minimum": 100, "maximum": 60000, "default": 12000},
        },
        ["command"],
    )

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        cwd = ctx.workspace.resolve(args.get("cwd") or ".")
        timeout = int(args.get("timeout_seconds") or 30)
        max_chars = int(args.get("max_output_chars") or 12000)
        cmd = args["command"]
        try:
            completed = subprocess.run(
                cmd,
                shell=True,
                cwd=str(cwd),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            out = (completed.stdout or "")
            err = (completed.stderr or "")
            combined = f"$ {cmd}\n[exit_code] {completed.returncode}\n\n[stdout]\n{out}\n[stderr]\n{err}"
            truncated = len(combined) > max_chars
            if truncated:
                combined = combined[:max_chars] + f"\n... truncated {len(combined)-max_chars} chars"
            return ToolResult(completed.returncode == 0, combined, data={"exit_code": completed.returncode, "truncated": truncated})
        except subprocess.TimeoutExpired as exc:
            return ToolResult(False, f"Command timed out after {timeout}s: {cmd}\nstdout={exc.stdout}\nstderr={exc.stderr}", error="timeout")


class PythonRunTool(BaseTool):
    name = "python_run"
    description = "Run a Python snippet in a temporary file inside the workspace. Prefer this for deterministic checks."
    parameters = _schema(
        {
            "code": {"type": "string"},
            "cwd": {"type": "string", "default": "."},
            "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 120, "default": 30},
            "max_output_chars": {"type": "integer", "minimum": 100, "maximum": 60000, "default": 12000},
        },
        ["code"],
    )

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        cwd = ctx.workspace.resolve(args.get("cwd") or ".")
        code = args.get("code") or ""
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, dir=str(ctx.workspace.root), encoding="utf-8") as f:
            f.write(code)
            temp_path = f.name
        try:
            shell_tool = ShellRunTool()
            rel = os.path.relpath(temp_path, ctx.workspace.root)
            return shell_tool.run(
                {
                    "command": f"{shlex.quote(sys.executable)} {'-S ' if getattr(sys.flags, 'no_site', 0) else ''}{shlex.quote(rel)}",
                    "cwd": str(cwd.relative_to(ctx.workspace.root)),
                    "timeout_seconds": args.get("timeout_seconds") or 30,
                    "max_output_chars": args.get("max_output_chars") or 12000,
                },
                ctx,
            )
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass
