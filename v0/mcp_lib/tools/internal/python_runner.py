"""
mcp/tools/internal/python_runner.py
─────────────────────────────────────
Execute Python code snippets in an isolated namespace.
Category: INTERNAL_UTILITY — immutable via governance API.

⚠  Production note: replace exec() with a proper sandbox
   (e.g. RestrictedPython, gVisor, or a subprocess with
   resource limits) before deploying in untrusted contexts.
"""

import io
import sys
import traceback
from mcp_lib.registry.models import ToolCategory
from mcp_lib.tools.base import tool_def


def _handle(code: str, timeout: int = 10) -> str:
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = buf_out = io.StringIO()
    sys.stderr = buf_err = io.StringIO()
    try:
        namespace: dict = {}
        exec(compile(code, "<mcp_tool>", "exec"), namespace)
        stdout = buf_out.getvalue()
        stderr = buf_err.getvalue()
        parts = []
        if stdout:
            parts.append(stdout)
        if stderr:
            parts.append(f"[stderr]\n{stderr}")
        return "\n".join(parts) if parts else "(code executed — no output)"
    except Exception:
        tb = traceback.format_exc()
        return f"[python_runner ERROR]\n{tb}"
    finally:
        sys.stdout, sys.stderr = old_out, old_err


def make_entry():
    return tool_def(
        name="run_python",
        description=(
            "Execute a Python code snippet in an isolated namespace. "
            "Returns stdout/stderr output. "
            "Useful for data processing, calculations, and quick scripts."
        ),
        handler=_handle,
        category=ToolCategory.INTERNAL_UTILITY,
        properties={
            "code": {
                "type": "string",
                "description": "Python source code to execute"
            },
        },
        required=["code"],
        tags=["python", "code", "execution", "utility"],
        aliases=["python", "exec_python"],
    )
