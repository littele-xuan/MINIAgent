"""
mcp/tools/internal/calculator.py
──────────────────────────────────
Safe math expression evaluator.
Category: INTERNAL_UTILITY — immutable via governance API.
"""

import math
import operator
from typing import Any

from mcp_lib.registry.models import ToolCategory
from mcp_lib.tools.base import tool_def, make_error_result

# Whitelist of safe builtins exposed to eval
_SAFE_GLOBALS: dict[str, Any] = {
    "__builtins__": {},
    "math": math,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "pow": pow,
    "int": int,
    "float": float,
}


def _handle(expression: str) -> str:
    try:
        result = eval(expression, _SAFE_GLOBALS, {})
        # Format: avoid scientific notation for reasonable numbers
        if isinstance(result, float) and result == int(result) and abs(result) < 1e15:
            return str(int(result))
        return str(result)
    except Exception as e:
        return make_error_result("calculator", e)


def make_entry():
    return tool_def(
        name="calculator",
        description=(
            "Evaluate a mathematical expression. "
            "Supports standard operators (+,-,*,/,**,%), "
            "math module (math.sqrt, math.sin, math.log, …), "
            "and built-ins (abs, round, min, max, sum, pow)."
        ),
        handler=_handle,
        category=ToolCategory.INTERNAL_UTILITY,
        properties={
            "expression": {
                "type": "string",
                "description": "A Python math expression, e.g. 'math.sqrt(144) + 2**8'"
            }
        },
        required=["expression"],
        tags=["math", "calculation", "utility"],
        aliases=["calc", "math"],
    )
