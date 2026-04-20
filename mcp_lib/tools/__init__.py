"""
mcp_lib/tools/__init__.py
──────────────────────────
Top-level bootstrap: registers internal tools + external seed tools.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp_lib.registry.registry import ToolRegistry

from .internal import bootstrap_internal_tools
from .external import bootstrap_external_tools


def bootstrap_all_tools(registry: "ToolRegistry") -> None:
    """
    Register all tools into the registry.

    Order:
      1. Internal utility tools (calculator, search, python, file, weather)
      2. Internal system / governance tools (tool_add, tool_list, …)
      3. External seed tools (random_joke, uuid_gen, …)
    """
    bootstrap_internal_tools(registry)
    bootstrap_external_tools(registry)
