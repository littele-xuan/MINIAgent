"""
mcp_lib/tools/external/__init__.py
────────────────────────────────────
Bootstrap external (user-manageable) seed tools.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp_lib.registry.registry import ToolRegistry

from . import seed_tools


def bootstrap_external_tools(registry: "ToolRegistry") -> None:
    """Register seed external tools."""
    for entry in seed_tools.make_entries():
        registry.register(entry, overwrite=False)
