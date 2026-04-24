"""
mcp_lib/tools/internal/__init__.py
────────────────────────────────────
Bootstrap: registers all internal tools (utility + system) into a registry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp_lib.registry.registry import ToolRegistry

from . import (
    calculator,
    file_ops,
    python_runner,
    registry_ops,
    weather,
    web_search,
)


def bootstrap_internal_tools(registry: "ToolRegistry") -> None:
    """Register all internal tools into the registry."""
    # ── INTERNAL_UTILITY tools ────────────────────────────────────────────────
    utility_entries = [
        calculator.make_entry(),
        python_runner.make_entry(),
        web_search.make_entry(),
        weather.make_entry(),
        *file_ops.make_entries(),
    ]
    for entry in utility_entries:
        registry.register(entry, overwrite=False)

    # ── INTERNAL_SYSTEM (governance) tools ───────────────────────────────────
    for entry in registry_ops.make_entries(registry):
        registry.register(entry, overwrite=False)
