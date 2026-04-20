"""
mcp/governance/manager.py
──────────────────────────
GovernanceManager — high-level Python API for tool lifecycle management.

Wraps ToolRegistry with:
  - Audit logging
  - Batch operations
  - Import / export
  - Plugin hooks

This layer is for programmatic (Python code) use.
For LLM-driven management use the registry_ops governance tools.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from mcp_lib.registry.models import ToolCategory, ToolEntry
from mcp_lib.registry.registry import ToolRegistry, ToolProtectedError
from mcp_lib.tools.base import tool_def

logger = logging.getLogger(__name__)


class GovernanceManager:
    """
    High-level governance API.

    Usage:
        mgr = GovernanceManager(registry)
        mgr.add_tool("my_tool", "desc", handler_fn, tags=["demo"])
        mgr.disable_tool("my_tool")
        mgr.export_external("backup.json")
    """

    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self._audit_log: list[dict] = []
        # Register audit hook
        registry.add_hook("on_register", self._audit)
        registry.add_hook("on_update",   self._audit)
        registry.add_hook("on_remove",   self._audit)
        registry.add_hook("on_call",     self._audit_call)

    # ── audit ────────────────────────────────────────────────────────────────

    def _audit(self, **kwargs) -> None:
        entry = kwargs.get("entry") or kwargs.get("name", "?")
        name  = entry.name if isinstance(entry, ToolEntry) else str(entry)
        self._audit_log.append({
            "ts":     datetime.utcnow().isoformat(),
            "event":  "mutation",
            "tool":   name,
            "detail": str(kwargs.get("changes", "")),
        })

    def _audit_call(self, **kwargs) -> None:
        self._audit_log.append({
            "ts":    datetime.utcnow().isoformat(),
            "event": "call",
            "tool":  kwargs.get("name", "?"),
        })

    def get_audit_log(self, last_n: int = 50) -> list[dict]:
        return self._audit_log[-last_n:]

    # ── tool lifecycle ───────────────────────────────────────────────────────

    def add_tool(
        self,
        name:        str,
        description: str,
        handler:     Callable,
        version:     str            = "1.0.0",
        tags:        list[str]      = None,
        aliases:     list[str]      = None,
        metadata:    dict[str, Any] = None,
        properties:  dict[str, Any] = None,
        required:    list[str]      = None,
    ) -> ToolEntry:
        entry = tool_def(
            name=name,
            description=description,
            handler=handler,
            category=ToolCategory.EXTERNAL,
            properties=properties or {},
            required=required or [],
            version=version,
            tags=tags or [],
            aliases=aliases or [],
            metadata=metadata or {},
            created_by="governance_manager",
        )
        self.registry.register(entry)
        logger.info(f"GovernanceManager: added tool '{name}'")
        return entry

    def update_tool(self, name: str, **kwargs) -> ToolEntry:
        return self.registry.update(name, **kwargs)

    def remove_tool(self, name: str) -> ToolEntry:
        return self.registry.remove(name)

    def enable_tool(self, name: str) -> ToolEntry:
        return self.registry.enable(name)

    def disable_tool(self, name: str) -> ToolEntry:
        return self.registry.disable(name)

    def deprecate_tool(self, name: str, replaced_by: Optional[str] = None) -> ToolEntry:
        return self.registry.deprecate(name, replaced_by=replaced_by)

    def alias_tool(self, name: str, alias: str) -> None:
        self.registry.add_alias(name, alias)

    def merge_tools(self, source: str, target: str, keep_source: bool = False) -> None:
        self.registry.merge(source, target, keep_source=keep_source)

    def list_versions(self, name: str) -> list[dict]:
        return self.registry.list_versions(name)

    # ── batch operations ─────────────────────────────────────────────────────

    def enable_by_tag(self, tag: str) -> list[str]:
        """Enable all tools matching a tag. Returns list of tool names."""
        tools = self.registry.list_tools(enabled_only=False, include_deprecated=True)
        changed = []
        for t in tools:
            if tag in t.tags and not t.enabled and not t.is_protected:
                try:
                    self.registry.enable(t.name)
                    changed.append(t.name)
                except ToolProtectedError:
                    pass
        return changed

    def disable_by_tag(self, tag: str) -> list[str]:
        """Disable all tools matching a tag."""
        tools = self.registry.list_tools(enabled_only=True, include_deprecated=False)
        changed = []
        for t in tools:
            if tag in t.tags and not t.is_protected:
                try:
                    self.registry.disable(t.name)
                    changed.append(t.name)
                except ToolProtectedError:
                    pass
        return changed

    # ── import / export ──────────────────────────────────────────────────────

    def export_external(self, path: Path) -> int:
        """Export EXTERNAL tool metadata to JSON (no handlers)."""
        tools = self.registry.list_tools(
            enabled_only=False,
            category=ToolCategory.EXTERNAL,
            include_deprecated=True,
        )
        data = [t.to_dict(include_history=True) for t in tools]
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str))
        return len(data)

    def summary(self) -> str:
        """Human-readable registry summary."""
        s = self.registry.stats()
        lines = ["=== Registry Summary ==="]
        lines.append(f"  Total tools : {s['total']}")
        lines.append(f"  Enabled     : {s['enabled']}")
        lines.append(f"  Disabled    : {s['disabled']}")
        lines.append(f"  Deprecated  : {s['deprecated']}")
        lines.append("  By category :")
        for cat, count in s["by_category"].items():
            lines.append(f"    {cat:20s}: {count}")
        lines.append(f"  Audit events: {len(self._audit_log)}")
        return "\n".join(lines)
