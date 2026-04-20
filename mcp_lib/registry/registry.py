"""
mcp/registry/registry.py
────────────────────────
ToolRegistry — the single source of truth for all tools.

Architecture:
  _tools    : dict[name → ToolEntry]        primary store
  _aliases  : dict[alias → canonical_name]  fast alias resolution
  _hooks    : dict[event → list[callback]]  plugin / observability hooks

Thread-safety: all mutations hold _lock (RLock, re-entrant so hooks can query).
"""

from __future__ import annotations

import inspect
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from .models import ToolCategory, ToolEntry, ToolStatus, ToolVersion


# ── Exceptions ────────────────────────────────────────────────────────────────

class RegistryError(Exception):
    """Base error for all registry operations."""

class ToolNotFoundError(RegistryError):
    pass

class ToolProtectedError(RegistryError):
    """Raised when governance API tries to mutate a protected tool."""

class ToolAlreadyExistsError(RegistryError):
    pass

class ToolDisabledError(RegistryError):
    pass


# ══════════════════════════════════════════════════════════════════════════════
# ToolRegistry
# ══════════════════════════════════════════════════════════════════════════════

class ToolRegistry:
    """
    Central tool registry.

    Lifecycle:
        registry = ToolRegistry()
        registry.register(entry)          # add tool
        result = await registry.call(name, args)  # dispatch

    Governance (external tools only):
        registry.update(name, description="…")
        registry.disable(name)
        registry.deprecate(name, replaced_by="…")
        registry.add_alias(name, alias)
        registry.merge(source, target)
        registry.remove(name)

    Extension hooks:
        registry.add_hook("on_call", fn)  # fn(name, arguments, result)
    """

    # ── constructor ──────────────────────────────────────────────────────────

    def __init__(self, persistence_path: Optional[Path] = None):
        self._tools:   dict[str, ToolEntry] = {}
        self._aliases: dict[str, str]       = {}   # alias → canonical name
        self._lock = threading.RLock()
        self._persistence_path = persistence_path

        # Hook slots — each value is a list of callables
        # Signature: fn(name, entry, **kwargs)
        self._hooks: dict[str, list[Callable]] = {
            "on_register": [],
            "on_remove":   [],
            "on_update":   [],
            "on_call":     [],
            "on_error":    [],
        }

    # ── hook management ──────────────────────────────────────────────────────

    def add_hook(self, event: str, fn: Callable) -> None:
        """Register a callback for a registry event."""
        self._hooks.setdefault(event, []).append(fn)

    def _fire(self, event: str, **kwargs) -> None:
        for fn in self._hooks.get(event, []):
            try:
                fn(**kwargs)
            except Exception:
                pass   # hooks must never crash the main path

    # ── core CRUD ────────────────────────────────────────────────────────────

    def register(self, entry: ToolEntry, overwrite: bool = False) -> None:
        """
        Add a ToolEntry to the registry.
        Used by internal bootstrap AND governance add_tool.
        """
        with self._lock:
            if entry.name in self._tools and not overwrite:
                raise ToolAlreadyExistsError(
                    f"Tool '{entry.name}' already exists. "
                    f"Pass overwrite=True to replace."
                )
            self._tools[entry.name] = entry
            for alias in entry.aliases:
                self._aliases[alias] = entry.name
        self._fire("on_register", name=entry.name, entry=entry)

    def get(self, name: str) -> ToolEntry:
        """Resolve name (or alias) → ToolEntry. Raises ToolNotFoundError."""
        with self._lock:
            canonical = self._aliases.get(name, name)
            entry = self._tools.get(canonical)
        if entry is None:
            raise ToolNotFoundError(f"Tool '{name}' not found")
        return entry

    def has(self, name: str) -> bool:
        with self._lock:
            canonical = self._aliases.get(name, name)
            return canonical in self._tools

    def remove(self, name: str) -> ToolEntry:
        """Remove a tool. Protected tools raise ToolProtectedError."""
        entry = self.get(name)
        if entry.is_protected:
            raise ToolProtectedError(
                f"'{name}' is a protected {entry.category.label} tool "
                f"and cannot be removed."
            )
        with self._lock:
            del self._tools[entry.name]
            dead_aliases = [a for a, c in self._aliases.items() if c == entry.name]
            for a in dead_aliases:
                del self._aliases[a]
        self._fire("on_remove", name=name, entry=entry)
        return entry

    def update(self, name: str, changelog: str = "", **kwargs) -> ToolEntry:
        """
        Update mutable fields of an external tool.

        Allowed kwargs:
            description, version, input_schema, handler,
            enabled, tags, aliases, deprecated, replaced_by,
            metadata, status
        """
        entry = self.get(name)
        if entry.is_protected:
            raise ToolProtectedError(
                f"'{name}' is protected and cannot be updated via governance API."
            )
        ALLOWED = {
            "description", "version", "input_schema", "handler",
            "enabled", "tags", "aliases", "deprecated",
            "replaced_by", "metadata", "status",
        }
        bad_keys = set(kwargs) - ALLOWED
        if bad_keys:
            raise RegistryError(f"Unknown update fields: {bad_keys}")

        with self._lock:
            # Archive current version snapshot when schema or handler changes
            if "handler" in kwargs or "input_schema" in kwargs:
                entry.version_history.append(ToolVersion(
                    version=entry.version,
                    description=entry.description,
                    input_schema=entry.input_schema,
                    handler=entry.handler,
                    created_at=entry.created_at,
                    changelog=changelog or "archived before update",
                ))
            for k, v in kwargs.items():
                setattr(entry, k, v)
            entry.updated_at = datetime.utcnow()
            # Re-index aliases if they changed
            if "aliases" in kwargs:
                stale = [a for a, c in self._aliases.items() if c == entry.name]
                for a in stale:
                    del self._aliases[a]
                for a in entry.aliases:
                    self._aliases[a] = entry.name
        self._fire("on_update", name=name, entry=entry, changes=kwargs)
        return entry

    # ── governance shortcuts ─────────────────────────────────────────────────

    def enable(self, name: str) -> ToolEntry:
        return self.update(name, enabled=True, status=ToolStatus.ACTIVE)

    def disable(self, name: str) -> ToolEntry:
        return self.update(name, enabled=False, status=ToolStatus.DISABLED)

    def deprecate(self, name: str, replaced_by: Optional[str] = None) -> ToolEntry:
        return self.update(
            name,
            deprecated=True,
            status=ToolStatus.DEPRECATED,
            replaced_by=replaced_by,
        )

    def add_alias(self, name: str, alias: str) -> None:
        entry = self.get(name)
        if entry.is_protected:
            raise ToolProtectedError(f"Cannot alias protected tool '{name}'")
        with self._lock:
            if alias not in entry.aliases:
                entry.aliases.append(alias)
            self._aliases[alias] = entry.name

    def merge(self, source: str, target: str, keep_source: bool = False) -> None:
        """
        Merge source tool's aliases and redirect source name → target.
        Optionally keep the source entry (disabled).
        """
        src = self.get(source)
        tgt = self.get(target)
        if src.is_protected or tgt.is_protected:
            raise ToolProtectedError("Cannot merge protected tools")
        with self._lock:
            for alias in src.aliases:
                if alias not in tgt.aliases:
                    tgt.aliases.append(alias)
                self._aliases[alias] = tgt.name
            self._aliases[src.name] = tgt.name  # old name now resolves to target
        if not keep_source:
            with self._lock:
                del self._tools[src.name]
        else:
            self.disable(source)

    # ── query ────────────────────────────────────────────────────────────────

    def list_tools(
        self,
        enabled_only:       bool = True,
        category:           Optional[ToolCategory] = None,
        tags:               Optional[list[str]] = None,
        include_deprecated: bool = False,
        include_system:     bool = True,
    ) -> list[ToolEntry]:
        with self._lock:
            tools = list(self._tools.values())
        if enabled_only:
            tools = [t for t in tools if t.enabled]
        if not include_deprecated:
            tools = [t for t in tools if not t.deprecated]
        if not include_system:
            tools = [t for t in tools if not t.category.is_governance]
        if category:
            tools = [t for t in tools if t.category == category]
        if tags:
            tools = [t for t in tools if any(tag in t.tags for tag in tags)]
        return tools

    def list_versions(self, name: str) -> list[dict]:
        entry = self.get(name)
        history = [v.to_dict() for v in entry.version_history]
        history.append({
            "version":    entry.version,
            "description": entry.description,
            "created_at": entry.updated_at.isoformat(),
            "changelog":  "(current)",
        })
        return history

    def search(self, query: str) -> list[ToolEntry]:
        q = query.lower()
        with self._lock:
            return [
                t for t in self._tools.values()
                if q in t.name.lower()
                or q in t.description.lower()
                or any(q in tag.lower() for tag in t.tags)
            ]

    def stats(self) -> dict:
        with self._lock:
            tools = list(self._tools.values())
        return {
            "total":      len(tools),
            "enabled":    sum(1 for t in tools if t.enabled),
            "disabled":   sum(1 for t in tools if not t.enabled),
            "deprecated": sum(1 for t in tools if t.deprecated),
            "by_category": {
                cat.value: sum(1 for t in tools if t.category == cat)
                for cat in ToolCategory
            },
        }

    # ── call dispatch ────────────────────────────────────────────────────────

    async def call(self, name: str, arguments: dict[str, Any]) -> Any:
        """
        Resolve name/alias → handler → execute.

        Supports both sync and async handlers transparently.
        """
        entry = self.get(name)
        if not entry.enabled:
            raise ToolDisabledError(f"Tool '{entry.name}' is disabled.")
        if entry.deprecated:
            repl = f" Use '{entry.replaced_by}' instead." if entry.replaced_by else ""
            raise RegistryError(f"Tool '{entry.name}' is deprecated.{repl}")

        try:
            if inspect.iscoroutinefunction(entry.handler):
                result = await entry.handler(**arguments)
            else:
                result = entry.handler(**arguments)
            self._fire("on_call", name=name, entry=entry,
                       arguments=arguments, result=result)
            return result
        except Exception as e:
            self._fire("on_error", name=name, entry=entry,
                       arguments=arguments, error=e)
            raise

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, path: Optional[Path] = None) -> None:
        """Persist registry metadata (no handlers) to JSON."""
        target = path or self._persistence_path
        if target is None:
            return
        with self._lock:
            data = {
                name: entry.to_dict(include_history=True)
                for name, entry in self._tools.items()
            }
        target.write_text(
            json.dumps(data, indent=2, ensure_ascii=False, default=str)
        )

    # ── dunder ───────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._tools)

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"ToolRegistry("
            f"{s['total']} tools, "
            f"{s['enabled']} enabled, "
            f"{s['by_category']})"
        )
