"""
mcp/registry/models.py
─────────────────────
Core data models for the tool registry.

Design principles:
  - ToolCategory controls mutability (protected vs manageable)
  - ToolEntry carries full lifecycle metadata
  - ToolVersion preserves history before each destructive update
  - All models are serializable to dict (handlers excluded)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional


# ══════════════════════════════════════════════════════════════════════════════
# Enumerations
# ══════════════════════════════════════════════════════════════════════════════

class ToolCategory(str, Enum):
    """
    Three-tier category hierarchy.

    ┌─────────────────────────────────────────────────────┐
    │  INTERNAL_SYSTEM   — governance/registry ops        │
    │    Cannot be listed for external modification       │
    │    Cannot be removed / updated / aliased via API    │
    ├─────────────────────────────────────────────────────┤
    │  INTERNAL_UTILITY  — built-in utilities             │
    │    (calculator, search, python, file, weather …)    │
    │    Read-only from governance perspective            │
    ├─────────────────────────────────────────────────────┤
    │  EXTERNAL          — user / LLM managed tools       │
    │    Fully manageable: add / update / remove / alias  │
    └─────────────────────────────────────────────────────┘

    Extension point: subclass or add new values here when
    future tool tiers are needed (e.g. PLUGIN, REMOTE, …).
    """
    INTERNAL_SYSTEM  = "internal_system"
    INTERNAL_UTILITY = "internal_utility"
    EXTERNAL         = "external"

    # ── convenience predicates ──────────────────────────────────────────────

    @property
    def is_protected(self) -> bool:
        """Protected tools cannot be mutated through governance operations."""
        return self in (ToolCategory.INTERNAL_SYSTEM, ToolCategory.INTERNAL_UTILITY)

    @property
    def is_governance(self) -> bool:
        """Governance tools manage the registry itself."""
        return self == ToolCategory.INTERNAL_SYSTEM

    @property
    def label(self) -> str:
        return {
            ToolCategory.INTERNAL_SYSTEM:  "🔒 system",
            ToolCategory.INTERNAL_UTILITY: "🔧 utility",
            ToolCategory.EXTERNAL:         "🌐 external",
        }[self]


class ToolStatus(str, Enum):
    ACTIVE     = "active"
    DISABLED   = "disabled"
    DEPRECATED = "deprecated"


# ══════════════════════════════════════════════════════════════════════════════
# Version snapshot (archived on each destructive update)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ToolVersion:
    """Immutable snapshot of a tool at a point in time."""
    version:      str
    description:  str
    input_schema: dict[str, Any]
    handler:      Callable           # not serialized to disk
    created_at:   datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    changelog:    str = ""

    def to_dict(self) -> dict:
        return {
            "version":     self.version,
            "description": self.description,
            "created_at":  self.created_at.isoformat(),
            "changelog":   self.changelog,
        }


# ══════════════════════════════════════════════════════════════════════════════
# ToolEntry — primary registry record
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ToolEntry:
    """
    Full metadata record for a single tool.

    Extension points:
      - `metadata` dict accepts arbitrary key-value pairs for future fields
      - `tags` can encode sub-categories, teams, environments, …
      - `version_history` stores archived ToolVersion snapshots
    """

    # ── identity ───────────────────────────────────────────────────────────
    name:         str
    version:      str
    description:  str
    input_schema: dict[str, Any]
    handler:      Callable           # callable(**arguments) → Any

    # ── classification ─────────────────────────────────────────────────────
    category:     ToolCategory

    # ── lifecycle ──────────────────────────────────────────────────────────
    enabled:      bool          = True
    status:       ToolStatus    = ToolStatus.ACTIVE
    deprecated:   bool          = False
    replaced_by:  Optional[str] = None

    # ── organisation ───────────────────────────────────────────────────────
    tags:         list[str]     = field(default_factory=list)
    aliases:      list[str]     = field(default_factory=list)

    # ── audit ──────────────────────────────────────────────────────────────
    id:           str           = field(default_factory=lambda: str(uuid.uuid4())[:8])
    created_at:   datetime      = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at:   datetime      = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by:   str           = "system"

    # ── extensibility ──────────────────────────────────────────────────────
    metadata:         dict[str, Any]  = field(default_factory=dict)
    version_history:  list[ToolVersion] = field(default_factory=list)

    # ── derived predicates ──────────────────────────────────────────────────

    @property
    def is_callable(self) -> bool:
        return self.enabled and not self.deprecated

    @property
    def is_protected(self) -> bool:
        return self.category.is_protected

    @property
    def summary(self) -> str:
        status = "✓" if self.enabled else "✗"
        dep = " [deprecated]" if self.deprecated else ""
        return f"{status} {self.name} v{self.version} ({self.category.label}){dep}"

    # ── serialization ───────────────────────────────────────────────────────

    def to_dict(self, include_history: bool = False) -> dict:
        """Serialize to JSON-safe dict (handler excluded)."""
        d = {
            "id":           self.id,
            "name":         self.name,
            "version":      self.version,
            "description":  self.description,
            "input_schema": self.input_schema,
            "category":     self.category.value,
            "enabled":      self.enabled,
            "status":       self.status.value,
            "deprecated":   self.deprecated,
            "replaced_by":  self.replaced_by,
            "tags":         self.tags,
            "aliases":      self.aliases,
            "created_at":   self.created_at.isoformat(),
            "updated_at":   self.updated_at.isoformat(),
            "created_by":   self.created_by,
            "metadata":     self.metadata,
        }
        if include_history:
            d["version_history"] = [v.to_dict() for v in self.version_history]
        return d
