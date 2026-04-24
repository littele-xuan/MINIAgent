from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass(slots=True)
class ToolPolicy:
    disabled_tools: set[str] = field(default_factory=set)
    allow_governance: bool = True
    blocked_risks: set[str] = field(default_factory=set)

    def is_allowed(self, name: str, *, visible_names: set[str], risk: str | None = None) -> bool:
        if name not in visible_names or name in self.disabled_tools:
            return False
        if risk and risk in self.blocked_risks:
            return False
        return True

    def disable(self, name: str) -> None:
        self.disabled_tools.add(name)

    def enable(self, name: str) -> None:
        self.disabled_tools.discard(name)

    def block_risks(self, risks: Iterable[str]) -> None:
        self.blocked_risks.update(str(r) for r in risks)
