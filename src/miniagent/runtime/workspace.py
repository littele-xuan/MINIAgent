from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..core.errors import WorkspaceSecurityError


@dataclass(slots=True)
class Workspace:
    """Path-safe workspace abstraction used by file and shell tools."""

    root: Path

    @classmethod
    def create(cls, root: str | Path) -> "Workspace":
        p = Path(root).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        return cls(root=p)

    def resolve(self, path: str | Path = ".") -> Path:
        raw = Path(path)
        candidate = raw if raw.is_absolute() else self.root / raw
        resolved = candidate.expanduser().resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise WorkspaceSecurityError(f"Path escapes workspace: {path}") from exc
        return resolved

    def ensure_parent(self, path: str | Path) -> Path:
        resolved = self.resolve(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        return resolved

    def display_path(self, path: str | Path) -> str:
        p = self.resolve(path)
        try:
            return str(p.relative_to(self.root))
        except ValueError:
            return str(p)
