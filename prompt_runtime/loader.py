from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from .base import BasePromptProvider


class FilePromptProvider(BasePromptProvider):
    """Loads prompts from a dedicated prompts/ directory."""

    def __init__(self, prompts_root: str | Path) -> None:
        self.prompts_root = Path(prompts_root)
        self._cache: dict[str, str] = {}

    def get(self, name: str, *, variables: Mapping[str, Any] | None = None) -> str:
        text = self._cache.get(name)
        if text is None:
            path = self.prompts_root / f"{name}.md"
            if not path.exists():
                raise FileNotFoundError(f"Prompt not found: {path}")
            text = path.read_text(encoding='utf-8').strip()
            self._cache[name] = text
        if not variables:
            return text
        return text.format(**variables)


@lru_cache(maxsize=1)
def get_default_prompt_provider() -> FilePromptProvider:
    root = Path(__file__).resolve().parent.parent
    return FilePromptProvider(root / 'prompts')


def render_prompt(name: str, **variables: Any) -> str:
    return get_default_prompt_provider().get(name, variables=variables or None)
