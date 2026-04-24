from __future__ import annotations

from typing import Any

from .base import BaseSkillManager


class LegacySkillManager(BaseSkillManager):
    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    async def load(self) -> None:
        if hasattr(self.runtime, 'load'):
            await self.runtime.load()

    async def select(self, query: str) -> dict[str, Any] | None:
        selector = getattr(self.runtime, 'select', None)
        if selector is None:
            return None
        result = await selector(query) if callable(selector) else None
        if result is None:
            return None
        return result.model_dump() if hasattr(result, 'model_dump') else result

    def local_tools_for(self, selected_skill: dict[str, Any] | None) -> list[Any]:
        resolver = getattr(self.runtime, 'local_tools_for', None)
        if resolver is None:
            return []
        return list(resolver(selected_skill))

    def filter_tool_catalog(self, selected_skill: dict[str, Any] | None, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        filt = getattr(self.runtime, 'filter_tool_catalog', None)
        if filt is None:
            return tools
        return list(filt(selected_skill, tools))
