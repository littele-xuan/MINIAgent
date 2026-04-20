from __future__ import annotations

import importlib.util
from types import ModuleType
from typing import Any

from mcp_lib.registry.models import ToolCategory, ToolEntry
from mcp_lib.tools.base import tool_def

from .models import SkillBundle, SkillLocalToolSpec


class SkillToolRegistrar:
    def build_tool_entries(self, bundle: SkillBundle) -> list[ToolEntry]:
        entries: list[ToolEntry] = []
        for spec in bundle.local_tools:
            handler = self._load_handler(spec)
            entry = tool_def(
                name=spec.name,
                description=spec.description,
                handler=handler,
                category=ToolCategory.EXTERNAL,
                properties=(spec.input_schema or {}).get('properties', {}),
                required=(spec.input_schema or {}).get('required', []),
                version=spec.version,
                tags=['skill-tool', bundle.name, *spec.tags],
                aliases=spec.aliases,
                metadata={
                    'skill': bundle.name,
                    'examples': spec.examples,
                    **dict(spec.metadata or {}),
                },
                created_by=f'skill:{bundle.name}',
            )
            entries.append(entry)
        return entries

    def _load_handler(self, spec: SkillLocalToolSpec):
        module_name = f'skill_tool_{spec.name}_{abs(hash(str(spec.handler_file)))}'
        loader = importlib.util.spec_from_file_location(module_name, spec.handler_file)
        if loader is None or loader.loader is None:
            raise RuntimeError(f'Cannot import skill tool handler from {spec.handler_file}')
        module = importlib.util.module_from_spec(loader)
        loader.loader.exec_module(module)
        handler = getattr(module, spec.handler_symbol)
        return handler
