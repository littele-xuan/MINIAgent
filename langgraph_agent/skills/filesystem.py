from __future__ import annotations

import importlib.util
import inspect
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..config.models import SkillConfig
from .base import BaseSkillManager
from .manifest import SkillManifest


@dataclass(slots=True)
class LoadedSkillTool:
    name: str
    description: str
    input_schema: dict[str, Any]
    metadata: dict[str, Any]
    handler: Any

    async def ainvoke(self, arguments: dict[str, Any]) -> Any:
        value = self._call_handler(arguments)
        return await value if inspect.isawaitable(value) else value

    def _call_handler(self, arguments: dict[str, Any]) -> Any:
        sig = inspect.signature(self.handler)
        params = list(sig.parameters.values())
        if len(params) == 1:
            param = params[0]
            if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
                if param.name in arguments:
                    return self.handler(arguments[param.name])
                if 'text' in arguments:
                    return self.handler(arguments['text'])
                if len(arguments) == 1:
                    return self.handler(next(iter(arguments.values())))
        try:
            return self.handler(**arguments)
        except TypeError:
            return self.handler(arguments)


class FilesystemSkillManager(BaseSkillManager):
    def __init__(self, config: SkillConfig) -> None:
        self.config = config
        root = config.skills_root or 'skills'
        self.skills_root = Path(root)
        self.skills: list[dict[str, Any]] = []
        self.tools_by_skill: dict[str, list[LoadedSkillTool]] = {}

    async def load(self) -> None:
        self.skills = []
        self.tools_by_skill = {}
        if not self.config.enabled or not self.skills_root.exists():
            return
        for skill_dir in sorted(self.skills_root.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill_md = skill_dir / 'SKILL.md'
            if not skill_md.exists():
                continue
            content = skill_md.read_text(encoding='utf-8')
            skill = self._parse_skill(skill_dir.name, content)
            self.skills.append(skill)
            self.tools_by_skill[skill['name']] = self._load_skill_tools(skill_dir, skill)

    def _split_front_matter(self, content: str) -> tuple[dict[str, Any], str]:
        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                try:
                    return yaml.safe_load(parts[1]) or {}, parts[2].strip()
                except Exception:
                    return {}, content
        return {}, content

    def _parse_skill(self, folder_name: str, content: str) -> dict[str, Any]:
        meta, body = self._split_front_matter(content)
        first_heading = re.search(r'^#\s+(.+)$', body, flags=re.MULTILINE)
        title = str(meta.get('title') or (first_heading.group(1).strip() if first_heading else folder_name))
        lower = (body + ' ' + str(meta)).lower()
        triggers = list(meta.get('triggers') or meta.get('keywords') or [])
        triggers.extend([folder_name.replace('-', ' '), title.lower()])
        if 'weather' in lower:
            triggers.extend(['weather', 'forecast'])
        if 'text' in lower or 'slug' in lower:
            triggers.extend(['text', 'transform', 'slug', 'slugify', 'reverse'])
        manifest = SkillManifest(
            name=str(meta.get('name') or folder_name),
            title=title,
            version=str(meta.get('version') or '0.1.0'),
            description=str(meta.get('description') or body[:600]),
            triggers=sorted({str(k) for k in triggers if k}),
            instructions=body,
            allowed_tools=list(meta.get('allowed_tools') or []),
            metadata=dict(meta.get('metadata') or {}),
        )
        data = manifest.model_dump(mode='json')
        data['policy'] = self.config.policy
        data['keywords'] = data['triggers']
        return data

    def _normalize_input_schema(self, spec: dict[str, Any]) -> dict[str, Any]:
        if 'input_schema' in spec:
            return spec['input_schema'] or {}
        if 'parameters' in spec and isinstance(spec['parameters'], dict):
            return spec['parameters']
        properties = spec.get('properties') or {}
        required = spec.get('required') or []
        return {'type': 'object', 'properties': properties, 'required': required}

    def _load_skill_tools(self, skill_dir: Path, skill: dict[str, Any]) -> list[LoadedSkillTool]:
        tools_dir = skill_dir / 'mcp_tools'
        if not tools_dir.exists():
            return []
        tools: list[LoadedSkillTool] = []
        for tool_dir in sorted(tools_dir.iterdir()):
            tool_yaml = tool_dir / 'tool.yaml'
            handler_py = tool_dir / 'handler.py'
            if not tool_yaml.exists() or not handler_py.exists():
                continue
            spec = yaml.safe_load(tool_yaml.read_text(encoding='utf-8')) or {}
            handler = self._load_handler(handler_py, skill['name'], tool_dir.name)
            tool = LoadedSkillTool(
                name=spec.get('name', tool_dir.name),
                description=spec.get('description', ''),
                input_schema=self._normalize_input_schema(spec),
                metadata={'source': 'skill-local', 'skill': skill['name'], **(spec.get('metadata') or {})},
                handler=handler,
            )
            tools.append(tool)
        return tools

    def _load_handler(self, path: Path, skill_name: str, tool_name: str):
        module_name = f'langgraph_skill_{skill_name}_{tool_name}'
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f'Unable to load skill tool handler: {path}')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for attr in ('handle', 'handler', 'run', 'main', 'invoke'):
            fn = getattr(module, attr, None)
            if callable(fn):
                return fn
        raise RuntimeError(f'No callable handler found in {path}')

    async def select(self, query: str) -> dict[str, Any] | None:
        if not self.config.enabled or not self.config.auto_select:
            return None
        query_l = query.lower()
        scored: list[tuple[int, dict[str, Any]]] = []
        for skill in self.skills:
            score = 0
            for keyword in skill.get('keywords', []):
                if keyword and str(keyword).lower() in query_l:
                    score += 2
            for tool in self.tools_by_skill.get(skill['name'], []):
                if tool.name.lower() in query_l:
                    score += 3
                meta_keywords = tool.metadata.get('keywords', []) if isinstance(tool.metadata.get('keywords'), list) else []
                for meta_kw in meta_keywords:
                    if str(meta_kw).lower() in query_l:
                        score += 1
            if score:
                scored.append((score, skill))
        if not scored:
            return None
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1]

    def local_tools_for(self, selected_skill: dict[str, Any] | None) -> list[Any]:
        if selected_skill is None:
            return []
        return list(self.tools_by_skill.get(selected_skill['name'], []))

    def filter_tool_catalog(self, selected_skill: dict[str, Any] | None, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if selected_skill is None or self.config.policy != 'restrictive':
            return tools
        local = {tool.name for tool in self.tools_by_skill.get(selected_skill['name'], [])}
        allowed = set(selected_skill.get('allowed_tools') or []) | local | {'tool_list', 'tool_inspect', 'tool_search', 'tool_enable', 'tool_disable', 'tool_upsert', 'tool_delete', 'tool_export'}
        filtered = [tool for tool in tools if tool.get('name') in allowed or tool.get('metadata', {}).get('skill') == selected_skill['name']]
        return filtered or tools
