from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..schemas import ToolDescriptor
from ..utils import json_safe
from .audit import ToolAuditLog
from .policy import ToolPolicy


@dataclass(slots=True)
class ToolRegistry:
    policy: ToolPolicy = field(default_factory=ToolPolicy)
    audit: ToolAuditLog = field(default_factory=ToolAuditLog)
    _tools: dict[str, ToolDescriptor] = field(default_factory=dict)
    protected_names: set[str] = field(default_factory=lambda: {'tool_list', 'tool_inspect', 'tool_search', 'tool_enable', 'tool_disable', 'tool_upsert', 'tool_delete', 'tool_export'})

    def refresh(self, tools: list[ToolDescriptor | dict[str, Any]]) -> None:
        for item in tools:
            descriptor = item if isinstance(item, ToolDescriptor) else ToolDescriptor.model_validate(item)
            self._tools[descriptor.name] = descriptor
        self.audit.append(event_type='refresh', tool_name='*', payload={'count': len(self._tools)})

    def upsert(self, descriptor: ToolDescriptor | dict[str, Any], *, source: str = 'governance') -> dict[str, Any]:
        tool = descriptor if isinstance(descriptor, ToolDescriptor) else ToolDescriptor.model_validate(descriptor)
        existing = self._tools.get(tool.name)
        self._tools[tool.name] = tool
        self.audit.append(event_type='upsert', tool_name=tool.name, payload={'source': source, 'created': existing is None})
        return {'tool_name': tool.name, 'created': existing is None, 'descriptor': tool.model_dump(mode='json')}

    def delete(self, name: str) -> dict[str, Any]:
        if name in self.protected_names:
            raise PermissionError(f'Cannot delete protected governance tool: {name}')
        if name not in self._tools:
            raise KeyError(f'Unknown tool: {name}')
        self._tools.pop(name)
        self.policy.enable(name)
        self.audit.append(event_type='delete', tool_name=name)
        return {'tool_name': name, 'deleted': True}

    def list(self, *, include_disabled: bool = False) -> list[ToolDescriptor]:
        out: list[ToolDescriptor] = []
        for name in sorted(self._tools):
            if not include_disabled and name in self.policy.disabled_tools:
                continue
            tool = self._tools[name]
            if not tool.enabled and not include_disabled:
                continue
            out.append(tool)
        return out

    def visible_dicts(self) -> list[dict[str, Any]]:
        return [tool.model_dump(mode='json') for tool in self.list()]

    def get(self, name: str) -> ToolDescriptor | None:
        return self._tools.get(name)

    def search(self, query: str) -> list[dict[str, Any]]:
        q = query.lower().strip()
        results = []
        for tool in self.list(include_disabled=True):
            blob = f"{tool.name}\n{tool.description}\n{tool.metadata}\n{tool.risk}".lower()
            if not q or q in blob:
                results.append(tool.model_dump(mode='json'))
        return results

    def disable(self, name: str) -> dict[str, Any]:
        if name not in self._tools:
            raise KeyError(f'Unknown tool: {name}')
        self.policy.disable(name)
        self.audit.append(event_type='disable', tool_name=name)
        return {'tool_name': name, 'enabled': False}

    def enable(self, name: str) -> dict[str, Any]:
        if name not in self._tools:
            raise KeyError(f'Unknown tool: {name}')
        self.policy.enable(name)
        self.audit.append(event_type='enable', tool_name=name)
        return {'tool_name': name, 'enabled': True}

    def export(self) -> dict[str, Any]:
        return {
            'tools': [tool.model_dump(mode='json') for tool in self.list(include_disabled=True)],
            'disabled_tools': sorted(self.policy.disabled_tools),
            'audit': json_safe(self.audit.recent()),
        }
