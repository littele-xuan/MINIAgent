from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable

from ..schemas import ToolDescriptor
from .registry import ToolRegistry


@dataclass(slots=True)
class GovernanceTool:
    name: str
    description: str
    input_schema: dict[str, Any]
    metadata: dict[str, Any]
    handler: Callable[[dict[str, Any]], Any]
    risk: str = 'safe_read'

    async def ainvoke(self, arguments: dict[str, Any]) -> Any:
        value = self.handler(arguments)
        return await value if inspect.isawaitable(value) else value


class RegistryGovernanceToolset:
    """LLM-visible governance tools for inspecting and managing the tool catalog."""

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    def tools(self) -> dict[str, GovernanceTool]:
        return {
            'tool_list': GovernanceTool(
                name='tool_list',
                description='List currently registered MCP tools and governance state.',
                input_schema={'type': 'object', 'properties': {'include_disabled': {'type': 'boolean'}}, 'required': []},
                metadata={'provider': 'governance', 'protected': True},
                handler=lambda args: self.registry.export() if args.get('include_disabled') else {'tools': [t.model_dump(mode='json') for t in self.registry.list()]},
            ),
            'tool_inspect': GovernanceTool(
                name='tool_inspect',
                description='Inspect one MCP tool descriptor by name.',
                input_schema={'type': 'object', 'properties': {'name': {'type': 'string'}}, 'required': ['name']},
                metadata={'provider': 'governance', 'protected': True},
                handler=lambda args: self._inspect(args['name']),
            ),
            'tool_search': GovernanceTool(
                name='tool_search',
                description='Search MCP tools by keyword over name, description, risk, and metadata.',
                input_schema={'type': 'object', 'properties': {'query': {'type': 'string'}}, 'required': ['query']},
                metadata={'provider': 'governance', 'protected': True},
                handler=lambda args: {'results': self.registry.search(args.get('query', ''))},
            ),
            'tool_disable': GovernanceTool(
                name='tool_disable',
                description='Disable a visible MCP tool for this agent runtime.',
                input_schema={'type': 'object', 'properties': {'name': {'type': 'string'}}, 'required': ['name']},
                metadata={'provider': 'governance', 'protected': True},
                handler=lambda args: self.registry.disable(args['name']),
            ),
            'tool_enable': GovernanceTool(
                name='tool_enable',
                description='Enable a previously disabled MCP tool for this agent runtime.',
                input_schema={'type': 'object', 'properties': {'name': {'type': 'string'}}, 'required': ['name']},
                metadata={'provider': 'governance', 'protected': True},
                handler=lambda args: self.registry.enable(args['name']),
            ),
            'tool_upsert': GovernanceTool(
                name='tool_upsert',
                description='Add or update a tool descriptor in the runtime registry. This only updates catalog metadata; executable tools still require a real provider/MCP server.',
                input_schema={
                    'type': 'object',
                    'properties': {
                        'name': {'type': 'string'},
                        'description': {'type': 'string'},
                        'input_schema': {'type': 'object'},
                        'metadata': {'type': 'object'},
                        'risk': {'type': 'string'},
                        'enabled': {'type': 'boolean'},
                        'version': {'type': 'string'},
                    },
                    'required': ['name', 'description'],
                },
                metadata={'provider': 'governance', 'protected': True},
                risk='external_write',
                handler=lambda args: self.registry.upsert(ToolDescriptor(**args), source='llm-governance'),
            ),
            'tool_delete': GovernanceTool(
                name='tool_delete',
                description='Delete a non-protected tool descriptor from the runtime registry.',
                input_schema={'type': 'object', 'properties': {'name': {'type': 'string'}}, 'required': ['name']},
                metadata={'provider': 'governance', 'protected': True},
                risk='external_write',
                handler=lambda args: self.registry.delete(args['name']),
            ),
            'tool_export': GovernanceTool(
                name='tool_export',
                description='Export tool registry including disabled tools and recent audit events.',
                input_schema={'type': 'object', 'properties': {}, 'required': []},
                metadata={'provider': 'governance', 'protected': True},
                handler=lambda args: self.registry.export(),
            ),
        }

    def descriptors(self) -> list[ToolDescriptor]:
        return [
            ToolDescriptor(name=t.name, description=t.description, input_schema=t.input_schema, metadata=t.metadata, risk=t.risk)
            for t in self.tools().values()
        ]

    def _inspect(self, name: str) -> dict[str, Any]:
        tool = self.registry.get(name)
        if tool is None:
            raise KeyError(f'Unknown tool: {name}')
        return tool.model_dump(mode='json')
