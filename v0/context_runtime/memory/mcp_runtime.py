from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from llm_runtime import BaseLLM, MCPStructuredResponse
from mcp_lib.registry.models import ToolCategory
from mcp_lib.registry.registry import ToolRegistry
from mcp_lib.tools.base import register_all, tool_def


@dataclass(slots=True)
class InternalMCPToolSpec:
    name: str
    description: str
    properties: dict[str, Any]
    required: list[str] = field(default_factory=list)
    handler: Callable[..., Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def catalog_entry(self) -> dict[str, Any]:
        return {
            'name': self.name,
            'description': self.description,
            'input_schema': {
                'type': 'object',
                'properties': self.properties,
                'required': self.required,
            },
            'metadata': self.metadata,
        }


class BaseInternalMCPRuntime(ABC):
    @abstractmethod
    def catalog_payload(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    async def execute(self, response: MCPStructuredResponse) -> list[dict[str, Any]]:
        raise NotImplementedError


class ToolRegistryMCPRuntime(BaseInternalMCPRuntime):
    """Local MCP-like runtime backed by the same ToolRegistry abstractions as the agent."""

    def __init__(self, specs: list[InternalMCPToolSpec]) -> None:
        self._registry = ToolRegistry()
        entries = []
        for spec in specs:
            if spec.handler is None:
                raise ValueError(f'tool handler missing for {spec.name}')
            entries.append(
                tool_def(
                    name=spec.name,
                    description=spec.description,
                    handler=spec.handler,
                    category=ToolCategory.INTERNAL_UTILITY,
                    properties=spec.properties,
                    required=spec.required,
                    metadata=spec.metadata,
                )
            )
        register_all(self._registry, entries)
        self._specs = list(specs)

    def catalog_payload(self) -> list[dict[str, Any]]:
        return [spec.catalog_entry() for spec in self._specs]

    async def execute(self, response: MCPStructuredResponse) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for call in response.mcp_calls:
            payload = await self._registry.call(call.tool_name, call.arguments)
            normalized = payload if isinstance(payload, dict) else {'value': payload}
            results.append({'tool_name': call.tool_name, 'arguments': call.arguments, 'result': normalized})
        return results


def render_internal_mcp_prompt(*, base_prompt: str, runtime: BaseInternalMCPRuntime) -> str:
    contract = (
        'Use the same MCP-style JSON envelope as the main agent. '
        'Return exactly one JSON object with keys: thought, mode, final, mcp_calls. '
        'When mode="mcp", choose only tools from the internal tool catalog below. '
        'No markdown. No prose outside JSON.'
    )
    catalog = json.dumps(runtime.catalog_payload(), ensure_ascii=False, indent=2)
    return f'{base_prompt}\n\n## internal-mcp-contract\n{contract}\n\n## internal-tool-catalog\n{catalog}'


def _normalize_internal_response(payload: dict[str, Any]) -> MCPStructuredResponse:
    normalized = dict(payload)
    if 'mcp_calls' not in normalized:
        if 'calls' in normalized:
            normalized['mcp_calls'] = normalized['calls']
        elif 'tool_calls' in normalized:
            normalized['mcp_calls'] = normalized['tool_calls']

    if isinstance(normalized.get('mcp_calls'), list):
        rewritten_calls: list[dict[str, Any] | Any] = []
        for item in normalized['mcp_calls']:
            if not isinstance(item, dict):
                rewritten_calls.append(item)
                continue
            call = dict(item)
            if 'tool_name' not in call and 'name' in call:
                call['tool_name'] = call.pop('name')
            if 'arguments' not in call:
                for legacy_key in ('args', 'input', 'parameters'):
                    if legacy_key in call:
                        call['arguments'] = call.pop(legacy_key)
                        break
            rewritten_calls.append(call)
        normalized['mcp_calls'] = rewritten_calls

    if 'mode' in normalized:
        return MCPStructuredResponse.model_validate(normalized)

    # Compatibility path: older memory modules returned task-specific JSON.
    if any(key in normalized for key in ('facts', 'memories', 'picks', 'answer', 'summary', 'label', 'normalized_content')):
        return MCPStructuredResponse.model_validate(
            {
                'thought': normalized.get('name', 'legacy-memory-response'),
                'mode': 'final',
                'final': {'output_mode': 'application/json', 'data': normalized},
                'mcp_calls': [],
            }
        )

    raise ValueError(f'unsupported internal MCP payload: {payload}')


async def run_internal_mcp_turn(
    *,
    llm: BaseLLM,
    system_prompt: str,
    payload: dict[str, Any],
    runtime: BaseInternalMCPRuntime,
    schema_name: str,
    max_output_tokens: int = 1000,
) -> tuple[MCPStructuredResponse, list[dict[str, Any]]]:
    raw = await llm.chat_json(
        messages=[
            {'role': 'system', 'content': render_internal_mcp_prompt(base_prompt=system_prompt, runtime=runtime)},
            {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False, indent=2)},
        ],
        schema={
            'type': 'object',
            'additionalProperties': True,
            'properties': {
                'thought': {'type': 'string'},
                'mode': {'type': 'string'},
                'final': {'type': ['object', 'null']},
                'mcp_calls': {'type': 'array'},
                'facts': {'type': 'array'},
                'memories': {'type': 'array'},
                'picks': {'type': 'array'},
                'answer': {},
                'summary': {'type': 'string'},
                'label': {'type': 'string'},
                'reason': {'type': 'string'},
                'normalized_content': {'type': 'string'},
                'name': {'type': 'string'},
            },
        },
        schema_name=schema_name,
        temperature=0.0,
        max_output_tokens=max_output_tokens,
    )
    response = _normalize_internal_response(raw)
    executed = await runtime.execute(response) if response.mode == 'mcp' else []
    return response, executed
