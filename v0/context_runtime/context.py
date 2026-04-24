from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from prompt_runtime import render_prompt


@dataclass(slots=True)
class MessageLayer:
    name: str
    content: str
    priority: int = 0


@dataclass(slots=True)
class PromptContext:
    system_prompt: str
    layers: list[MessageLayer] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseContextManager(ABC):
    @abstractmethod
    def build_prompt_context(
        self,
        *,
        agent: Any,
        query: str,
        active_skill: Any,
        visible_tools: list[Any],
        observations: list[dict[str, Any]],
        accepted_output_modes: list[str] | None = None,
        memory_packet: Any | None = None,
    ) -> PromptContext:
        raise NotImplementedError


def _compact_text(text: str, *, limit: int = 220) -> str:
    normalized = re.sub(r'\s+', ' ', text).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + '...'


def _tool_guide(tool_catalog: list[dict[str, Any]]) -> str:
    if not tool_catalog:
        return 'No live MCP tools are currently available. You must answer from memory and prompt context only.'
    lines: list[str] = []
    for item in tool_catalog:
        required = ((item.get('input_schema') or {}).get('required') or [])
        required_text = ', '.join(required) if required else 'none'
        lines.append(
            f"- {item.get('name', '')}: {_compact_text(str(item.get('description') or ''), limit=120)} | required_args={required_text}"
        )
    return '\n'.join(lines)


def _render_memory_items(items: list[Any], *, limit: int) -> str:
    lines: list[str] = []
    for item in items[:limit]:
        metadata = getattr(item, 'metadata', {}) or {}
        scope = metadata.get('scope')
        category = metadata.get('category')
        key = metadata.get('key')
        label_parts = [part for part in [scope, category, key] if part]
        label = '.'.join(str(part) for part in label_parts) if label_parts else getattr(item, 'source_type', 'memory')
        text = _compact_text(getattr(item, 'text', ''), limit=220)
        if text:
            lines.append(f'- [{label}] {text}')
    return '\n'.join(lines)


def _render_summary_items(items: list[Any], *, limit: int) -> str:
    lines: list[str] = []
    for node in items[-limit:]:
        summary_id = getattr(node, 'id', '')
        level = getattr(node, 'level', '')
        content = _compact_text(getattr(node, 'content', ''), limit=260)
        if content:
            lines.append(f'- summary_id={summary_id} level={level} {content}')
    return '\n'.join(lines)


def _render_tool_tail(observations: list[dict[str, Any]], *, limit: int = 4) -> str:
    lines: list[str] = []
    for item in observations[-limit:]:
        mode = item.get('mode') or 'unknown'
        calls = item.get('calls', []) or []
        call_names = ', '.join(str(call.get('tool_name') or '') for call in calls if isinstance(call, dict) and call.get('tool_name'))
        observation = _compact_text(str(item.get('observation') or ''), limit=240)
        prefix = f'- [{mode}]'
        if call_names:
            prefix += f' tools={call_names}'
        if observation:
            lines.append(f'{prefix} {observation}')
    return '\n'.join(lines)


def _render_agent_profile(agent: Any) -> str:
    return '\n'.join(
        [
            f'name: {agent.config.name}',
            f'description: {agent.config.description}',
            f'role: {agent.config.role}',
            f'planner_mode: {agent.config.planner}',
        ]
    )


def _semantic_memory_tail(
    *,
    query: str,
    memory_packet: Any | None,
    observations: list[dict[str, Any]],
) -> str:
    lines = [f'current_user_goal: {_compact_text(query, limit=240)}']

    if memory_packet is not None:
        long_term: list[str] = []
        task_context: list[str] = []
        for item in getattr(memory_packet, 'retrieved_memories', []) or []:
            text = _compact_text(getattr(item, 'text', ''), limit=180)
            if not text:
                continue
            scope = (getattr(item, 'metadata', {}) or {}).get('scope')
            if getattr(item, 'source_type', '') == 'fact' and scope == 'cross_session':
                if text not in long_term:
                    long_term.append(text)
            else:
                if text not in task_context:
                    task_context.append(text)

        if long_term:
            lines.append('durable_memory:')
            lines.extend(f'- {item}' for item in long_term[:4])

        if task_context:
            lines.append('task_context:')
            lines.extend(f'- {item}' for item in task_context[:6])

        summaries = getattr(memory_packet, 'active_summaries', []) or []
        if summaries:
            lines.append('active_summaries:')
            for node in summaries[-2:]:
                content = _compact_text(getattr(node, 'content', ''), limit=200)
                if content:
                    lines.append(f'- {content}')

        warnings = getattr(memory_packet, 'warnings', []) or []
        if warnings:
            lines.append('runtime_warnings:')
            lines.extend(f'- {_compact_text(str(item), limit=180)}' for item in warnings[:4])

    if observations:
        lines.append('recent_effective_observations:')
        for item in observations[-3:]:
            observation = _compact_text(str(item.get('observation') or ''), limit=220)
            if not observation:
                continue
            mode = item.get('mode') or 'unknown'
            lines.append(f'- [{mode}] {observation}')

    return '\n'.join(lines)


class LayeredContextManager(BaseContextManager):
    def build_prompt_context(
        self,
        *,
        agent: Any,
        query: str,
        active_skill: Any,
        visible_tools: list[Any],
        observations: list[dict[str, Any]],
        accepted_output_modes: list[str] | None = None,
        memory_packet: Any | None = None,
    ) -> PromptContext:
        layers: list[MessageLayer] = []

        layers.append(
            MessageLayer(
                name='operating-contract',
                priority=110,
                content=render_prompt('agent/context_operating_contract'),
            )
        )

        if memory_packet is not None:
            layers.append(
                MessageLayer(
                    name='memory-operating-rules',
                    priority=108,
                    content=render_prompt('agent/memory_operating_rules'),
                )
            )

        layers.append(
            MessageLayer(
                name='agent-profile',
                priority=100,
                content=_render_agent_profile(agent),
            )
        )

        tool_catalog = []
        for tool in visible_tools:
            tool_catalog.append(tool.catalog_entry() if hasattr(tool, 'catalog_entry') else {
                'name': tool.name,
                'description': tool.description,
                'input_schema': tool.input_schema,
                'metadata': tool.metadata,
            })
        layers.append(
            MessageLayer(
                name='mcp-tool-guide',
                priority=98,
                content=_tool_guide(tool_catalog),
            )
        )
        layers.append(
            MessageLayer(
                name='mcp-tool-catalog',
                priority=95,
                content=json.dumps(tool_catalog, ensure_ascii=False, indent=2),
            )
        )

        layers.append(
            MessageLayer(
                name='tool-usage-rules',
                priority=94,
                content=render_prompt('agent/tool_usage_rules'),
            )
        )

        if active_skill is not None:
            rendered = active_skill.rendered_body or active_skill.body
            layers.append(
                MessageLayer(
                    name='active-skill',
                    priority=97,
                    content=json.dumps(
                        {
                            'name': active_skill.name,
                            'description': active_skill.description,
                            'when_to_use': active_skill.when_to_use,
                            'allowed_tools': list(active_skill.allowed_tools),
                            'local_tools': [tool.name for tool in active_skill.local_tools],
                            'output_modes': list(active_skill.output_modes),
                            'accepted_output_modes': list(active_skill.accepted_output_modes),
                            'mcp': dict(active_skill.frontmatter.mcp or {}),
                            'a2a': dict(active_skill.frontmatter.a2a or {}),
                            'instructions': rendered,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                )
            )

        if accepted_output_modes:
            layers.append(
                MessageLayer(
                    name='output-negotiation',
                    priority=92,
                    content='accepted_output_modes: ' + ', '.join(accepted_output_modes),
                )
            )

        if memory_packet is not None:
            layers.append(
                MessageLayer(
                    name='memory-session-stats',
                    priority=96,
                    content=', '.join(f'{key}={value}' for key, value in memory_packet.stats.items()),
                )
            )
            if memory_packet.pinned_notes:
                layers.append(MessageLayer(name='memory-pinned-notes', priority=95, content='\n\n'.join(memory_packet.pinned_notes)))
            if memory_packet.warnings:
                layers.append(MessageLayer(name='memory-warnings', priority=94, content='\n'.join(f'- {item}' for item in memory_packet.warnings)))
            if memory_packet.retrieved_memories:
                long_term = []
                task_context = []
                for item in memory_packet.retrieved_memories:
                    scope = (item.metadata or {}).get('scope')
                    if item.source_type == 'fact' and scope == 'cross_session':
                        long_term.append(item)
                    else:
                        task_context.append(item)
                if long_term:
                    layers.append(
                        MessageLayer(
                            name='memory-long-term',
                            priority=93,
                            content=_render_memory_items(long_term, limit=8),
                        )
                    )
                if task_context:
                    layers.append(
                        MessageLayer(
                            name='memory-task-context',
                            priority=92,
                            content=_render_memory_items(task_context, limit=10),
                        )
                    )
            if memory_packet.active_summaries:
                layers.append(
                    MessageLayer(
                        name='memory-active-summaries',
                        priority=91,
                        content=_render_summary_items(memory_packet.active_summaries, limit=3),
                    )
                )
        if observations:
            layers.append(
                MessageLayer(
                    name='tool-output-tail',
                    priority=80,
                    content=_render_tool_tail(observations),
                )
            )

        layers.append(
            MessageLayer(
                name='working-context-tail',
                priority=10,
                content=_semantic_memory_tail(query=query, memory_packet=memory_packet, observations=observations),
            )
        )

        ordered = sorted(layers, key=lambda item: item.priority, reverse=True)
        system_prompt = '\n\n'.join(f'## {layer.name}\n{layer.content}' for layer in ordered)
        return PromptContext(
            system_prompt=system_prompt,
            layers=ordered,
            metadata={
                'query': query,
                'accepted_output_modes': accepted_output_modes or ['text/plain', 'application/json'],
                'tool_catalog': tool_catalog,
                'memory_packet': memory_packet,
            },
        )
