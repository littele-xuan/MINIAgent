from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


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
    ) -> PromptContext:
        raise NotImplementedError


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
    ) -> PromptContext:
        layers: list[MessageLayer] = []

        layers.append(
            MessageLayer(
                name='operating-contract',
                priority=110,
                content=(
                    'This agent is API-first and MCP-native. All executable capabilities must come from the live MCP tool catalog below.\n'
                    'Do not invent tools. Do not rely on hidden local routing, heuristic tool picking, or ad hoc execution paths.\n'
                    'When mode="mcp", every tool_name must exist in the live catalog and every arguments object must satisfy that tool\'s input_schema.\n'
                    'Use mode="final" only when the answer can be produced directly from the user request, active skill instructions, live catalog metadata, or prior observations.\n'
                    'The output must always be a single strict JSON object matching the planner schema. No markdown. No prose outside JSON.'
                ),
            )
        )

        layers.append(
            MessageLayer(
                name='agent-profile',
                priority=100,
                content=json.dumps(
                    {
                        'name': agent.config.name,
                        'description': agent.config.description,
                        'role': agent.config.role,
                        'planner_mode': agent.config.planner,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
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
                name='mcp-tool-catalog',
                priority=95,
                content=json.dumps(tool_catalog, ensure_ascii=False, indent=2),
            )
        )

        layers.append(
            MessageLayer(
                name='tool-usage-rules',
                priority=94,
                content=(
                    'Prefer the live MCP catalog over generic narration.\n'
                    'Batch independent MCP calls in the same turn when they do not depend on each other.\n'
                    'For sequential workflows, issue the next MCP call only after inspecting the previous observation.\n'
                    'When the user asks to inspect or mutate tools, translate that natural-language request into governance-tool JSON arguments instead of assuming the caller already executed runtime.call_tool(...).\n'
                    'If a registry governance tool mutates the catalog, assume the runtime will refresh the live MCP catalog before the next turn.'
                ),
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
                    content=json.dumps({'accepted_output_modes': accepted_output_modes}, ensure_ascii=False),
                )
            )

        if observations:
            layers.append(
                MessageLayer(
                    name='execution-trace',
                    priority=90,
                    content=json.dumps(observations[-4:], ensure_ascii=False, indent=2),
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
            },
        )
