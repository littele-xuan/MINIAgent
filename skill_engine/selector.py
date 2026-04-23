from __future__ import annotations

import json
from typing import Iterable, Sequence

from pydantic import BaseModel, ConfigDict

from llm_runtime import OpenAICompatibleLLM

from .models import SkillBundle


class SkillSelectionDecision(BaseModel):
    model_config = ConfigDict(extra='forbid')

    skill_name: str | None = None
    skill_arguments: str = ''
    reason: str = ''


class SkillSelector:
    """LLM-based skill selector.

    No keyword scoring is used. The model receives the live skill catalog and
    decides whether a skill should be activated for the current request.
    """

    def __init__(self, llm: OpenAICompatibleLLM) -> None:
        self.llm = llm

    async def choose(self, query: str, bundles: Iterable[SkillBundle]) -> SkillBundle | None:
        items = list(bundles)
        if not items:
            return None
        decision = await self.decide(query, items)
        if not decision.skill_name:
            return None
        for bundle in items:
            if bundle.name == decision.skill_name:
                return bundle.clone_for_arguments(decision.skill_arguments)
        return None

    async def decide(self, query: str, bundles: Sequence[SkillBundle]) -> SkillSelectionDecision:
        catalog = [
            {
                'name': bundle.name,
                'description': bundle.description,
                'when_to_use': bundle.when_to_use,
                'allowed_tools': list(bundle.allowed_tools),
                'output_modes': list(bundle.output_modes),
                'accepted_output_modes': list(bundle.accepted_output_modes),
                'examples': list(bundle.frontmatter.examples or []),
                'a2a': dict(bundle.frontmatter.a2a or {}),
                'mcp': dict(bundle.frontmatter.mcp or {}),
            }
            for bundle in bundles
        ]
        messages = [
            {
                'role': 'system',
                'content': (
                    'You are the skill selection module for an MCP/A2A agent. '
                    'Choose a skill only when it materially improves the handling of the request. '
                    'Do not force a skill when the base agent can answer directly. '
                    'Do not invent skill names. If no skill is clearly appropriate, return skill_name=null.'
                ),
            },
            {
                'role': 'user',
                'content': json.dumps(
                    {
                        'query': query,
                        'skills': catalog,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            },
        ]
        return await self.llm.chat_json_model(
            messages=messages,
            model_type=SkillSelectionDecision,
            schema_name='skill_selection_decision',
            temperature=0.0,
            max_output_tokens=600,
        )
