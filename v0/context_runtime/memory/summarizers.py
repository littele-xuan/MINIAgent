from __future__ import annotations

import json
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from llm_runtime import BaseLLM
from prompt_runtime import render_prompt

from .base import BaseSummaryGenerator
from .models import MessageRecord, SummaryNode
from .store import new_id
from .token_counter import estimate_tokens


@dataclass(slots=True)
class SummaryPlan:
    level: int
    target_tokens: int


class SummaryEnvelope(BaseModel):
    model_config = ConfigDict(extra='forbid')

    summary: str


class LLMSummaryGenerator(BaseSummaryGenerator):
    """Summary compaction is runtime-owned; the model only emits a compact summary string."""

    def __init__(self, llm: BaseLLM) -> None:
        self.llm = llm

    async def summarize_block(
        self,
        *,
        session_id: str,
        created_at: str,
        items: list[MessageRecord | SummaryNode],
        plans: list[SummaryPlan],
    ) -> SummaryNode:
        if not items:
            raise ValueError('cannot summarize empty block')
        original_text = '\n'.join(self._render_item(item) for item in items)
        original_tokens = estimate_tokens(original_text)
        leaf_ids = self._collect_leaf_message_ids(items)
        source_item_ids = [item.id for item in items]

        chosen_text = ''
        chosen_level = plans[-1].level if plans else 1
        for idx, plan in enumerate(plans or [SummaryPlan(level=1, target_tokens=max(160, original_tokens // 2))]):
            summary_text = await self._generate_summary(items=items, target_tokens=plan.target_tokens, pass_index=idx)
            if estimate_tokens(summary_text) < original_tokens:
                chosen_text = summary_text
                chosen_level = plan.level
                break
            if not chosen_text or estimate_tokens(summary_text) < estimate_tokens(chosen_text):
                chosen_text = summary_text
                chosen_level = plan.level

        if estimate_tokens(chosen_text) >= original_tokens:
            chosen_text = self._clip_to_tokens(chosen_text, max(96, min(original_tokens - 1, plans[-1].target_tokens if plans else original_tokens // 2)))

        return SummaryNode(
            id=new_id('sum'),
            session_id=session_id,
            level=chosen_level,
            content=chosen_text,
            created_at=created_at,
            token_count=estimate_tokens(chosen_text),
            source_item_ids=source_item_ids,
            leaf_message_ids=leaf_ids,
            metadata={'original_tokens': original_tokens},
        )

    async def _generate_summary(self, *, items: list[MessageRecord | SummaryNode], target_tokens: int, pass_index: int) -> str:
        envelope = await self.llm.chat_json_model(
            messages=[
                {'role': 'system', 'content': render_prompt('memory/context_summary_system')},
                {
                    'role': 'user',
                    'content': json.dumps(
                        {
                            'target_tokens': target_tokens,
                            'pass_index': pass_index,
                            'items': [
                                {
                                    'id': item.id,
                                    'type': 'summary' if isinstance(item, SummaryNode) else 'message',
                                    'role': None if isinstance(item, SummaryNode) else item.role,
                                    'content': item.content,
                                    'metadata': item.metadata,
                                }
                                for item in items
                            ],
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                },
            ],
            model_type=SummaryEnvelope,
            schema_name='memory_summary_generation',
            temperature=0.0,
            max_output_tokens=max(240, min(1200, target_tokens * 4)),
        )
        return self._clip_to_tokens(envelope.summary.strip(), target_tokens)

    def _clip_to_tokens(self, text: str, target_tokens: int) -> str:
        if estimate_tokens(text) <= target_tokens:
            return text
        words = text.split()
        kept: list[str] = []
        for word in words:
            candidate = ' '.join(kept + [word])
            if estimate_tokens(candidate) > target_tokens:
                break
            kept.append(word)
        clipped = ' '.join(kept).strip()
        if clipped:
            return clipped
        return text[: max(80, target_tokens * 4)]

    def _render_item(self, item: MessageRecord | SummaryNode) -> str:
        if isinstance(item, SummaryNode):
            return f'summary:{item.content}'
        return f'{item.role}:{item.content}'

    def _collect_leaf_message_ids(self, items: list[MessageRecord | SummaryNode]) -> list[str]:
        leaf_ids: list[str] = []
        for item in items:
            if isinstance(item, SummaryNode):
                leaf_ids.extend(item.leaf_message_ids)
            else:
                leaf_ids.append(item.id)
        deduped: list[str] = []
        seen: set[str] = set()
        for message_id in leaf_ids:
            if message_id in seen:
                continue
            seen.add(message_id)
            deduped.append(message_id)
        return deduped
