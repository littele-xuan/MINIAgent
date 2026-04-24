from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from ..config.models import MemoryConfig
from ..llm.base import BaseStructuredLLM
from ..utils import compact_text, json_safe
from .classifier import HeuristicMemoryClassifier


class MemoryCandidate(BaseModel):
    model_config = ConfigDict(extra='forbid')

    bucket: str = Field(description='profile, task, procedural, episodic, or memories')
    kind: str = Field(description='preference, profile, task_constraint, procedural, episodic, explicit_memory')
    text: str
    confidence: float = 0.8
    source: str = 'conversation'
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryExtractionResult(BaseModel):
    model_config = ConfigDict(extra='forbid')

    memories: list[MemoryCandidate] = Field(default_factory=list)
    should_update_summary: bool = False


class BaseMemoryExtractor(Protocol):
    async def extract(self, text: str, *, config: MemoryConfig, user_id: str, thread_id: str) -> list[tuple[str, dict[str, Any]]]:
        ...


class HeuristicMemoryExtractor:
    def __init__(self) -> None:
        self.classifier = HeuristicMemoryClassifier()

    async def extract(self, text: str, *, config: MemoryConfig, user_id: str, thread_id: str) -> list[tuple[str, dict[str, Any]]]:
        items = self.classifier.classify(text, long_term_bucket=config.long_term_namespace)
        return [
            (
                item.bucket,
                {
                    'text': item.text,
                    'kind': item.kind,
                    'confidence': item.confidence,
                    'source': 'conversation',
                    'extractor': 'heuristic',
                },
            )
            for item in items
        ]


class LLMMemoryExtractor:
    """Hot-path LLM memory extractor with a deterministic fallback.

    The schema is intentionally simple so any OpenAI-compatible model that
    supports structured output can perform Mem0-style extraction. Consolidation
    remains pluggable and conservative at write time.
    """

    def __init__(self, llm: BaseStructuredLLM, *, fallback: HeuristicMemoryExtractor | None = None) -> None:
        self.llm = llm
        self.fallback = fallback or HeuristicMemoryExtractor()

    async def extract(self, text: str, *, config: MemoryConfig, user_id: str, thread_id: str) -> list[tuple[str, dict[str, Any]]]:
        if not text.strip():
            return []
        messages = [
            {
                'role': 'system',
                'content': (
                    'Extract only durable, useful agent memories from the user message. '
                    'Do not store trivial one-off chatter. Prefer short atomic facts. '
                    'Use bucket profile for user preferences/profile, task for active task constraints, '
                    'procedural for standing instructions, memories for explicit long-term memories. '
                    'Return JSON only via the provided schema.'
                ),
            },
            {
                'role': 'user',
                'content': f'user_id={user_id}\nthread_id={thread_id}\nmessage:\n{compact_text(text, limit=4000)}',
            },
        ]
        try:
            result = await self.llm.ainvoke_json_model(
                messages,
                model_type=MemoryExtractionResult,
                schema_name='memory_extraction_result',
                max_output_tokens=900,
                invoke_config=None,
            )
            out: list[tuple[str, dict[str, Any]]] = []
            for item in result.memories:
                bucket = item.bucket if item.bucket in {'profile', 'task', 'procedural', 'episodic', config.long_term_namespace, 'memories'} else config.long_term_namespace
                if bucket == 'memories':
                    bucket = config.long_term_namespace
                if not item.text.strip():
                    continue
                out.append((bucket, json_safe({**item.model_dump(mode='json'), 'extractor': 'llm'})))
            if out:
                return out
        except Exception:
            pass
        return await self.fallback.extract(text, config=config, user_id=user_id, thread_id=thread_id)


class HybridMemoryExtractor:
    def __init__(self, llm: BaseStructuredLLM | None = None) -> None:
        self.heuristic = HeuristicMemoryExtractor()
        self.llm = LLMMemoryExtractor(llm, fallback=self.heuristic) if llm is not None else None

    async def extract(self, text: str, *, config: MemoryConfig, user_id: str, thread_id: str) -> list[tuple[str, dict[str, Any]]]:
        heuristic = await self.heuristic.extract(text, config=config, user_id=user_id, thread_id=thread_id)
        if config.extractor == 'heuristic' or self.llm is None:
            return heuristic
        llm_items = await self.llm.extract(text, config=config, user_id=user_id, thread_id=thread_id)
        merged: list[tuple[str, dict[str, Any]]] = []
        seen: set[tuple[str, str, str]] = set()
        for bucket, item in [*heuristic, *llm_items]:
            key = (bucket, str(item.get('kind') or ''), str(item.get('text') or '').strip().lower())
            if key in seen or not key[2]:
                continue
            seen.add(key)
            merged.append((bucket, item))
        return merged


# Backwards-compatible sync-ish helper used by old imports. It intentionally
# returns heuristic output because old callers cannot await an LLM.
def extract_memories(text: str, *, config: MemoryConfig) -> list[tuple[str, dict[str, Any]]]:
    classifier = HeuristicMemoryClassifier()
    items = classifier.classify(text, long_term_bucket=config.long_term_namespace)
    return [
        (
            item.bucket,
            {
                'text': item.text,
                'kind': item.kind,
                'confidence': item.confidence,
                'source': 'conversation',
                'extractor': 'heuristic',
            },
        )
        for item in items
    ]
