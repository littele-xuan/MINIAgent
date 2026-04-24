from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from llm_runtime import BaseLLM
from prompt_runtime import render_prompt

from .base import BaseMemoryRetriever
from .models import FactRecord, RetrievedMemory
from .store import SQLiteMemoryStore


@dataclass(slots=True)
class RetrievalCandidate:
    source_type: str
    source_id: str
    text: str
    metadata: dict[str, Any]


class RetrievalPick(BaseModel):
    model_config = ConfigDict(extra='forbid')

    source_type: str
    source_id: str
    relevance: float = 0.0
    reason: str = ''


class RetrievalSelection(BaseModel):
    model_config = ConfigDict(extra='forbid')

    picks: list[RetrievalPick] = Field(default_factory=list)


class LLMRetriever(BaseMemoryRetriever):
    """Memory retrieval stays inside the runtime; the model only ranks candidates."""

    def __init__(
        self,
        store: SQLiteMemoryStore,
        llm: BaseLLM,
        *,
        candidate_limit: int = 72,
    ) -> None:
        self.store = store
        self.llm = llm
        self.candidate_limit = candidate_limit

    async def retrieve(self, *, namespace: str, session_id: str, query: str, limit: int = 8) -> list[RetrievedMemory]:
        candidates = self._build_candidates(namespace=namespace, session_id=session_id, query=query)
        if not candidates:
            return []
        ranked = sorted(
            candidates,
            key=lambda item: self._heuristic_score(query=query, candidate=item),
            reverse=True,
        )
        trimmed = ranked[: self.candidate_limit]
        selection = await self.llm.chat_json_model(
            messages=[
                {'role': 'system', 'content': render_prompt('memory/retrieval_selection_system')},
                {
                    'role': 'user',
                    'content': json.dumps(
                        {
                            'query': query,
                            'max_results': limit,
                            'candidates': [
                                {
                                    'source_type': item.source_type,
                                    'source_id': item.source_id,
                                    'text': item.text,
                                    'metadata': item.metadata,
                                }
                                for item in trimmed
                            ],
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                },
            ],
            model_type=RetrievalSelection,
            schema_name='memory_retrieval_selection',
            temperature=0.0,
            max_output_tokens=1100,
        )
        index = {(item.source_type, item.source_id): item for item in trimmed}
        results: list[RetrievedMemory] = []
        seen: set[tuple[str, str]] = set()
        for pick in selection.picks:
            key = (pick.source_type, pick.source_id)
            if key in seen:
                continue
            candidate = index.get(key)
            if candidate is None:
                continue
            seen.add(key)
            results.append(
                RetrievedMemory(
                    source_type=candidate.source_type,  # type: ignore[arg-type]
                    source_id=candidate.source_id,
                    score=max(0.0, min(1.0, float(pick.relevance))),
                    text=candidate.text,
                    metadata={**candidate.metadata, 'selection_reason': pick.reason},
                )
            )
            if len(results) >= limit:
                break
        return self._fill_retrieval_gaps(query=query, ranked_candidates=trimmed, selected=results, limit=limit)

    def _build_candidates(self, *, namespace: str, session_id: str, query: str) -> list[RetrievalCandidate]:
        by_key: dict[tuple[str, str], RetrievalCandidate] = {}

        def add(candidate: RetrievalCandidate) -> None:
            by_key[(candidate.source_type, candidate.source_id)] = candidate

        facts = self.store.list_active_facts(
            namespace=namespace,
            session_id=session_id,
            include_session=True,
            include_cross_session=True,
            limit=120,
        )
        for fact in facts:
            add(self._fact_candidate(fact))

        for message in self.store.search_message_candidates(session_id, query, limit=30):
            add(
                RetrievalCandidate(
                    source_type='message',
                    source_id=message.id,
                    text=f'[{message.role}] {message.content}',
                    metadata={'created_at': message.created_at, 'scope': 'session', **message.metadata},
                )
            )
        for message in self.store.list_recent_messages(session_id, limit=24):
            add(
                RetrievalCandidate(
                    source_type='message',
                    source_id=message.id,
                    text=f'[{message.role}] {message.content}',
                    metadata={'created_at': message.created_at, 'scope': 'session', **message.metadata},
                )
            )

        for summary in self.store.search_summary_candidates(session_id, query, limit=16):
            add(
                RetrievalCandidate(
                    source_type='summary',
                    source_id=summary.id,
                    text=summary.content,
                    metadata={'level': summary.level, 'created_at': summary.created_at, 'scope': 'session', **summary.metadata},
                )
            )
        for summary in self.store.list_recent_summaries(session_id, limit=12):
            add(
                RetrievalCandidate(
                    source_type='summary',
                    source_id=summary.id,
                    text=summary.content,
                    metadata={'level': summary.level, 'created_at': summary.created_at, 'scope': 'session', **summary.metadata},
                )
            )

        for artifact in self.store.search_artifact_candidates(session_id, query, limit=10):
            add(
                RetrievalCandidate(
                    source_type='artifact',
                    source_id=artifact.id,
                    text=f'[{artifact.kind}] {artifact.path} :: {artifact.preview}',
                    metadata={'path': artifact.path, 'mtime': artifact.mtime, 'scope': 'session', **artifact.metadata},
                )
            )
        for artifact in self.store.list_recent_artifacts(session_id, limit=8):
            add(
                RetrievalCandidate(
                    source_type='artifact',
                    source_id=artifact.id,
                    text=f'[{artifact.kind}] {artifact.path} :: {artifact.preview}',
                    metadata={'path': artifact.path, 'mtime': artifact.mtime, 'scope': 'session', **artifact.metadata},
                )
            )

        for event in self.store.list_recent_events(session_id, limit=10):
            add(
                RetrievalCandidate(
                    source_type='failure',
                    source_id=event.id,
                    text=f'[{event.classifier}] {event.content}',
                    metadata={**event.metadata, 'created_at': event.created_at, 'scope': 'session'},
                )
            )

        return list(by_key.values())

    def _fact_candidate(self, fact: FactRecord) -> RetrievalCandidate:
        return RetrievalCandidate(
            source_type='fact',
            source_id=fact.id,
            text=f'[{fact.scope}] {fact.category}.{fact.key} = {fact.value}',
            metadata={
                'scope': fact.scope,
                'category': fact.category,
                'key': fact.key,
                'importance': fact.importance,
                'created_at': fact.created_at,
                **fact.metadata,
            },
        )

    def _fill_retrieval_gaps(
        self,
        *,
        query: str,
        ranked_candidates: list[RetrievalCandidate],
        selected: list[RetrievedMemory],
        limit: int,
    ) -> list[RetrievedMemory]:
        results = list(selected)
        seen = {(item.source_type, item.source_id) for item in results}

        wants_long_term = self._query_mentions_long_term(query)
        wants_task = self._query_mentions_task(query)
        wants_preferences = self._query_mentions_preferences(query)

        has_long_term = any((item.metadata or {}).get('scope') == 'cross_session' for item in results)
        has_task = any((item.metadata or {}).get('scope') == 'session' for item in results)
        has_preferences = any(self._is_preference_payload(item.text, item.metadata) for item in results)

        for candidate in ranked_candidates:
            if len(results) >= limit:
                break
            key = (candidate.source_type, candidate.source_id)
            if key in seen:
                continue
            scope = (candidate.metadata or {}).get('scope')
            need_long_term = wants_long_term and not has_long_term
            need_task = wants_task and not has_task
            if need_long_term and need_task:
                if scope not in {'cross_session', 'session'}:
                    continue
            elif need_long_term:
                if scope != 'cross_session':
                    continue
            elif need_task:
                if scope != 'session':
                    continue
            if wants_long_term and has_long_term and scope == 'cross_session':
                continue
            if wants_task and has_task and scope == 'session':
                continue
            score = self._heuristic_score(query=query, candidate=candidate)
            if score <= 0:
                continue
            results.append(
                RetrievedMemory(
                    source_type=candidate.source_type,  # type: ignore[arg-type]
                    source_id=candidate.source_id,
                    score=self._normalize_score(score),
                    text=candidate.text,
                    metadata={**candidate.metadata, 'selection_reason': 'heuristic_fallback'},
                )
            )
            seen.add(key)
            has_long_term = has_long_term or scope == 'cross_session'
            has_task = has_task or scope == 'session'
            has_preferences = has_preferences or self._is_preference_payload(candidate.text, candidate.metadata)

        if wants_preferences and not has_preferences:
            for candidate in ranked_candidates:
                if len(results) >= limit:
                    break
                key = (candidate.source_type, candidate.source_id)
                if key in seen or not self._is_preference_payload(candidate.text, candidate.metadata):
                    continue
                score = self._heuristic_score(query=query, candidate=candidate)
                if score <= 0:
                    continue
                results.append(
                    RetrievedMemory(
                        source_type=candidate.source_type,  # type: ignore[arg-type]
                        source_id=candidate.source_id,
                        score=self._normalize_score(score),
                        text=candidate.text,
                        metadata={**candidate.metadata, 'selection_reason': 'heuristic_preference_cover'},
                    )
                )
                seen.add(key)
                has_preferences = True
                break

        desired_task_keys = self._desired_task_fact_keys(query)
        if wants_task and desired_task_keys:
            present_task_keys = {
                str((item.metadata or {}).get('key') or '')
                for item in results
                if (item.metadata or {}).get('scope') == 'session'
            }
            for desired_key in desired_task_keys:
                if len(results) >= limit or desired_key in present_task_keys:
                    continue
                for candidate in ranked_candidates:
                    if len(results) >= limit:
                        break
                    key = (candidate.source_type, candidate.source_id)
                    metadata = candidate.metadata or {}
                    if key in seen:
                        continue
                    if metadata.get('scope') != 'session':
                        continue
                    if metadata.get('key') != desired_key:
                        continue
                    score = self._heuristic_score(query=query, candidate=candidate)
                    if score <= 0:
                        score = 1.0
                    results.append(
                        RetrievedMemory(
                            source_type=candidate.source_type,  # type: ignore[arg-type]
                            source_id=candidate.source_id,
                            score=self._normalize_score(score),
                            text=candidate.text,
                            metadata={**metadata, 'selection_reason': 'heuristic_task_key_cover'},
                        )
                    )
                    seen.add(key)
                    present_task_keys.add(desired_key)
                    break

        if results:
            return results[:limit]

        for candidate in ranked_candidates[:limit]:
            score = self._heuristic_score(query=query, candidate=candidate)
            if score <= 0:
                continue
            results.append(
                RetrievedMemory(
                    source_type=candidate.source_type,  # type: ignore[arg-type]
                    source_id=candidate.source_id,
                    score=self._normalize_score(score),
                    text=candidate.text,
                    metadata={**candidate.metadata, 'selection_reason': 'heuristic_backstop'},
                )
            )
        return results[:limit]

    def _heuristic_score(self, *, query: str, candidate: RetrievalCandidate) -> float:
        query_lower = query.lower()
        text_lower = candidate.text.lower()
        metadata = candidate.metadata or {}

        score = 0.0
        if candidate.source_type == 'fact':
            score += 0.4

        importance = metadata.get('importance')
        try:
            score += max(0.0, min(1.0, float(importance))) * 0.6
        except Exception:
            pass

        for token in self._keyword_tokens(query):
            if token in text_lower:
                score += 0.7

        scope = metadata.get('scope')
        category = str(metadata.get('category') or '')
        fact_key = str(metadata.get('key') or '')

        if self._query_mentions_long_term(query):
            if scope == 'cross_session':
                score += 1.3
            if category in {'profile', 'preference'}:
                score += 0.8
            if fact_key in {'long_term_preferences', 'preferred_technologies'}:
                score += 0.8
            if fact_key in {'occupation', 'profession'}:
                score += 0.3

        if self._query_mentions_task(query):
            if scope == 'session':
                score += 1.2
            if category == 'task':
                score += 0.8
            if fact_key in {'current_task', 'project_name', 'tech_stack', 'constraints'}:
                score += 0.8
            if fact_key in self._desired_task_fact_keys(query):
                score += 1.1

        return score

    def _keyword_tokens(self, text: str) -> list[str]:
        tokens = re.findall(r'[a-z0-9_+\-#.]{2,}', text.lower())
        deduped: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            if token in seen:
                continue
            seen.add(token)
            deduped.append(token)
        return deduped

    def _query_mentions_long_term(self, query: str) -> bool:
        return any(token in query for token in ['长期', '偏好', '喜欢', '记得', '风格', '习惯'])

    def _query_mentions_task(self, query: str) -> bool:
        return any(token in query for token in ['当前任务', '任务', '项目', '项目名', '技术栈', '约束', '限制', '文档编号', '状态枚举', '禁止项'])

    def _query_mentions_preferences(self, query: str) -> bool:
        return any(token in query for token in ['偏好', '喜欢', '喜好'])

    def _desired_task_fact_keys(self, query: str) -> list[str]:
        desired: list[str] = []
        mappings = [
            ('project_name', ['项目名', 'project name']),
            ('design_doc_id', ['设计文档编号', 'design_doc_id', 'doc_id', '文档编号']),
            ('order_status_enum', ['订单状态枚举', 'order_status_enum', '状态枚举']),
            ('constraints', ['禁止项', '约束', '限制', 'prohibitions', 'constraints']),
            ('tech_stack', ['技术栈', 'tech_stack']),
            ('current_task', ['当前任务', '任务']),
        ]
        for key, markers in mappings:
            if any(marker in query for marker in markers):
                desired.append(key)
        if self._query_mentions_task(query):
            for fallback in ['project_name', 'design_doc_id', 'order_status_enum', 'constraints', 'tech_stack']:
                if fallback not in desired:
                    desired.append(fallback)
        return desired

    def _is_preference_payload(self, text: str, metadata: dict[str, Any] | None) -> bool:
        metadata = metadata or {}
        category = str(metadata.get('category') or '')
        fact_key = str(metadata.get('key') or '')
        return category == 'preference' or fact_key in {'long_term_preferences', 'preferred_technologies'} or '偏好' in text

    def _normalize_score(self, raw: float) -> float:
        return max(0.05, min(1.0, raw / 4.0 if raw else 0.0))
