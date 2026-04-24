from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_runtime import BaseLLM

from .base import BaseMemoryEngine
from .factory import MemoryRuntimeComponents, MemoryRuntimeFactory
from .models import ContextPacket, FileArtifact, MessageRecord, RetrievedMemory, SummaryNode
from .repository import now_iso
from .store import new_id
from .summarizers import SummaryPlan
from .token_counter import estimate_tokens


@dataclass(slots=True)
class MemoryEngineConfig:
    root_dir: str
    namespace: str = 'default'
    session_id: str | None = None
    soft_token_limit: int = 2200
    hard_token_limit: int = 3200
    keep_recent_messages: int = 6
    summary_target_tokens: int = 650
    large_observation_tokens: int = 500
    retrieve_limit: int = 8
    auto_git_commit: bool = False
    api_base: str | None = None
    api_key: str | None = None
    model: str | None = None
    temperature: float = 0.0
    connect_timeout_seconds: float = 20.0
    request_timeout_seconds: float = 90.0
    retrieval_candidate_limit: int = 72


class ContextMemoryEngine(BaseMemoryEngine):
    """Production-oriented memory runtime.

    Structure:
    - immutable log + active-context projection
    - session memory and cross-session durable memory
    - summary DAG for compaction / expansion
    - artifacts for large observations
    - swappable implementations composed through MemoryRuntimeFactory
    """

    def __init__(
        self,
        config: MemoryEngineConfig,
        *,
        llm: BaseLLM | None = None,
        components: MemoryRuntimeComponents | None = None,
    ) -> None:
        self.config = config
        self.root = Path(config.root_dir)
        self.root.mkdir(parents=True, exist_ok=True)

        bundle = components or MemoryRuntimeFactory.create(config=config, llm=llm)
        self.llm = bundle.llm
        self.store = bundle.store
        self.repository = bundle.repository
        self.extractor = bundle.extractor
        self.failure_classifier = bundle.failure_classifier
        self.summarizer = bundle.summarizer
        self.retriever = bundle.retriever
        self.resolver = bundle.resolver
        self._owns_llm = bundle.owns_llm

        self.session_id = config.session_id or new_id('task')
        self.namespace = config.namespace
        self._lock = asyncio.Lock()
        self._pending_compaction: asyncio.Task[None] | None = None
        self._current_turn_id: str | None = None
        self._current_query_message_id: str | None = None
        self.store.ensure_session(self.session_id, namespace=self.namespace, created_at=now_iso(), metadata={'engine': 'context-runtime-llm'})
        self._persist_session_state()

    async def begin_turn(self, query: str) -> None:
        async with self._lock:
            now = now_iso()
            self._current_turn_id = new_id('turn')
            record = MessageRecord(
                id=new_id('msg'),
                session_id=self.session_id,
                turn_id=self._current_turn_id,
                role='user',
                content=query,
                created_at=now,
                token_count=estimate_tokens(query),
                metadata={'phase': 'query'},
            )
            self._current_query_message_id = record.id
            self.store.append_message(record)
            self.repository.append_message(record)
            await self._promote_facts(record, now)
            self._persist_session_state()
            self._schedule_compaction_if_needed()

    async def record_observation(self, event: dict[str, Any]) -> None:
        text = event.get('observation') or ''
        if not text:
            return
        async with self._lock:
            now = now_iso()
            content = text
            metadata = {'mode': event.get('mode')}
            if event.get('mode') == 'mcp':
                metadata['calls'] = event.get('calls', [])
            if estimate_tokens(content) >= self.config.large_observation_tokens:
                artifact = self._persist_large_blob(kind='tool_output', content=content, created_at=now, metadata=metadata)
                content = f'[artifact:{artifact.id}] {artifact.preview}'
                metadata = {**metadata, 'artifact_id': artifact.id, 'artifact_path': artifact.path}
            record = MessageRecord(
                id=new_id('msg'),
                session_id=self.session_id,
                turn_id=self._current_turn_id or new_id('turn'),
                role='tool',
                content=content,
                created_at=now,
                token_count=estimate_tokens(content),
                metadata=metadata,
            )
            self.store.append_message(record)
            self.repository.append_message(record)
            await self._promote_facts(record, now)
            event_obj = await self.failure_classifier.from_observation(
                session_id=self.session_id,
                content=content,
                created_at=now,
                metadata=metadata,
            )
            if event_obj is not None:
                self.store.add_event(event_obj)
                self.repository.write_event(event_obj)
            self._persist_session_state()
            self._schedule_compaction_if_needed()

    async def finalize_turn(self, *, answer: str, output_mode: str, payload: Any | None = None) -> None:
        async with self._lock:
            now = now_iso()
            content = answer if output_mode == 'text/plain' else json.dumps(payload, ensure_ascii=False, default=str)
            record = MessageRecord(
                id=new_id('msg'),
                session_id=self.session_id,
                turn_id=self._current_turn_id or new_id('turn'),
                role='assistant',
                content=content,
                created_at=now,
                token_count=estimate_tokens(content),
                metadata={'phase': 'final', 'output_mode': output_mode},
            )
            self.store.append_message(record)
            self.repository.append_message(record)
            await self._promote_facts(record, now)
            self._schedule_compaction_if_needed(force_blocking=False)
            self.repository.maybe_commit(f'context-runtime: turn {self._current_turn_id or "unknown"}')
            self._current_turn_id = None
            self._current_query_message_id = None
            self._persist_session_state()

    async def build_context_packet(self, *, query: str) -> ContextPacket:
        stats = self.store.session_stats(self.session_id)
        recent = self.store.list_recent_messages(self.session_id, limit=max(self.config.keep_recent_messages, 8))
        active_items = self.store.list_active_items_with_content(self.session_id)
        summaries = [payload for _, payload in active_items if isinstance(payload, SummaryNode)]
        retrieved = await self.retriever.retrieve(namespace=self.namespace, session_id=self.session_id, query=query, limit=self.config.retrieve_limit)
        warnings: list[str] = []
        if stats['active_tokens'] >= self.config.soft_token_limit:
            warnings.append('active context is above soft threshold; engine compaction is managing older state automatically')
        if stats['active_tokens'] >= self.config.hard_token_limit:
            warnings.append('active context exceeded hard threshold; next turn will block on compaction before planning')
        warnings.extend(self._stale_file_warnings(retrieved))
        warnings.extend(self._global_stale_warnings())
        return ContextPacket(
            session_id=self.session_id,
            recent_messages=recent,
            active_summaries=list(summaries),
            retrieved_memories=retrieved,
            pinned_notes=self.repository.system_notes(),
            warnings=warnings,
            stats=stats,
        )

    async def answer_memory_query(self, query: str) -> str | None:
        packet = await self.build_context_packet(query=query)
        return await self.resolver.answer(namespace=self.namespace, session_id=self.session_id, query=query, warnings=packet.warnings)

    async def inspect_state(self) -> dict[str, Any]:
        packet = await self.build_context_packet(query='state inspection')
        return {
            'session_id': self.session_id,
            'namespace': self.namespace,
            'stats': packet.stats,
            'warnings': packet.warnings,
            'recent_messages': [msg.content for msg in packet.recent_messages[-4:]],
            'retrieved_preview': [item.text for item in packet.retrieved_memories[:4]],
        }

    async def ensure_hard_limit(self) -> None:
        stats = self.store.session_stats(self.session_id)
        if stats['active_tokens'] < self.config.hard_token_limit:
            return
        async with self._lock:
            await self._compact_until_within_limit(blocking=True)
            self._persist_session_state()

    def expand_summary(self, summary_id: str) -> list[dict[str, Any]]:
        messages = self.store.expand_summary_to_messages(summary_id)
        return [
            {
                'id': message.id,
                'role': message.role,
                'content': message.content,
                'created_at': message.created_at,
            }
            for message in messages
        ]

    def snapshot_file(self, path: str | Path) -> dict[str, Any]:
        snap = self.repository.snapshot_file(path)
        return {
            'path': snap.path,
            'exists': snap.exists,
            'mtime': snap.mtime,
            'size_bytes': snap.size_bytes,
            'checksum': snap.checksum,
        }

    async def record_file_reference(self, *, path: str | Path, kind: str = 'file_ref', note: str = '') -> FileArtifact:
        async with self._lock:
            snap = self.repository.snapshot_file(path)
            preview = note or f'{Path(path).name} mtime={snap.mtime} checksum={snap.checksum}'
            artifact = FileArtifact(
                id=new_id('art'),
                session_id=self.session_id,
                kind=kind,
                path=str(path),
                preview=preview,
                checksum=snap.checksum or 'missing',
                created_at=now_iso(),
                size_bytes=snap.size_bytes or 0,
                mtime=snap.mtime,
                metadata={'tracked_file': True},
            )
            self.store.add_artifact(artifact)
            self.repository.write_artifact(artifact)
            self._persist_session_state()
            return artifact

    async def close(self) -> None:
        task = self._pending_compaction
        if task is not None and not task.done():
            await asyncio.wait([task])
        if self._owns_llm:
            await self.llm.close()

    async def _promote_facts(self, message: MessageRecord, created_at: str) -> None:
        for extracted in await self.extractor.materialize(namespace=self.namespace, session_id=self.session_id, message=message, created_at=created_at):
            if extracted.op == 'revoke':
                self.store.expire_active_facts(
                    namespace=self.namespace,
                    category=extracted.category,
                    key=extracted.key,
                    scope=extracted.scope,
                    session_id=self.session_id if extracted.scope == 'session' else None,
                    expired_at=created_at,
                    invalid_at=created_at,
                )
                continue
            fact = self.extractor.to_record(namespace=self.namespace, session_id=self.session_id, extracted=extracted, source_message_id=message.id, created_at=created_at)
            if extracted.replace_existing:
                self.store.expire_active_facts(
                    namespace=self.namespace,
                    category=fact.category,
                    key=fact.key,
                    scope=fact.scope,
                    session_id=fact.session_id,
                    expired_at=created_at,
                    invalid_at=created_at,
                )
            self.store.add_fact(fact)
            self.repository.write_fact(fact)

    def _persist_large_blob(self, *, kind: str, content: str, created_at: str, metadata: dict[str, Any]) -> FileArtifact:
        checksum = hashlib.sha256(content.encode('utf-8')).hexdigest()
        path = self.repository.artifact_dir / f'{kind}_{checksum[:12]}.txt'
        artifact = FileArtifact(
            id=new_id('art'),
            session_id=self.session_id,
            kind=kind,
            path=str(path),
            preview=content[:240].replace('\n', ' '),
            checksum=checksum,
            created_at=created_at,
            size_bytes=len(content.encode('utf-8')),
            mtime=path.stat().st_mtime if path.exists() else None,
            metadata=metadata,
        )
        self.repository.write_artifact(artifact, body=content)
        artifact.mtime = Path(artifact.path).stat().st_mtime
        self.store.add_artifact(artifact)
        return artifact

    def _schedule_compaction_if_needed(self, *, force_blocking: bool = False) -> None:
        stats = self.store.session_stats(self.session_id)
        if force_blocking or stats['active_tokens'] >= self.config.hard_token_limit:
            return
        if stats['active_tokens'] < self.config.soft_token_limit:
            return
        if self._pending_compaction is not None and not self._pending_compaction.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._pending_compaction = loop.create_task(self._compact_until_within_limit(blocking=False))

    async def _compact_until_within_limit(self, *, blocking: bool) -> None:
        while True:
            stats = self.store.session_stats(self.session_id)
            target = self.config.soft_token_limit if not blocking else max(self.config.soft_token_limit - 400, int(self.config.soft_token_limit * 0.8))
            if stats['active_tokens'] <= target:
                return
            active_items = self.store.list_active_items_with_content(self.session_id)
            raw_candidates = [pair for pair in active_items[:-self.config.keep_recent_messages] if pair[1] is not None]
            if len(raw_candidates) < 2:
                return
            block_pairs = raw_candidates[: min(6, len(raw_candidates))]
            start_position = block_pairs[0][0].position
            end_position = block_pairs[-1][0].position
            payloads = [pair[1] for pair in block_pairs if pair[1] is not None]
            summary = await self.summarizer.summarize_block(
                session_id=self.session_id,
                created_at=now_iso(),
                items=payloads,  # type: ignore[arg-type]
                plans=[
                    SummaryPlan(level=1, target_tokens=self.config.summary_target_tokens),
                    SummaryPlan(level=2, target_tokens=max(280, self.config.summary_target_tokens // 2)),
                    SummaryPlan(level=3, target_tokens=180),
                ],
            )
            self.store.insert_summary(summary)
            self.repository.write_summary(summary)
            self.store.replace_active_items_with_summary(
                self.session_id,
                start_position=start_position,
                end_position=end_position,
                summary_id=summary.id,
                token_count=summary.token_count,
            )
            if not blocking:
                await asyncio.sleep(0)

    def _persist_session_state(self) -> None:
        self.repository.write_session_state(
            session_id=self.session_id,
            payload={
                'session_id': self.session_id,
                'namespace': self.namespace,
                'stats': self.store.session_stats(self.session_id),
            },
        )

    def _global_stale_warnings(self) -> list[str]:
        warnings: list[str] = []
        for artifact in self.store.list_recent_artifacts(self.session_id, limit=20):
            if not artifact.metadata.get('tracked_file'):
                continue
            snap = self.repository.snapshot_file(artifact.path)
            if snap.exists and artifact.mtime is not None and snap.mtime is not None and snap.mtime > float(artifact.mtime):
                warnings.append(f'file changed since memory capture: {artifact.path}. re-read before trusting cached context')
        return warnings

    def _stale_file_warnings(self, retrieved: list[RetrievedMemory]) -> list[str]:
        warnings: list[str] = []
        for item in retrieved:
            path = item.metadata.get('path')
            old_mtime = item.metadata.get('mtime')
            if not path:
                continue
            snap = self.repository.snapshot_file(path)
            if snap.exists and old_mtime is not None and snap.mtime is not None and snap.mtime > float(old_mtime):
                warnings.append(f'file changed since memory capture: {path}. re-read before trusting cached context')
        return warnings
