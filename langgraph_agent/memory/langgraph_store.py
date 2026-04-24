from __future__ import annotations

from typing import Any

from ..config.models import MemoryConfig
from ..llm.base import BaseStructuredLLM
from ..utils import compact_text, json_safe
from .base import BaseMemoryManager
from .event_log import MemoryEventLog
from .extractors import HybridMemoryExtractor
from .resolver import MemoryResolver


class LangGraphStoreMemoryManager(BaseMemoryManager):
    """Long-term memory manager backed by LangGraph Store namespaces.

    Short-term conversational state remains in LangGraph checkpoints. This class
    manages cross-thread JSON memories, recent events, and optional LLM-based
    extraction behind a replaceable interface.
    """

    def __init__(self, store: Any, config: MemoryConfig, *, llm: BaseStructuredLLM | None = None) -> None:
        self.store = store
        self.config = config
        self.resolver = MemoryResolver()
        self.extractor = HybridMemoryExtractor(llm if config.extractor in {'llm', 'hybrid'} else None)
        self.event_log = MemoryEventLog(store, namespace_prefix=config.namespace)
        self._cache: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}

    def _namespace(self, user_id: str, bucket: str) -> tuple[str, str, str]:
        return (self.config.namespace, user_id, bucket)

    def _put(self, namespace: tuple[str, str, str], key: str, value: dict[str, Any]) -> None:
        safe = json_safe(value)
        self._cache.setdefault(namespace, {})[key] = safe
        try:
            self.store.put(namespace, key, safe)
        except TypeError:
            self.store.put(namespace, key, safe, index=False)
        except Exception as exc:
            raise RuntimeError(f'Failed to persist memory in namespace={namespace}: {exc}') from exc

    def _search_store(self, namespace: tuple[str, str, str], query: str) -> list[Any]:
        try:
            return self.store.search(namespace, query=query, limit=self.config.retrieval_limit)
        except TypeError:
            try:
                return self.store.search(namespace, query=query)
            except Exception:
                return []
        except Exception:
            return []

    def _value_from_item(self, item: Any) -> dict[str, Any] | None:
        value = getattr(item, 'value', None) or item
        if isinstance(value, dict):
            return json_safe(value)
        if value:
            return {'text': str(value), 'kind': 'unknown'}
        return None

    async def load_context(self, *, user_id: str, thread_id: str, query: str) -> dict[str, Any]:
        memories: list[dict[str, Any]] = []
        buckets = ('profile', 'task', self.config.long_term_namespace, 'procedural', 'episodic', 'assistant')
        for bucket in buckets:
            namespace = self._namespace(user_id, bucket)
            for value in self._cache.get(namespace, {}).values():
                memories.append({**value, 'bucket': value.get('bucket') or bucket})
            for item in self._search_store(namespace, query) or []:
                value = self._value_from_item(item)
                if value:
                    memories.append({**value, 'bucket': value.get('bucket') or bucket})

        deduped: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in memories:
            text = str(item.get('text') or item.get('value') or item.get('content') or '').strip()
            if not text:
                continue
            key = f"{item.get('bucket')}::{item.get('kind')}::{text.lower()}"
            if key in seen:
                continue
            seen.add(key)
            deduped.append({**item, 'text': compact_text(text, limit=1200)})
            if len(deduped) >= self.config.retrieval_limit:
                break
        return {
            'memories': deduped,
            'recent_events': self.event_log.recent(user_id=user_id, limit=self.config.event_log_limit),
            'thread_id': thread_id,
            'warnings': [],
        }

    async def record_user_turn(self, *, user_id: str, thread_id: str, text: str) -> None:
        self.event_log.append(user_id=user_id, thread_id=thread_id, event_type='user', text=text)
        if not self.config.enabled or not self.config.extract_on_hot_path:
            return
        items = await self.extractor.extract(text, config=self.config, user_id=user_id, thread_id=thread_id)
        for bucket, raw in items:
            key, record = self.resolver.make_record(raw=raw, user_id=user_id, thread_id=thread_id, bucket=bucket)
            self._put(self._namespace(user_id, bucket), key, record)

    async def record_tool_result(self, *, user_id: str, thread_id: str, tool_name: str, content: str) -> None:
        self.event_log.append(
            user_id=user_id,
            thread_id=thread_id,
            event_type='tool',
            text=f'{tool_name}: {compact_text(content, limit=1000)}',
            payload={'tool_name': tool_name},
        )
        if not self.config.enabled or not content:
            return
        bucket = 'episodic'
        raw = {'text': f'{tool_name}: {compact_text(content, limit=800)}', 'kind': 'tool_result', 'source': 'tool'}
        key, record = self.resolver.make_record(raw=raw, user_id=user_id, thread_id=thread_id, bucket=bucket)
        self._put(self._namespace(user_id, bucket), key, record)

    async def record_assistant_turn(self, *, user_id: str, thread_id: str, text: str) -> None:
        self.event_log.append(user_id=user_id, thread_id=thread_id, event_type='assistant', text=text)
        if not self.config.enabled or not self.config.record_assistant_turns or not text:
            return
        bucket = 'assistant'
        raw = {'text': compact_text(text, limit=800), 'kind': 'assistant_turn', 'source': 'assistant'}
        key, record = self.resolver.make_record(raw=raw, user_id=user_id, thread_id=thread_id, bucket=bucket)
        self._put(self._namespace(user_id, bucket), key, record)

    async def consolidate(self, *, user_id: str, thread_id: str) -> None:
        # Interface hook for future Mem0/LangMem/Graphiti-style consolidation.
        # Hot-path extraction already writes atomic memories; keep consolidation
        # conservative so it cannot accidentally delete research evidence.
        return None
