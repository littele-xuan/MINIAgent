from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from ..utils import compact_text, json_safe


class MemoryEventLog:
    def __init__(self, store: Any, *, namespace_prefix: str = 'default') -> None:
        self.store = store
        self.namespace_prefix = namespace_prefix
        self._cache: dict[tuple[str, str, str], list[dict[str, Any]]] = {}

    def _namespace(self, user_id: str, bucket: str) -> tuple[str, str, str]:
        return (self.namespace_prefix, user_id, bucket)

    def append(self, *, user_id: str, thread_id: str, event_type: str, text: str, payload: dict[str, Any] | None = None) -> None:
        namespace = self._namespace(user_id, 'events')
        record = {
            'event_type': event_type,
            'text': compact_text(text, limit=4000),
            'thread_id': thread_id,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'payload': json_safe(payload or {}),
        }
        self._cache.setdefault(namespace, []).append(record)
        try:
            self.store.put(namespace, str(uuid.uuid4()), record)
        except TypeError:
            self.store.put(namespace, str(uuid.uuid4()), record, index=False)
        except Exception:
            pass

    def _search_store(self, namespace: tuple[str, str, str], limit: int) -> list[dict[str, Any]]:
        try:
            items = self.store.search(namespace, query='', limit=limit)
        except TypeError:
            try:
                items = self.store.search(namespace, query='')
            except Exception:
                items = []
        except Exception:
            items = []
        out: list[dict[str, Any]] = []
        for item in items or []:
            value = getattr(item, 'value', None) or item
            if isinstance(value, dict):
                out.append(json_safe(value))
        return out

    def recent(self, *, user_id: str, limit: int = 12) -> list[dict[str, Any]]:
        namespace = self._namespace(user_id, 'events')
        combined = [*self._search_store(namespace, limit), *self._cache.get(namespace, [])]
        seen: set[str] = set()
        deduped: list[dict[str, Any]] = []
        for event in combined:
            key = f"{event.get('created_at')}::{event.get('thread_id')}::{event.get('event_type')}::{event.get('text')}"
            if key in seen:
                continue
            seen.add(key)
            deduped.append(event)
        deduped.sort(key=lambda item: str(item.get('created_at') or ''))
        return deduped[-limit:]
