from __future__ import annotations

from .store import FileMemoryStore
from .models import MemoryRecallResult


def recall_memory(store: FileMemoryStore, query: str, max_items: int = 6) -> MemoryRecallResult:
    return store.recall(query, max_items=max_items)
