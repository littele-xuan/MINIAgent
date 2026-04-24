from __future__ import annotations

from .models import MemoryItem
from .store import FileMemoryStore


def propose_memory(store: FileMemoryStore, *, content: str, evidence: str, layer: str = "facts", source: str = "agent", tags: list[str] | None = None):
    item = MemoryItem(layer=layer, content=content, evidence=evidence, source=source, tags=tags or [])
    return store.propose_update(item)
