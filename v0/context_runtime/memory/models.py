from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


Role = Literal["user", "assistant", "tool", "system"]
ContextItemKind = Literal["message", "summary"]
MemoryScope = Literal["session", "cross_session"]
RetrievedSourceType = Literal["fact", "summary", "message", "artifact", "failure"]


@dataclass(slots=True)
class MessageRecord:
    id: str
    session_id: str
    turn_id: str
    role: Role
    content: str
    created_at: str
    token_count: int
    kind: str = 'message'
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SummaryNode:
    id: str
    session_id: str
    level: int
    content: str
    created_at: str
    token_count: int
    source_item_ids: list[str] = field(default_factory=list)
    leaf_message_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ActiveContextItem:
    position: int
    kind: ContextItemKind
    item_id: str
    token_count: int


@dataclass(slots=True)
class FactRecord:
    id: str
    namespace: str
    session_id: str | None
    category: str
    key: str
    value: str
    scope: MemoryScope
    created_at: str
    valid_at: str | None = None
    invalid_at: str | None = None
    expired_at: str | None = None
    importance: float = 0.5
    status: str = 'active'
    source_message_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def temporal_label(self) -> str:
        start = self.valid_at or self.created_at
        end = self.invalid_at or 'present'
        return f'{start} -> {end}'


@dataclass(slots=True)
class RetrievedMemory:
    source_type: RetrievedSourceType
    source_id: str
    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ContextPacket:
    session_id: str
    recent_messages: list[MessageRecord] = field(default_factory=list)
    active_summaries: list[SummaryNode] = field(default_factory=list)
    retrieved_memories: list[RetrievedMemory] = field(default_factory=list)
    pinned_notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MemoryEvent:
    id: str
    session_id: str
    event_type: str
    classifier: str
    content: str
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FileArtifact:
    id: str
    session_id: str
    kind: str
    path: str
    preview: str
    checksum: str
    created_at: str
    size_bytes: int = 0
    mtime: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
