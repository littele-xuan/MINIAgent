from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

MemoryKind = Literal['profile', 'preference', 'task_constraint', 'episodic', 'procedural', 'tool_result', 'assistant_turn', 'explicit_memory']
MemoryBucket = Literal['profile', 'task', 'memories', 'episodic', 'assistant', 'events', 'procedural']


class MemoryRecord(BaseModel):
    model_config = ConfigDict(extra='forbid')

    text: str
    kind: MemoryKind
    bucket: str
    user_id: str
    thread_id: str | None = None
    source: str = 'conversation'
    confidence: float = 1.0
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationEvent(BaseModel):
    model_config = ConfigDict(extra='forbid')

    event_type: Literal['user', 'assistant', 'tool', 'system']
    text: str
    user_id: str
    thread_id: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    payload: dict[str, Any] = Field(default_factory=dict)
