from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TraceEvent(BaseModel):
    model_config = ConfigDict(extra='forbid')

    mode: str
    thought: str = ''
    observation: str = ''
    payload: dict[str, Any] = Field(default_factory=dict)
