from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SkillManifest(BaseModel):
    model_config = ConfigDict(extra='allow')

    name: str
    title: str | None = None
    version: str = '0.1.0'
    description: str = ''
    triggers: list[str] = Field(default_factory=list)
    instructions: str = ''
    allowed_tools: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
