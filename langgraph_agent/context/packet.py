from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ContextPacket(BaseModel):
    model_config = ConfigDict(extra='forbid')

    system_contract: str
    task_state: dict[str, Any] = Field(default_factory=dict)
    selected_skill: dict[str, Any] | None = None
    memory_packet: dict[str, Any] = Field(default_factory=dict)
    tool_catalog: list[dict[str, Any]] = Field(default_factory=list)
    recent_history: list[dict[str, Any]] = Field(default_factory=list)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    token_budget_report: dict[str, Any] = Field(default_factory=dict)
