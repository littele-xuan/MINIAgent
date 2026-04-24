from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


ToolRisk = Literal['safe_read', 'network_read', 'filesystem_read', 'filesystem_write', 'code_exec', 'external_write', 'destructive']


class ToolDescriptor(BaseModel):
    model_config = ConfigDict(extra='forbid')

    name: str
    description: str = ''
    input_schema: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    risk: ToolRisk = 'safe_read'
    enabled: bool = True
    version: str = '0.1.0'


class ArtifactRef(BaseModel):
    model_config = ConfigDict(extra='forbid')

    artifact_id: str
    kind: str = 'text'
    summary: str = ''
    uri: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolError(BaseModel):
    model_config = ConfigDict(extra='forbid')

    code: str = 'tool_error'
    message: str
    retryable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class ToolObservation(BaseModel):
    model_config = ConfigDict(extra='forbid')

    tool_name: str
    call_id: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    text: str = ''
    structured_content: dict[str, Any] | list[Any] | None = None
    payload: Any = None
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    status: Literal['ok', 'error'] = 'ok'
    error: str | None = None
    error_obj: ToolError | None = None
    latency_ms: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
