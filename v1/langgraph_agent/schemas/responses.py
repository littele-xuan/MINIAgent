from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_serializer, model_validator


def _parse_json_object(raw: Any, *, fallback: dict[str, Any] | None = None) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if raw in (None, ''):
        return fallback or {}
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
        except Exception as exc:
            raise ValueError(f'arguments_json is not valid JSON: {exc}') from exc
        if not isinstance(value, dict):
            raise ValueError('arguments_json must decode to a JSON object')
        return value
    return fallback or {}


def _parse_json_payload(raw: Any) -> Any:
    if raw is None or raw == '':
        return None
    if isinstance(raw, (dict, list, int, float, bool)):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return raw
    return raw


class MCPCall(BaseModel):
    """Canonical MCP-compatible tool call emitted by the planner.

    The canonical field is ``arguments`` as a JSON object. ``arguments_json`` is
    accepted only for backwards compatibility with the older refactor.
    """

    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    tool_name: str = Field(validation_alias=AliasChoices('tool_name', 'name', 'tool'))
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices('arguments', 'args', 'input', 'parameters', 'arguments_json'),
    )
    reason: str = ''

    @model_validator(mode='before')
    @classmethod
    def _normalize(cls, raw: Any) -> Any:
        if not isinstance(raw, dict):
            return raw
        value = dict(raw)
        if 'tool_name' not in value:
            for key in ('name', 'tool'):
                if key in value:
                    value['tool_name'] = value.pop(key)
                    break
        if 'arguments' not in value:
            for key in ('args', 'input', 'parameters', 'arguments_json'):
                if key in value:
                    value['arguments'] = _parse_json_object(value.pop(key))
                    break
        elif isinstance(value.get('arguments'), str):
            value['arguments'] = _parse_json_object(value['arguments'])
        return value

    @property
    def arguments_json(self) -> str:
        return json.dumps(self.arguments, ensure_ascii=False, default=str)


ToolCallSpec = MCPCall


class DelegateSpec(BaseModel):
    model_config = ConfigDict(extra='forbid')

    peer_name: str
    message: str
    reason: str = ''


class FinalResponse(BaseModel):
    model_config = ConfigDict(extra='forbid')

    output_mode: Literal['text/plain', 'application/json'] = 'text/plain'
    text: str | None = None
    data: Any = Field(default=None, validation_alias=AliasChoices('data', 'payload', 'data_json'))

    @model_validator(mode='before')
    @classmethod
    def _normalize(cls, raw: Any) -> Any:
        if not isinstance(raw, dict):
            return raw
        value = dict(raw)
        if 'data' not in value:
            if 'payload' in value:
                value['data'] = value.pop('payload')
            elif 'data_json' in value:
                value['data'] = _parse_json_payload(value.pop('data_json'))
        elif isinstance(value.get('data'), str):
            value['data'] = _parse_json_payload(value.get('data'))
        return value

    @property
    def payload(self) -> Any:
        return self.data

    @property
    def data_json(self) -> str | None:
        if self.data is None:
            return None
        if isinstance(self.data, str):
            return self.data
        return json.dumps(self.data, ensure_ascii=False, default=str)

    @field_serializer('data')
    def _serialize_data(self, value: Any) -> Any:
        return value

    @model_validator(mode='after')
    def _validate_payload(self) -> 'FinalResponse':
        if self.output_mode == 'application/json' and self.data is None:
            raise ValueError('application/json output requires data')
        if self.output_mode == 'text/plain' and self.text is None and self.data is None:
            raise ValueError('text/plain output requires text or data')
        return self


class AgentDecision(BaseModel):
    """Strict planner contract for the LangGraph while-loop agent."""

    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    thought: str = ''
    mode: Literal['final', 'mcp', 'delegate', 'ask_human', 'memory_update', 'reflect', 'error_recovery']
    final: FinalResponse | None = None
    mcp_calls: list[MCPCall] = Field(default_factory=list, validation_alias=AliasChoices('mcp_calls', 'tool_calls', 'calls'))
    delegate: DelegateSpec | None = None
    question: str | None = None
    memory_ops: list[dict[str, Any]] = Field(default_factory=list)
    confidence: float | None = None

    @model_validator(mode='before')
    @classmethod
    def _normalize(cls, raw: Any) -> Any:
        if not isinstance(raw, dict):
            return raw
        value = dict(raw)
        if value.get('mode') == 'tool':
            value['mode'] = 'mcp'
        if 'mcp_calls' not in value:
            calls = value.get('tool_calls') or value.get('calls')
            if calls is not None:
                value['mcp_calls'] = calls
        value.pop('tool_calls', None)
        value.pop('calls', None)
        return value

    @property
    def tool_calls(self) -> list[MCPCall]:
        return self.mcp_calls

    @model_validator(mode='after')
    def _validate(self) -> 'AgentDecision':
        if self.mode == 'final' and self.final is None:
            raise ValueError('mode=final requires final')
        if self.mode == 'mcp' and not self.mcp_calls:
            raise ValueError('mode=mcp requires at least one mcp_call')
        if self.mode == 'delegate' and self.delegate is None:
            raise ValueError('mode=delegate requires delegate')
        if self.mode == 'ask_human' and not self.question:
            raise ValueError('mode=ask_human requires question')
        if self.mode != 'mcp' and self.mcp_calls:
            raise ValueError('mcp_calls are only allowed when mode=mcp')
        return self


PlanDecision = AgentDecision
