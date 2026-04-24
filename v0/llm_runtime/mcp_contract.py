from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator


def _parse_json_object(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if raw in (None, ''):
        return {}
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


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


class MCPToolCallEnvelope(BaseModel):
    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    tool_name: str = Field(min_length=1, validation_alias=AliasChoices('tool_name', 'name', 'tool'), serialization_alias='tool_name')
    arguments_json: str = Field(
        default='{}',
        validation_alias=AliasChoices('arguments_json', 'arguments', 'args', 'input', 'parameters'),
        serialization_alias='arguments_json',
    )
    reason: str = ''

    @model_validator(mode='before')
    @classmethod
    def _compat_normalize(cls, raw: Any) -> Any:
        if not isinstance(raw, dict):
            return raw
        call = dict(raw)
        if 'tool_name' not in call:
            for key in ('name', 'tool'):
                if key in call:
                    call['tool_name'] = call.pop(key)
                    break
        if 'arguments_json' not in call:
            for key in ('arguments', 'args', 'input', 'parameters'):
                if key in call:
                    value = call.pop(key)
                    call['arguments_json'] = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                    break
        return call

    @property
    def arguments(self) -> dict[str, Any]:
        return _parse_json_object(self.arguments_json)


class MCPFinalEnvelope(BaseModel):
    model_config = ConfigDict(extra='forbid')

    output_mode: Literal['text/plain', 'application/json'] = 'application/json'
    text: str | None = None
    data_json: str | None = None

    @model_validator(mode='before')
    @classmethod
    def _compat_normalize(cls, raw: Any) -> Any:
        if not isinstance(raw, dict):
            return raw
        payload = dict(raw)
        if 'data_json' not in payload and 'data' in payload:
            data = payload.get('data')
            if isinstance(data, str):
                payload['data_json'] = data
            elif data is not None:
                payload['data_json'] = json.dumps(data, ensure_ascii=False)
            else:
                payload['data_json'] = None
        return payload

    @property
    def data(self) -> Any:
        return _parse_json_payload(self.data_json)

    @model_validator(mode='after')
    def _validate_payload(self) -> 'MCPFinalEnvelope':
        if self.output_mode == 'application/json' and self.data_json is None:
            raise ValueError('application/json final output requires data_json')
        if self.output_mode == 'text/plain' and self.text is None and self.data_json is None:
            raise ValueError('text/plain final output requires text or data_json')
        return self


class MCPStructuredResponse(BaseModel):
    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    thought: str = ''
    mode: Literal['mcp', 'final']
    final: MCPFinalEnvelope | None = None
    mcp_calls: list[MCPToolCallEnvelope] = Field(
        default_factory=list,
        validation_alias=AliasChoices('mcp_calls', 'calls', 'tool_calls'),
        serialization_alias='mcp_calls',
    )

    @model_validator(mode='before')
    @classmethod
    def _compat_normalize(cls, raw: Any) -> Any:
        if not isinstance(raw, dict):
            return raw
        payload = dict(raw)
        if 'mcp_calls' not in payload:
            calls = payload.get('calls') or payload.get('tool_calls')
            if isinstance(calls, list):
                payload['mcp_calls'] = [MCPToolCallEnvelope.model_validate(item).model_dump() if isinstance(item, dict) else item for item in calls]
        if 'final' in payload and isinstance(payload['final'], dict):
            payload['final'] = MCPFinalEnvelope.model_validate(payload['final']).model_dump()
        return payload

    @model_validator(mode='after')
    def _validate_mode(self) -> 'MCPStructuredResponse':
        if self.mode == 'mcp' and not self.mcp_calls:
            raise ValueError('mode=mcp requires at least one tool call')
        if self.mode == 'final' and self.final is None:
            raise ValueError('mode=final requires final payload')
        if self.mode != 'mcp' and self.mcp_calls:
            raise ValueError('mcp_calls are only allowed when mode=mcp')
        if self.mode != 'final' and self.final is not None:
            raise ValueError('final is only allowed when mode=final')
        return self
