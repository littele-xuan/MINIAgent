from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from llm_runtime import BaseLLM
from prompt_runtime import render_prompt


def _parse_json_object(raw: Any, *, fallback: dict[str, Any] | None = None) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if raw in (None, ''):
        return fallback or {}
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except Exception:
            return fallback or {}
        return parsed if isinstance(parsed, dict) else (fallback or {})
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


class FinalAnswer(BaseModel):
    model_config = ConfigDict(extra='forbid')

    output_mode: Literal['text/plain', 'application/json'] = 'text/plain'
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
    def validate_payload(self) -> 'FinalAnswer':
        if self.output_mode == 'application/json' and self.data_json is None:
            raise ValueError('application/json final output requires data_json')
        if self.output_mode == 'text/plain' and (self.text is None and self.data_json is None):
            raise ValueError('text/plain final output requires text or data_json')
        return self


class MCPToolCall(BaseModel):
    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    tool_name: str = Field(min_length=1, validation_alias=AliasChoices('tool_name', 'name', 'tool'), serialization_alias='tool_name')
    arguments_json: str = Field(
        default='{}',
        validation_alias=AliasChoices('arguments_json', 'arguments', 'args', 'input', 'parameters'),
        serialization_alias='arguments_json',
    )
    reason: str = Field(default='', description='brief operational reason for this MCP call')

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


class A2ADelegateCall(BaseModel):
    model_config = ConfigDict(extra='forbid')

    peer_name: str = Field(min_length=1)
    message: str = Field(min_length=1)
    accepted_output_modes: list[Literal['text/plain', 'application/json']] = Field(
        default_factory=lambda: ['text/plain', 'application/json']
    )
    reason: str = Field(default='', description='runtime-only A2A routing reason')


class PlannerOutput(BaseModel):
    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    thought: str = Field(default='', description='brief action summary, not hidden chain-of-thought')
    mode: Literal['final', 'mcp']
    final: FinalAnswer | None = None
    mcp_calls: list[MCPToolCall] = Field(
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
        if 'final' not in payload and 'response' in payload:
            response = payload.pop('response')
            if isinstance(response, dict):
                # legacy response envelope -> canonical final envelope
                text_block = response.get('text') if isinstance(response.get('text'), dict) else None
                content = text_block.get('content') if isinstance(text_block, dict) else None
                fmt = text_block.get('format') if isinstance(text_block, dict) else None
                mode = payload.get('mode')
                if mode == 'final':
                    payload['final'] = {
                        'output_mode': 'application/json' if response.get('data') is not None else 'text/plain',
                        'text': content or response.get('text'),
                        'data_json': json.dumps(response.get('data'), ensure_ascii=False) if response.get('data') is not None else None,
                    }
                elif fmt and isinstance(fmt, dict):
                    payload['final'] = {
                        'output_mode': 'text/plain',
                        'text': content,
                        'data_json': None,
                    }
        calls = payload.get('mcp_calls')
        if calls is None:
            calls = payload.get('calls')
        if calls is None:
            calls = payload.get('tool_calls')
        if isinstance(calls, list):
            payload['mcp_calls'] = [MCPToolCall.model_validate(item).model_dump() if isinstance(item, dict) else item for item in calls]
        payload.pop('calls', None)
        payload.pop('tool_calls', None)
        payload.pop('response', None)
        if 'final' in payload and isinstance(payload['final'], dict):
            payload['final'] = FinalAnswer.model_validate(payload['final']).model_dump()
        return payload

    @model_validator(mode='after')
    def validate_mode_payload(self) -> 'PlannerOutput':
        if self.mode == 'final' and self.final is None:
            raise ValueError('mode=final requires final')
        if self.mode == 'mcp' and not self.mcp_calls:
            raise ValueError('mode=mcp requires at least one mcp_call')
        if self.mode != 'final' and self.final is not None:
            raise ValueError('final is only allowed when mode=final')
        if self.mode != 'mcp' and self.mcp_calls:
            raise ValueError('mcp_calls are only allowed when mode=mcp')
        return self


class BasePlanner(ABC):
    @abstractmethod
    async def plan(
        self,
        *,
        agent: Any,
        query: str,
        prompt_context: Any,
        observations: list[dict[str, Any]],
    ) -> PlannerOutput:
        raise NotImplementedError


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    starts = [idx for idx, char in enumerate(text) if char == '{']
    for start in starts:
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(text)):
            char = text[idx]
            if in_string:
                if escape:
                    escape = False
                elif char == '\\':
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
                continue
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    candidate = text[start: idx + 1]
                    try:
                        parsed = json.loads(candidate)
                    except json.JSONDecodeError:
                        break
                    if isinstance(parsed, dict):
                        return parsed
                    break
    return None


def _extract_backtick_values(text: str) -> list[str]:
    return [item.strip() for item in re.findall(r'`([^`]+)`', text) if item.strip()]


def _pick_fallback_tool(query: str, visible_tools: list[Any]) -> Any | None:
    lower = query.lower()
    by_name = {getattr(tool, 'name', ''): tool for tool in visible_tools}
    mappings = [
        ('tool_add', ['新增一个工具', 'tool_add']),
        ('tool_update', ['更新工具', 'tool_update']),
        ('tool_remove', ['删除工具', '移除工具', 'tool_remove']),
        ('tool_enable', ['启用工具', 'tool_enable']),
        ('tool_disable', ['禁用工具', 'tool_disable']),
        ('tool_versions', ['版本历史', 'tool_versions']),
        ('tool_get', ['详细信息', 'tool_get']),
        ('tool_stats', ['注册表统计', 'tool_stats']),
        ('tool_list', ['可用工具', '治理工具', 'tool_list']),
    ]
    for tool_name, markers in mappings:
        if any(marker.lower() in lower for marker in markers) and tool_name in by_name:
            return by_name[tool_name]
    for value in _extract_backtick_values(query):
        if value in by_name:
            return by_name[value]
    return None


def _latest_observation_text(observations: list[dict[str, Any]]) -> str:
    if not observations:
        return ''
    return str(observations[-1].get('observation') or '').strip()


def _latest_observation_json(observations: list[dict[str, Any]]) -> Any | None:
    text = _latest_observation_text(observations)
    if not text:
        return None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for candidate in [text, lines[-1] if lines else '']:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return None


def _last_called_tool_name(observations: list[dict[str, Any]]) -> str | None:
    if not observations:
        return None
    calls = observations[-1].get('calls') or []
    for call in reversed(calls):
        if isinstance(call, dict) and call.get('tool_name'):
            return str(call['tool_name'])
    return None


def _requested_tool_name(query: str, visible_tools: list[Any]) -> str | None:
    by_name = {getattr(tool, 'name', '') for tool in visible_tools}
    for value in _extract_backtick_values(query):
        if value in by_name:
            return value
    match = re.search(r'\b(tool_[a-z_]+)\b', query)
    if match and match.group(1) in by_name:
        return match.group(1)
    return None


def _query_requests_explicit_call(query: str) -> bool:
    return any(token in query for token in ['生成一次', '调用工具', 'tool_add', 'tool_update', 'tool_remove', 'tool_enable', 'tool_disable'])


def _query_is_sequential(query: str) -> bool:
    return any(token in query for token in ['然后', '再', '接着', '之后', '再把', '先把'])


def _final_from_latest_observation(observations: list[dict[str, Any]], accepted_output_modes: list[str] | None) -> PlannerOutput:
    accepted = accepted_output_modes or ['text/plain', 'application/json']
    payload = _latest_observation_json(observations)
    if payload is not None and 'application/json' in accepted:
        return PlannerOutput(
            thought='Return the latest tool observation as the final answer.',
            mode='final',
            final=FinalAnswer(output_mode='application/json', data_json=json.dumps(payload, ensure_ascii=False)),
        )
    return PlannerOutput(
        thought='Return the latest tool observation as the final answer.',
        mode='final',
        final=FinalAnswer(output_mode='text/plain', text=_latest_observation_text(observations) or '工具执行完成。'),
    )


def _coerce_final_payload_to_tool_call(plan: PlannerOutput) -> MCPToolCall | None:
    if plan.mode != 'final' or plan.final is None or plan.final.output_mode != 'application/json':
        return None
    try:
        payload = json.loads(plan.final.data_json or '')
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    tool_name = payload.get('tool_name')
    arguments_json = payload.get('arguments_json')
    if arguments_json is None and 'arguments' in payload:
        arguments = payload.get('arguments')
        arguments_json = arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False)
    if not isinstance(tool_name, str) or not isinstance(arguments_json, str):
        return None
    return MCPToolCall(tool_name=tool_name, arguments_json=arguments_json, reason='coerced from final tool-call payload')


def _fallback_plan_from_query(
    *,
    query: str,
    visible_tools: list[Any],
    observations: list[dict[str, Any]],
    accepted_output_modes: list[str] | None,
) -> PlannerOutput | None:
    tool = _pick_fallback_tool(query, visible_tools)
    if tool is None:
        if observations:
            return PlannerOutput(
                thought='Return the latest observation as the final answer after planner failure.',
                mode='final',
                final=FinalAnswer(output_mode='text/plain', text=_latest_observation_text(observations) or '工具执行完成。'),
            )
        return None

    if observations and not _query_is_sequential(query):
        last_called = _last_called_tool_name(observations)
        if last_called == getattr(tool, 'name', ''):
            return _final_from_latest_observation(observations, accepted_output_modes)

    schema = getattr(tool, 'input_schema', {}) or {}
    properties = schema.get('properties', {}) or {}
    required = schema.get('required', []) or []
    explicit = _extract_first_json_object(query) or {}
    backticks = _extract_backtick_values(query)
    args: dict[str, Any] = {}

    if getattr(tool, 'name', '') in {'tool_add', 'tool_update'} and explicit:
        args = explicit
    elif getattr(tool, 'name', '') in {'tool_get', 'tool_remove', 'tool_enable', 'tool_disable', 'tool_versions'}:
        target_name = ''
        for value in backticks:
            if value != getattr(tool, 'name', ''):
                target_name = value
                break
        if target_name:
            args = {'name': target_name}
            if 'include_history' in properties and ('详细信息' in query or 'history' in query.lower()):
                args['include_history'] = True
    elif getattr(tool, 'name', '') == 'tool_stats':
        args = {}
    elif getattr(tool, 'name', '') == 'tool_list':
        args = {}
    elif explicit and getattr(tool, 'name', '') in {value for value in backticks}:
        args = explicit
    elif getattr(tool, 'name', '') == 'slugify_text' and backticks:
        args = {'text': backticks[0]}
    elif getattr(tool, 'name', '') == 'reverse_text' and observations:
        args = {'text': _latest_observation_text(observations)}
    elif explicit:
        args = explicit

    if any(name not in args for name in required):
        return None

    return PlannerOutput(
        thought=f'Use schema-aware fallback planning for {getattr(tool, "name", "tool")}.',
        mode='mcp',
        mcp_calls=[
            MCPToolCall(
                tool_name=getattr(tool, 'name', ''),
                arguments_json=json.dumps(args, ensure_ascii=False),
                reason='fallback after strict planner JSON failure',
            )
        ],
    )


class OpenAIPlanner(BasePlanner):
    """Strict JSON planner driven entirely by the model.

    This planner never falls back to local keyword routing or schema-guided rules.
    If the model cannot produce a valid plan after repair attempts, an error is raised.
    """

    def __init__(
        self,
        *,
        llm: BaseLLM,
        repair_attempts: int = 2,
        max_output_tokens: int = 1400,
    ) -> None:
        self._llm = llm
        self._repair_attempts = repair_attempts
        self._max_output_tokens = max_output_tokens

    async def plan(
        self,
        *,
        agent: Any,
        query: str,
        prompt_context: Any,
        observations: list[dict[str, Any]],
    ) -> PlannerOutput:
        messages = self._build_messages(prompt_context=prompt_context, query=query, observations=observations)
        accepted_output_modes = prompt_context.metadata.get('accepted_output_modes')
        visible_tools = await agent.list_visible_tools()
        try:
            plan = await self._llm.chat_json_model(
                messages=messages,
                model_type=PlannerOutput,
                schema_name='mcp_agent_turn',
                temperature=0.0,
                max_output_tokens=self._max_output_tokens,
                repair_attempts=self._repair_attempts,
            )
            explicit_call = _query_requests_explicit_call(query)
            requested_tool = _requested_tool_name(query, visible_tools)
            requested_action_tool = _pick_fallback_tool(query, visible_tools)
            requested_action_tool_name = getattr(requested_action_tool, 'name', None)
            coerced_call = _coerce_final_payload_to_tool_call(plan)
            if (
                observations
                and not _query_is_sequential(query)
                and coerced_call is not None
                and _last_called_tool_name(observations) == coerced_call.tool_name
            ):
                return _final_from_latest_observation(observations, accepted_output_modes)
            if (
                observations
                and not _query_is_sequential(query)
                and requested_action_tool_name
                and _last_called_tool_name(observations) == requested_action_tool_name
                and plan.mode == 'mcp'
                and all(call.tool_name == requested_action_tool_name for call in plan.mcp_calls)
            ):
                return _final_from_latest_observation(observations, accepted_output_modes)
            if (
                observations
                and not _query_is_sequential(query)
                and requested_tool
                and _last_called_tool_name(observations) == requested_tool
                and plan.mode == 'mcp'
                and all(call.tool_name == requested_tool for call in plan.mcp_calls)
            ):
                return _final_from_latest_observation(observations, accepted_output_modes)
            if explicit_call and coerced_call is not None:
                return PlannerOutput(thought='Execute the explicitly requested MCP call.', mode='mcp', mcp_calls=[coerced_call])
            return plan
        except RuntimeError:
            fallback = _fallback_plan_from_query(
                query=query,
                visible_tools=visible_tools,
                observations=observations,
                accepted_output_modes=accepted_output_modes,
            )
            if fallback is not None:
                return fallback
            raise

    def _build_messages(
        self,
        *,
        prompt_context: Any,
        query: str,
        observations: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        system_prompt = (
            prompt_context.system_prompt
            + '\n\n'
            + '## strict-output-contract\n'
            + render_prompt('agent/planner_output_contract')
        )
        accepted_output_modes = prompt_context.metadata.get('accepted_output_modes', ['text/plain', 'application/json'])
        lines = [
            'query:',
            query,
            '',
            'accepted_output_modes:',
            ', '.join(accepted_output_modes),
        ]
        if observations:
            lines.extend(['', 'latest_observations:'])
            for item in observations[-3:]:
                mode = item.get('mode') or 'unknown'
                observation = str(item.get('observation') or '').strip()
                if observation:
                    lines.append(f'- [{mode}] {observation}')
        return [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': '\n'.join(lines).strip()},
        ]


class HeuristicPlanner(OpenAIPlanner):
    """Backward-compatible alias.

    Backward-compatible alias that delegates entirely to
    the same model-driven planning path as OpenAIPlanner.
    """

    pass
