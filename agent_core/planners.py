from __future__ import annotations

import inspect
import json
import re
from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class FinalAnswer(BaseModel):
    model_config = ConfigDict(extra='forbid')

    output_mode: Literal['text/plain', 'application/json'] = 'text/plain'
    text: str | None = None
    data: Any | None = None

    @model_validator(mode='after')
    def validate_payload(self) -> 'FinalAnswer':
        if self.output_mode == 'application/json' and self.data is None:
            raise ValueError('application/json final output requires data')
        if self.output_mode == 'text/plain' and (self.text is None and self.data is None):
            raise ValueError('text/plain final output requires text or data')
        return self


class MCPToolCall(BaseModel):
    model_config = ConfigDict(extra='forbid')

    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    reason: str = Field(default='', description='brief operational reason for this MCP call')


class A2ADelegateCall(BaseModel):
    model_config = ConfigDict(extra='forbid')

    peer_name: str = Field(min_length=1)
    message: str = Field(min_length=1)
    accepted_output_modes: list[Literal['text/plain', 'application/json']] = Field(
        default_factory=lambda: ['text/plain', 'application/json']
    )
    reason: str = Field(default='', description='runtime-only A2A routing reason')


class PlannerOutput(BaseModel):
    model_config = ConfigDict(extra='forbid')

    thought: str = Field(default='', description='brief action summary, not hidden chain-of-thought')
    mode: Literal['final', 'mcp']
    final: FinalAnswer | None = None
    mcp_calls: list[MCPToolCall] = Field(default_factory=list)

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


_SEQUENTIAL_MARKERS = ('然后', '再', '接着', '之后', 'then', 'next', 'after that', 'and then')
_FREE_TEXT_PREFIXES = [
    '请', '帮我', '麻烦', '请你', '请先', '请再', '帮我把', '请把', '请用', '请查看', '请查询', '请列出',
    'please', 'could you', 'can you', 'list', 'show', 'search', 'find', 'call', 'invoke',
]
_BOOLEAN_TRUE = {'true', 'yes', 'on', 'enable', 'enabled', '是', '开启', '启用', '需要'}
_BOOLEAN_FALSE = {'false', 'no', 'off', 'disable', 'disabled', '否', '关闭', '禁用', '不要'}

_INTENT_HINTS = [
    (('列出', '列表', '可用'), 'list show available'),
    (('查询', '查看', '详情', '信息'), 'inspect info describe'),
    (('统计',), 'stats summary'),
    (('版本', '历史'), 'versions history'),
    (('新增', '添加', '创建'), 'add create'),
    (('更新', '升级', '修改'), 'update upgrade modify'),
    (('删除', '移除'), 'remove delete'),
    (('启用', '恢复'), 'enable'),
    (('禁用', '停用'), 'disable'),
    (('调用', '使用'), 'call invoke'),
    (('搜索', '查找'), 'search find'),
    (('工具', '注册表', '治理'), 'tool registry governance'),
    (('反转',), 'reverse'),
    (('转成', '转换'), 'convert transform'),
]


def _query_signal_text(query: str) -> str:
    lowered = query.lower()
    signals = [query]
    for tokens, english in _INTENT_HINTS:
        if any(token in lowered for token in tokens):
            signals.append(english)
    return ' '.join(signals)


_SAFE_OPTIONAL_PROPS = {
    'query', 'keyword', 'search', 'name', 'text', 'input', 'message', 'value', 'prompt', 'expression',
    'city', 'location', 'description', 'changelog', 'handler_mode', 'tags', 'aliases',
    'input_schema_json', 'metadata_json', 'code', 'module_path', 'include_history'
}


def _tokenize(text: str) -> list[str]:
    return re.findall(r'[a-z0-9_]+|[\u4e00-\u9fff]+', text.lower())


def _flatten_metadata(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return ' '.join(_flatten_metadata(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return ' '.join(_flatten_metadata(item) for item in value)
    return str(value)


def _tool_signal_text(tool: Any) -> str:
    return ' '.join(
        filter(
            None,
            [
                getattr(tool, 'name', ''),
                getattr(tool, 'description', ''),
                _flatten_metadata(getattr(tool, 'metadata', {})),
                ' '.join((getattr(tool, 'input_schema', {}) or {}).get('properties', {}).keys()),
            ],
        )
    )


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


def _extract_fenced_block(text: str, *, language: str) -> str | None:
    match = re.search(rf'```{language}\s*(.*?)```', text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _extract_quoted_segments(text: str) -> list[str]:
    segments: list[str] = []
    for pattern in [r'`([^`]+)`', r'"([^"]+)"', r'“([^”]+)”']:
        segments.extend(item.strip() for item in re.findall(pattern, text) if item.strip())
    deduped: list[str] = []
    seen: set[str] = set()
    for item in segments:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def _extract_identifier_candidates(text: str) -> list[str]:
    quoted = [item for item in _extract_quoted_segments(text) if re.fullmatch(r'[A-Za-z0-9_.-]+', item)]
    contextual = re.findall(
        r'(?:工具|tool)\s*[：:\s]*["“]?([A-Za-z0-9_.-]+)["”]?' \
        r'|(?:查看|查询|删除|移除|启用|禁用|更新|升级|调用|使用|inspect|describe|remove|delete|enable|disable|update|call|invoke)\s*["“`]?([A-Za-z0-9_.-]+)["”`]?',
        text,
        flags=re.IGNORECASE,
    )
    items = quoted[:]
    for pair in contextual:
        for value in pair:
            if value:
                items.append(value)
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def _clean_free_text(query: str) -> str:
    text = query.strip()
    for prefix in _FREE_TEXT_PREFIXES:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip(' ：:，,。')
            break
    explicit_json = _extract_first_json_object(text)
    if explicit_json is not None:
        try:
            serialized = json.dumps(explicit_json, ensure_ascii=False)
            text = text.replace(serialized, '').strip()
        except Exception:
            pass
    return text.strip(' ：:，,。')


def _parse_json_text(text: str) -> Any | None:
    candidate = text.strip()
    if not candidate:
        return None
    if not ((candidate.startswith('{') and candidate.endswith('}')) or (candidate.startswith('[') and candidate.endswith(']'))):
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _last_observation_bundle(observations: list[dict[str, Any]]) -> tuple[str, Any | None, list[str]]:
    if not observations:
        return '', None, []
    last = observations[-1]
    results = last.get('results') or []
    texts = [item.get('primary_text', '') for item in results if item.get('primary_text')]
    text = '\n'.join(texts).strip() or str(last.get('observation', '') or '')
    parsed = _parse_json_text(text)
    called_tools = [item.get('tool_name', '') for step in observations for item in (step.get('calls') or []) if item.get('tool_name')]
    return text, parsed, called_tools


def _final_from_observations(observations: list[dict[str, Any]], accepted_output_modes: list[str] | None) -> PlannerOutput:
    text, parsed, _ = _last_observation_bundle(observations)
    accepted = accepted_output_modes or ['text/plain', 'application/json']
    if parsed is not None and 'application/json' in accepted:
        return PlannerOutput(
            thought='Returning the latest structured observation as the final answer.',
            mode='final',
            final=FinalAnswer(output_mode='application/json', data=parsed),
        )
    return PlannerOutput(
        thought='Returning the latest observation as the final answer.',
        mode='final',
        final=FinalAnswer(output_mode='text/plain', text=text or '工具执行完成。'),
    )


def _score_tool(query: str, tool: Any) -> float:
    signal_query = _query_signal_text(query)
    query_lower = signal_query.lower()
    tool_name = getattr(tool, 'name', '').lower()
    tool_tokens = set(_tokenize(_tool_signal_text(tool)))
    query_tokens = set(_tokenize(signal_query))
    overlap = len(query_tokens & tool_tokens)
    score = float(overlap)
    metadata = getattr(tool, 'metadata', {}) or {}
    registry_surface = metadata.get('surface') == 'registry'

    management_tokens = {'inspect', 'info', 'describe', 'stats', 'summary', 'versions', 'history', 'list', 'search', 'add', 'create', 'update', 'upgrade', 'modify', 'remove', 'delete', 'enable', 'disable'}
    invocation_tokens = {'call', 'invoke'}
    explicit_identifiers = _extract_identifier_candidates(query)
    if tool_name and tool_name in query_lower:
        score += 100.0 if (query_tokens & invocation_tokens) else 15.0
    if tool_name in explicit_identifiers:
        score += 50.0 if (query_tokens & invocation_tokens) else 8.0
    if (query_tokens & management_tokens) and not registry_surface:
        score -= 20.0
    if registry_surface and ({'tool', 'registry', 'governance'} & query_tokens):
        score += 40.0
    if registry_surface and ({'inspect', 'info', 'describe', 'stats', 'summary', 'versions', 'history', 'list', 'search'} & query_tokens):
        score += 20.0
    if registry_surface and ({'inspect', 'info', 'describe'} & query_tokens) and ('info' in tool_name or 'get' in tool_name):
        score += 120.0
    if registry_surface and ({'versions', 'history'} & query_tokens) and 'version' in tool_name:
        score += 120.0
    if registry_surface and ({'stats', 'summary'} & query_tokens) and 'stat' in tool_name:
        score += 120.0
    if registry_surface and ({'list', 'search'} & query_tokens) and ('list' in tool_name or 'search' in tool_name):
        score += 120.0
    properties = (getattr(tool, 'input_schema', {}) or {}).get('properties', {})
    score += 0.25 * len(set(properties.keys()) & query_tokens)
    return score


def _infer_boolean(query: str) -> bool | None:
    tokens = set(_tokenize(query))
    if tokens & _BOOLEAN_TRUE:
        return True
    if tokens & _BOOLEAN_FALSE:
        return False
    return None


def _infer_property_value(
    *,
    property_name: str,
    property_schema: dict[str, Any],
    query: str,
    observations: list[dict[str, Any]],
    explicit_json: dict[str, Any],
    quoted: list[str],
    identifiers: list[str],
    cleaned_text: str,
) -> Any | None:
    if property_name in explicit_json:
        return explicit_json[property_name]
    if 'default' in property_schema:
        return property_schema.get('default')

    latest_text, latest_json, _ = _last_observation_bundle(observations)
    if isinstance(latest_json, dict) and property_name in latest_json:
        return latest_json[property_name]

    prop_type = property_schema.get('type')
    if prop_type == 'boolean':
        if property_name == 'include_history' and ({'versions', 'history'} & set(_tokenize(_query_signal_text(query)))):
            return True
        return _infer_boolean(query)

    if prop_type == 'array':
        if property_name in explicit_json and isinstance(explicit_json[property_name], list):
            return explicit_json[property_name]
        if property_name in {'tags', 'aliases'} and identifiers:
            return identifiers
        return None

    if prop_type == 'object':
        if property_name in explicit_json and isinstance(explicit_json[property_name], dict):
            return explicit_json[property_name]
        json_block = _extract_fenced_block(query, language='json')
        if json_block:
            try:
                return json.loads(json_block)
            except json.JSONDecodeError:
                return None
        return None

    if property_name == 'code':
        code_block = _extract_fenced_block(query, language='python') or _extract_fenced_block(query, language='py')
        if code_block:
            return code_block

    if property_name == 'handler_mode':
        if explicit_json.get('code') is not None or _extract_fenced_block(query, language='python') or _extract_fenced_block(query, language='py'):
            return 'python_inline'
        if explicit_json.get('module_path') is not None:
            return 'python_module'

    if property_name.endswith('_json'):
        json_block = _extract_fenced_block(query, language='json')
        if json_block:
            try:
                return json.loads(json_block)
            except json.JSONDecodeError:
                return None

    if property_name == 'name':
        return identifiers[0] if identifiers else None
    if property_name in {'source', 'target', 'alias', 'replaced_by', 'version'}:
        if property_name == 'source' and len(identifiers) >= 1:
            return identifiers[0]
        if property_name == 'target' and len(identifiers) >= 2:
            return identifiers[1]
        if property_name in {'alias', 'replaced_by'} and identifiers:
            return identifiers[-1]
        if property_name == 'version':
            match = re.search(r'\b\d+\.\d+\.\d+\b', query)
            return match.group(0) if match else None
        return None
    if property_name in {'query', 'keyword', 'search'}:
        if quoted:
            return quoted[-1]
        return None
    if property_name in {'text', 'input', 'message', 'value', 'prompt', 'expression', 'city', 'location'}:
        if latest_text:
            return latest_text
        if quoted:
            return quoted[-1]
        return cleaned_text or query
    if property_name in {'description', 'changelog'}:
        if quoted:
            return quoted[-1]
        return cleaned_text or query

    if prop_type == 'string':
        if latest_text:
            return latest_text
        if quoted:
            return quoted[-1]
        if identifiers:
            return identifiers[-1]
        return cleaned_text or query

    return None


def _extract_arguments_for_tool(tool: Any, query: str, observations: list[dict[str, Any]]) -> dict[str, Any] | None:
    schema = getattr(tool, 'input_schema', {}) or {}
    properties = schema.get('properties', {}) or {}
    required = schema.get('required', []) or []

    explicit_json = _extract_first_json_object(query) or {}
    quoted = _extract_quoted_segments(query)
    identifiers = _extract_identifier_candidates(query)
    cleaned_text = _clean_free_text(query)

    arguments: dict[str, Any] = {}
    for name in properties:
        if name in explicit_json:
            arguments[name] = explicit_json[name]

    for name, prop_schema in properties.items():
        if name in arguments:
            continue
        if name not in required and name not in _SAFE_OPTIONAL_PROPS and 'default' not in prop_schema:
            continue
        value = _infer_property_value(
            property_name=name,
            property_schema=prop_schema,
            query=query,
            observations=observations,
            explicit_json=explicit_json,
            quoted=quoted,
            identifiers=identifiers,
            cleaned_text=cleaned_text,
        )
        if value is not None:
            arguments[name] = value

    if any(name not in arguments for name in required):
        return None
    return arguments


def _build_schema_guided_plan(
    *,
    query: str,
    visible_tools: list[Any],
    observations: list[dict[str, Any]],
    accepted_output_modes: list[str] | None,
) -> PlannerOutput | None:
    if not visible_tools:
        if observations:
            return _final_from_observations(observations, accepted_output_modes)
        return None

    sequential = any(marker in query.lower() for marker in _SEQUENTIAL_MARKERS)
    _, _, called_tools = _last_observation_bundle(observations)
    ranked = sorted(
        ((tool, _score_tool(query, tool)) for tool in visible_tools),
        key=lambda item: item[1],
        reverse=True,
    )
    ranked = [(tool, score) for tool, score in ranked if score > 0]

    if observations and not sequential:
        return _final_from_observations(observations, accepted_output_modes)

    for tool, score in ranked:
        if observations and tool.name in called_tools:
            continue
        arguments = _extract_arguments_for_tool(tool, query, observations)
        if arguments is None:
            continue
        return PlannerOutput(
            thought=f'Schema-guided routing selected {tool.name} (score={score:.2f}).',
            mode='mcp',
            mcp_calls=[MCPToolCall(tool_name=tool.name, arguments=arguments, reason='schema-guided fallback from the live MCP catalog')],
        )

    if observations:
        return _final_from_observations(observations, accepted_output_modes)
    return None


class HeuristicPlanner(BasePlanner):
    """Generic schema-guided fallback planner for local tests and offline runs."""

    async def plan(
        self,
        *,
        agent: Any,
        query: str,
        prompt_context: Any,
        observations: list[dict[str, Any]],
    ) -> PlannerOutput:
        normalized = query.strip()
        lower = normalized.lower()

        if any(token in lower for token in ['list skills', 'show skills', '有哪些skill', '哪些skill', 'skills', '技能列表']):
            skills = ', '.join(skill['name'] for skill in agent.list_skills()) or '(none loaded)'
            return PlannerOutput(
                thought='Returning the explicitly loaded skill catalog.',
                mode='final',
                final=FinalAnswer(output_mode='text/plain', text='已加载 skills：' + skills),
            )

        visible_tools = await agent.list_visible_tools()
        guided = _build_schema_guided_plan(
            query=normalized,
            visible_tools=visible_tools,
            observations=observations,
            accepted_output_modes=prompt_context.metadata.get('accepted_output_modes'),
        )
        if guided is not None:
            return guided

        return PlannerOutput(
            thought='No safe tool plan was inferred; returning a direct answer.',
            mode='final',
            final=FinalAnswer(output_mode='text/plain', text=f'{agent.config.name} 收到：{normalized}'),
        )


class OpenAIPlanner(BasePlanner):
    def __init__(
        self,
        *,
        api_base: str,
        api_key: str,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 1200,
        connect_timeout_seconds: float = 20.0,
        request_timeout_seconds: float = 90.0,
        max_retries: int = 0,
        repair_attempts: int = 2,
        client: Any | None = None,
    ) -> None:
        if client is None:
            try:
                import httpx
                from openai import AsyncOpenAI
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("OpenAIPlanner requires the 'openai' package") from exc
            client = AsyncOpenAI(
                base_url=api_base,
                api_key=api_key,
                timeout=httpx.Timeout(timeout=request_timeout_seconds, connect=connect_timeout_seconds),
                max_retries=max_retries,
            )
        self._client = client
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._request_timeout_seconds = request_timeout_seconds
        self._repair_attempts = repair_attempts

    async def plan(
        self,
        *,
        agent: Any,
        query: str,
        prompt_context: Any,
        observations: list[dict[str, Any]],
    ) -> PlannerOutput:
        schema = PlannerOutput.model_json_schema()
        messages = self._build_messages(prompt_context=prompt_context, query=query, observations=observations, schema=schema)
        last_error: Exception | None = None
        last_raw = '{}'

        for attempt in range(self._repair_attempts + 1):
            raw = await self._complete(messages, schema)
            last_raw = raw
            try:
                return PlannerOutput.model_validate(json.loads(raw))
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
                if attempt >= self._repair_attempts:
                    break
                messages = messages + [
                    {
                        'role': 'user',
                        'content': (
                            'Your previous reply violated the schema or was not valid JSON. '
                            'Return exactly one corrected JSON object and nothing else.\n'
                            f'validation_error={exc}\n'
                            f'previous_reply={raw}'
                        ),
                    }
                ]

        visible_tools = await agent.list_visible_tools()
        guided = _build_schema_guided_plan(
            query=query,
            visible_tools=visible_tools,
            observations=observations,
            accepted_output_modes=prompt_context.metadata.get('accepted_output_modes'),
        )
        if guided is not None:
            return guided

        if observations:
            return _final_from_observations(observations, prompt_context.metadata.get('accepted_output_modes'))

        error_text = f'planner returned invalid JSON after retries: {last_error}; raw={last_raw}' if last_error else last_raw
        return PlannerOutput(
            thought='Planner repair fallback produced a direct final answer.',
            mode='final',
            final=FinalAnswer(output_mode='text/plain', text=error_text),
        )

    def _build_messages(
        self,
        *,
        prompt_context: Any,
        query: str,
        observations: list[dict[str, Any]],
        schema: dict[str, Any],
    ) -> list[dict[str, str]]:
        system_prompt = (
            prompt_context.system_prompt
            + '\n\n'
            + '## strict-output-contract\n'
            + 'Return exactly one JSON object matching the schema below. '
            + 'Do not emit markdown, code fences, explanations, or prose outside JSON. '
            + 'The `thought` field must be a short action summary, not hidden chain-of-thought.\n'
            + json.dumps(schema, ensure_ascii=False, indent=2)
        )
        messages = [
            {'role': 'system', 'content': system_prompt},
            {
                'role': 'user',
                'content': json.dumps(
                    {
                        'query': query,
                        'accepted_output_modes': prompt_context.metadata.get('accepted_output_modes', ['text/plain', 'application/json']),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            },
        ]
        if observations:
            messages.append({'role': 'user', 'content': 'Execution observations:\n' + json.dumps(observations, ensure_ascii=False, indent=2)})
        return messages

    async def _complete(self, messages: list[dict[str, str]], schema: dict[str, Any]) -> str:
        response_format = {
            'type': 'json_schema',
            'json_schema': {
                'name': 'mcp_agent_turn',
                'strict': True,
                'schema': schema,
            },
        }
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                response_format=response_format,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                timeout=self._request_timeout_seconds,
            )
            if inspect.isawaitable(resp):
                resp = await resp
        except Exception:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                response_format={'type': 'json_object'},
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                timeout=self._request_timeout_seconds,
            )
            if inspect.isawaitable(resp):
                resp = await resp
        return resp.choices[0].message.content or '{}'
