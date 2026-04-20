from __future__ import annotations

import asyncio
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


class HeuristicPlanner(BasePlanner):
    """Deterministic fallback kept for tests and local debugging only."""

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
        visible_tools = await agent.list_visible_tools()
        visible_names = {tool.name for tool in visible_tools}

        if observations:
            last = observations[-1]
            if last.get('mode') == 'mcp':
                batch_results = last.get('results') or []
                text = '\n'.join(item.get('primary_text', '') for item in batch_results if item.get('primary_text')).strip()
                return PlannerOutput(
                    thought='MCP observations are available; returning the consolidated result.',
                    mode='final',
                    final=FinalAnswer(output_mode='text/plain', text=text or '工具执行完成。'),
                )

        if any(token in lower for token in ['list tools', 'show tools', '有哪些工具', '哪些工具', '可用工具', 'tools']):
            names = ', '.join(tool.name for tool in visible_tools) or '(none)'
            return PlannerOutput(
                thought='Returning the live MCP catalog directly.',
                mode='final',
                final=FinalAnswer(output_mode='text/plain', text='可用 MCP 工具：' + names),
            )

        if any(token in lower for token in ['list skills', 'show skills', '有哪些skill', '哪些skill', 'skills', '技能列表']):
            skills = ', '.join(skill['name'] for skill in agent.list_skills()) or '(none)'
            return PlannerOutput(
                thought='Returning the discovered Anthropic-style skill catalog directly.',
                mode='final',
                final=FinalAnswer(output_mode='text/plain', text='可用 skills：' + skills),
            )

        if 'slugify' in lower and 'slugify_text' in visible_names:
            text = self._extract_quoted_or_tail_text(normalized)
            return PlannerOutput(
                thought='Routing to the live MCP slugify tool.',
                mode='mcp',
                mcp_calls=[MCPToolCall(tool_name='slugify_text', arguments={'text': text}, reason='normalize text into a slug')],
            )
        if any(token in lower for token in ['reverse', '反转', '倒序']) and 'reverse_text' in visible_names:
            text = self._extract_quoted_or_tail_text(normalized)
            return PlannerOutput(
                thought='Routing to the live MCP reverse tool.',
                mode='mcp',
                mcp_calls=[MCPToolCall(tool_name='reverse_text', arguments={'text': text}, reason='reverse the provided text')],
            )

        expr = self._extract_math_expression(normalized)
        if expr and 'calculator' in visible_names:
            return PlannerOutput(
                thought='Routing to the live MCP calculator tool.',
                mode='mcp',
                mcp_calls=[MCPToolCall(tool_name='calculator', arguments={'expression': expr}, reason='evaluate the math expression')],
            )

        if any(token in lower for token in ['天气', 'weather', 'forecast']) and 'get_weather' in visible_names:
            city = self._extract_city(normalized)
            return PlannerOutput(
                thought='Routing to the live MCP weather tool.',
                mode='mcp',
                mcp_calls=[MCPToolCall(tool_name='get_weather', arguments={'city': city}, reason='retrieve the requested weather report')],
            )

        for tool in visible_tools:
            keywords = tool.metadata.get('keywords', []) if tool.metadata else []
            if any(str(keyword).lower() in lower for keyword in keywords):
                sample_args = {}
                props = (tool.input_schema or {}).get('properties', {})
                if 'text' in props:
                    sample_args['text'] = normalized
                return PlannerOutput(
                    thought=f'Routing to the live MCP tool {tool.name}.',
                    mode='mcp',
                    mcp_calls=[MCPToolCall(tool_name=tool.name, arguments=sample_args, reason='metadata keyword match')],
                )

        skill = agent.active_skill
        if skill is not None:
            return PlannerOutput(
                thought='No MCP execution is needed; answering inside the active skill contract.',
                mode='final',
                final=FinalAnswer(output_mode='text/plain', text=f'[{skill.name}] {normalized}'),
            )

        return PlannerOutput(
            thought='No MCP execution is needed; returning a direct answer.',
            mode='final',
            final=FinalAnswer(output_mode='text/plain', text=f'{agent.config.name} 收到：{normalized}'),
        )

    def _extract_math_expression(self, text: str) -> str | None:
        normalized = text.replace('×', '*').replace('÷', '/')
        matches = re.findall(r'[0-9\s\+\-\*/\(\)\.%]{3,}', normalized)
        if matches:
            expr = max(matches, key=len).strip()
            if re.fullmatch(r'[0-9\s\+\-\*/\(\)\.%]+', expr):
                return expr
        return None

    def _extract_city(self, text: str) -> str:
        match = re.search(r'(?:weather|天气|forecast)\s*(?:for|in)?\s*([A-Za-z\u4e00-\u9fff\- ]{2,})', text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip(' .，。')
        return text.strip()

    def _extract_quoted_or_tail_text(self, text: str) -> str:
        quoted = re.findall(r'[`"“](.*?)[`"”]', text)
        if quoted:
            return quoted[-1].strip()
        for marker in ('reverse', 'slugify', '反转', '倒序'):
            idx = text.lower().find(marker)
            if idx >= 0:
                return text[idx + len(marker):].strip(' ：:') or text
        return text


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
        raw = await self._complete(messages, schema)
        try:
            return PlannerOutput.model_validate(json.loads(raw))
        except (json.JSONDecodeError, ValidationError) as exc:
            repair_messages = messages + [
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
            repaired = await self._complete(repair_messages, schema)
            try:
                return PlannerOutput.model_validate(json.loads(repaired))
            except (json.JSONDecodeError, ValidationError):
                return PlannerOutput(
                    thought=f'planner parse fallback: {exc}',
                    mode='final',
                    final=FinalAnswer(output_mode='text/plain', text=raw),
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
