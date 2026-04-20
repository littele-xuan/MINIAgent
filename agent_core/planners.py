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
    """Deterministic fallback kept for tests and local debugging only.

    It includes a lightweight governance parser so that tool-registry CRUD can
    still flow through the same "user query -> planner JSON -> MCP tool call"
    path instead of bypassing the agent with direct runtime calls.
    """

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

        follow_up = self._plan_followup_from_observations(
            query=normalized,
            lower=lower,
            observations=observations,
            visible_names=visible_names,
        )
        if follow_up is not None:
            return follow_up

        governance = self._plan_governance_query(
            query=normalized,
            lower=lower,
            visible_tools=visible_tools,
            visible_names=visible_names,
        )
        if governance is not None:
            return governance

        if any(token in lower for token in ['list skills', 'show skills', '有哪些skill', '哪些skill', 'skills', '技能列表']):
            skills = ', '.join(skill['name'] for skill in agent.list_skills()) or '(none)'
            return PlannerOutput(
                thought='Returning the discovered Anthropic-style skill catalog directly.',
                mode='final',
                final=FinalAnswer(output_mode='text/plain', text='可用 skills：' + skills),
            )

        if ('slugify' in lower or 'slug' in lower or '转成 slug' in lower) and 'slugify_text' in visible_names:
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

    def _plan_followup_from_observations(
        self,
        *,
        query: str,
        lower: str,
        observations: list[dict[str, Any]],
        visible_names: set[str],
    ) -> PlannerOutput | None:
        if not observations:
            return None
        last = observations[-1]
        if last.get('mode') != 'mcp':
            return None

        last_calls = last.get('calls') or []
        last_results = last.get('results') or []
        last_tool_name = last_calls[-1].get('tool_name') if last_calls else ''
        last_text = '\n'.join(item.get('primary_text', '') for item in last_results if item.get('primary_text')).strip()

        wants_slug_then_reverse = (
            ('slugify' in lower or 'slug' in lower or '转成 slug' in query.lower())
            and any(token in lower for token in ['reverse', '反转', '倒序'])
        )
        if wants_slug_then_reverse and last_tool_name == 'slugify_text' and 'reverse_text' in visible_names:
            return PlannerOutput(
                thought='Continuing the sequential text-transform workflow with reverse_text.',
                mode='mcp',
                mcp_calls=[
                    MCPToolCall(
                        tool_name='reverse_text',
                        arguments={'text': last_text},
                        reason='reverse the slugified text from the previous step',
                    )
                ],
            )

        return PlannerOutput(
            thought='MCP observations are available; returning the consolidated result.',
            mode='final',
            final=FinalAnswer(output_mode='text/plain', text=last_text or '工具执行完成。'),
        )

    def _plan_governance_query(
        self,
        *,
        query: str,
        lower: str,
        visible_tools: list[Any],
        visible_names: set[str],
    ) -> PlannerOutput | None:
        explicit_call = self._plan_explicit_tool_invocation(
            query=query,
            lower=lower,
            visible_tools=visible_tools,
            visible_names=visible_names,
        )
        if explicit_call is not None:
            return explicit_call

        registryish = any(
            token in lower
            for token in [
                'tool', 'tools', 'registry', 'mcp', '工具', '注册表', '治理', 'governance', 'schema', '版本', 'alias', '合并',
                '启用', '禁用', '删除', '新增', '添加', '更新', '升级',
            ]
        )
        if not registryish:
            return None

        list_tool_name = self._pick_visible(visible_names, 'tool_search', 'tool_list')
        info_tool_name = self._pick_visible(visible_names, 'tool_info', 'tool_get')
        stats_tool_name = self._pick_visible(visible_names, 'registry_stats', 'tool_stats')

        if stats_tool_name and any(token in lower for token in ['统计', 'stats', '概况', 'overview']) and any(token in lower for token in ['registry', '注册表', '工具', 'tool']):
            return PlannerOutput(
                thought='Inspecting registry statistics through the governance surface.',
                mode='mcp',
                mcp_calls=[MCPToolCall(tool_name=stats_tool_name, arguments={}, reason='summarize registry status')],
            )

        if 'tool_versions' in visible_names and any(token in lower for token in ['版本历史', 'versions', '历史版本']):
            name = self._extract_tool_name(query)
            if name:
                return PlannerOutput(
                    thought='Inspecting archived versions for the requested tool.',
                    mode='mcp',
                    mcp_calls=[MCPToolCall(tool_name='tool_versions', arguments={'name': name}, reason='retrieve version history for the requested tool')],
                )

        if 'tool_add' in visible_names and ((any(token in lower for token in ['新增工具', '添加工具', '创建工具', 'add tool', 'create tool'])) or ('新增' in lower and '工具' in lower) or ('添加' in lower and '工具' in lower)):
            payload = self._extract_governance_payload(query, operation='add')
            if payload:
                return PlannerOutput(
                    thought='Translating the natural-language add request into a tool_add JSON call.',
                    mode='mcp',
                    mcp_calls=[MCPToolCall(tool_name='tool_add', arguments=payload, reason='add a new external tool from the user request')],
                )

        if 'tool_update' in visible_names and ((any(token in lower for token in ['更新工具', '修改工具', '升级工具', 'update tool', 'upgrade tool'])) or ('更新' in lower and '工具' in lower) or ('升级' in lower and '工具' in lower) or ('修改' in lower and '工具' in lower)):
            payload = self._extract_governance_payload(query, operation='update')
            if payload:
                return PlannerOutput(
                    thought='Translating the natural-language update request into a tool_update JSON call.',
                    mode='mcp',
                    mcp_calls=[MCPToolCall(tool_name='tool_update', arguments=payload, reason='update an existing external tool from the user request')],
                )

        if 'tool_enable' in visible_names and ((any(token in lower for token in ['启用工具', 'enable tool', '重新启用', '恢复工具'])) or ('启用' in lower and '工具' in lower) or ('恢复' in lower and '工具' in lower)):
            name = self._extract_tool_name(query)
            if name:
                return PlannerOutput(
                    thought='Enabling the requested tool through governance tools.',
                    mode='mcp',
                    mcp_calls=[MCPToolCall(tool_name='tool_enable', arguments={'name': name}, reason='re-enable an existing external tool')],
                )

        if 'tool_disable' in visible_names and ((any(token in lower for token in ['禁用工具', 'disable tool', '停用工具'])) or ('禁用' in lower and '工具' in lower) or ('停用' in lower and '工具' in lower)):
            name = self._extract_tool_name(query)
            if name:
                return PlannerOutput(
                    thought='Disabling the requested tool through governance tools.',
                    mode='mcp',
                    mcp_calls=[MCPToolCall(tool_name='tool_disable', arguments={'name': name}, reason='disable an existing external tool')],
                )

        if 'tool_remove' in visible_names and ((any(token in lower for token in ['删除工具', '移除工具', 'remove tool', 'delete tool'])) or ('删除' in lower and '工具' in lower) or ('移除' in lower and '工具' in lower)):
            name = self._extract_tool_name(query)
            if name:
                return PlannerOutput(
                    thought='Removing the requested tool through governance tools.',
                    mode='mcp',
                    mcp_calls=[MCPToolCall(tool_name='tool_remove', arguments={'name': name}, reason='remove an external tool from the registry')],
                )

        if 'tool_deprecate' in visible_names and any(token in lower for token in ['废弃工具', '弃用工具', 'deprecate tool']):
            name = self._extract_tool_name(query)
            if name:
                args = {'name': name}
                replaced_by = self._extract_replaced_by(query)
                if replaced_by:
                    args['replaced_by'] = replaced_by
                return PlannerOutput(
                    thought='Deprecating the requested tool through governance tools.',
                    mode='mcp',
                    mcp_calls=[MCPToolCall(tool_name='tool_deprecate', arguments=args, reason='deprecate an external tool and optionally point to a replacement')],
                )

        if 'tool_alias' in visible_names and any(token in lower for token in ['别名', 'alias']):
            parsed = self._extract_alias_pair(query)
            if parsed is not None:
                name, alias = parsed
                return PlannerOutput(
                    thought='Adding the requested alias through governance tools.',
                    mode='mcp',
                    mcp_calls=[MCPToolCall(tool_name='tool_alias', arguments={'name': name, 'alias': alias}, reason='create an alias for an external tool')],
                )

        if 'tool_merge' in visible_names and any(token in lower for token in ['合并工具', 'merge tool', 'merge']):
            parsed = self._extract_merge_pair(query)
            if parsed is not None:
                source, target = parsed
                keep_source = any(token in lower for token in ['保留源工具', 'keep source', '保留原工具'])
                return PlannerOutput(
                    thought='Merging the requested tools through governance tools.',
                    mode='mcp',
                    mcp_calls=[MCPToolCall(tool_name='tool_merge', arguments={'source': source, 'target': target, 'keep_source': keep_source}, reason='merge one external tool into another')],
                )

        if info_tool_name and any(token in lower for token in ['详情', '信息', 'schema', '入参', '参数', 'tool info', 'describe']):
            name = self._extract_tool_name(query)
            if name:
                return PlannerOutput(
                    thought='Inspecting one tool through the governance surface.',
                    mode='mcp',
                    mcp_calls=[MCPToolCall(tool_name=info_tool_name, arguments={'name': name}, reason='inspect one tool from the registry')],
                )

        if list_tool_name and ((any(token in lower for token in ['查找工具', '搜索工具', 'search tool', 'find tool', '列出工具', '工具列表', '可用工具', '治理工具', 'list tools', 'show tools']) or (any(token in lower for token in ['列出', '列表', '可用', 'show', 'list']) and any(token in lower for token in ['工具', 'tool', 'registry', '治理'])))):
            tool_query = self._extract_search_query(query)
            args = {'query': tool_query} if tool_query else {}
            return PlannerOutput(
                thought='Inspecting the live registry catalog through governance tools.',
                mode='mcp',
                mcp_calls=[MCPToolCall(tool_name=list_tool_name, arguments=args, reason='list or search registry tools from the user request')],
            )

        return None

    def _plan_explicit_tool_invocation(
        self,
        *,
        query: str,
        lower: str,
        visible_tools: list[Any],
        visible_names: set[str],
    ) -> PlannerOutput | None:
        if not any(token in lower for token in ['调用工具', 'call tool', '使用工具', 'invoke tool']):
            return None
        tool_name = self._extract_explicit_tool_name(query)
        if not tool_name or tool_name not in visible_names:
            return None
        args = self._extract_first_json_object(query) or {}
        return PlannerOutput(
            thought=f'Calling the explicitly requested tool {tool_name}.',
            mode='mcp',
            mcp_calls=[MCPToolCall(tool_name=tool_name, arguments=args, reason='explicit user-directed tool invocation')],
        )

    def _pick_visible(self, visible_names: set[str], *candidates: str) -> str | None:
        for name in candidates:
            if name in visible_names:
                return name
        return None

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

    def _extract_explicit_tool_name(self, text: str) -> str | None:
        patterns = [
            r'(?:调用工具|使用工具|invoke tool|call tool)\s*[：:\s]*[`"“]?([A-Za-z0-9_.-]+)[`"”]?',
            r'[`"“]([A-Za-z0-9_.-]+)[`"”]\s*(?:工具|tool)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return match.group(1)
        return self._extract_tool_name(text)

    def _extract_tool_name(self, text: str) -> str | None:
        backticked = re.findall(r'`([A-Za-z0-9_.-]+)`', text)
        if backticked:
            return backticked[0]
        contextual_patterns = [
            r'(?:工具|tool)\s*[：:\s]*["“]?([A-Za-z0-9_.-]+)["”]?',
            r'(?:禁用|启用|删除|移除|废弃|弃用|更新|升级|查看|查询|inspect|describe|enable|disable|remove|delete|update|upgrade)\s*["“]?([A-Za-z0-9_.-]+)["”]?',
            r'["“]([A-Za-z0-9_.-]+)["”]?\s*(?:工具|tool)',
        ]
        for pattern in contextual_patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return match.group(1)
        if '{' in text:
            return None
        quoted = re.findall(r'["“]([A-Za-z0-9_.-]+)["”]', text)
        if quoted:
            return quoted[0]
        return None

    def _extract_replaced_by(self, text: str) -> str | None:
        patterns = [
            r'(?:replaced_by|替代为|替换为|改用)\s*[：:\s]*[`"“]?([A-Za-z0-9_.-]+)[`"”]?',
            r'(?:use|改用)\s*[`"“]?([A-Za-z0-9_.-]+)[`"”]?\s*(?:instead)?',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def _extract_alias_pair(self, text: str) -> tuple[str, str] | None:
        patterns = [
            r'给\s*[`"“]?([A-Za-z0-9_.-]+)[`"”]?\s*添加别名\s*[`"“]?([A-Za-z0-9_.-]+)[`"”]?',
            r'alias\s*[`"“]?([A-Za-z0-9_.-]+)[`"”]?\s*(?:for|to)\s*[`"“]?([A-Za-z0-9_.-]+)[`"”]?',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                first, second = match.group(1), match.group(2)
                if 'alias' in pattern:
                    return second, first
                return first, second
        return None

    def _extract_merge_pair(self, text: str) -> tuple[str, str] | None:
        patterns = [
            r'把\s*[`"“]?([A-Za-z0-9_.-]+)[`"”]?\s*合并到\s*[`"“]?([A-Za-z0-9_.-]+)[`"”]?',
            r'merge\s*[`"“]?([A-Za-z0-9_.-]+)[`"”]?\s*into\s*[`"“]?([A-Za-z0-9_.-]+)[`"”]?',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return match.group(1), match.group(2)
        return None

    def _extract_search_query(self, text: str) -> str:
        quoted = re.findall(r'[`"“](.*?)[`"”]', text)
        if quoted:
            return quoted[-1].strip()
        for marker in ('查找工具', '搜索工具', 'search tool', 'find tool'):
            idx = text.lower().find(marker)
            if idx >= 0:
                return text[idx + len(marker):].strip(' ：:，,。')
        return ''

    def _extract_governance_payload(self, text: str, *, operation: str) -> dict[str, Any] | None:
        explicit = self._extract_first_json_object(text)
        if explicit is not None:
            return explicit

        payload: dict[str, Any] = {}
        name = self._extract_tool_name(text)
        if name:
            payload['name'] = name

        desc_match = re.search(r'(?:描述|description)\s*(?:为|是|:|：)?\s*([^\n。]+)', text, flags=re.IGNORECASE)
        if desc_match and operation == 'add':
            payload['description'] = desc_match.group(1).strip()

        version_match = re.search(r'(?:版本|version)\s*(?:为|是|:|：)?\s*([A-Za-z0-9_.-]+)', text, flags=re.IGNORECASE)
        if version_match:
            payload['version'] = version_match.group(1).strip()

        code_block = self._extract_fenced_block(text, language='python') or self._extract_fenced_block(text, language='py')
        if code_block:
            payload['code'] = code_block

        schema_block = self._extract_fenced_block(text, language='json')
        if schema_block:
            try:
                payload['input_schema_json'] = json.loads(schema_block)
            except json.JSONDecodeError:
                pass

        if operation == 'add' and {'name', 'description'} <= set(payload):
            return payload
        if operation == 'update' and 'name' in payload:
            return payload
        return None

    def _extract_fenced_block(self, text: str, *, language: str) -> str | None:
        pattern = rf'```{language}\s*(.*?)```'
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def _extract_first_json_object(self, text: str) -> dict[str, Any] | None:
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
