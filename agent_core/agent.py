from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .context import BaseContextManager, LayeredContextManager
from .planners import BasePlanner, FinalAnswer, HeuristicPlanner, OpenAIPlanner, PlannerOutput
from .tool_runtime import BaseToolRuntime, MCPClientToolRuntime, ToolDescriptor
from a2a_runtime import A2AClient, AgentCardBuilder, PeerAgent, SendMessageConfiguration, SendMessageRequest, build_a2a_app
from skill_engine import AnthropicSkillLoader, SkillBundle, SkillSelector, SkillToolRegistrar


_REGISTRY_MUTATION_TOOLS = {
    'tool_add',
    'tool_update',
    'tool_remove',
    'tool_enable',
    'tool_disable',
    'tool_deprecate',
    'tool_alias',
    'tool_merge',
}


class AgentDecision(BaseModel):
    model_config = ConfigDict(extra='forbid')

    thought: str = ''
    mode: str
    final: dict[str, Any] | None = None
    mcp_calls: list[dict[str, Any]] = Field(default_factory=list)
    a2a: dict[str, Any] | None = None


class AgentRunResult(BaseModel):
    model_config = ConfigDict(extra='forbid')

    answer: str
    output_mode: str = 'text/plain'
    payload: Any | None = None
    selected_skill: str | None = None
    trace: list[dict[str, Any]] = Field(default_factory=list)


@dataclass(slots=True)
class AgentConfig:
    name: str = 'agent'
    description: str = 'API-first MCP agent with strict JSON planning and optional runtime A2A routing.'
    role: str = 'general-purpose'
    verbose: bool = False
    skills_root: str | None = None
    skill_tool_policy: str = 'advisory'  # advisory | restrictive
    auto_load_skills: bool = False
    auto_activate_skills: bool = False
    planner: str = 'api'  # api | openai | heuristic
    api_base: str = field(default_factory=lambda: os.getenv('MCP_API_BASE')  or 'https://api.openai.com/v1')
    api_key: str = field(default_factory=lambda: os.getenv('MCP_API_KEY') or '')
    model: str = field(default_factory=lambda: os.getenv('MCP_MODEL')  or '')
    temperature: float = 0.1
    max_steps: int = 6
    connect_timeout_seconds: float = 20.0
    planner_timeout_seconds: float = 90.0
    tool_timeout_seconds: float = 90.0
    observation_preview_chars: int = 600
    a2a_routing_threshold: float = 3.0


class Agent:
    def __init__(
        self,
        config: AgentConfig,
        *,
        tool_runtime: BaseToolRuntime | None = None,
        planner: BasePlanner | None = None,
        context_manager: BaseContextManager | None = None,
        scheduler: Any | None = None,
        subagent_manager: Any | None = None,
        memory_manager: Any | None = None,
        a2a_client: A2AClient | None = None,
    ) -> None:
        self.config = config
        self.tool_runtime = tool_runtime
        self.context_manager = context_manager or LayeredContextManager()
        self.scheduler = scheduler
        self.subagent_manager = subagent_manager
        self.memory_manager = memory_manager
        self.skill_loader = AnthropicSkillLoader(config.skills_root) if config.skills_root else None
        self.skill_selector = SkillSelector()
        self.skill_registrar = SkillToolRegistrar()
        self.skill_catalog: list[SkillBundle] = []
        self.active_skill: SkillBundle | None = None
        self.delegation_skill: SkillBundle | None = None
        self._tools: list[ToolDescriptor] = []
        self.a2a_enabled = False
        self.a2a_client = a2a_client or A2AClient()
        self.peers: dict[str, PeerAgent] = {}
        self.card_builder: AgentCardBuilder | None = None

        if planner is not None:
            self.planner = planner
        elif config.planner in {'api', 'openai'}:
            if not (config.api_base and config.api_key and config.model):
                raise ValueError('API planner requires api_base/api_key/model. Provide env vars or pass a custom planner.')
            self.planner = OpenAIPlanner(
                api_base=config.api_base,
                api_key=config.api_key,
                model=config.model,
                temperature=config.temperature,
                connect_timeout_seconds=config.connect_timeout_seconds,
                request_timeout_seconds=config.planner_timeout_seconds,
            )
        else:
            self.planner = HeuristicPlanner()

    async def attach_tool_runtime(self, tool_runtime: BaseToolRuntime) -> None:
        self.tool_runtime = tool_runtime
        await self.refresh_tools()
        if self.config.auto_load_skills:
            await self.load_skills()

    async def connect(self, server_script: str, *, env: dict[str, str] | None = None) -> None:
        runtime = MCPClientToolRuntime()
        server_path = Path(server_script).resolve()
        if not server_path.exists():
            raise FileNotFoundError(f'MCP server script not found: {server_path}')

        merged_env = dict(env or {})
        if self.config.skills_root and 'AGENT_SKILLS_ROOT' not in merged_env and 'MCP_SKILLS_ROOT' not in merged_env:
            merged_env['AGENT_SKILLS_ROOT'] = str(Path(self.config.skills_root).resolve())

        self._log(f'connecting MCP stdio runtime: {server_path}')
        await runtime.connect(str(server_path), env=merged_env)
        self._log('MCP session initialized; refreshing live tool catalog')
        await self.attach_tool_runtime(runtime)
        self._log(f'MCP ready with {len(self._tools)} live tools')

    async def disconnect(self) -> None:
        if self.tool_runtime is not None:
            self._log('closing MCP runtime')
            await self.tool_runtime.close()
        await self.a2a_client.close()

    async def refresh_tools(self) -> list[ToolDescriptor]:
        if self.tool_runtime is None:
            self._tools = []
            return []
        self._tools = await asyncio.wait_for(self.tool_runtime.list_tools(), timeout=self.config.tool_timeout_seconds)
        return list(self._tools)

    async def discover_skills(self) -> list[SkillBundle]:
        if self.skill_loader is None:
            return []
        return self.skill_loader.discover_bundles()

    async def load_skills(self, skill_names: list[str] | None = None) -> list[SkillBundle]:
        bundles = await self.discover_skills()
        if not bundles:
            self.skill_catalog = []
            return []
        selected_names = set(skill_names or [])
        selected = [bundle for bundle in bundles if not selected_names or bundle.name in selected_names]

        existing = {bundle.name for bundle in self.skill_catalog}
        for bundle in selected:
            if bundle.name in existing:
                continue
            self.skill_catalog.append(bundle)
            if self.tool_runtime is not None and self.tool_runtime.supports_dynamic_registration():
                for entry in self.skill_registrar.build_tool_entries(bundle):
                    try:
                        self.tool_runtime.register_tool_entry(entry)  # type: ignore[attr-defined]
                    except Exception:
                        continue
        if self.tool_runtime is not None and self.tool_runtime.supports_dynamic_registration() and selected:
            await self.refresh_tools()
        self._log(f'loaded {len(selected)} skills explicitly; total loaded={len(self.skill_catalog)}')
        return list(self.skill_catalog)

    def clear_loaded_skills(self) -> None:
        self.skill_catalog = []
        self.active_skill = None
        self.delegation_skill = None

    def list_skills(self) -> list[dict[str, Any]]:
        return [bundle.catalog_entry() for bundle in self.skill_catalog]

    def activate_skill(self, query: str, *, skill_name: str | None = None, skill_arguments: str = '') -> SkillBundle | None:
        self.active_skill = None
        self.delegation_skill = None
        if not self.skill_catalog:
            return None
        if not skill_name:
            return None

        selected = None
        for bundle in self.skill_catalog:
            if bundle.name == skill_name:
                selected = bundle.clone_for_arguments(skill_arguments)
                break

        if selected is None:
            return None
        if self._is_a2a_skill(selected):
            self.delegation_skill = selected
            self._log(f'runtime A2A skill selected explicitly: {selected.name}')
        else:
            self.active_skill = selected
            self._log(f'active prompt skill selected explicitly: {selected.name}')
        return selected

    async def list_visible_tools(self) -> list[ToolDescriptor]:
        tools = list(self._tools)
        if self.config.skill_tool_policy != 'restrictive' or self.active_skill is None:
            return tools
        allowed = set(self.active_skill.allowed_tools) | {tool.name for tool in self.active_skill.local_tools}
        if not allowed:
            return tools
        return [tool for tool in tools if tool.name in allowed]

    async def relay_to_peer(
        self,
        peer_name: str,
        text: str,
        *,
        accepted_output_modes: list[str] | None = None,
    ) -> dict[str, Any]:
        peer = self.peers.get(peer_name)
        if peer is None:
            raise KeyError(f'Unknown peer: {peer_name}')
        configuration = SendMessageConfiguration(acceptedOutputModes=accepted_output_modes or ['text/plain', 'application/json'])
        response = await asyncio.wait_for(
            self.a2a_client.send_text(peer.agent_card_url, text, configuration=configuration),
            timeout=self.config.tool_timeout_seconds,
        )
        if response.task and response.task.artifacts:
            first = response.task.artifacts[0].parts[0]
            return self._part_to_payload(first)
        if response.message and response.message.parts:
            first = response.message.parts[0]
            return self._part_to_payload(first)
        return {'output_mode': 'text/plain', 'text': '', 'primary_text': ''}

    async def _execute_plan(self, plan: PlannerOutput) -> tuple[bool, dict[str, Any] | None, dict[str, Any]]:
        if plan.mode == 'final':
            final_payload = self._normalize_final(plan.final or FinalAnswer(output_mode='text/plain', text=''))
            return True, final_payload, {
                'mode': 'final',
                'observation': final_payload.get('answer', ''),
                'output_mode': final_payload.get('output_mode', 'text/plain'),
            }

        if self.tool_runtime is None:
            raise RuntimeError('No MCP tool runtime attached')
        visible_names = {tool.name for tool in await self.list_visible_tools()}
        batch = []
        for call in plan.mcp_calls:
            if call.tool_name not in visible_names:
                raise RuntimeError(f'MCP tool not visible for current policy: {call.tool_name}')
            batch.append((call.tool_name, call.arguments or {}))
        results = await asyncio.wait_for(
            self.tool_runtime.call_tools_batch(batch),
            timeout=self.config.tool_timeout_seconds,
        )
        catalog_refreshed = False
        if any(call.tool_name in _REGISTRY_MUTATION_TOOLS for call in plan.mcp_calls):
            await self.refresh_tools()
            catalog_refreshed = True
        return False, None, {
            'mode': 'mcp',
            'calls': [
                {
                    'tool_name': result.tool_name,
                    'arguments': result.arguments,
                }
                for result in results
            ],
            'results': [result.to_observation() for result in results],
            'catalog_refreshed': catalog_refreshed,
            'observation': '\n\n'.join(result.primary_text for result in results if result.primary_text),
        }

    async def run(self, query: str, *, max_steps: int | None = None, skill_name: str | None = None, skill_arguments: str = '') -> str:
        result = await self.run_detailed(query, max_steps=max_steps, skill_name=skill_name, skill_arguments=skill_arguments)
        return result.answer

    async def run_detailed(
        self,
        query: str,
        *,
        max_steps: int | None = None,
        skill_name: str | None = None,
        skill_arguments: str = '',
        accepted_output_modes: list[str] | None = None,
    ) -> AgentRunResult:
        selected_skill = self.activate_skill(query, skill_name=skill_name, skill_arguments=skill_arguments)
        self._log(f'run start | query={query}')

        delegated = await self._maybe_run_a2a(
            query,
            accepted_output_modes=accepted_output_modes,
            explicit_skill_name=skill_name,
            selected_skill=selected_skill,
        )
        if delegated is not None:
            return delegated

        observations: list[dict[str, Any]] = []
        max_steps = max_steps or self.config.max_steps

        for step in range(1, max_steps + 1):
            visible_tools = await self.list_visible_tools()
            prompt_context = self.context_manager.build_prompt_context(
                agent=self,
                query=query,
                active_skill=self.active_skill,
                visible_tools=visible_tools,
                observations=observations,
                accepted_output_modes=accepted_output_modes,
            )
            self._log(f'step {step}/{max_steps} | visible_tools={len(visible_tools)} | accepted_output_modes={prompt_context.metadata.get("accepted_output_modes")}')
            plan = await asyncio.wait_for(
                self.planner.plan(agent=self, query=query, prompt_context=prompt_context, observations=observations),
                timeout=self.config.planner_timeout_seconds,
            )
            self._log(f'step {step} plan | mode={plan.mode} | thought={plan.thought}')
            if plan.mode == 'mcp':
                for call in plan.mcp_calls:
                    self._log(f'step {step} MCP call | {call.tool_name}({json.dumps(call.arguments, ensure_ascii=False)})')

            done, final_payload, event = await self._execute_plan(plan)
            self._log(f'step {step} observation | {self._preview(event.get("observation", ""))}')
            observations.append({'thought': plan.thought, **event})

            if done:
                coerced = self._coerce_output(final_payload or {'output_mode': 'text/plain', 'answer': ''}, accepted_output_modes)
                self._log(f'run complete | output_mode={coerced["output_mode"]} | answer={self._preview(coerced["answer"])}')
                return AgentRunResult(
                    answer=coerced['answer'],
                    output_mode=coerced['output_mode'],
                    payload=coerced.get('payload'),
                    selected_skill=self.active_skill.name if self.active_skill else None,
                    trace=observations,
                )

        fallback = {'output_mode': 'text/plain', 'answer': '已达到最大执行步数。', 'payload': None}
        coerced = self._coerce_output(fallback, accepted_output_modes)
        self._log('run stopped at max_steps')
        return AgentRunResult(
            answer=coerced['answer'],
            output_mode=coerced['output_mode'],
            payload=coerced.get('payload'),
            selected_skill=self.active_skill.name if self.active_skill else None,
            trace=observations,
        )

    async def handle_a2a_request(self, request: SendMessageRequest):
        part = request.message.parts[0]
        text = part.text if part.text is not None else json.dumps(part.data, ensure_ascii=False)
        accepted_output_modes = list(request.configuration.accepted_output_modes) if request.configuration and request.configuration.accepted_output_modes else None
        result = await self.run_detailed(text, accepted_output_modes=accepted_output_modes)
        return result

    def enable_a2a(self, *, base_url: str, peers: list[PeerAgent] | None = None, provider_org: str = 'local', provider_url: str = 'https://localhost', documentation_url: str | None = None) -> None:
        self.a2a_enabled = True
        self.peers = {peer.name: peer for peer in (peers or [])}
        self.card_builder = AgentCardBuilder(
            agent_name=self.config.name,
            description=self.config.description,
            provider_org=provider_org,
            provider_url=provider_url,
            agent_version='1.0.0',
            documentation_url=documentation_url,
            skills_root=self.config.skills_root,
        )
        self._a2a_base_url = base_url.rstrip('/')
        self._log(f'A2A runtime routing enabled with {len(self.peers)} peers')

    def disable_a2a(self) -> None:
        self.a2a_enabled = False
        self.peers = {}
        self.card_builder = None

    def add_peer(self, peer: PeerAgent) -> None:
        self.peers[peer.name] = peer

    def build_a2a_app(self, *, base_url: str | None = None):
        if not self.a2a_enabled:
            raise RuntimeError('A2A is not enabled for this agent')
        if self.card_builder is None:
            raise RuntimeError('A2A card builder is not configured')
        return build_a2a_app(agent=self, card_builder=self.card_builder, base_url=(base_url or getattr(self, '_a2a_base_url', None) or 'http://localhost'))

    async def _maybe_run_a2a(
        self,
        query: str,
        *,
        accepted_output_modes: list[str] | None,
        explicit_skill_name: str | None,
        selected_skill: SkillBundle | None,
    ) -> AgentRunResult | None:
        if not self.a2a_enabled or not self.peers:
            return None
        if explicit_skill_name and self.delegation_skill is None:
            return None

        peer, reason = self._select_peer_for_query(query, force=self.delegation_skill is not None)
        if peer is None:
            if self.delegation_skill is not None:
                raise RuntimeError('A2A routing skill was selected, but no peer matched the current request.')
            return None

        self._log(f'runtime A2A route | peer={peer.name} | reason={reason}')
        result = await self.relay_to_peer(peer.name, query, accepted_output_modes=accepted_output_modes)
        payload = self._normalize_external_payload(result)
        coerced = self._coerce_output(payload, accepted_output_modes)
        trace = [{
            'thought': 'runtime A2A routing executed before MCP planning',
            'mode': 'a2a',
            'delegate': {
                'peer_name': peer.name,
                'reason': reason,
                'message': query,
                'accepted_output_modes': accepted_output_modes or ['text/plain', 'application/json'],
            },
            'result': result,
            'observation': result.get('primary_text') or result.get('text') or coerced['answer'],
        }]
        routed_skill = selected_skill.name if selected_skill is not None else self._default_a2a_skill_name()
        return AgentRunResult(
            answer=coerced['answer'],
            output_mode=coerced['output_mode'],
            payload=coerced.get('payload'),
            selected_skill=routed_skill,
            trace=trace,
        )

    def _select_peer_for_query(self, query: str, *, force: bool) -> tuple[PeerAgent | None, str]:
        lower = query.lower()
        terms = self._routing_terms(query)
        best_peer = None
        best_score = 0.0
        best_reason = ''
        math_like = self._looks_like_math(query)
        registry_like = any(token in lower for token in ['registry', 'tool', 'tools', 'skill', 'skills', 'mcp', '工具', '技能'])

        for peer in self.peers.values():
            score = 0.0
            reasons: list[str] = []
            name_lower = peer.name.lower()
            desc_lower = peer.description.lower()
            tags_lower = [tag.lower() for tag in peer.tags]

            if name_lower in lower:
                score += 6.0
                reasons.append('peer name matched the request')
            for tag in tags_lower:
                if tag and tag in lower:
                    score += 3.0
                    reasons.append(f'tag matched: {tag}')
            for term in terms:
                if term in desc_lower:
                    score += 0.5
            if math_like and {'math', 'calculator'} & set(tags_lower):
                score += 4.0
                reasons.append('math-shaped request matched peer tags')
            if registry_like and {'registry', 'tools', 'skills', 'mcp'} & set(tags_lower):
                score += 4.0
                reasons.append('registry-shaped request matched peer tags')

            if score > best_score:
                best_score = score
                best_peer = peer
                best_reason = '; '.join(dict.fromkeys(reasons)) or 'best peer score'

        if best_peer is None:
            return None, ''
        if not force and best_score < self.config.a2a_routing_threshold:
            return None, ''
        return best_peer, best_reason or 'forced by runtime A2A skill'

    def _routing_terms(self, text: str) -> set[str]:
        return set(re.findall(r'[a-zA-Z0-9_-]+|[一-鿿]{2,}', text.lower()))

    def _looks_like_math(self, text: str) -> bool:
        return bool(re.search(r'[0-9][0-9\s\+\-\*/\(\)\.%]{1,}', text)) or any(
            token in text.lower() for token in ['compute', 'calculate', 'math', '算一下', '计算']
        )

    def _is_a2a_skill(self, bundle: SkillBundle) -> bool:
        return bool((bundle.frontmatter.a2a or {}).get('enabled'))

    def _default_a2a_skill_name(self) -> str | None:
        for bundle in self.skill_catalog:
            if self._is_a2a_skill(bundle):
                return bundle.name
        return None

    def _normalize_final(self, final: FinalAnswer) -> dict[str, Any]:
        if final.output_mode == 'application/json':
            payload = final.data
            return {
                'output_mode': 'application/json',
                'payload': payload,
                'answer': json.dumps(payload, ensure_ascii=False, default=str),
            }
        if final.text is not None:
            return {
                'output_mode': 'text/plain',
                'payload': final.data,
                'answer': final.text,
            }
        payload = final.data
        return {
            'output_mode': 'text/plain',
            'payload': payload,
            'answer': json.dumps(payload, ensure_ascii=False, default=str),
        }

    def _normalize_external_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        output_mode = payload.get('output_mode', 'text/plain')
        if output_mode == 'application/json':
            data = payload.get('payload')
            return {
                'output_mode': 'application/json',
                'payload': data,
                'answer': json.dumps(data, ensure_ascii=False, default=str),
            }
        answer = payload.get('primary_text') or payload.get('text') or json.dumps(payload.get('payload'), ensure_ascii=False, default=str)
        return {
            'output_mode': 'text/plain',
            'payload': payload.get('payload'),
            'answer': answer,
        }

    def _coerce_output(self, payload: dict[str, Any], accepted_output_modes: list[str] | None) -> dict[str, Any]:
        accepted = accepted_output_modes or ['text/plain', 'application/json']
        if payload['output_mode'] in accepted:
            return payload
        return {
            'output_mode': 'text/plain',
            'payload': payload.get('payload'),
            'answer': payload['answer'],
        }

    def _part_to_payload(self, part: Any) -> dict[str, Any]:
        media_type = getattr(part, 'media_type', None) or getattr(part, 'mediaType', None) or 'text/plain'
        if media_type == 'application/json' and getattr(part, 'data', None) is not None:
            text = json.dumps(part.data, ensure_ascii=False, default=str)
            return {
                'output_mode': 'application/json',
                'payload': part.data,
                'text': text,
                'primary_text': text,
            }
        text = getattr(part, 'text', None)
        if text is None and getattr(part, 'data', None) is not None:
            text = json.dumps(part.data, ensure_ascii=False, default=str)
        return {
            'output_mode': 'text/plain',
            'payload': getattr(part, 'data', None),
            'text': text or '',
            'primary_text': text or '',
        }

    def _log(self, message: str) -> None:
        if self.config.verbose:
            print(f'[{self.config.name}] {message}', flush=True)

    def _preview(self, value: Any) -> str:
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
        if len(text) <= self.config.observation_preview_chars:
            return text
        return text[: self.config.observation_preview_chars] + '...'

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.disconnect()
