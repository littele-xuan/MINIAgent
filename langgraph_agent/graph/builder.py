from __future__ import annotations

import json
import uuid
from typing import Any, Literal

from ..context.base import BaseContextAssembler
from ..memory.base import BaseMemoryManager
from ..schemas import AgentDecision, FinalResponse
from ..skills.base import BaseSkillManager
from ..tools import RegistryGovernanceToolset, ToolArgumentValidator, ToolRegistry
from ..tools.base import BaseToolProvider
from ..tools.composite import CompositeToolExecutor
from ..utils import compact_text, json_safe
from .heuristics import HeuristicPlanner
from .state import AgentGraphState


class GraphAgentRunner:
    """LangGraph orchestration shell.

    The graph owns only serializable state. Runtime resources (LLM clients,
    Langfuse callbacks, MCP sessions, tool handlers, stores) live on this runner
    or are passed through LangGraph config.
    """

    def __init__(
        self,
        *,
        agent: Any,
        llm: Any,
        tool_provider: BaseToolProvider,
        memory_manager: BaseMemoryManager,
        context_assembler: BaseContextAssembler,
        skill_manager: BaseSkillManager,
        delegation_manager: Any,
        checkpointer: Any | None,
        store: Any | None,
    ) -> None:
        self.agent = agent
        self.llm = llm
        self.tool_provider = tool_provider
        self.memory_manager = memory_manager
        self.context_assembler = context_assembler
        self.skill_manager = skill_manager
        self.delegation_manager = delegation_manager
        self.checkpointer = checkpointer
        self.store = store
        self.registry = ToolRegistry()
        self.governance = RegistryGovernanceToolset(self.registry)
        self.validator = ToolArgumentValidator()
        self._executors: dict[str, CompositeToolExecutor] = {}
        self.heuristic_planner = HeuristicPlanner()
        self.graph = self._build_graph()

    def _build_graph(self):
        from langgraph.graph import END, START, StateGraph

        builder = StateGraph(AgentGraphState)
        builder.add_node('ingest_input', self.ingest_input)
        builder.add_node('memory_read', self.memory_read)
        builder.add_node('skill_select', self.skill_select)
        builder.add_node('tool_catalog', self.tool_catalog)
        builder.add_node('context_build', self.context_build)
        builder.add_node('plan', self.plan)
        builder.add_node('execute_mcp_tools', self.execute_mcp_tools)
        builder.add_node('memory_writeback', self.memory_writeback)
        builder.add_node('delegate', self.delegate)
        builder.add_node('finalize', self.finalize)

        builder.add_edge(START, 'ingest_input')
        builder.add_edge('ingest_input', 'memory_read')
        builder.add_edge('memory_read', 'skill_select')
        builder.add_edge('skill_select', 'tool_catalog')
        builder.add_edge('tool_catalog', 'context_build')
        builder.add_edge('context_build', 'plan')
        builder.add_conditional_edges('plan', self.route_after_plan, {
            'mcp': 'execute_mcp_tools',
            'delegate': 'delegate',
            'final': 'finalize',
        })
        builder.add_edge('execute_mcp_tools', 'memory_writeback')
        builder.add_edge('memory_writeback', 'tool_catalog')
        builder.add_edge('delegate', 'finalize')
        builder.add_edge('finalize', END)

        compile_kwargs: dict[str, Any] = {}
        if self.checkpointer is not None:
            compile_kwargs['checkpointer'] = self.checkpointer
        if self.store is not None:
            compile_kwargs['store'] = self.store
        return builder.compile(**compile_kwargs)

    async def ingest_input(self, state: AgentGraphState) -> dict[str, Any]:
        query = state['query']
        thread_id = state.get('thread_id') or str(uuid.uuid4())
        user_id = state.get('user_id') or self.agent.config.memory.user_id
        run_id = state.get('run_id') or str(uuid.uuid4())
        existing_history = list(state.get('history', []) or [])
        existing_history.append({'role': 'user', 'content': query})
        await self.memory_manager.record_user_turn(user_id=user_id, thread_id=thread_id, text=query)
        trace = list(state.get('trace', []) or [])
        trace.append({'mode': 'ingest', 'thought': 'Initialized serializable LangGraph state.', 'observation': query, 'payload': {'run_id': run_id}})
        return {
            'run_id': run_id,
            'thread_id': thread_id,
            'user_id': user_id,
            'step_count': 0,
            'history': json_safe(existing_history),
            'trace': json_safe(trace),
            'errors': [],
            'task_state': json_safe(state.get('task_state') or {}),
            'plan': None,
            'pending_mcp_calls': [],
            'final_response': None,
            'tool_results': [],
            'delegated_result': None,
            'memory_context': {},
            'visible_tools': [],
            'context_packet': {},
        }

    async def memory_read(self, state: AgentGraphState) -> dict[str, Any]:
        memory_context = await self.memory_manager.load_context(user_id=state['user_id'], thread_id=state['thread_id'], query=state['query'])
        trace = list(state.get('trace', []) or [])
        trace.append({
            'mode': 'memory_read',
            'thought': 'Retrieved long-term/task/episodic memory.',
            'observation': f"{len(memory_context.get('memories', []))} memories",
            'payload': json_safe(memory_context),
        })
        return {'memory_context': json_safe(memory_context), 'trace': trace}

    async def skill_select(self, state: AgentGraphState) -> dict[str, Any]:
        selected = await self.skill_manager.select(state['query'])
        trace = list(state.get('trace', []) or [])
        if selected is not None:
            trace.append({
                'mode': 'skill',
                'thought': 'Selected skill for current request.',
                'observation': selected.get('name', ''),
                'payload': json_safe(selected),
            })
        return {'selected_skill': json_safe(selected) if selected else None, 'trace': trace}

    def _local_tools(self, selected_skill: dict[str, Any] | None) -> dict[str, Any]:
        local_tools = {tool.name: tool for tool in self.skill_manager.local_tools_for(selected_skill)}
        local_tools.update(self.governance.tools())
        return local_tools

    async def tool_catalog(self, state: AgentGraphState) -> dict[str, Any]:
        local_tools = self._local_tools(state.get('selected_skill'))
        executor = CompositeToolExecutor(self.tool_provider, local_tools=local_tools)
        all_tools = await executor.list_tools()
        self.registry.refresh(all_tools)
        visible = self.registry.visible_dicts()
        visible = self.skill_manager.filter_tool_catalog(state.get('selected_skill'), visible)
        self._executors[state['thread_id']] = executor
        trace = list(state.get('trace', []) or [])
        trace.append({'mode': 'tool_catalog', 'thought': 'Refreshed MCP tool catalog.', 'observation': f'{len(visible)} visible tools', 'payload': {'tool_names': [t.get('name') for t in visible]}})
        return {'visible_tools': json_safe(visible), 'trace': trace}

    async def context_build(self, state: AgentGraphState) -> dict[str, Any]:
        packet = self.context_assembler.build_packet(
            agent=self.agent,
            state=state,
            query=state['query'],
            visible_tools=state.get('visible_tools', []),
            memory_context=state.get('memory_context'),
            selected_skill=state.get('selected_skill'),
        )
        return {'context_packet': packet.model_dump(mode='json')}

    async def plan(self, state: AgentGraphState, config: Any | None = None) -> dict[str, Any]:
        step = int(state.get('step_count', 0)) + 1
        trace = list(state.get('trace', []) or [])
        if step > int(state.get('max_steps', self.agent.config.max_steps)):
            final = FinalResponse(output_mode='text/plain', text='已达到最大执行步数，停止循环。')
            plan = AgentDecision(mode='final', thought='Reached max steps.', final=final)
            trace.append({'mode': 'plan', 'thought': plan.thought, 'observation': 'final:max_steps', 'payload': plan.model_dump(mode='json')})
            return {'step_count': step, 'plan': plan.model_dump(mode='json'), 'trace': json_safe(trace)}

        messages = self.context_assembler.build_messages(
            agent=self.agent,
            state=state,
            query=state['query'],
            visible_tools=state.get('visible_tools', []),
            memory_context=state.get('memory_context'),
            selected_skill=state.get('selected_skill'),
        )
        try:
            plan = await self.llm.ainvoke_json_model(
                messages,
                model_type=AgentDecision,
                schema_name='langgraph_agent_mcp_decision',
                max_output_tokens=1600,
                invoke_config=self.agent.monitor.runnable_config(
                    run_name=f"{self.agent.config.name}:llm:planner",
                    user_id=state.get('user_id'),
                    session_id=state.get('thread_id'),
                    tags=['llm', 'planner'],
                    metadata={'run_id': state.get('run_id'), 'step': step},
                ),
            )
        except Exception as exc:
            plan = self.heuristic_planner.build_plan(
                query=state['query'],
                visible_tools=state.get('visible_tools', []),
                memory_context=state.get('memory_context'),
                tool_results=state.get('tool_results', []),
            )
            trace.append({
                'mode': 'plan-fallback',
                'thought': 'Structured LLM planning failed; deterministic fallback used for debuggability.',
                'observation': compact_text(str(exc), limit=1000),
                'payload': {'error': compact_text(str(exc), limit=2000)},
            })
        trace.append({'mode': 'plan', 'thought': plan.thought, 'observation': plan.mode, 'payload': plan.model_dump(mode='json')})
        return {
            'step_count': step,
            'plan': plan.model_dump(mode='json'),
            'pending_mcp_calls': [call.model_dump(mode='json') for call in plan.mcp_calls] if plan.mode == 'mcp' else [],
            'trace': json_safe(trace),
        }

    def route_after_plan(self, state: AgentGraphState) -> Literal['mcp', 'delegate', 'final']:
        plan = AgentDecision.model_validate(state.get('plan') or {})
        if plan.mode == 'mcp':
            return 'mcp'
        if plan.mode == 'delegate':
            return 'delegate'
        return 'final'

    async def execute_mcp_tools(self, state: AgentGraphState) -> dict[str, Any]:
        plan = AgentDecision.model_validate(state.get('plan') or {})
        executor = self._executors.get(state['thread_id'])
        if executor is None:
            executor = CompositeToolExecutor(self.tool_provider, local_tools=self._local_tools(state.get('selected_skill')))
            self._executors[state['thread_id']] = executor

        visible_by_name = {tool.get('name'): tool for tool in state.get('visible_tools', [])}
        visible_names = {name for name in visible_by_name if name}
        calls: list[tuple[str, dict[str, Any]]] = []
        immediate_errors: list[dict[str, Any]] = []
        for item in plan.mcp_calls:
            try:
                descriptor = visible_by_name.get(item.tool_name, {})
                risk = descriptor.get('risk')
                if risk in set(getattr(self.agent.config.tools, 'require_approval_for_risks', ()) or ()): 
                    raise RuntimeError(f'Tool {item.tool_name} has risk={risk} and requires explicit human approval.')
                if not self.registry.policy.is_allowed(item.tool_name, visible_names=visible_names, risk=risk):
                    raise RuntimeError(f'Tool not visible or disabled under current policy: {item.tool_name}')
                self.validator.validate(tool_name=item.tool_name, input_schema=descriptor.get('input_schema') or {}, arguments=item.arguments)
                calls.append((item.tool_name, item.arguments))
            except Exception as exc:
                immediate_errors.append({'tool_name': item.tool_name, 'arguments': json_safe(item.arguments), 'text': '', 'payload': None, 'status': 'error', 'error': str(exc)})

        results = immediate_errors + await executor.execute_batch(calls)
        history = list(state.get('history', []) or [])
        trace = list(state.get('trace', []) or [])
        accumulated_results = list(state.get('tool_results', []) or [])
        for result in results:
            text = result.get('text') or result.get('error') or ''
            history.append({'role': 'tool', 'name': result.get('tool_name'), 'content': text})
            trace.append({
                'mode': 'tool',
                'thought': f"Executed MCP tool {result.get('tool_name')}",
                'observation': compact_text(text, limit=1200),
                'payload': json_safe({'protocol': 'mcp', **result}),
            })
            accumulated_results.append(json_safe(result))
            await self.memory_manager.record_tool_result(
                user_id=state['user_id'],
                thread_id=state['thread_id'],
                tool_name=str(result.get('tool_name') or ''),
                content=text,
            )
        return {'tool_results': json_safe(accumulated_results), 'history': json_safe(history), 'trace': json_safe(trace)}

    async def memory_writeback(self, state: AgentGraphState) -> dict[str, Any]:
        await self.memory_manager.consolidate(user_id=state['user_id'], thread_id=state['thread_id'])
        return {}

    async def delegate(self, state: AgentGraphState) -> dict[str, Any]:
        plan = AgentDecision.model_validate(state.get('plan') or {})
        assert plan.delegate is not None
        result = await self.delegation_manager.delegate(
            peer_name=plan.delegate.peer_name,
            message=plan.delegate.message,
            accepted_output_modes=state.get('accepted_output_modes') or ['text/plain', 'application/json'],
        )
        answer = result.get('primary_text') or result.get('text') or json.dumps(result.get('payload'), ensure_ascii=False, default=str)
        history = list(state.get('history', []) or [])
        history.append({'role': 'assistant', 'content': answer})
        trace = list(state.get('trace', []) or [])
        trace.append({'mode': 'delegate', 'thought': plan.thought, 'observation': compact_text(answer, limit=1200), 'payload': json_safe(result)})
        return {
            'delegated_result': json_safe(result),
            'history': json_safe(history),
            'trace': json_safe(trace),
            'final_response': {
                'output_mode': result.get('output_mode', 'text/plain'),
                'text': answer,
                'data': result.get('payload') if result.get('output_mode') == 'application/json' else None,
            },
        }

    async def finalize(self, state: AgentGraphState) -> dict[str, Any]:
        if state.get('final_response') is not None:
            final = FinalResponse.model_validate(state['final_response'])
        else:
            plan = AgentDecision.model_validate(state.get('plan') or {})
            if plan.mode == 'final' and plan.final is not None:
                final = plan.final
            elif plan.mode == 'ask_human':
                final = FinalResponse(output_mode='text/plain', text=plan.question or '需要更多信息。')
            elif state.get('tool_results'):
                latest = state['tool_results'][-1]
                final = FinalResponse(output_mode='text/plain', text=latest.get('text') or latest.get('error') or '')
            else:
                final = FinalResponse(output_mode='text/plain', text='')
        answer = final.text if final.output_mode == 'text/plain' else json.dumps(final.payload, ensure_ascii=False, default=str)
        await self.memory_manager.record_assistant_turn(user_id=state['user_id'], thread_id=state['thread_id'], text=answer)
        history = list(state.get('history', []) or [])
        history.append({'role': 'assistant', 'content': answer})
        trace = list(state.get('trace', []) or [])
        trace.append({'mode': 'final', 'thought': 'Completed LangGraph MCP run.', 'observation': compact_text(answer, limit=1200), 'payload': final.model_dump(mode='json')})
        return {
            'history': json_safe(history),
            'final_response': final.model_dump(mode='json'),
            'trace': json_safe(trace),
        }
