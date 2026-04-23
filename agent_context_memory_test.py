import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path

from agent_core import Agent, AgentConfig
from agent_core.tool_runtime import LocalRegistryToolRuntime
from mcp_lib.registry.models import ToolCategory
from mcp_lib.registry.registry import ToolRegistry
from mcp_lib.tools.base import tool_def


LOCAL_RUNTIME_ROOT = (Path(__file__).resolve().parent / '.agent_context_memory_lab').resolve()


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')


def _build_registry() -> ToolRegistry:
    registry = ToolRegistry()

    registry.register(
        tool_def(
            name='project_brief_lookup',
            description='Read the authoritative project brief for a known project slug and return the linked design_doc_id.',
            category=ToolCategory.INTERNAL_UTILITY,
            properties={
                'project_slug': {
                    'type': 'string',
                    'description': 'Known project slug. Use aurora for this test.',
                }
            },
            required=['project_slug'],
            handler=lambda project_slug: {
                'project': 'Aurora',
                'project_slug': project_slug,
                'design_doc_id': 'aurora-order-v2',
                'service': 'order-service',
                'task_scope': 'Migrate the Aurora order service to the new platform stack.',
            },
            metadata={'surface': 'context-memory-test'},
        )
    )

    registry.register(
        tool_def(
            name='design_doc_lookup',
            description='Read the authoritative design document for a known design_doc_id and return task-critical constraints.',
            category=ToolCategory.INTERNAL_UTILITY,
            properties={
                'doc_id': {
                    'type': 'string',
                    'description': 'Design document id returned by project_brief_lookup.',
                }
            },
            required=['doc_id'],
            handler=lambda doc_id: {
                'doc_id': doc_id,
                'project': 'Aurora',
                'tech_stack': ['FastAPI', 'PostgreSQL'],
                'order_status_enum': ['PENDING', 'PAID', 'CANCELED'],
                'constraints': ['Do not use Redis', 'Keep the enum exactly as documented'],
                'notes': 'This document is the authoritative source for the Aurora migration task.',
            },
            metadata={'surface': 'context-memory-test'},
        )
    )
    return registry


async def _make_agent(*, namespace: str) -> Agent:
    runtime = LocalRegistryToolRuntime(_build_registry())
    agent = Agent(
        AgentConfig(
            name='context-memory-real-loop',
            planner='api',
            verbose=True,
            memory_root=str(LOCAL_RUNTIME_ROOT / 'agent_memory'),
            memory_namespace=namespace,
            memory_soft_token_limit=520,
            memory_hard_token_limit=720,
            memory_keep_recent_messages=3,
            memory_summary_target_tokens=180,
            auto_activate_skills=False,
            api_base=os.getenv('MCP_API_BASE') or os.getenv('OPENAI_BASE_URL') or 'https://api.openai.com/v1',
            api_key=os.getenv('MCP_API_KEY') or os.getenv('OPENAI_API_KEY') or '',
            model=os.getenv('MCP_MODEL') or os.getenv('OPENAI_MODEL') or '',
        ),
        tool_runtime=runtime,
    )
    await agent.refresh_tools()
    return agent


async def run_case(agent: Agent, query: str, *, retries: int = 1):
    print(f'\n=== Query ===\n{query}')
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            result = await agent.run_detailed(query)
            print('--- Answer ---')
            print(result.answer)
            print('--- Trace Modes ---')
            print([step.get('mode') for step in result.trace])
            return result
        except RuntimeError as exc:
            last_error = exc
            if attempt >= retries:
                raise
            print(f'--- Retry {attempt + 1} after planner/JSON error ---')
            print(str(exc))
    raise RuntimeError(f'run_case exhausted retries: {last_error}')


async def main() -> None:
    api_key = os.getenv('MCP_API_KEY') or os.getenv('OPENAI_API_KEY') or ''
    model = os.getenv('MCP_MODEL') or os.getenv('OPENAI_MODEL') or ''
    if not api_key or not model:
        raise SystemExit('This script requires MCP_API_KEY/OPENAI_API_KEY and MCP_MODEL/OPENAI_MODEL in the environment.')

    LOCAL_RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    namespace = f'context-real-loop-{_run_id()}'

    print('[context-test] phase 1: real agent loop with live tool catalog')
    task1 = await _make_agent(namespace=namespace)
    await run_case(task1, '记住我是一名平台工程师，长期偏好 Rust、PostgreSQL 和 OpenTelemetry。')

    tool_result = await run_case(
        task1,
        '当前任务：迁移 Aurora 订单系统。'
        '你必须先调用 `project_brief_lookup` 读取 Aurora 项目概览，再根据返回的 design_doc_id 调用 `design_doc_lookup`。'
        '最后告诉我项目名、设计文档编号、订单状态枚举、禁止项和技术栈，并把这些任务信息保留在当前会话上下文。',
    )
    trace_modes = [step.get('mode') for step in tool_result.trace]
    trace_calls = [step.get('calls', []) for step in tool_result.trace if step.get('mode') == 'mcp']
    assert trace_modes.count('mcp') >= 2, f'expected at least two mcp steps, got {trace_modes}'
    assert trace_calls[0][0]['tool_name'] == 'project_brief_lookup', f'unexpected first tool call: {trace_calls[0]}'
    assert trace_calls[1][0]['tool_name'] == 'design_doc_lookup', f'unexpected second tool call: {trace_calls[1]}'
    assert 'Aurora' in tool_result.answer, 'project name missing from tool-informed answer'
    assert 'aurora-order-v2' in tool_result.answer, 'design document id missing from tool-informed answer'
    assert 'PENDING' in tool_result.answer, 'order status enum missing from tool-informed answer'
    assert ('Redis' in tool_result.answer or 'redis' in tool_result.answer), 'task constraint missing from tool-informed answer'

    durable_packet = await task1.memory_manager.build_context_packet(query='我的长期偏好是什么？')
    durable_prompt_context = task1.context_manager.build_prompt_context(
        agent=task1,
        query='我的长期偏好是什么？',
        active_skill=None,
        visible_tools=await task1.list_visible_tools(),
        observations=tool_result.trace,
        memory_packet=durable_packet,
    )
    task_packet = await task1.memory_manager.build_context_packet(query='请总结当前 Aurora 任务。')
    task_prompt_context = task1.context_manager.build_prompt_context(
        agent=task1,
        query='请总结当前 Aurora 任务。',
        active_skill=None,
        visible_tools=await task1.list_visible_tools(),
        observations=tool_result.trace,
        memory_packet=task_packet,
    )
    layer_names = [layer.name for layer in task_prompt_context.layers]
    print('--- Prompt Layers ---')
    print(layer_names)
    assert 'mcp-tool-guide' in task_prompt_context.system_prompt, 'tool guide not injected into system prompt'
    assert 'project_brief_lookup' in task_prompt_context.system_prompt, 'tool names were not exposed in the prompt'
    assert 'design_doc_lookup' in task_prompt_context.system_prompt, 'tool descriptions were not exposed in the prompt'
    assert 'memory-long-term' in durable_prompt_context.system_prompt, 'retrieved durable memory was not injected into prompt context'
    assert 'memory-task-context' in task_prompt_context.system_prompt, 'retrieved task memory was not injected into prompt context'
    assert layer_names[-1] == 'working-context-tail', f'expected semantic working context at prompt tail, got {layer_names[-1]}'

    summary_result = await run_case(
        task1,
        '不要再调用工具。请直接总结我的长期偏好，以及当前 Aurora 任务的项目名、设计文档编号、订单状态枚举、禁止项和技术栈。',
    )
    assert ('Rust' in summary_result.answer or 'OpenTelemetry' in summary_result.answer), 'cross-session durable memory missing'
    assert 'Aurora' in summary_result.answer, 'task memory missing'
    assert 'aurora-order-v2' in summary_result.answer, 'tool observation did not survive into prompt-retrieved context'
    assert 'PENDING' in summary_result.answer, 'order status enum did not survive into prompt-retrieved context'

    state1 = await task1.memory_manager.inspect_state()
    i = 0
    while state1['stats']['summary_count'] == 0 and i < 4:
        await run_case(
            task1,
            '不要调用工具。请仅用一句话重申当前 Aurora 任务的关键约束，保留项目名、文档编号、状态枚举、禁止项和技术栈。',
            retries=2,
        )
        state1 = await task1.memory_manager.inspect_state()
        i += 1
    print('[task1 state]', state1)
    assert state1['stats']['summary_count'] > 0, 'summary compaction did not happen in task1'
    await task1.disconnect()

    print('\n[context-test] phase 2: fresh agent instance with same namespace')
    task2 = await _make_agent(namespace=namespace)
    answer2 = (
        await run_case(
            task2,
            '新的会话里，不要调用工具。请直接列出我长期偏好的技术名称，用逗号分隔，不要补充任何项目名、文档编号、状态枚举或任务细节。',
        )
    ).answer
    assert ('Rust' in answer2 or 'OpenTelemetry' in answer2), 'cross-session durable memory did not survive a fresh agent instance'
    assert 'aurora-order-v2' not in answer2, 'session-scoped task memory should not leak into a fresh task session'
    assert 'PENDING' not in answer2, 'tool-derived task context should not leak into a fresh task session'
    await task2.disconnect()

    print('\n[context-test] all checks passed')
    print(f'[context-test] local memory root: {LOCAL_RUNTIME_ROOT}')
    print(f'[context-test] namespace: {namespace}')


if __name__ == '__main__':
    asyncio.run(main())
