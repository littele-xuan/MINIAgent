from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Awaitable, Callable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from langgraph_agent.agent.runtime import LangGraphAgent
from langgraph_agent.config.models import LangGraphAgentConfig, MemoryConfig, SkillConfig
from langgraph_agent.testing.demo_tools import DemoToolProvider, DemoToolProviderAdapter


ScenarioFn = Callable[[LangGraphAgent], Awaitable[None]]


def _require_real_api(args: argparse.Namespace) -> None:
    if not args.api_key or not args.model:
        raise SystemExit(
            'Real API credentials are required. Set MCP_API_KEY/OPENAI_API_KEY and MCP_MODEL/OPENAI_MODEL, '
            'or pass --api-key and --model. This runner intentionally does not mock the LLM.'
        )


def _assert_contains(text: str, needles: list[str], *, label: str) -> None:
    if not any(n.lower() in text.lower() for n in needles):
        raise AssertionError(f'{label} failed. Expected one of {needles!r} in answer:\n{text}')


async def _print_run(agent: LangGraphAgent, query: str, *, thread_id: str, user_id: str = 'research-user', max_steps: int | None = None):
    print('\n=== USER ===')
    print(query)
    result = await agent.run_detailed(query, thread_id=thread_id, user_id=user_id, max_steps=max_steps)
    print('--- ANSWER ---')
    print(result.answer)
    print('--- TRACE MODES ---')
    print([step.get('mode') for step in result.trace])
    return result


async def scenario_memory_cross_thread(agent: LangGraphAgent) -> None:
    print('\n[scenario] memory_cross_thread')
    await _print_run(agent, '记住我长期偏好 Rust 和 PostgreSQL；以后涉及后端方案时优先考虑它们。', thread_id='mem-A')
    await _print_run(agent, '当前任务：实现一个离线 CLI。必须保留审计日志，不要依赖网络调用。', thread_id='mem-A')
    result = await _print_run(agent, '请根据长期记忆总结我的技术偏好和当前任务约束。', thread_id='mem-B')
    _assert_contains(result.answer, ['Rust', 'PostgreSQL'], label='long-term preference recall')
    _assert_contains(result.answer, ['CLI', '审计', '网络'], label='task constraint recall')


async def scenario_tool_multi_step(agent: LangGraphAgent) -> None:
    print('\n[scenario] tool_multi_step')
    result = await _print_run(
        agent,
        '请使用工具 write_note 写入内容 `release=2026.04\nowner=agent`，然后使用 read_note 读回来，并最终告诉我读回的完整内容。',
        thread_id='tool-A',
        max_steps=6,
    )
    _assert_contains(result.answer, ['release=2026.04'], label='write/read note workflow')
    if not any(step.get('mode') == 'tool' for step in result.trace):
        raise AssertionError('Expected at least one real tool execution in trace')


async def scenario_tool_governance(agent: LangGraphAgent) -> None:
    print('\n[scenario] tool_governance')
    result = await _print_run(
        agent,
        '请先用 tool_search 搜索 note 相关工具，再用 tool_inspect 查看 read_note 的 schema，最后总结它的用途和参数。',
        thread_id='tool-gov-A',
        max_steps=5,
    )
    _assert_contains(result.answer, ['read_note', 'note', 'schema', '参数'], label='tool governance inspection')


async def scenario_context_compression(agent: LangGraphAgent) -> None:
    print('\n[scenario] context_compression')
    result = await _print_run(
        agent,
        '请调用 huge_log，topic 设为 context-compression，然后不要复述全部日志，只总结日志主题、行数和你观察到的格式。',
        thread_id='ctx-A',
        max_steps=4,
    )
    _assert_contains(result.answer, ['context-compression', '1600', '日志', 'log'], label='large observation compression')


async def scenario_skill_slugify(agent: LangGraphAgent) -> None:
    print('\n[scenario] skill_slugify')
    result = await _print_run(
        agent,
        '请使用 text-transform skill，把文字 Hello LangGraph Agent 做 slugify 处理，只返回结果。',
        thread_id='skill-A',
        max_steps=4,
    )
    _assert_contains(result.answer, ['hello-langgraph-agent'], label='skill-local tool slugify')


async def scenario_json_output(agent: LangGraphAgent) -> None:
    print('\n[scenario] structured_json_output')
    result = await agent.run_detailed(
        '请用 application/json 输出一个对象，字段包括 stack、database、reason，其中 stack 必须根据我的长期偏好填写。',
        thread_id='mem-B',
        user_id='research-user',
        accepted_output_modes=['application/json'],
        max_steps=3,
    )
    print('--- ANSWER JSON ---')
    print(result.answer)
    data: Any = result.payload if result.payload is not None else json.loads(result.answer)
    blob = json.dumps(data, ensure_ascii=False)
    _assert_contains(blob, ['Rust', 'PostgreSQL'], label='structured JSON memory use')


SCENARIOS: dict[str, ScenarioFn] = {
    'memory': scenario_memory_cross_thread,
    'tools': scenario_tool_multi_step,
    'governance': scenario_tool_governance,
    'context': scenario_context_compression,
    'skill': scenario_skill_slugify,
    'json': scenario_json_output,
}


async def main() -> None:
    parser = argparse.ArgumentParser(description='Run real API LangGraph agent scenarios. No pytest, no mocked LLM.')
    parser.add_argument('--api-base', default=os.getenv('MCP_API_BASE') or os.getenv('OPENAI_BASE_URL') or 'https://api.openai.com/v1')
    parser.add_argument('--api-key', default=os.getenv('MCP_API_KEY') or os.getenv('OPENAI_API_KEY') or '')
    parser.add_argument('--model', default=os.getenv('MCP_MODEL') or os.getenv('OPENAI_MODEL') or '')
    parser.add_argument('--root', default='.real_agent_scenarios')
    parser.add_argument('--reset', action='store_true')
    parser.add_argument('--scenario', choices=['all', *SCENARIOS.keys()], default='all')
    parser.add_argument('--memory-extractor', choices=['heuristic', 'llm', 'hybrid'], default='hybrid')
    args = parser.parse_args()
    _require_real_api(args)

    project_root = Path(__file__).resolve().parents[1]
    root = Path(args.root).resolve()
    if args.reset and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    provider = DemoToolProviderAdapter(DemoToolProvider(root / 'tool_sandbox'))
    config = LangGraphAgentConfig(
        name='research-langgraph-baseline',
        description='Research-grade LangGraph while-loop MCP agent baseline.',
        role='MCP tool-using research agent',
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        verbose=True,
        working_dir=str(root),
        memory=MemoryConfig(
            user_id='research-user',
            namespace='research-baseline',
            sqlite_path=str(root / 'checkpoints.sqlite'),
            extractor=args.memory_extractor,
        ),
        skills=SkillConfig(enabled=True, skills_root=str(project_root / 'skills')),
    )

    selected = list(SCENARIOS) if args.scenario == 'all' else [args.scenario]
    async with LangGraphAgent(config, tool_provider=provider) as agent:
        for name in selected:
            await SCENARIOS[name](agent)
    print('\n[real-scenarios] all selected scenarios passed ✅')


if __name__ == '__main__':
    asyncio.run(main())
