import argparse
import asyncio
import json
import os
from pathlib import Path

from agent_core import Agent, AgentConfig


DEFAULT_CASES = [
    '请列出当前可用工具。',
    '请查看工具注册表统计。',
    '请查询 `calculator` 工具的详细信息。',
    '请计算 12*(3+4)。',
    '请先把 `Hello World 2026` 转成 slug，再把结果反转。',
]


ADD_TOOL_QUERY = r'''请新增一个工具。请根据下面 JSON 生成一次 tool_add 调用：
{
  "name": "echo_governance_demo",
  "description": "Echo the provided text with a governance marker",
  "code": "def handler(text: str):\n    return {\"message\": f\"governed::{text}\"}",
  "input_schema_json": {
    "type": "object",
    "properties": {
      "text": {"type": "string"}
    },
    "required": ["text"],
    "additionalProperties": false
  },
  "tags": ["demo", "governance"]
}
'''

UPDATE_TOOL_QUERY = r'''请更新工具 `echo_governance_demo`。请根据下面 JSON 生成一次 tool_update 调用：
{
  "name": "echo_governance_demo",
  "description": "Echo text with an updated governance marker",
  "code": "def handler(text: str):\n    return {\"message\": f\"governed-v2::{text.upper()}\"}",
  "changelog": "upgrade governance smoke test tool"
}
'''


CALL_TOOL_QUERY = '请调用工具 `echo_governance_demo`，参数 {"text": "hello"}。'
DISABLE_TOOL_QUERY = '请禁用工具 `echo_governance_demo`。'
ENABLE_TOOL_QUERY = '请启用工具 `echo_governance_demo`。'
REMOVE_TOOL_QUERY = '请删除工具 `echo_governance_demo`。'
VERSIONS_QUERY = '请查看 `echo_governance_demo` 的版本历史。'


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run MCP/Skill integration and governance smoke tests through user queries.')
    parser.add_argument('--query', default='请计算 12*(3+4)。')
    parser.add_argument('--api-base', default=os.getenv('MCP_API_BASE') or 'https://api.openai.com/v1')
    parser.add_argument('--api-key', default=os.getenv('MCP_API_KEY') or os.getenv('OPENAI_API_KEY') or '')
    parser.add_argument('--model', default=os.getenv('MCP_MODEL') or os.getenv('OPENAI_MODEL') or '')
    parser.add_argument('--planner', choices=['api', 'heuristic'], default='api')
    parser.add_argument('--max-steps', type=int, default=6)
    parser.add_argument('--connect-timeout', type=float, default=90)
    parser.add_argument('--planner-timeout', type=float, default=90.0)
    parser.add_argument('--tool-timeout', type=float, default=90.0)
    parser.add_argument('--verbose', action='store_true', default=True)
    parser.add_argument('--skip-governance-smoke', action='store_true')
    parser.add_argument('--skip-extra-cases', action='store_true')
    return parser


async def run_case(agent: Agent, query: str) -> None:
    print(f'\n=== Query ===\n{query}', flush=True)
    result = await agent.run_detailed(query)
    print('--- Final Answer ---', flush=True)
    print(result.answer, flush=True)
    print(f'output_mode={result.output_mode} selected_skill={result.selected_skill}', flush=True)
    print('--- Trace ---', flush=True)
    for idx, step in enumerate(result.trace, start=1):
        print(f'{idx}. mode={step.get("mode")} thought={step.get("thought", "")}', flush=True)
        print(f'   observation={step.get("observation", "")}', flush=True)


async def governance_smoke(agent: Agent) -> None:
    print('\n=== Governance Smoke (query-driven) ===', flush=True)
    for query in [
        '请列出当前可用的治理工具。',
        ADD_TOOL_QUERY,
        CALL_TOOL_QUERY,
        UPDATE_TOOL_QUERY,
        CALL_TOOL_QUERY,
        VERSIONS_QUERY,
        DISABLE_TOOL_QUERY,
        ENABLE_TOOL_QUERY,
        REMOVE_TOOL_QUERY,
    ]:
        await run_case(agent, query)


async def main() -> None:
    args = build_parser().parse_args()
    if args.planner == 'api' and (not args.api_key or not args.model):
        raise SystemExit('planner=api requires MCP_API_KEY/OPENAI_API_KEY and MCP_MODEL/OPENAI_MODEL')

    root = Path(__file__).resolve().parent
    server_script = root / 'mcp_lib' / 'server' / 'mcp_server.py'
    skills_root = root / 'skills'

    agent = Agent(
        AgentConfig(
            name='demo-agent',
            description='MCP + Skill integration demo agent',
            skills_root=str(skills_root),
            planner=args.planner,
            api_base=args.api_base,
            api_key=args.api_key,
            model=args.model,
            max_steps=args.max_steps,
            connect_timeout_seconds=args.connect_timeout,
            planner_timeout_seconds=args.planner_timeout,
            tool_timeout_seconds=args.tool_timeout,
            verbose=args.verbose,
        )
    )

    async with agent:
        print(f'[demo] planner={args.planner}', flush=True)
        print(f'[demo] connecting to MCP server: {server_script}', flush=True)
        await agent.connect(str(server_script))
        live_tools = await agent.refresh_tools()
        print(f'[demo] live MCP tools: {len(live_tools)}', flush=True)
        print(f'[demo] loaded skills: {len(agent.list_skills())}', flush=True)
        print('[demo] skill catalog:', json.dumps(agent.list_skills(), ensure_ascii=False, indent=2), flush=True)

        await run_case(agent, args.query)

        if not args.skip_extra_cases:
            for query in DEFAULT_CASES:
                await run_case(agent, query)

        if not args.skip_governance_smoke:
            await governance_smoke(agent)


if __name__ == '__main__':
    asyncio.run(main())
