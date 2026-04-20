import argparse
import asyncio
import os
from pathlib import Path

from agent_core import Agent, AgentConfig, LocalRegistryToolRuntime
from mcp_lib.registry.registry import ToolRegistry
from mcp_lib.tools import bootstrap_all_tools


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run the real API + skill integration demo.')
    parser.add_argument('--query', default='请先把 `Hello World 2026` 转成 slug，再把结果反转，并说明你选择了哪个 skill。')
    parser.add_argument('--api-base', default=os.getenv('MCP_API_BASE') or os.getenv('OPENAI_BASE_URL') or 'https://api.openai.com/v1')
    parser.add_argument('--api-key', default=os.getenv('MCP_API_KEY') or os.getenv('OPENAI_API_KEY') or '')
    parser.add_argument('--model', default=os.getenv('MCP_MODEL') or os.getenv('OPENAI_MODEL') or '')
    parser.add_argument('--max-steps', type=int, default=6)
    parser.add_argument('--planner-timeout', type=float, default=90.0)
    parser.add_argument('--tool-timeout', type=float, default=90.0)
    parser.add_argument('--verbose', action='store_true', default=True)
    return parser


async def main() -> None:
    args = build_parser().parse_args()
    if not args.api_key or not args.model:
        raise SystemExit('Set MCP_API_KEY/MCP_MODEL (or OPENAI_API_KEY/OPENAI_MODEL) before running the real API skill demo.')

    registry = ToolRegistry()
    bootstrap_all_tools(registry)
    root = Path(__file__).resolve().parent

    agent = Agent(
        AgentConfig(
            name='skill-demo-agent',
            description='Real API + skill routing demo',
            skills_root=str(root / 'skills'),
            skill_tool_policy='restrictive',
            planner='api',
            api_base=args.api_base,
            api_key=args.api_key,
            model=args.model,
            max_steps=args.max_steps,
            planner_timeout_seconds=args.planner_timeout,
            tool_timeout_seconds=args.tool_timeout,
            verbose=args.verbose,
        ),
        tool_runtime=LocalRegistryToolRuntime(registry),
    )

    await agent.refresh_tools()
    await agent.load_skills()
    result = await agent.run_detailed(args.query)

    print('\n=== Skill Result ===', flush=True)
    print(result.answer, flush=True)
    print(f'output_mode={result.output_mode} selected_skill={result.selected_skill}', flush=True)
    print('\n=== Trace ===', flush=True)
    for idx, step in enumerate(result.trace, start=1):
        print(f'{idx}. mode={step.get("mode")} thought={step.get("thought", "")}', flush=True)
        print(f'   observation={step.get("observation", "")}', flush=True)


if __name__ == '__main__':
    asyncio.run(main())
