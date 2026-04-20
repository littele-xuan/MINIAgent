import argparse
import asyncio
import os
from pathlib import Path

import httpx

from agent_core import Agent, AgentConfig, LocalRegistryToolRuntime
from a2a_runtime import A2AClient, PeerAgent, SendMessageConfiguration
from a2a_runtime.testing import URLRouterTransport
from mcp_lib.registry.registry import ToolRegistry
from mcp_lib.tools import bootstrap_all_tools


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run the real API + A2A demo on top of MCP runtimes.')
    parser.add_argument('--query', default='请把这个数学问题委派给 math-agent，计算 12*(3+4)，并返回 JSON。')
    parser.add_argument('--api-base', default=os.getenv('MCP_API_BASE')  or 'https://api.openai.com/v1')
    parser.add_argument('--api-key', default=os.getenv('MCP_API_KEY') or '')
    parser.add_argument('--model', default=os.getenv('MCP_MODEL')  or '')
    parser.add_argument('--connect-timeout', type=float, default=20.0)
    parser.add_argument('--planner-timeout', type=float, default=90.0)
    parser.add_argument('--tool-timeout', type=float, default=90.0)
    return parser


async def build_agent(
    name: str,
    role: str,
    skills_root: Path,
    *,
    api_base: str,
    api_key: str,
    model: str,
    connect_timeout_seconds: float,
    planner_timeout_seconds: float,
    tool_timeout_seconds: float,
) -> Agent:
    registry = ToolRegistry()
    bootstrap_all_tools(registry)
    agent = Agent(
        AgentConfig(
            name=name,
            description=f'{name} agent',
            role=role,
            skills_root=str(skills_root),
            planner='api',
            api_base=api_base,
            api_key=api_key,
            model=model,
            connect_timeout_seconds=connect_timeout_seconds,
            planner_timeout_seconds=planner_timeout_seconds,
            tool_timeout_seconds=tool_timeout_seconds,
            verbose=True,
        ),
        tool_runtime=LocalRegistryToolRuntime(registry),
    )
    await agent.refresh_tools()
    await agent.load_skills()
    return agent


async def main() -> None:
    args = build_parser().parse_args()
    if not args.api_key or not args.model:
        raise SystemExit('Set MCP_API_KEY/MCP_MODEL (or OPENAI_API_KEY/OPENAI_MODEL) before running the real API A2A demo.')

    skills_root = Path(__file__).resolve().parent / 'skills'
    router = await build_agent(
        'router-agent',
        'router',
        skills_root,
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        connect_timeout_seconds=args.connect_timeout,
        planner_timeout_seconds=args.planner_timeout,
        tool_timeout_seconds=args.tool_timeout,
    )
    math_agent = await build_agent(
        'math-agent',
        'math-specialist',
        skills_root,
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        connect_timeout_seconds=args.connect_timeout,
        planner_timeout_seconds=args.planner_timeout,
        tool_timeout_seconds=args.tool_timeout,
    )

    math_agent.enable_a2a(base_url='http://math.local')
    math_app = math_agent.build_a2a_app(base_url='http://math.local')

    router.enable_a2a(
        base_url='http://router.local',
        peers=[PeerAgent(name='math-agent', description='math specialist', agent_card_url='http://math.local/.well-known/agent-card.json', tags=['math', 'calculator'])],
    )
    router_app = router.build_a2a_app(base_url='http://router.local')

    transport = URLRouterTransport({'http://math.local': math_app, 'http://router.local': router_app})
    http_client = httpx.AsyncClient(transport=transport, base_url='http://router.local')
    router.a2a_client = A2AClient(client=http_client)

    client = A2AClient(client=http_client)
    response = await client.send_text(
        'http://router.local/.well-known/agent-card.json',
        args.query,
        configuration=SendMessageConfiguration(acceptedOutputModes=['application/json']),
    )

    print('\n=== Router Response ===', flush=True)
    if response.task and response.task.artifacts:
        print(response.task.artifacts[0].parts[0].data, flush=True)
        print('metadata:', response.task.metadata, flush=True)
    elif response.message and response.message.parts:
        print(response.message.parts[0].data or response.message.parts[0].text, flush=True)

    await http_client.aclose()
    await transport.aclose()
    await router.disconnect()
    await math_agent.disconnect()


if __name__ == '__main__':
    asyncio.run(main())
