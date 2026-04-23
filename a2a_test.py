import argparse
import asyncio
import json
import os
from pathlib import Path

import httpx

from agent_core import Agent, AgentConfig, LocalRegistryToolRuntime
from a2a_runtime import A2AClient, PeerAgent, SendMessageConfiguration
from a2a_runtime.testing import URLRouterTransport
from mcp_lib.registry.registry import ToolRegistry
from mcp_lib.tools import bootstrap_all_tools


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run A2A discovery / send / task / rpc smoke tests on top of MCP runtimes.')
    parser.add_argument('--query', default='请把这个数学问题委派给 math-agent，计算 12*(3+4)。')
    parser.add_argument('--api-base', default=os.getenv('MCP_API_BASE') or 'https://api.openai.com/v1')
    parser.add_argument('--api-key', default=os.getenv('MCP_API_KEY') or os.getenv('OPENAI_API_KEY') or '')
    parser.add_argument('--model', default=os.getenv('MCP_MODEL') or os.getenv('OPENAI_MODEL') or '')
    parser.add_argument('--planner', choices=['api', 'openai'], default='api')
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
    planner: str,
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
            planner=planner,
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
    await agent.load_skills(['text-transform', 'registry-maintainer', 'a2a-router', 'math-specialist'])
    return agent


async def print_card(client: A2AClient, url: str, label: str) -> None:
    card = await client.fetch_agent_card(url)
    skills = card.get('skills') or []
    print(f'\n=== {label} Agent Card ===', flush=True)
    print(json.dumps({'name': card.get('name'), 'description': card.get('description'), 'skill_count': len(skills)}, ensure_ascii=False, indent=2), flush=True)


async def main() -> None:
    args = build_parser().parse_args()
    if not args.api_key or not args.model:
        raise SystemExit('This script requires MCP_API_KEY/OPENAI_API_KEY and MCP_MODEL/OPENAI_MODEL because planning is fully LLM-driven.')

    skills_root = Path(__file__).resolve().parent / 'skills'
    router = await build_agent(
        'router-agent',
        'router',
        skills_root,
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        planner=args.planner,
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
        planner=args.planner,
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

    try:
        print(f'[a2a-demo] planner={args.planner}', flush=True)
        await print_card(client, 'http://router.local/.well-known/agent-card.json', 'Router')
        await print_card(client, 'http://math.local/.well-known/agent-card.json', 'Math')

        direct_math = await client.send_text(
            'http://math.local/.well-known/agent-card.json',
            '请计算 12*(3+4)。',
            configuration=SendMessageConfiguration(acceptedOutputModes=['text/plain']),
        )
        print('\n=== Direct Math Agent Response ===', flush=True)
        if direct_math.task and direct_math.task.artifacts:
            print(direct_math.task.artifacts[0].parts[0].text or direct_math.task.artifacts[0].parts[0].data, flush=True)
            print('task_id:', direct_math.task.id, flush=True)

        routed = await client.send_text(
            'http://router.local/.well-known/agent-card.json',
            args.query,
            configuration=SendMessageConfiguration(acceptedOutputModes=['text/plain']),
        )
        print('\n=== Routed Router Response ===', flush=True)
        if routed.task and routed.task.artifacts:
            artifact = routed.task.artifacts[0].parts[0]
            print(artifact.data or artifact.text, flush=True)
            print('metadata:', json.dumps(routed.task.metadata or {}, ensure_ascii=False, indent=2), flush=True)
            task_id = routed.task.id
        elif routed.message and routed.message.parts:
            print(routed.message.parts[0].data or routed.message.parts[0].text, flush=True)
            task_id = routed.message.task_id
        else:
            raise RuntimeError('Unexpected routed response shape')

        print('\n=== REST GetTask ===', flush=True)
        rest_task = await http_client.get(f'http://router.local/a2a/v1/tasks/{task_id}', headers={'A2A-Version': '1.0'})
        print(rest_task.status_code, flush=True)
        print(json.dumps(rest_task.json(), ensure_ascii=False, indent=2), flush=True)

        print('\n=== JSON-RPC GetTask ===', flush=True)
        rpc_task = await http_client.post(
            'http://router.local/a2a/v1/rpc',
            headers={'A2A-Version': '1.0'},
            json={
                'jsonrpc': '2.0',
                'id': 'get-task-1',
                'method': 'GetTask',
                'params': {'id': task_id},
            },
        )
        print(rpc_task.status_code, flush=True)
        print(json.dumps(rpc_task.json(), ensure_ascii=False, indent=2), flush=True)

        print('\n=== JSON-RPC ListTasks ===', flush=True)
        rpc_list = await http_client.post(
            'http://router.local/a2a/v1/rpc',
            headers={'A2A-Version': '1.0'},
            json={
                'jsonrpc': '2.0',
                'id': 'list-tasks-1',
                'method': 'ListTasks',
                'params': {},
            },
        )
        print(rpc_list.status_code, flush=True)
        print(json.dumps(rpc_list.json(), ensure_ascii=False, indent=2), flush=True)
    finally:
        await client.close()
        await http_client.aclose()
        await transport.aclose()
        await router.disconnect()
        await math_agent.disconnect()


if __name__ == '__main__':
    asyncio.run(main())
