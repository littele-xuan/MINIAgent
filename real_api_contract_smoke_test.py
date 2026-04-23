from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from agent_core import Agent, AgentConfig


async def main() -> None:
    parser = argparse.ArgumentParser(description='Run a real-API smoke test for planner MCP + runtime-owned memory JSON contracts.')
    parser.add_argument('--root', default='.real_api_contract_smoke')
    parser.add_argument('--api-base', default=os.getenv('MCP_API_BASE') or os.getenv('OPENAI_BASE_URL') or 'https://api.openai.com/v1')
    parser.add_argument('--api-key', default=os.getenv('MCP_API_KEY') or os.getenv('OPENAI_API_KEY') or '')
    parser.add_argument('--model', default=os.getenv('MCP_MODEL') or os.getenv('OPENAI_MODEL') or '')
    args = parser.parse_args()

    if not args.api_key or not args.model:
        raise SystemExit('Please set MCP_API_KEY/OPENAI_API_KEY and MCP_MODEL/OPENAI_MODEL before running this smoke test.')

    if os.getenv('LANGFUSE_PUBLIC_KEY') and os.getenv('LANGFUSE_SECRET_KEY'):
        print('[smoke] Langfuse tracing is enabled via environment variables.')

    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    agent = Agent(
        AgentConfig(
            name='mcp-contract-smoke',
            planner='api',
            verbose=True,
            memory_root=str(root / 'memory'),
            memory_namespace='smoke-user',
            auto_activate_skills=False,
            api_base=args.api_base,
            api_key=args.api_key,
            model=args.model,
        )
    )
    try:
        answer1 = await agent.run('记住我是一名平台工程师，长期偏好 Rust、PostgreSQL 和 OpenTelemetry。')
        answer2 = await agent.run('请总结我的长期偏好。')
        print(json.dumps({'answer1': answer1, 'answer2': answer2}, ensure_ascii=False, indent=2))
    finally:
        await agent.disconnect()


if __name__ == '__main__':
    asyncio.run(main())
