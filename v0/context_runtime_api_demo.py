import argparse
import asyncio
import os
from pathlib import Path

from context_runtime.memory import ContextRuntimeAPI


async def main() -> None:
    parser = argparse.ArgumentParser(description='Direct API demo for the standalone LLM-driven context runtime.')
    parser.add_argument('--root', default='.context_runtime_api_demo')
    parser.add_argument('--reset', action='store_true')
    parser.add_argument('--api-base', default=os.getenv('MCP_API_BASE') or os.getenv('OPENAI_BASE_URL') or 'https://api.openai.com/v1')
    parser.add_argument('--api-key', default=os.getenv('MCP_API_KEY') or os.getenv('OPENAI_API_KEY') or '')
    parser.add_argument('--model', default=os.getenv('MCP_MODEL') or os.getenv('OPENAI_MODEL') or '')
    args = parser.parse_args()

    if not args.api_key or not args.model:
        raise SystemExit('This demo requires MCP_API_KEY/OPENAI_API_KEY and MCP_MODEL/OPENAI_MODEL.')

    root = Path(args.root).resolve()
    if args.reset and root.exists():
        import shutil
        shutil.rmtree(root)

    api = ContextRuntimeAPI.create(
        root_dir=str(root),
        namespace='demo-user',
        session_id='demo-task',
        soft_token_limit=320,
        hard_token_limit=420,
        summary_target_tokens=140,
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
    )

    await api.ingest_turn(user_query='记住我是平台工程师，我长期偏好 Rust 和 SQLite。', final_answer='收到')
    await api.ingest_turn(user_query='当前任务：实现一个 CLI。技术约束：离线运行，不要网络调用。', final_answer='收到')

    print('--- memory answer ---')
    print(await api.ask_memory('请总结我的长期偏好和当前任务约束。'))
    print('--- packet ---')
    print(await api.context_packet('你记得我的长期偏好吗？'))
    await api.close()


if __name__ == '__main__':
    asyncio.run(main())
