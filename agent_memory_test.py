import argparse
import asyncio
import json
import os
import shutil
from pathlib import Path
from typing import Any

from agent_core import Agent, AgentConfig
from agent_core.tool_runtime import BaseToolRuntime, ToolCallResult, ToolDescriptor


class DemoToolRuntime(BaseToolRuntime):
    def __init__(self, workdir: Path) -> None:
        self.workdir = workdir
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.note_file = self.workdir / 'demo_note.txt'

    async def list_tools(self) -> list[ToolDescriptor]:
        return [
            ToolDescriptor(
                name='huge_log',
                description='Generate a very large text log for artifact/offloading tests. Input: topic string.',
                input_schema={'type': 'object', 'properties': {'topic': {'type': 'string'}}, 'required': ['topic']},
                metadata={'category': 'testing'},
            ),
            ToolDescriptor(
                name='write_note',
                description='Write the provided content into a demo note file and return the file path.',
                input_schema={'type': 'object', 'properties': {'content': {'type': 'string'}}, 'required': ['content']},
                metadata={'category': 'testing'},
            ),
            ToolDescriptor(
                name='read_note',
                description='Read the current demo note file and return its content.',
                input_schema={'type': 'object', 'properties': {}},
                metadata={'category': 'testing'},
            ),
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolCallResult:
        if name == 'huge_log':
            topic = arguments.get('topic', 'demo')
            text = '\n'.join(f'[{i:04d}] {topic} log payload chunk {i}' for i in range(1600))
            return ToolCallResult(tool_name=name, arguments=arguments, content=[{'type': 'text', 'text': text}])
        if name == 'write_note':
            self.note_file.write_text(arguments['content'], encoding='utf-8')
            text = json.dumps({'path': str(self.note_file), 'bytes': self.note_file.stat().st_size}, ensure_ascii=False)
            return ToolCallResult(tool_name=name, arguments=arguments, content=[{'type': 'text', 'text': text}])
        if name == 'read_note':
            text = self.note_file.read_text(encoding='utf-8') if self.note_file.exists() else ''
            return ToolCallResult(tool_name=name, arguments=arguments, content=[{'type': 'text', 'text': text}])
        raise KeyError(name)


async def run_case(agent: Agent, query: str) -> str:
    print(f'\n=== Query ===\n{query}')
    result = await agent.run_detailed(query)
    print('--- Answer ---')
    print(result.answer)
    print('--- Trace Modes ---')
    print([step.get('mode') for step in result.trace])
    return result.answer


async def main() -> None:
    parser = argparse.ArgumentParser(description='Real-LLM integration smoke test for agent memory/context.')
    parser.add_argument('--memory-root', default='.agent_memory_demo')
    parser.add_argument('--reset', action='store_true')
    parser.add_argument('--api-base', default=os.getenv('MCP_API_BASE') or os.getenv('OPENAI_BASE_URL') or 'https://api.openai.com/v1')
    parser.add_argument('--api-key', default=os.getenv('MCP_API_KEY') or os.getenv('OPENAI_API_KEY') or '')
    parser.add_argument('--model', default=os.getenv('MCP_MODEL') or os.getenv('OPENAI_MODEL') or '')
    args = parser.parse_args()

    if not args.api_key or not args.model:
        raise SystemExit('This integration test requires MCP_API_KEY/OPENAI_API_KEY and MCP_MODEL/OPENAI_MODEL.')

    memory_root = Path(args.memory_root).resolve()
    if args.reset and memory_root.exists():
        shutil.rmtree(memory_root)
    tool_workdir = memory_root / 'tool_sandbox'

    runtime = DemoToolRuntime(tool_workdir)
    agent = Agent(
        AgentConfig(
            name='memory-demo-agent',
            planner='api',
            verbose=True,
            memory_root=str(memory_root),
            memory_namespace='demo-user',
            memory_soft_token_limit=520,
            memory_hard_token_limit=720,
            memory_keep_recent_messages=4,
            memory_summary_target_tokens=180,
            memory_large_observation_tokens=120,
            auto_activate_skills=False,
            api_base=args.api_base,
            api_key=args.api_key,
            model=args.model,
        ),
        tool_runtime=runtime,
    )
    await agent.refresh_tools()

    print('\n[memory-test] phase 1: durable memory + task memory')
    await run_case(agent, '记住我长期偏好 Rust 和 PostgreSQL。')
    await run_case(agent, '当前任务：实现一个离线 CLI。必须保留审计日志，不要网络调用。')
    answer = await run_case(agent, '请根据记忆总结我的长期偏好以及当前任务约束。')
    assert ('Rust' in answer or 'PostgreSQL' in answer), 'long-term memory was not recalled'
    assert ('CLI' in answer or '网络' in answer or '审计' in answer), 'task memory was not recalled'

    print('\n[memory-test] phase 2: compaction with model-driven summaries')
    for i in range(10):
        await run_case(agent, f'补充上下文消息 {i}: 这是连续会话细节，用于触发长上下文压缩，但不能丢失关键约束。')
    state = await agent.memory_manager.inspect_state()
    print('[state after compaction]', state)
    assert state['stats']['summary_count'] > 0, 'compaction did not create summary nodes'

    print('\n[memory-test] phase 3: tool usage through model planning')
    result = await agent.run_detailed('请使用工具 huge_log 生成一个很大的日志，topic 设为 integration-memory-test。')
    print(result.answer)
    assert any(step.get('mode') == 'mcp' for step in result.trace), 'planner did not call MCP tools'
    artifacts = agent.memory_manager.store.list_recent_artifacts(agent.memory_manager.session_id, limit=5)
    assert any(item.kind == 'tool_output' for item in artifacts), 'large tool output was not offloaded as artifact'

    print('\n[memory-test] phase 4: multi-step tool workflow')
    answer = await run_case(agent, '请先使用 write_note 写入内容 `release=2026.04\nowner=agent`，然后使用 read_note 读回来并告诉我结果。')
    assert 'release=2026.04' in answer, 'multi-step tool workflow did not complete'

    print('\n[memory-test] all checks passed ✅')
    await agent.disconnect()


if __name__ == '__main__':
    asyncio.run(main())
