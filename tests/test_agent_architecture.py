from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_core import Agent, AgentConfig, LocalRegistryToolRuntime
from mcp_lib.registry.registry import ToolDisabledError, ToolRegistry
from mcp_lib.tools import bootstrap_all_tools


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


def run(coro):
    return asyncio.run(coro)


async def build_agent() -> Agent:
    registry = ToolRegistry()
    bootstrap_all_tools(registry)
    root = Path(__file__).resolve().parents[1]
    agent = Agent(
        AgentConfig(
            name='test-agent',
            description='heuristic architecture test agent',
            skills_root=str(root / 'skills'),
            skill_tool_policy='restrictive',
            planner='heuristic',
            verbose=False,
        ),
        tool_runtime=LocalRegistryToolRuntime(registry),
    )
    await agent.refresh_tools()
    await agent.load_skills()
    return agent


def test_registry_skill_exposes_real_governance_tools() -> None:
    async def scenario() -> None:
        agent = await build_agent()
        try:
            selected = agent.activate_skill('请查看工具注册表统计，并查询 calculator 的 schema。')
            assert selected is not None
            assert selected.name == 'registry-maintainer'
            visible_names = {tool.name for tool in await agent.list_visible_tools()}
            assert 'tool_search' in visible_names
            assert 'tool_info' in visible_names
            assert 'registry_stats' in visible_names
            assert 'tool_add' in visible_names
        finally:
            await agent.disconnect()

    run(scenario())


def test_query_driven_governance_crud_flow() -> None:
    async def scenario() -> None:
        agent = await build_agent()
        try:
            add_result = await agent.run_detailed(ADD_TOOL_QUERY)
            assert any(step.get('mode') == 'mcp' for step in add_result.trace)
            assert 'echo_governance_demo' in {tool.name for tool in await agent.refresh_tools()}

            call_result = await agent.run_detailed('请调用工具 `echo_governance_demo`，参数 {"text": "hello"}。')
            assert 'governed::hello' in call_result.answer

            update_result = await agent.run_detailed(UPDATE_TOOL_QUERY)
            assert any(step.get('mode') == 'mcp' for step in update_result.trace)

            upgraded_call = await agent.run_detailed('请调用工具 `echo_governance_demo`，参数 {"text": "hello"}。')
            assert 'governed-v2::HELLO' in upgraded_call.answer

            disable_result = await agent.run_detailed('请禁用工具 `echo_governance_demo`。')
            assert 'disabled' in disable_result.answer.lower() or '禁用' in disable_result.answer
            await agent.refresh_tools()
            assert 'echo_governance_demo' not in {tool.name for tool in await agent.list_visible_tools()}
            try:
                await agent.tool_runtime.call_tool('echo_governance_demo', {'text': 'hello'})
            except ToolDisabledError:
                pass
            else:
                raise AssertionError('disabled tool should not be callable')

            enable_result = await agent.run_detailed('请启用工具 `echo_governance_demo`。')
            assert 'enabled' in enable_result.answer.lower() or '启用' in enable_result.answer
            await agent.refresh_tools()
            enabled_call = await agent.run_detailed('请调用工具 `echo_governance_demo`，参数 {"text": "hello"}。')
            assert 'governed-v2::HELLO' in enabled_call.answer

            versions_result = await agent.run_detailed('请查看 `echo_governance_demo` 的版本历史。')
            assert 'versions' in versions_result.answer.lower() or 'current' in versions_result.answer.lower()

            remove_result = await agent.run_detailed('请删除工具 `echo_governance_demo`。')
            assert 'removed' in remove_result.answer.lower() or '删除' in remove_result.answer
            assert 'echo_governance_demo' not in {tool.name for tool in await agent.refresh_tools()}
        finally:
            await agent.disconnect()

    run(scenario())


def test_multi_step_text_transform_stays_inside_agent_loop() -> None:
    async def scenario() -> None:
        agent = await build_agent()
        try:
            result = await agent.run_detailed('请先把 `Hello World 2026` 转成 slug，再把结果反转。')
            assert result.answer.strip() == '6202-dlrow-olleh'
            mcp_steps = [step for step in result.trace if step.get('mode') == 'mcp']
            assert len(mcp_steps) == 2
            assert mcp_steps[0]['calls'][0]['tool_name'] == 'slugify_text'
            assert mcp_steps[1]['calls'][0]['tool_name'] == 'reverse_text'
        finally:
            await agent.disconnect()

    run(scenario())


def test_registry_queries_use_governance_surface() -> None:
    async def scenario() -> None:
        agent = await build_agent()
        try:
            list_result = await agent.run_detailed('请列出当前可用的治理工具。')
            assert 'tool_' in list_result.answer or 'registry_' in list_result.answer

            info_result = await agent.run_detailed('请查询 `calculator` 工具的详细信息。')
            assert 'calculator' in info_result.answer
            assert 'input_schema' in info_result.answer

            stats_result = await agent.run_detailed('请查看工具注册表统计。')
            assert 'total' in stats_result.answer
            assert 'by_category' in stats_result.answer
        finally:
            await agent.disconnect()

    run(scenario())
