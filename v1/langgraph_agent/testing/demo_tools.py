from __future__ import annotations

import ast
import json
import operator
from pathlib import Path
from typing import Any

from ..config.models import LangGraphAgentConfig, MCPServerConfig, SkillConfig
from ..schemas import ToolDescriptor


_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_eval_expr(expr: str) -> float | int:
    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
            return _ALLOWED_BINOPS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
            return _ALLOWED_UNARY[type(node.op)](walk(node.operand))
        raise ValueError(f'Unsupported arithmetic expression: {expr}')
    return walk(ast.parse(expr, mode='eval'))


class DemoToolProvider:
    """Deterministic executable tools for real-LLM integration scenarios.

    The LLM is always real in the scenario runner. These tools are intentionally
    local and auditable so failures mostly reflect agent planning/context rather
    than flaky external services.
    """

    def __init__(self, workdir: str | Path) -> None:
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.note_file = self.workdir / 'demo_note.txt'

    async def connect(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def list_tools(self):
        return [
            ToolDescriptor(
                name='calculator',
                description='Calculate a basic arithmetic expression. Supports +, -, *, /, //, %, ** and parentheses.',
                input_schema={'type': 'object', 'properties': {'expression': {'type': 'string'}}, 'required': ['expression']},
                metadata={'provider': 'demo-executable'},
                risk='safe_read',
            ),
            ToolDescriptor(
                name='huge_log',
                description='Generate a huge text log for testing context compression and observation cleaning.',
                input_schema={'type': 'object', 'properties': {'topic': {'type': 'string'}}, 'required': ['topic']},
                metadata={'provider': 'demo-executable'},
                risk='safe_read',
            ),
            ToolDescriptor(
                name='write_note',
                description='Write a note to disk in the test sandbox.',
                input_schema={'type': 'object', 'properties': {'content': {'type': 'string'}}, 'required': ['content']},
                metadata={'provider': 'demo-executable'},
                risk='filesystem_write',
            ),
            ToolDescriptor(
                name='read_note',
                description='Read the current note from the test sandbox.',
                input_schema={'type': 'object', 'properties': {}},
                metadata={'provider': 'demo-executable'},
                risk='filesystem_read',
            ),
        ]

    async def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == 'calculator':
            expression = str(arguments.get('expression', '')).strip()
            value = _safe_eval_expr(expression)
            return {'tool_name': name, 'arguments': arguments, 'text': str(value), 'payload': {'expression': expression, 'value': value}}
        if name == 'huge_log':
            topic = arguments.get('topic', 'demo')
            text = '\n'.join(f'[{i:04d}] {topic} log payload chunk {i}' for i in range(1600))
            return {'tool_name': name, 'arguments': arguments, 'text': text, 'payload': {'topic': topic, 'lines': 1600}}
        if name == 'write_note':
            self.note_file.write_text(arguments['content'], encoding='utf-8')
            payload = {'path': str(self.note_file), 'bytes': self.note_file.stat().st_size}
            return {'tool_name': name, 'arguments': arguments, 'text': json.dumps(payload, ensure_ascii=False), 'payload': payload}
        if name == 'read_note':
            text = self.note_file.read_text(encoding='utf-8') if self.note_file.exists() else ''
            return {'tool_name': name, 'arguments': arguments, 'text': text, 'payload': {'path': str(self.note_file), 'exists': self.note_file.exists()}}
        raise KeyError(name)


class DemoToolProviderAdapter:
    def __init__(self, demo: DemoToolProvider) -> None:
        self.demo = demo

    async def connect(self) -> None:
        await self.demo.connect()

    async def close(self) -> None:
        await self.demo.close()

    async def list_tools(self):
        return await self.demo.list_tools()

    async def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self.demo.execute(name, arguments)


def build_local_mcp_agent_config(root: Path, *, model: str, api_key: str, api_base: str) -> LangGraphAgentConfig:
    server_script = root / 'v0' / 'mcp_lib' / 'server' / 'mcp_server.py'
    return LangGraphAgentConfig(
        name='langgraph-mcp-agent',
        api_base=api_base,
        api_key=api_key,
        model=model,
        mcp_servers=[
            MCPServerConfig(
                name='local',
                transport='stdio',
                command='python',
                args=[str(server_script)],
                stateful_session=False,
            )
        ],
        skills=SkillConfig(enabled=False),
    )
