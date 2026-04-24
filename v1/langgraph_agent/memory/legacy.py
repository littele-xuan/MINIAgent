from __future__ import annotations

from typing import Any

from .base import BaseMemoryManager


class LegacyMemoryManager(BaseMemoryManager):
    def __init__(self, engine: Any) -> None:
        self.engine = engine

    async def load_context(self, *, user_id: str, thread_id: str, query: str) -> dict[str, Any]:
        packet = await self.engine.build_context_packet(query=query)
        return {
            'memories': [getattr(item, 'text', '') for item in getattr(packet, 'retrieved_memories', [])],
            'warnings': list(getattr(packet, 'warnings', []) or []),
        }

    async def record_user_turn(self, *, user_id: str, thread_id: str, text: str) -> None:
        if hasattr(self.engine, 'begin_turn'):
            await self.engine.begin_turn(text)

    async def record_tool_result(self, *, user_id: str, thread_id: str, tool_name: str, content: str) -> None:
        if hasattr(self.engine, 'record_observation'):
            await self.engine.record_observation({'mode': 'mcp', 'observation': content, 'calls': [{'tool_name': tool_name}]})

    async def record_assistant_turn(self, *, user_id: str, thread_id: str, text: str) -> None:
        if hasattr(self.engine, 'finalize_turn'):
            await self.engine.finalize_turn(answer=text, output_mode='text/plain', payload=None)
