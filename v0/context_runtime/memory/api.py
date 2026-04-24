from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llm_runtime import BaseLLM

from .engine import ContextMemoryEngine, MemoryEngineConfig
from .factory import MemoryRuntimeComponents


@dataclass(slots=True)
class ContextRuntimeAPI:
    """Direct async API over the context/memory runtime, independent of Agent orchestration."""

    engine: ContextMemoryEngine

    @classmethod
    def create(
        cls,
        *,
        root_dir: str,
        namespace: str,
        session_id: str | None = None,
        soft_token_limit: int = 2200,
        hard_token_limit: int = 3200,
        keep_recent_messages: int = 6,
        summary_target_tokens: int = 650,
        large_observation_tokens: int = 500,
        retrieve_limit: int = 8,
        auto_git_commit: bool = False,
        api_base: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        llm: BaseLLM | None = None,
        components: MemoryRuntimeComponents | None = None,
    ) -> 'ContextRuntimeAPI':
        engine = ContextMemoryEngine(
            MemoryEngineConfig(
                root_dir=root_dir,
                namespace=namespace,
                session_id=session_id,
                soft_token_limit=soft_token_limit,
                hard_token_limit=hard_token_limit,
                keep_recent_messages=keep_recent_messages,
                summary_target_tokens=summary_target_tokens,
                large_observation_tokens=large_observation_tokens,
                retrieve_limit=retrieve_limit,
                auto_git_commit=auto_git_commit,
                api_base=api_base,
                api_key=api_key,
                model=model,
            ),
            llm=llm,
            components=components,
        )
        return cls(engine=engine)

    async def ingest_turn(self, *, user_query: str, tool_observation: str | None = None, final_answer: str = 'ok') -> None:
        await self.engine.begin_turn(user_query)
        if tool_observation:
            await self.engine.record_observation({'mode': 'mcp', 'observation': tool_observation, 'calls': []})
        await self.engine.finalize_turn(answer=final_answer, output_mode='text/plain')

    async def ask_memory(self, query: str) -> str | None:
        return await self.engine.answer_memory_query(query)

    async def context_packet(self, query: str) -> dict[str, Any]:
        packet = await self.engine.build_context_packet(query=query)
        return {
            'session_id': packet.session_id,
            'stats': packet.stats,
            'warnings': packet.warnings,
            'retrieved_memories': [item.text for item in packet.retrieved_memories],
            'recent_messages': [msg.content for msg in packet.recent_messages],
        }

    async def close(self) -> None:
        await self.engine.close()
