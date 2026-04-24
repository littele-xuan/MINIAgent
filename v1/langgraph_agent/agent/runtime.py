from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import BaseAgent
from .result import GraphRunResult
from ..config.models import LangGraphAgentConfig
from ..context.assembler import DefaultContextAssembler
from ..graph.builder import GraphAgentRunner
from ..llm.factory import ChatModelFactory
from ..memory.langgraph_store import LangGraphStoreMemoryManager
from ..observability.langfuse import LangfuseMonitor
from ..skills.filesystem import FilesystemSkillManager
from ..tools.providers.langgraph_mcp import LangGraphMCPToolProvider
from ..utils import json_safe


class _NoopDelegationManager:
    async def delegate(self, *, peer_name: str, message: str, accepted_output_modes: list[str]) -> dict[str, Any]:
        return {
            'peer_name': peer_name,
            'text': f'Delegation is not configured. peer={peer_name}; message={message}',
            'output_mode': 'text/plain',
            'payload': None,
        }


class LangGraphAgent(BaseAgent):
    def __init__(
        self,
        config: LangGraphAgentConfig,
        *,
        llm: Any | None = None,
        tool_provider: Any | None = None,
        memory_manager: Any | None = None,
        context_assembler: Any | None = None,
        skill_manager: Any | None = None,
        delegation_manager: Any | None = None,
        store: Any | None = None,
        checkpointer: Any | None = None,
    ) -> None:
        self.config = config
        self._owns_llm = llm is None
        self.llm = llm or ChatModelFactory.create(config)
        self.tool_provider = tool_provider or LangGraphMCPToolProvider(config.mcp_servers)
        self.store = store or self._build_store()
        self._checkpointer_input = checkpointer
        self._checkpointer_cm: Any | None = None
        self.checkpointer: Any | None = None
        self.memory_manager = memory_manager or LangGraphStoreMemoryManager(self.store, config.memory, llm=self.llm)
        self.context_assembler = context_assembler or DefaultContextAssembler()
        self.skill_manager = skill_manager or FilesystemSkillManager(config.skills)
        self.delegation_manager = delegation_manager or _NoopDelegationManager()
        self.runner: GraphAgentRunner | None = None
        self.monitor = LangfuseMonitor(config)

    def _log(self, message: str) -> None:
        if self.config.verbose:
            print(f'[{self.config.name}] {message}', flush=True)

    def _build_store(self):
        if self.config.memory.store_type == 'postgres' and self.config.memory.store_conn_string:
            try:
                from langgraph.store.postgres import PostgresStore
                store = PostgresStore.from_conn_string(self.config.memory.store_conn_string)
                setup = getattr(store, 'setup', None)
                if callable(setup):
                    setup()
                return store
            except Exception as exc:  # pragma: no cover
                raise RuntimeError('Failed to initialize PostgresStore for long-term memory.') from exc
        try:
            from langgraph.store.memory import InMemoryStore
            return InMemoryStore()
        except Exception:
            # Tiny fallback for static analysis environments without LangGraph installed.
            class _FallbackStore:
                def __init__(self) -> None:
                    self.data: dict[tuple[str, ...], dict[str, Any]] = {}
                def put(self, namespace, key, value, *args, **kwargs):
                    self.data.setdefault(tuple(namespace), {})[key] = value
                def search(self, namespace, query=None, limit=10):
                    class Item:
                        def __init__(self, value): self.value = value
                    return [Item(v) for v in list(self.data.get(tuple(namespace), {}).values())[:limit]]
            return _FallbackStore()

    def _build_checkpointer(self):
        if self.config.memory.checkpointer_type == 'sqlite':
            path = self.config.memory.sqlite_path or str(Path(self.config.working_dir) / '.langgraph' / f'{self.config.name}.sqlite')
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            # Prefer the async checkpointer for async graph execution. Some
            # versions expose it as an async context manager, so connect() enters it.
            try:
                from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
                return AsyncSqliteSaver.from_conn_string(path)
            except Exception:
                try:
                    from langgraph.checkpoint.sqlite import SqliteSaver
                    return SqliteSaver.from_conn_string(path)
                except Exception:
                    pass
        try:
            from langgraph.checkpoint.memory import InMemorySaver
            return InMemorySaver()
        except Exception:
            return None

    async def _enter_checkpointer(self, candidate: Any) -> Any:
        if candidate is None:
            return None
        if hasattr(candidate, '__aenter__') and hasattr(candidate, '__aexit__'):
            self._checkpointer_cm = candidate
            return await candidate.__aenter__()
        return candidate

    async def connect(self) -> None:
        await self.tool_provider.connect()
        await self.skill_manager.load()
        if self.checkpointer is None:
            candidate = self._checkpointer_input if self._checkpointer_input is not None else self._build_checkpointer()
            self.checkpointer = await self._enter_checkpointer(candidate)
        self.runner = GraphAgentRunner(
            agent=self,
            llm=self.llm,
            tool_provider=self.tool_provider,
            memory_manager=self.memory_manager,
            context_assembler=self.context_assembler,
            skill_manager=self.skill_manager,
            delegation_manager=self.delegation_manager,
            checkpointer=self.checkpointer,
            store=self.store,
        )

    async def close(self) -> None:
        await self.tool_provider.close()
        if self._checkpointer_cm is not None:
            try:
                await self._checkpointer_cm.__aexit__(None, None, None)
            finally:
                self._checkpointer_cm = None
                self.checkpointer = None
        if self._owns_llm and hasattr(self.llm, 'close'):
            await self.llm.close()

    async def run(self, query: str, *, thread_id: str | None = None, user_id: str | None = None, accepted_output_modes: list[str] | None = None, max_steps: int | None = None) -> str:
        result = await self.run_detailed(query, thread_id=thread_id, user_id=user_id, accepted_output_modes=accepted_output_modes, max_steps=max_steps)
        return result.answer

    async def run_detailed(self, query: str, *, thread_id: str | None = None, user_id: str | None = None, accepted_output_modes: list[str] | None = None, max_steps: int | None = None) -> GraphRunResult:
        if self.runner is None:
            await self.connect()
        assert self.runner is not None
        effective_user = user_id or self.config.memory.user_id
        effective_thread = thread_id or f'{self.config.name}-thread'
        state = {
            'query': query,
            'thread_id': effective_thread,
            'user_id': effective_user,
            'accepted_output_modes': accepted_output_modes or list(self.config.output_modes),
            'max_steps': max_steps or self.config.max_steps,
        }
        # Keep graph invocation config checkpoint-safe: no callback handlers here.
        graph_config: dict[str, Any] = {
            'run_name': f"{self.config.observability.trace_name_prefix}:{self.config.name}",
            'tags': ['langgraph-agent', self.config.name],
            'metadata': {
                'agent': self.config.name,
                'model': self.config.model,
                'user_id': effective_user,
                'session_id': effective_thread,
            },
        }
        if self.checkpointer is not None:
            graph_config.setdefault('configurable', {})
            graph_config['configurable']['thread_id'] = effective_thread
        self._log(f'run start | query={query}')
        result = await self.runner.graph.ainvoke(state, config=graph_config)
        final = result.get('final_response') or {'output_mode': 'text/plain', 'text': ''}
        output_mode = final.get('output_mode', 'text/plain')
        payload = None
        if output_mode == 'application/json' and (final.get('data_json') or final.get('data') is not None):
            try:
                payload = json.loads(final['data_json']) if final.get('data_json') else final.get('data')
            except Exception:
                payload = final.get('data_json')
            answer = json.dumps(payload, ensure_ascii=False, default=str)
        else:
            answer = final.get('text') or ''
        selected_skill = (result.get('selected_skill') or {}).get('name') if isinstance(result.get('selected_skill'), dict) else None
        return GraphRunResult(
            answer=answer,
            output_mode=output_mode,
            payload=payload,
            selected_skill=selected_skill,
            thread_id=result.get('thread_id') or effective_thread,
            trace=list(json_safe(result.get('trace') or [])),
            history=list(json_safe(result.get('history') or [])),
        )

    async def __aenter__(self) -> 'LangGraphAgent':
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()
