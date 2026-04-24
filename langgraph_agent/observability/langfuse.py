from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config.models import LangGraphAgentConfig


@dataclass(slots=True)
class LangfuseMonitor:
    config: LangGraphAgentConfig

    def enabled(self) -> bool:
        obs = self.config.observability
        return bool(obs.provider == 'langfuse' and obs.enabled and obs.public_key and obs.secret_key)

    def _build_handler(self):
        if not self.enabled():
            return None
        try:
            from langfuse.langchain import CallbackHandler
            return CallbackHandler()
        except Exception:
            return None

    def runnable_config(
        self,
        *,
        run_name: str,
        user_id: str | None,
        session_id: str | None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # Langfuse v3/v4 compatible pattern: callback handler plus trace
        # attributes propagated through metadata fields. This config is for LLM
        # calls only; do not pass it to a checkpointed LangGraph graph invoke.
        md = dict(metadata or {})
        if user_id:
            md['langfuse_user_id'] = user_id
        if session_id:
            md['langfuse_session_id'] = session_id
        if tags:
            md['langfuse_tags'] = list(tags)
        config: dict[str, Any] = {'run_name': run_name, 'tags': list(tags or []), 'metadata': md}
        handler = self._build_handler()
        if handler is not None:
            config['callbacks'] = [handler]
        return config
