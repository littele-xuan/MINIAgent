from __future__ import annotations

from .base import BaseStructuredLLM
from .providers.openai_langchain import LangChainStructuredLLM
from ..config.models import LangGraphAgentConfig


class ChatModelFactory:
    @staticmethod
    def create(config: LangGraphAgentConfig) -> BaseStructuredLLM:
        if config.provider != 'openai-compatible':
            raise ValueError(f'Unsupported provider: {config.provider}')
        try:
            from langchain_openai import ChatOpenAI
        except Exception as exc:  # pragma: no cover
            raise RuntimeError('langchain-openai is required for LangGraphAgent.') from exc

        model = ChatOpenAI(
            model=config.model,
            api_key=config.api_key,
            base_url=config.api_base,
            temperature=config.temperature,
            timeout=config.request_timeout_seconds,
            max_retries=config.max_retries,
        )
        return LangChainStructuredLLM(model)
