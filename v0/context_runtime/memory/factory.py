from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_runtime import BaseLLM, LLMClientConfig, OpenAICompatibleLLM, require_llm_config

from .extractors import FailureClassifier, LLMTurnFactExtractor
from .query_resolver import MemoryQueryResolver
from .repository import MemoryRepository
from .retriever import LLMRetriever
from .store import SQLiteMemoryStore
from .summarizers import LLMSummaryGenerator


@dataclass(slots=True)
class MemoryRuntimeComponents:
    llm: BaseLLM
    store: Any
    repository: Any
    extractor: Any
    failure_classifier: Any
    summarizer: Any
    retriever: Any
    resolver: Any
    owns_llm: bool = False


class MemoryRuntimeFactory:
    """Composes the default production memory runtime from swappable components."""

    @staticmethod
    def create(*, config: Any, llm: BaseLLM | None = None) -> MemoryRuntimeComponents:
        root = Path(config.root_dir)
        root.mkdir(parents=True, exist_ok=True)

        owns_llm = llm is None
        runtime_llm = llm
        if runtime_llm is None:
            client_cfg = require_llm_config(api_base=config.api_base, api_key=config.api_key, model=config.model)
            runtime_llm = OpenAICompatibleLLM(
                LLMClientConfig(
                    api_base=client_cfg.api_base,
                    api_key=client_cfg.api_key,
                    model=client_cfg.model,
                    temperature=config.temperature,
                    connect_timeout_seconds=config.connect_timeout_seconds,
                    request_timeout_seconds=config.request_timeout_seconds,
                    max_retries=client_cfg.max_retries,
                    max_output_tokens=client_cfg.max_output_tokens,
                    enable_langfuse=client_cfg.enable_langfuse,
                    langfuse_public_key=client_cfg.langfuse_public_key,
                    langfuse_secret_key=client_cfg.langfuse_secret_key,
                    langfuse_base_url=client_cfg.langfuse_base_url,
                    langfuse_session_id=client_cfg.langfuse_session_id,
                    langfuse_user_id=client_cfg.langfuse_user_id,
                    langfuse_tags=client_cfg.langfuse_tags,
                    use_responses_api=client_cfg.use_responses_api,
                )
            )

        store = SQLiteMemoryStore(root / 'memory.sqlite3')
        repository = MemoryRepository(root / 'repository', namespace=config.namespace, auto_git_commit=config.auto_git_commit)
        extractor = LLMTurnFactExtractor(runtime_llm)
        failure_classifier = FailureClassifier(runtime_llm)
        summarizer = LLMSummaryGenerator(runtime_llm)
        retriever = LLMRetriever(store, runtime_llm, candidate_limit=config.retrieval_candidate_limit)
        resolver = MemoryQueryResolver(store, retriever, runtime_llm)
        return MemoryRuntimeComponents(
            llm=runtime_llm,
            store=store,
            repository=repository,
            extractor=extractor,
            failure_classifier=failure_classifier,
            summarizer=summarizer,
            retriever=retriever,
            resolver=resolver,
            owns_llm=owns_llm,
        )
