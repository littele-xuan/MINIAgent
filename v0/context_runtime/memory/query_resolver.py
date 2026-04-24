from __future__ import annotations

import json

from pydantic import BaseModel, ConfigDict

from llm_runtime import BaseLLM
from prompt_runtime import render_prompt

from .base import BaseMemoryQueryResolver
from .retriever import LLMRetriever
from .store import SQLiteMemoryStore


class MemoryAnswerEnvelope(BaseModel):
    model_config = ConfigDict(extra='forbid')

    answer: str | None = None


class MemoryQueryResolver(BaseMemoryQueryResolver):
    def __init__(self, store: SQLiteMemoryStore, retriever: LLMRetriever, llm: BaseLLM) -> None:
        self.store = store
        self.retriever = retriever
        self.llm = llm

    async def answer(self, *, namespace: str, session_id: str, query: str, warnings: list[str] | None = None) -> str | None:
        warnings = warnings or []
        retrieved = await self.retriever.retrieve(namespace=namespace, session_id=session_id, query=query, limit=8)
        envelope = await self.llm.chat_json_model(
            messages=[
                {'role': 'system', 'content': render_prompt('memory/query_answer_system')},
                {
                    'role': 'user',
                    'content': json.dumps(
                        {
                            'query': query,
                            'warnings': warnings,
                            'retrieved_memories': [
                                {
                                    'source_type': item.source_type,
                                    'source_id': item.source_id,
                                    'score': item.score,
                                    'text': item.text,
                                    'metadata': item.metadata,
                                }
                                for item in retrieved
                            ],
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                },
            ],
            model_type=MemoryAnswerEnvelope,
            schema_name='memory_query_answer',
            temperature=0.0,
            max_output_tokens=500,
        )
        return envelope.answer.strip() if envelope.answer else None
