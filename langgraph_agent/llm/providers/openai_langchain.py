from __future__ import annotations

import json
from typing import Any, TypeVar

from pydantic import BaseModel

from ..base import BaseStructuredLLM
from ..parsing import extract_first_json_object, normalize_ai_message_content

ModelT = TypeVar('ModelT', bound=BaseModel)


class LangChainStructuredLLM(BaseStructuredLLM):
    def __init__(self, model: Any) -> None:
        self._model = model

    async def ainvoke_text(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int | None = None,
        invoke_config: dict[str, Any] | None = None,
    ) -> str:
        kwargs: dict[str, Any] = {}
        if max_output_tokens is not None:
            kwargs['max_output_tokens'] = max_output_tokens
        response = await self._model.ainvoke(messages, config=invoke_config, **kwargs)
        return normalize_ai_message_content(getattr(response, 'content', response))

    async def _structured_once(
        self,
        messages: list[dict[str, str]],
        *,
        model_type: type[ModelT],
        invoke_config: dict[str, Any] | None,
    ) -> ModelT | None:
        if not hasattr(self._model, 'with_structured_output'):
            return None
        methods: list[dict[str, Any]] = [
            {},
            {'method': 'json_schema'},
            {'method': 'function_calling'},
            {'strict': True},
        ]
        for options in methods:
            try:
                structured_model = self._model.with_structured_output(model_type, **options)
                result = await structured_model.ainvoke(messages, config=invoke_config)
                if isinstance(result, model_type):
                    return result
                return model_type.model_validate(result)
            except Exception:
                continue
        return None

    async def _repair_to_schema(
        self,
        *,
        broken_output: str,
        schema: dict[str, Any],
        schema_name: str,
        invoke_config: dict[str, Any] | None,
        max_output_tokens: int | None,
    ) -> dict[str, Any] | None:
        repair_messages = [
            {
                'role': 'system',
                'content': (
                    'Convert the provided content into valid JSON only. '
                    f'The JSON must validate against schema {schema_name}. '
                    'Do not include markdown fences or explanations.\n\n'
                    + json.dumps(schema, ensure_ascii=False, indent=2)
                ),
            },
            {'role': 'user', 'content': broken_output or ''},
        ]
        text = await self.ainvoke_text(
            repair_messages,
            max_output_tokens=max_output_tokens,
            invoke_config=invoke_config,
        )
        return extract_first_json_object(text)

    async def ainvoke_json_model(
        self,
        messages: list[dict[str, str]],
        *,
        model_type: type[ModelT],
        schema_name: str,
        max_output_tokens: int | None = None,
        invoke_config: dict[str, Any] | None = None,
    ) -> ModelT:
        structured = await self._structured_once(
            messages,
            model_type=model_type,
            invoke_config=invoke_config,
        )
        if structured is not None:
            return structured

        schema = model_type.model_json_schema()
        contract = (
            'Return JSON only. Do not include markdown fences or extra commentary. '
            f'The JSON must validate against this Pydantic schema named {schema_name}:\n'
            + json.dumps(schema, ensure_ascii=False, indent=2)
        )
        fallback_messages = list(messages) + [{'role': 'system', 'content': contract}]
        text = await self.ainvoke_text(
            fallback_messages,
            max_output_tokens=max_output_tokens,
            invoke_config=invoke_config,
        )
        payload = extract_first_json_object(text)
        if payload is None:
            payload = await self._repair_to_schema(
                broken_output=text,
                schema=schema,
                schema_name=schema_name,
                invoke_config=invoke_config,
                max_output_tokens=max_output_tokens,
            )
        if payload is None:
            raise RuntimeError(
                f'Failed to parse JSON for schema {schema_name}. Raw output: {text}'
            )
        return model_type.model_validate(payload)

    async def close(self) -> None:
        close = getattr(self._model, 'aclose', None)
        if callable(close):
            await close()
