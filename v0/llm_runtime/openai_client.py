from __future__ import annotations

import inspect
import json
import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, ValidationError

from .base import BaseLLM
from .schema_utils import build_openai_responses_json_schema


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {'1', 'true', 'yes', 'on'}


def _default_use_responses_api(api_base: str) -> bool:
    parsed = urlparse(api_base)
    host = (parsed.netloc or '').lower()
    if not host:
        return False
    return host == 'api.openai.com' or host.endswith('.openai.com')


def _default_enable_langfuse_wrapper(api_base: str) -> bool:
    return _default_use_responses_api(api_base)


def _error_text(exc: Exception) -> str:
    return str(exc).lower()


def _mentions_unsupported_parameter(exc: Exception, parameter: str) -> bool:
    text = _error_text(exc)
    return 'unsupported parameter' in text and parameter.lower() in text


@dataclass(slots=True)
class LLMClientConfig:
    api_base: str
    api_key: str
    model: str
    temperature: float = 0.0
    connect_timeout_seconds: float = 20.0
    request_timeout_seconds: float = 90.0
    max_retries: int = 0
    max_output_tokens: int = 1600
    enable_langfuse: bool = True
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_base_url: str | None = None
    langfuse_session_id: str | None = None
    langfuse_user_id: str | None = None
    langfuse_tags: tuple[str, ...] = ()
    use_responses_api: bool = True


class OpenAICompatibleLLM(BaseLLM):
    """Async OpenAI-compatible client wrapper.

    Default behavior:
    - prefer the Responses API for GPT-5+/MCP-native structured execution
    - prefer Langfuse's OpenAI wrapper when available
    - keep transport concerns here and leave planning/memory semantics to runtime models
    """

    def __init__(self, config: LLMClientConfig, *, client: Any | None = None) -> None:
        self.config = config
        self._langfuse_enabled = False
        self._client_kind = 'openai'
        self._responses_api_disabled = False
        self._responses_api_disabled_reason: str | None = None
        if client is None:
            try:
                import httpx
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("OpenAICompatibleLLM requires the 'httpx' package") from exc

            timeout = httpx.Timeout(timeout=config.request_timeout_seconds, connect=config.connect_timeout_seconds)

            if config.enable_langfuse:
                try:
                    from langfuse.openai import AsyncOpenAI as LangfuseAsyncOpenAI

                    client = LangfuseAsyncOpenAI(
                        base_url=config.api_base,
                        api_key=config.api_key,
                        timeout=timeout,
                        max_retries=config.max_retries,
                    )
                    self._langfuse_enabled = True
                    self._client_kind = 'langfuse'
                except Exception:
                    client = None
                    self._langfuse_enabled = False

            if client is None:
                try:
                    from openai import AsyncOpenAI
                except Exception as exc:  # pragma: no cover
                    raise RuntimeError("OpenAICompatibleLLM requires the 'openai' package") from exc
                client = AsyncOpenAI(
                    base_url=config.api_base,
                    api_key=config.api_key,
                    timeout=timeout,
                    max_retries=config.max_retries,
                )
                self._client_kind = 'openai'

        self._client = client
        self._prefer_responses_api = bool(config.use_responses_api and hasattr(self._client, 'responses'))
        self._responses_supports_max_output_tokens = True

    async def close(self) -> None:
        close = getattr(self._client, 'close', None)
        if close is not None:
            result = close()
            if inspect.isawaitable(result):
                await result
        if self._langfuse_enabled:
            try:
                from langfuse import get_client

                get_client().flush()
            except Exception:
                pass

    def _trace_kwargs(self, *, name: str) -> dict[str, Any]:
        if not self._langfuse_enabled:
            return {}
        metadata: dict[str, Any] = {
            'schema_name': name,
            'llm_provider': 'openai-compatible',
            'client_kind': self._client_kind,
        }
        if self.config.langfuse_session_id:
            metadata['langfuse_session_id'] = self.config.langfuse_session_id
        if self.config.langfuse_user_id:
            metadata['langfuse_user_id'] = self.config.langfuse_user_id
        if self.config.langfuse_tags:
            metadata['langfuse_tags'] = list(self.config.langfuse_tags)
        return {'name': name, 'metadata': metadata}

    def _messages_to_input(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        for message in messages:
            converted.append(
                {
                    'role': message['role'],
                    'content': message['content'],
                }
            )
        return converted

    def _extract_output_text(self, response: Any) -> str:
        direct = getattr(response, 'output_text', None)
        if isinstance(direct, str):
            return direct.strip()

        output = getattr(response, 'output', None)
        if isinstance(output, list):
            chunks: list[str] = []
            for item in output:
                if isinstance(item, dict):
                    content = item.get('content')
                    if isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict):
                                text = part.get('text') or part.get('value')
                                if isinstance(text, str):
                                    chunks.append(text)
                    elif isinstance(content, str):
                        chunks.append(content)
                    elif item.get('type') == 'message' and isinstance(item.get('text'), str):
                        chunks.append(item['text'])
                else:
                    content = getattr(item, 'content', None)
                    if isinstance(content, list):
                        for part in content:
                            text = getattr(part, 'text', None) or getattr(part, 'value', None)
                            if isinstance(text, str):
                                chunks.append(text)
                    else:
                        text = getattr(item, 'text', None)
                        if isinstance(text, str):
                            chunks.append(text)
            return ''.join(chunks).strip()

        if hasattr(response, 'model_dump_json'):
            try:
                dumped = json.loads(response.model_dump_json())
                if isinstance(dumped, dict) and isinstance(dumped.get('output_text'), str):
                    return dumped['output_text'].strip()
            except Exception:
                pass
        return ''

    def _api_modes(self) -> list[str]:
        modes: list[str] = []
        has_responses = hasattr(self._client, 'responses') and not self._responses_api_disabled
        has_chat_completions = hasattr(getattr(self._client, 'chat', None), 'completions')
        if self._prefer_responses_api and has_responses:
            modes.append('responses')
        if has_chat_completions:
            modes.append('chat_completions')
        if not modes and has_responses:
            modes.append('responses')
        return modes

    def _disable_responses_api(self, exc: Exception) -> None:
        if self._responses_api_disabled:
            return
        self._responses_api_disabled = True
        self._responses_api_disabled_reason = str(exc)

    async def _responses_create(self, *, messages: list[dict[str, Any]], name: str, text_config: dict[str, Any] | None, temperature: float | None, max_output_tokens: int | None) -> Any:
        kwargs: dict[str, Any] = {
            'model': self.config.model,
            'input': self._messages_to_input(messages),
            'temperature': self.config.temperature if temperature is None else temperature,
            'timeout': self.config.request_timeout_seconds,
            **self._trace_kwargs(name=name),
        }
        if self._responses_supports_max_output_tokens and (max_output_tokens is not None or self.config.max_output_tokens):
            kwargs['max_output_tokens'] = max_output_tokens or self.config.max_output_tokens
        if text_config is not None:
            kwargs['text'] = text_config
        try:
            resp = self._client.responses.create(**kwargs)
            if inspect.isawaitable(resp):
                resp = await resp
            return resp
        except Exception as exc:
            if 'max_output_tokens' in kwargs and _mentions_unsupported_parameter(exc, 'max_output_tokens'):
                self._responses_supports_max_output_tokens = False
                retry_kwargs = dict(kwargs)
                retry_kwargs.pop('max_output_tokens', None)
                resp = self._client.responses.create(**retry_kwargs)
                if inspect.isawaitable(resp):
                    resp = await resp
                return resp
            raise

    async def _chat_completions_create(
        self,
        *,
        messages: list[dict[str, Any]],
        name: str,
        response_format: dict[str, Any] | None,
        temperature: float | None,
        max_output_tokens: int | None,
    ) -> Any:
        resp = self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            response_format=response_format,
            temperature=self.config.temperature if temperature is None else temperature,
            max_tokens=max_output_tokens or self.config.max_output_tokens,
            timeout=self.config.request_timeout_seconds,
            **self._trace_kwargs(name=name),
        )
        if inspect.isawaitable(resp):
            resp = await resp
        return resp

    async def chat_text(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> str:
        last_error: Exception | None = None
        for mode in self._api_modes():
            try:
                if mode == 'responses':
                    resp = await self._responses_create(
                        messages=messages,
                        name='chat_text',
                        text_config={'format': {'type': 'text'}},
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                    )
                    return self._extract_output_text(resp)

                resp = await self._chat_completions_create(
                    messages=messages,
                    name='chat_text',
                    response_format=None,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                )
                return (resp.choices[0].message.content or '').strip()
            except Exception as exc:
                if mode == 'responses':
                    self._disable_responses_api(exc)
                last_error = exc
                continue
        raise RuntimeError(f'LLM text generation failed across all available API modes: {last_error}')

    async def chat_json(
        self,
        *,
        messages: list[dict[str, Any]],
        schema: dict[str, Any],
        schema_name: str,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        repair_attempts: int = 2,
    ) -> dict[str, Any]:
        strict_schema = build_openai_responses_json_schema(schema)
        text_format = {
            'type': 'json_schema',
            'name': schema_name,
            'schema': strict_schema,
            'strict': True,
        }
        last_error: Exception | None = None
        working_messages = list(messages)
        raw = '{}'
        for attempt in range(repair_attempts + 1):
            for mode in self._api_modes():
                try:
                    if mode == 'responses':
                        resp = await self._responses_create(
                            messages=working_messages,
                            name=schema_name,
                            text_config={'format': text_format},
                            temperature=temperature,
                            max_output_tokens=max_output_tokens,
                        )
                        raw = self._extract_output_text(resp) or '{}'
                    else:
                        resp = await self._chat_completions_create(
                            messages=working_messages,
                            name=schema_name,
                            response_format={
                                'type': 'json_schema',
                                'json_schema': {
                                    'name': schema_name,
                                    'strict': True,
                                    'schema': strict_schema,
                                },
                            },
                            temperature=temperature,
                            max_output_tokens=max_output_tokens,
                        )
                        raw = (resp.choices[0].message.content or '').strip() or '{}'
                    parsed = json.loads(raw)
                    if not isinstance(parsed, dict):
                        raise ValueError('structured response must be a JSON object')
                    return parsed
                except Exception as exc:
                    if mode == 'responses':
                        self._disable_responses_api(exc)
                    last_error = exc
                    continue
            if attempt >= repair_attempts:
                break
            working_messages = working_messages + [
                {
                    'role': 'user',
                    'content': (
                        'Your previous reply or API attempt was invalid for the required strict JSON schema. '
                        'Return exactly one corrected JSON object that matches the schema. '
                        'Use canonical keys only. No markdown. No prose.\n'
                        f'validation_error={last_error}\n'
                        f'previous_reply={raw}'
                    ),
                }
            ]

        # Last-resort JSON mode fallback.
        for attempt in range(repair_attempts + 1):
            for mode in self._api_modes():
                try:
                    if mode == 'responses':
                        resp = await self._responses_create(
                            messages=working_messages,
                            name=f'{schema_name}_json_fallback',
                            text_config={'format': {'type': 'json_object'}},
                            temperature=temperature,
                            max_output_tokens=max_output_tokens,
                        )
                        raw = self._extract_output_text(resp) or '{}'
                    else:
                        resp = await self._chat_completions_create(
                            messages=working_messages,
                            name=f'{schema_name}_repair',
                            response_format={'type': 'json_object'},
                            temperature=temperature,
                            max_output_tokens=max_output_tokens,
                        )
                        raw = (resp.choices[0].message.content or '').strip() or '{}'
                    parsed = json.loads(raw)
                    if not isinstance(parsed, dict):
                        raise ValueError('json_object mode did not return an object')
                    return parsed
                except Exception as exc:
                    if mode == 'responses':
                        self._disable_responses_api(exc)
                    last_error = exc
                    continue
            if attempt >= repair_attempts:
                break
            working_messages = working_messages + [
                {
                    'role': 'user',
                    'content': (
                        'Return valid JSON only. Match the target schema exactly. '
                        f'validation_error={last_error}\nprevious_reply={raw}'
                    ),
                }
            ]

        raise RuntimeError(f'LLM JSON generation failed: {last_error}; raw={raw}; schema_name={schema_name}; schema={strict_schema}')

    async def chat_json_model(
        self,
        *,
        messages: list[dict[str, Any]],
        model_type: type[BaseModel],
        schema_name: str,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        repair_attempts: int = 2,
    ) -> BaseModel:
        working_messages = list(messages)
        payload: dict[str, Any] = {}
        last_error: ValidationError | None = None
        for attempt in range(repair_attempts + 1):
            payload = await self.chat_json(
                messages=working_messages,
                schema=model_type.model_json_schema(),
                schema_name=schema_name,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                repair_attempts=max(0, repair_attempts - attempt),
            )
            try:
                return model_type.model_validate(payload)
            except ValidationError as exc:  # pragma: no cover
                last_error = exc
                if attempt >= repair_attempts:
                    break
                working_messages = working_messages + [
                    {
                        'role': 'user',
                        'content': (
                            'Your previous JSON object failed Pydantic validation for the required schema. '
                            'Return exactly one corrected JSON object. No markdown. No prose.\n'
                            f'validation_error={exc}\n'
                            f'previous_reply={json.dumps(payload, ensure_ascii=False)}'
                        ),
                    }
                ]
        raise RuntimeError(f'LLM returned JSON that failed model validation: {last_error}; payload={payload}') from last_error


def require_llm_config(*, api_base: str | None = None, api_key: str | None = None, model: str | None = None) -> LLMClientConfig:
    resolved_base = api_base or os.getenv('MCP_API_BASE') or os.getenv('OPENAI_BASE_URL') or 'https://api.openai.com/v1'
    resolved_key = api_key or os.getenv('MCP_API_KEY') or os.getenv('OPENAI_API_KEY') or ''
    resolved_model = model or os.getenv('MCP_MODEL') or os.getenv('OPENAI_MODEL') or ''
    if not resolved_key or not resolved_model:
        raise ValueError('Real LLM execution requires api_key and model. Set MCP_API_KEY/OPENAI_API_KEY and MCP_MODEL/OPENAI_MODEL.')
    langfuse_disabled = _env_flag('LANGFUSE_DISABLED', False)
    force_langfuse_wrapper = _env_flag('LANGFUSE_FORCE_WRAPPER', False)
    has_langfuse_credentials = bool((os.getenv('LANGFUSE_PUBLIC_KEY') or '').strip() and (os.getenv('LANGFUSE_SECRET_KEY') or '').strip())
    enable_langfuse = True
    tags_raw = os.getenv('LANGFUSE_TAGS', '')
    tags = tuple(item.strip() for item in tags_raw.split(',') if item.strip())
    force_chat_completions = _env_flag('OPENAI_FORCE_CHAT_COMPLETIONS', False)
    force_responses_api = _env_flag('OPENAI_FORCE_RESPONSES_API', False) or _env_flag('OPENAI_USE_RESPONSES', False)
    use_responses_api = force_responses_api or (not force_chat_completions and _default_use_responses_api(resolved_base))
    return LLMClientConfig(
        api_base=resolved_base,
        api_key=resolved_key,
        model=resolved_model,
        enable_langfuse=enable_langfuse,
        langfuse_public_key=os.getenv('LANGFUSE_PUBLIC_KEY') or None,
        langfuse_secret_key=os.getenv('LANGFUSE_SECRET_KEY') or None,
        langfuse_base_url=os.getenv('LANGFUSE_BASE_URL') or None,
        langfuse_session_id=os.getenv('LANGFUSE_SESSION_ID') or None,
        langfuse_user_id=os.getenv('LANGFUSE_USER_ID') or None,
        langfuse_tags=tags,
        use_responses_api=use_responses_api,
    )
