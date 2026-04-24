from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from ..core.outcome import ToolCall
from ..runtime.config import AgentConfig
from ..runtime.env import load_dotenv_if_present
from .types import LLMResponse


def _is_openai_host(base_url: str | None) -> bool:
    if not base_url:
        return True
    host = (urlparse(base_url).netloc or "").lower()
    return host == "api.openai.com" or host.endswith(".openai.com")


@dataclass(slots=True)
class OpenAIResponsesClient:
    """OpenAI SDK client for OpenAI-compatible endpoints.

    Despite the historical class name, this client supports both APIs:
    - chat_completions: preferred default for MCP_API_BASE gateways
    - responses: retained for native OpenAI Responses API experiments

    Langfuse is enabled by default when LANGFUSE_PUBLIC_KEY and
    LANGFUSE_SECRET_KEY are present.  The Langfuse OpenAI wrapper is used as a
    drop-in replacement; if it cannot be imported, the official OpenAI SDK is
    used without failing local execution.
    """

    config: AgentConfig
    client: Any
    langfuse_enabled: bool = False
    client_name: str = "openai"
    api_mode: str = "chat_completions"

    @classmethod
    def create(cls, config: AgentConfig) -> "OpenAIResponsesClient":
        load_dotenv_if_present()
        api_key = config.llm_api_key() or ""
        base_url = config.llm_base_url()
        model = config.llm_model()
        if not api_key:
            raise RuntimeError(
                "MCP_API_KEY is missing. Set MCP_API_KEY in the environment or in the project .env file. "
                "OPENAI_API_KEY is accepted only as a compatibility fallback. Run `python agent.py --doctor` for diagnostics."
            )
        if not model:
            raise RuntimeError(
                "MCP_MODEL is missing. Set MCP_MODEL in the environment or in the project .env file. "
                "OPENAI_MODEL is accepted only as a compatibility fallback."
            )

        kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": config.llm.request_timeout_seconds,
            "max_retries": config.llm.max_retries,
        }
        if base_url:
            kwargs["base_url"] = base_url

        client = None
        langfuse_enabled = False
        client_name = "openai"
        if config.langfuse_configured():
            os.environ.setdefault(config.observability.base_url_env, config.observability.default_base_url)
            client = _try_create_langfuse_client(kwargs)
            if client is not None:
                langfuse_enabled = True
                client_name = "langfuse.openai"
        if client is None:
            try:
                from openai import OpenAI  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("Install openai to use MINIAgent: pip install openai") from exc
            client = OpenAI(**kwargs)

        api_mode = (config.llm.api_mode or "chat_completions").strip().lower()
        if api_mode not in {"chat_completions", "responses", "auto"}:
            api_mode = "chat_completions"
        if api_mode == "auto":
            # Native OpenAI hosts can use Responses first; custom MCP gateways
            # are most likely to support /v1/chat/completions.
            api_mode = "responses" if _is_openai_host(base_url) else "chat_completions"
        return cls(
            config=config,
            client=client,
            langfuse_enabled=langfuse_enabled,
            client_name=client_name,
            api_mode=api_mode,
        )

    def create_response(
        self,
        *,
        instructions: str,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> LLMResponse:
        if self.api_mode == "responses":
            return self._create_responses_response(instructions=instructions, input_items=input_items, tools=tools, metadata=metadata)
        return self._create_chat_response(instructions=instructions, input_items=input_items, tools=tools, metadata=metadata)

    def _create_responses_response(
        self,
        *,
        instructions: str,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.config.llm_model(),
            "instructions": instructions,
            "input": input_items,
            "tools": tools,
            "store": bool(self.config.llm.store),
        }
        if self.config.llm.temperature is not None:
            kwargs["temperature"] = self.config.llm.temperature
        if self.config.llm.max_output_tokens:
            kwargs["max_output_tokens"] = self.config.llm.max_output_tokens
        if metadata:
            kwargs["metadata"] = metadata
        try:
            response = self.client.responses.create(**kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"Responses API call failed using model={self.config.llm_model()!r}. "
                "Check MCP_API_BASE, MCP_API_KEY, MCP_MODEL, and whether the endpoint supports /v1/responses. "
                "For MCP/OpenAI-compatible gateways, set llm.api_mode: chat_completions. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc
        return self._normalize_responses_response(response)

    def _create_chat_response(
        self,
        *,
        instructions: str,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> LLMResponse:
        messages = _input_items_to_chat_messages(instructions, input_items)
        chat_tools = _responses_tools_to_chat_tools(tools)
        kwargs: dict[str, Any] = {
            "model": self.config.llm_model(),
            "messages": messages,
        }
        if chat_tools:
            kwargs["tools"] = chat_tools
            kwargs["tool_choice"] = "auto"
        if self.config.llm.temperature is not None:
            kwargs["temperature"] = self.config.llm.temperature
        if self.config.llm.max_output_tokens:
            # Newer OpenAI models prefer max_completion_tokens; many compatible
            # gateways still accept max_tokens.  Try modern first and fall back.
            kwargs["max_completion_tokens"] = self.config.llm.max_output_tokens
        if metadata and self.langfuse_enabled:
            kwargs["metadata"] = metadata
        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as first_exc:
            if "max_completion_tokens" in kwargs:
                kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
                try:
                    response = self.client.chat.completions.create(**kwargs)
                except Exception as second_exc:
                    raise RuntimeError(
                        f"Chat Completions API call failed using model={self.config.llm_model()!r}. "
                        "Check MCP_API_BASE, MCP_API_KEY, MCP_MODEL, and endpoint compatibility. "
                        f"Original error: {type(second_exc).__name__}: {second_exc}"
                    ) from second_exc
            else:
                raise RuntimeError(
                    f"Chat Completions API call failed using model={self.config.llm_model()!r}. "
                    "Check MCP_API_BASE, MCP_API_KEY, MCP_MODEL, and endpoint compatibility. "
                    f"Original error: {type(first_exc).__name__}: {first_exc}"
                ) from first_exc
        return self._normalize_chat_response(response)

    def _normalize_responses_response(self, response: Any) -> LLMResponse:
        response_dict = _to_dict(response)
        raw_output = response_dict.get("output") or []
        output_items = [_to_dict(item) for item in raw_output]
        tool_calls: list[ToolCall] = []
        for item in output_items:
            if item.get("type") == "function_call":
                raw_args = item.get("arguments") or "{}"
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
                except Exception:
                    args = {"_raw_arguments": raw_args}
                call_id = str(item.get("call_id") or item.get("id") or "")
                tool_calls.append(
                    ToolCall(
                        id=str(item.get("id") or call_id),
                        call_id=call_id,
                        name=str(item.get("name") or ""),
                        arguments=args,
                        raw_arguments=raw_args if isinstance(raw_args, str) else json.dumps(raw_args, ensure_ascii=False, default=str),
                    )
                )
        text = response_dict.get("output_text") or _extract_text(output_items)
        usage = _to_dict(response_dict.get("usage") or {})
        return LLMResponse(text=text, output_items=output_items, tool_calls=tool_calls, usage=usage, raw_id=response_dict.get("id"))

    def _normalize_chat_response(self, response: Any) -> LLMResponse:
        response_dict = _to_dict(response)
        choices = response_dict.get("choices") or []
        message = _to_dict((choices[0] or {}).get("message")) if choices else {}
        text = _extract_chat_message_text(message)
        raw_tool_calls = message.get("tool_calls") or []
        tool_calls: list[ToolCall] = []
        output_items: list[dict[str, Any]] = []
        if text:
            output_items.append(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text}],
                }
            )
        for raw in raw_tool_calls:
            item = _to_dict(raw)
            function = _to_dict(item.get("function") or {})
            raw_args = function.get("arguments") or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            except Exception:
                args = {"_raw_arguments": raw_args}
            call_id = str(item.get("id") or "")
            name = str(function.get("name") or item.get("name") or "")
            output_items.append(
                {
                    "type": "function_call",
                    "id": call_id,
                    "call_id": call_id,
                    "name": name,
                    "arguments": raw_args if isinstance(raw_args, str) else json.dumps(raw_args, ensure_ascii=False, default=str),
                }
            )
            tool_calls.append(
                ToolCall(
                    id=call_id,
                    call_id=call_id,
                    name=name,
                    arguments=args,
                    raw_arguments=raw_args if isinstance(raw_args, str) else json.dumps(raw_args, ensure_ascii=False, default=str),
                )
            )
        usage = _to_dict(response_dict.get("usage") or {})
        return LLMResponse(text=str(text or ""), output_items=output_items, tool_calls=tool_calls, usage=usage, raw_id=response_dict.get("id"))


def _try_create_langfuse_client(kwargs: dict[str, Any]) -> Any | None:
    try:
        from langfuse.openai import OpenAI as LangfuseOpenAI  # type: ignore

        return LangfuseOpenAI(**kwargs)
    except Exception:
        pass
    try:
        from langfuse.openai import openai as langfuse_openai  # type: ignore

        return langfuse_openai.OpenAI(**kwargs)
    except Exception:
        return None


def _to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return {"value": value}


def _extract_text(output_items: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for item in output_items:
        if item.get("type") != "message":
            continue
        for c in item.get("content") or []:
            cdict = c if isinstance(c, dict) else _to_dict(c)
            if cdict.get("type") in {"output_text", "text"}:
                parts.append(str(cdict.get("text") or ""))
    return "".join(parts).strip()


def _responses_tools_to_chat_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for tool in tools:
        if not tool:
            continue
        if "function" in tool:
            converted.append(tool)
            continue
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": tool.get("name"),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
                },
            }
        )
    return converted


def _extract_chat_message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    text = _message_content_to_text(content).strip()
    if text:
        return text
    for key in ("output_text", "text", "answer", "refusal"):
        value = message.get(key)
        if value:
            return _message_content_to_text(value).strip()
    return ""


def _input_items_to_chat_messages(instructions: str, input_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if instructions:
        messages.append({"role": "system", "content": instructions})
    pending_tool_calls: list[dict[str, Any]] = []

    def flush_pending_tool_calls() -> None:
        nonlocal pending_tool_calls
        if pending_tool_calls:
            messages.append({"role": "assistant", "content": None, "tool_calls": pending_tool_calls})
            pending_tool_calls = []

    for item in input_items:
        item = _to_dict(item)
        item_type = item.get("type")
        role = item.get("role")
        if item_type == "function_call":
            raw_args = item.get("arguments") or "{}"
            if not isinstance(raw_args, str):
                raw_args = json.dumps(raw_args, ensure_ascii=False, default=str)
            call_id = str(item.get("call_id") or item.get("id") or "")
            pending_tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": str(item.get("name") or ""), "arguments": raw_args},
                }
            )
            continue
        if item_type == "function_call_output":
            flush_pending_tool_calls()
            messages.append({"role": "tool", "tool_call_id": str(item.get("call_id") or ""), "content": str(item.get("output") or "")})
            continue
        flush_pending_tool_calls()
        if item_type == "message":
            messages.append({"role": role or "assistant", "content": _message_content_to_text(item.get("content"))})
        elif role in {"user", "assistant", "system", "developer"}:
            chat_role = "system" if role == "developer" else role
            messages.append({"role": chat_role, "content": _message_content_to_text(item.get("content"))})
        else:
            messages.append({"role": "user", "content": json.dumps(item, ensure_ascii=False, default=str)})
    flush_pending_tool_calls()
    return messages


def _message_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            pdict = _to_dict(part)
            if "text" in pdict:
                parts.append(str(pdict.get("text") or ""))
            elif "content" in pdict:
                parts.append(str(pdict.get("content") or ""))
            else:
                parts.append(json.dumps(pdict, ensure_ascii=False, default=str))
        return "".join(parts)
    return str(content)
