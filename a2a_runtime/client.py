from __future__ import annotations

import uuid
from typing import Any

import httpx

from .errors import InvalidAgentResponseError
from .models import (
    A2A_PROTOCOL_VERSION,
    AgentCard,
    JsonRpcRequest,
    JsonRpcResponse,
    Message,
    Part,
    Role,
    SendMessageConfiguration,
    SendMessageRequest,
    SendMessageResponse,
)


class A2AJsonRpcRemoteError(Exception):
    def __init__(self, code: int, message: str, data: list[dict[str, Any]] | None = None):
        super().__init__(f'A2A JSON-RPC error {code}: {message}')
        self.code = code
        self.message = message
        self.data = data or []


class A2AClient:
    def __init__(self, *, timeout: float = 20.0, client: httpx.AsyncClient | None = None):
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = client
        self._owns_client = client is None

    async def __aenter__(self) -> 'A2AClient':
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
            self._owns_client = True
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def close(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
            self._owns_client = True
        return self._client

    async def fetch_agent_card(self, agent_card_url: str) -> AgentCard:
        resp = await self.client.get(agent_card_url, headers={'Accept': 'application/json', 'A2A-Version': A2A_PROTOCOL_VERSION})
        resp.raise_for_status()
        return AgentCard.model_validate(resp.json())

    async def send_text(self, agent_card_url: str, text: str, *, configuration: SendMessageConfiguration | None = None, context_id: str | None = None, task_id: str | None = None) -> SendMessageResponse:
        request = SendMessageRequest(
            message=Message(
                messageId=str(uuid.uuid4()),
                contextId=context_id,
                taskId=task_id,
                role=Role.ROLE_USER,
                parts=[Part(text=text, mediaType='text/plain')],
            ),
            configuration=configuration or SendMessageConfiguration(),
        )
        return await self.send_message(agent_card_url, request)

    async def send_message(self, agent_card_url: str, request: SendMessageRequest) -> SendMessageResponse:
        card = await self.fetch_agent_card(agent_card_url)
        for interface in card.supported_interfaces:
            if interface.protocol_binding == 'HTTP+JSON' and interface.protocol_version == A2A_PROTOCOL_VERSION:
                return await self._send_http_json(interface.url, request)
        for interface in card.supported_interfaces:
            if interface.protocol_binding == 'JSONRPC' and interface.protocol_version == A2A_PROTOCOL_VERSION:
                result = await self.rpc_call(interface.url, 'SendMessage', request.model_dump(by_alias=True, exclude_none=True))
                return SendMessageResponse.model_validate(result)
        raise RuntimeError('No compatible A2A interface found in Agent Card')

    async def _send_http_json(self, base_url: str, request: SendMessageRequest) -> SendMessageResponse:
        resp = await self.client.post(
            f"{base_url.rstrip('/')}/message:send",
            headers={'Content-Type': 'application/json', 'Accept': 'application/json', 'A2A-Version': A2A_PROTOCOL_VERSION},
            json=request.model_dump(by_alias=True, exclude_none=True),
        )
        resp.raise_for_status()
        return SendMessageResponse.model_validate(resp.json())

    async def get_task(self, base_or_rpc_url: str, task_id: str, *, history_length: int | None = None) -> dict[str, Any]:
        if base_or_rpc_url.endswith('/rpc'):
            params = {'id': task_id}
            if history_length is not None:
                params['historyLength'] = history_length
            return await self.rpc_call(base_or_rpc_url, 'GetTask', params)
        resp = await self.client.get(
            f"{base_or_rpc_url.rstrip('/')}/tasks/{task_id}",
            headers={'Accept': 'application/json', 'A2A-Version': A2A_PROTOCOL_VERSION},
            params={'historyLength': history_length} if history_length is not None else None,
        )
        resp.raise_for_status()
        return resp.json()

    async def list_tasks(self, base_or_rpc_url: str, **params: Any) -> dict[str, Any]:
        if base_or_rpc_url.endswith('/rpc'):
            return await self.rpc_call(base_or_rpc_url, 'ListTasks', params)
        resp = await self.client.get(
            f"{base_or_rpc_url.rstrip('/')}/tasks",
            headers={'Accept': 'application/json', 'A2A-Version': A2A_PROTOCOL_VERSION},
            params=params,
        )
        resp.raise_for_status()
        return resp.json()

    async def cancel_task(self, base_or_rpc_url: str, task_id: str) -> dict[str, Any]:
        if base_or_rpc_url.endswith('/rpc'):
            return await self.rpc_call(base_or_rpc_url, 'CancelTask', {'id': task_id})
        resp = await self.client.post(
            f"{base_or_rpc_url.rstrip('/')}/tasks/{task_id}:cancel",
            headers={'Accept': 'application/json', 'A2A-Version': A2A_PROTOCOL_VERSION},
        )
        resp.raise_for_status()
        return resp.json()

    async def rpc_call(self, rpc_url: str, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        request = JsonRpcRequest(id=str(uuid.uuid4()), method=method, params=params or {})
        resp = await self.client.post(
            rpc_url,
            headers={'Content-Type': 'application/json', 'Accept': 'application/json', 'A2A-Version': A2A_PROTOCOL_VERSION},
            json=request.model_dump(by_alias=True, exclude_none=True),
        )
        try:
            body = JsonRpcResponse.model_validate(resp.json())
        except Exception as exc:
            raise InvalidAgentResponseError(str(exc)) from exc
        if body.error is not None:
            raise A2AJsonRpcRemoteError(body.error.code, body.error.message, body.error.data)
        if body.result is None:
            raise A2AJsonRpcRemoteError(-32006, 'Agent returned neither result nor error.')
        return body.result
