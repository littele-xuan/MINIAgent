from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

import httpx

from .models import Message, Part, Role, SendMessageConfiguration, SendMessageRequest, SendMessageResponse


class A2AClient:
    """Compatibility wrapper.

    Preferred discovery path uses the official a2a-sdk card resolver when available.
    Message sending remains protocol-level HTTP so the framework stays stable across
    SDK releases while still producing/consuming standard A2A payloads.
    """

    def __init__(self, *, timeout: float = 30.0, client: httpx.AsyncClient | None = None) -> None:
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = client is None

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def fetch_agent_card(self, agent_card_url: str) -> dict[str, Any]:
        base = agent_card_url.rstrip('/')
        if base.endswith('/.well-known/agent-card.json'):
            base = base[: -len('/.well-known/agent-card.json')]
        try:
            from a2a.client import A2ACardResolver
            from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH

            resolver = A2ACardResolver(httpx_client=self._client, base_url=base)
            card = await resolver.get_agent_card()
            dump = getattr(card, 'model_dump', None)
            if callable(dump):
                return dump(mode='json', by_alias=True, exclude_none=True)
            return dict(card)
        except Exception:
            response = await self._client.get(base.rstrip('/') + '/.well-known/agent-card.json')
            response.raise_for_status()
            return response.json()

    async def send_text(
        self,
        agent_card_url: str,
        text: str,
        *,
        configuration: SendMessageConfiguration | None = None,
    ) -> SendMessageResponse:
        card = await self.fetch_agent_card(agent_card_url)
        base_url = self._extract_message_base(card, agent_card_url)
        request = SendMessageRequest(
            message=Message(
                messageId=str(uuid4()),
                role=Role.ROLE_USER,
                parts=[Part(text=text, mediaType='text/plain')],
            ),
            configuration=configuration,
        )
        endpoint = base_url.rstrip('/') + '/message:send'
        response = await self._client.post(
            endpoint,
            json=request.model_dump(by_alias=True, exclude_none=True),
            headers={'A2A-Version': '1.0'},
        )
        response.raise_for_status()
        return SendMessageResponse.model_validate(response.json())

    def _extract_message_base(self, card: dict[str, Any], fallback_url: str) -> str:
        if isinstance(card.get('url'), str) and card['url']:
            return str(card['url']).rstrip('/')
        interfaces = card.get('supportedInterfaces') or []
        if isinstance(interfaces, list):
            for item in interfaces:
                if isinstance(item, dict) and item.get('url'):
                    return str(item['url']).rstrip('/')
        fallback = fallback_url.rstrip('/')
        if fallback.endswith('/.well-known/agent-card.json'):
            fallback = fallback[: -len('/.well-known/agent-card.json')]
        return fallback.rstrip('/') + '/a2a/v1'
