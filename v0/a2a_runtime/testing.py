from __future__ import annotations

from typing import Any

import httpx


class URLRouterTransport(httpx.AsyncBaseTransport):
    def __init__(self, mapping: dict[str, Any]):
        self._clients = {}
        for base_url, app in mapping.items():
            transport = httpx.ASGITransport(app=app)
            self._clients[base_url.rstrip('/')] = httpx.AsyncClient(transport=transport, base_url=base_url.rstrip('/'))

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        base = f"{request.url.scheme}://{request.url.host}"
        if request.url.port:
            base += f":{request.url.port}"
        client = self._clients.get(base)
        if client is None:
            raise RuntimeError(f'No ASGI app mapped for {base}')
        relative = request.url.raw_path.decode()
        if request.url.query:
            relative += '?' + request.url.query.decode()
        proxied = client.build_request(request.method, relative, headers=request.headers, content=request.content)
        return await client.send(proxied)

    async def aclose(self) -> None:
        for client in self._clients.values():
            await client.aclose()
