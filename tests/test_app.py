from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi import Request as FastAPIRequest
from httpx import ASGITransport, AsyncClient
from httpx import Request as HTTPXRequest

from app import create_app
from manager import ProxySession
from protocols import ActiveBackendProtocol
from schemas import (
    AppConfig,
    BackendStatus,
    ModelPresetConfig,
    ModelSource,
    ServiceStatus,
)


class FakeManager:
    def __init__(self, backend: BackendStatus) -> None:
        self._backend = backend

    def status(self) -> ServiceStatus:
        return ServiceStatus(
            healthy=True,
            api_prefix="/api",
            active_backend=self._backend.name,
            backends=[self._backend],
        )

    def backend_statuses(self) -> list[BackendStatus]:
        return [self._backend]

    async def load(self, name: str) -> BackendStatus:
        assert name == self._backend.name
        return self._backend

    async def unload(self, name: str | None = None) -> BackendStatus:
        assert name in {None, self._backend.name}
        return self._backend

    def active_backend(self) -> ActiveBackendProtocol | None:
        return SimpleNamespace(
            config=SimpleNamespace(name=self._backend.name),
        )

    def find_backend_for_model(self, model_name: str) -> str | None:
        if model_name == self._backend.name:
            return self._backend.name
        return None

    def backends(self) -> list[ModelPresetConfig]:
        return [
            ModelPresetConfig(
                name=self._backend.name,
                model=ModelSource(local_path=Path("/models/dummy.gguf")),
                backend_family="rocm",
            )
        ]

    async def get_logs(self, name: str) -> list[str]:
        return []

    async def cleanup(self) -> None:
        pass

    async def open_proxy_session(
        self,
        path: str,
        request: HTTPXRequest,
    ) -> ProxySession:
        backend = FastAPI()

        @backend.post("/v1/chat/completions")
        async def chat_completions(request: FastAPIRequest) -> dict[str, object]:
            return {
                "ok": "yes",
                "path": path,
                "query": dict(request.query_params),
            }

        client = AsyncClient(
            transport=ASGITransport(app=backend),
            base_url="http://backend",
        )
        upstream_request = client.build_request(
            "POST",
            path,
            content=request.content,
            params=request.url.params,
        )
        response = await client.send(upstream_request, stream=True)
        return ProxySession(client=client, response=response)


@pytest.mark.asyncio
async def test_app_routes_and_proxy(tmp_path: Path) -> None:
    backend = BackendStatus(
        name="primary",
        state="running",
        model="repo/model.gguf",
        speculative_type="draft-mtp",
        host="127.0.0.1",
        port=8080,
        base_url="http://127.0.0.1:8080",
        active=True,
    )
    app = create_app(
        app_config=AppConfig(models=[], api_prefix="/api", hf_cache_dir=tmp_path),
        manager=FakeManager(backend),
    )
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        health = await client.get("/api/healthz")
        status = await client.get("/api/status")
        proxied = await client.post(
            "/api/v1/chat/completions?foo=bar",
            json={"model": "primary"},
        )

    assert health.json() == {"ok": True}
    assert status.json()["active_backend"] == "primary"
    assert proxied.json() == {
        "ok": "yes",
        "path": "/v1/chat/completions",
        "query": {"foo": "bar"},
    }
