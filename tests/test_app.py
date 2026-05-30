from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi import Request as FastAPIRequest
from httpx import ASGITransport, AsyncClient
from httpx import Request as HTTPXRequest

from app import create_app
from manager import ProxySession
from protocols import ActiveModelProtocol
from schemas import (
    AppConfig,
    ModelConfig,
    ModelResource,
    ModelSource,
    ModelStatus,
    ServiceStatus,
    SpeculativeConfig,
)
from tests.helpers import make_app_config, make_model


class FakeManager:
    def __init__(self, model_resource: ModelResource) -> None:
        self._model_resource = model_resource

    def status(self) -> ServiceStatus:
        return ServiceStatus(
            healthy=True,
            api_prefix="/api",
            config_path="config.json",
            active_model=self._model_resource.name,
            active_runtime="rocm",
            models=[self._model_resource],
        )

    def model_statuses(self) -> list[ModelStatus]:
        return [self._model_resource.status]

    def model_resources(self) -> list[ModelResource]:
        return [self._model_resource]

    def model_resource(self, name: str) -> ModelResource:
        assert name == self._model_resource.name
        return self._model_resource

    async def load(self, name: str, runtime: str) -> ModelResource:
        assert name == self._model_resource.name
        assert runtime == "rocm"
        return self._model_resource

    async def unload(self, name: str | None = None) -> ModelResource:
        assert name in {None, self._model_resource.name}
        return self._model_resource

    def active_model(self) -> ActiveModelProtocol | None:
        return SimpleNamespace(
            config=SimpleNamespace(name=self._model_resource.name),
        )

    def find_model_for_name(self, model_name: str) -> str | None:
        if model_name == self._model_resource.name:
            return self._model_resource.name
        return None

    def models(self) -> list[ModelConfig]:
        return [
            make_model(
                self._model_resource.name,
                Path("/models/dummy.gguf"),
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
        mock_app = FastAPI()

        @mock_app.post("/v1/chat/completions")
        async def chat_completions(request: FastAPIRequest) -> dict[str, object]:
            return {
                "ok": "yes",
                "path": path,
                "query": dict(request.query_params),
            }

        client = AsyncClient(
            transport=ASGITransport(app=mock_app),
            base_url="http://mock_upstream",
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
    model_resource = ModelResource(
        name="primary",
        config=make_model(
            "primary",
            Path("/models/dummy.gguf"),
            speculative=SpeculativeConfig(type="draft-mtp"),
        ),
        status=ModelStatus(
            name="primary",
            state="running",
            model="repo/model.gguf",
            speculative_type="draft-mtp",
            host="127.0.0.1",
            port=8080,
            base_url="http://127.0.0.1:8080",
            active=True,
            active_runtime="rocm",
        ),
    )
    app = create_app(
        app_config=make_app_config(api_prefix="/api", hf_cache_dir=tmp_path),
        manager=FakeManager(model_resource),
    )
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        health = await client.get("/api/health")
        status = await client.get("/api/status")
        models = await client.get("/api/models")
        model = await client.get("/api/models/primary")
        loaded = await client.post("/api/models/primary/load", json={"runtime": "rocm"})
        unloaded = await client.post("/api/models/primary/unload")
        proxied = await client.post(
            "/api/v1/chat/completions?foo=bar",
            json={"model": "primary"},
        )

    assert health.json() == {"ok": True}
    assert status.json()["active_model"] == "primary"
    assert models.json()["models"][0]["name"] == "primary"
    assert model.json()["name"] == "primary"
    assert loaded.json()["name"] == "primary"
    assert unloaded.json()["name"] == "primary"
    assert proxied.json() == {
        "ok": "yes",
        "path": "/v1/chat/completions",
        "query": {"foo": "bar"},
    }
