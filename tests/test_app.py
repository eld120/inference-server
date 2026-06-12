import asyncio
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
        self.load_calls: list[tuple[str, str]] = []

    def status(self) -> ServiceStatus:
        return ServiceStatus(
            healthy=True,
            api_prefix="/api",
            config_path="config.json",
            active_model=self._model_resource.name,
            active_runtime="rocm",
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
        self.load_calls.append((name, runtime))
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

    async def handle_runtime_communication_failure(self, exc: Exception) -> None:
        self.last_runtime_failure = str(exc)

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


class LoadingManager(FakeManager):
    def status(self) -> ServiceStatus:
        return ServiceStatus(
            healthy=True,
            api_prefix="/api",
            config_path="config.json",
            active_model=None,
            active_runtime="rocm",
        )

    def active_model(self) -> ActiveModelProtocol | None:
        return None


class WarmRuntimeNoModelManager(FakeManager):
    def status(self) -> ServiceStatus:
        return ServiceStatus(
            healthy=True,
            api_prefix="/api",
            config_path="config.json",
            active_model=None,
            active_runtime="rocm",
            active_container_id="warm-container-id",
        )

    def model_statuses(self) -> list[ModelStatus]:
        status = self._model_resource.status.model_copy(
            update={
                "active": False,
                "active_runtime": None,
                "container_id": None,
                "state": "stopped",
            }
        )
        return [status]

    def model_resource(self, name: str) -> ModelResource:
        assert name == self._model_resource.name
        return self._model_resource.model_copy(update={"status": self.model_statuses()[0]})

    def active_model(self) -> ActiveModelProtocol | None:
        return None


class CancellationSafeLoadingManager(FakeManager):
    def __init__(self, model_resource: ModelResource) -> None:
        super().__init__(model_resource)
        self.started = asyncio.Event()
        self.finished = asyncio.Event()
        self.completed = False

    async def load(self, name: str, runtime: str) -> ModelResource:
        assert name == self._model_resource.name
        assert runtime == "rocm"
        self.started.set()
        await self.finished.wait()
        self.completed = True
        return self._model_resource


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
    assert "models" not in status.json()
    assert models.json()["models"][0]["name"] == "primary"
    assert model.json()["name"] == "primary"
    assert loaded.json()["name"] == "primary"
    assert unloaded.json()["name"] == "primary"
    assert proxied.json() == {
        "ok": "yes",
        "path": "/v1/chat/completions",
        "query": {"foo": "bar"},
    }


@pytest.mark.asyncio
async def test_load_continues_after_request_cancellation(tmp_path: Path) -> None:
    model_resource = ModelResource(
        name="primary",
        config=make_model("primary", Path("/models/dummy.gguf")),
        status=ModelStatus(
            name="primary",
            state="starting",
            model="repo/model.gguf",
            host="127.0.0.1",
            port=8080,
            base_url="http://127.0.0.1:8080",
            active=False,
            active_runtime="rocm",
        ),
    )
    manager = CancellationSafeLoadingManager(model_resource)
    app = create_app(
        app_config=make_app_config(api_prefix="/api", hf_cache_dir=tmp_path),
        manager=manager,
    )

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        request_task = asyncio.create_task(
            client.post("/api/models/primary/load", json={"runtime": "rocm"})
        )
        await manager.started.wait()

        request_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request_task

        assert manager.completed is False

        manager.finished.set()
        for _ in range(20):
            if manager.completed:
                break
            await asyncio.sleep(0)

    assert manager.completed is True


@pytest.mark.asyncio
async def test_app_rejects_completions_while_model_is_loading(tmp_path: Path) -> None:
    model_resource = ModelResource(
        name="primary",
        config=make_model("primary", Path("/models/dummy.gguf")),
        status=ModelStatus(
            name="primary",
            state="starting",
            model="repo/model.gguf",
            host="127.0.0.1",
            port=8080,
            base_url="http://127.0.0.1:8080",
            active=False,
            active_runtime="rocm",
        ),
    )
    app = create_app(
        app_config=make_app_config(api_prefix="/api", hf_cache_dir=tmp_path),
        manager=LoadingManager(model_resource),
    )

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/v1/chat/completions",
            json={"model": "primary"},
        )

    assert response.status_code == 503
    assert response.json() == {"detail": "Model 'primary' is still loading."}


@pytest.mark.asyncio
async def test_app_rejects_completions_when_runtime_is_warm_but_no_model_loaded(
    tmp_path: Path,
) -> None:
    model_resource = ModelResource(
        name="primary",
        config=make_model("primary", Path("/models/dummy.gguf")),
        status=ModelStatus(
            name="primary",
            state="stopped",
            model="repo/model.gguf",
            host="127.0.0.1",
            port=8080,
            base_url="http://127.0.0.1:8080",
            active=False,
        ),
    )
    manager = WarmRuntimeNoModelManager(model_resource)
    app = create_app(
        app_config=make_app_config(api_prefix="/api", hf_cache_dir=tmp_path),
        manager=manager,
    )

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/v1/chat/completions",
            json={"model": "primary"},
        )

    assert response.status_code == 200
    assert response.json() == {"ok": "yes", "path": "/v1/chat/completions", "query": {}}
    assert manager.load_calls == [("primary", "rocm")]


class IdempotentUnloadManager(FakeManager):
    def __init__(self, model_resource: ModelResource) -> None:
        super().__init__(model_resource)
        self.unload_calls: list[str | None] = []

    async def unload(self, name: str | None = None) -> ModelResource:
        self.unload_calls.append(name)
        return self._model_resource.model_copy(
            update={
                "status": self._model_resource.status.model_copy(
                    update={
                        "state": "stopped",
                        "active": False,
                        "active_runtime": None,
                        "container_id": None,
                        "model_path": None,
                        "draft_model_path": None,
                    }
                )
            }
        )


@pytest.mark.asyncio
async def test_unload_route_is_idempotent_and_status_is_compact(tmp_path: Path) -> None:
    model_resource = ModelResource(
        name="primary",
        config=make_model("primary", Path("/models/dummy.gguf")),
        status=ModelStatus(
            name="primary",
            state="running",
            model="repo/model.gguf",
            host="127.0.0.1",
            port=8080,
            base_url="http://127.0.0.1:8080",
            active=True,
            active_runtime="rocm",
            container_id="container-1",
        ),
    )
    app = create_app(
        app_config=make_app_config(api_prefix="/api", hf_cache_dir=tmp_path),
        manager=IdempotentUnloadManager(model_resource),
    )

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        first = await client.post("/api/models/primary/unload")
        second = await client.post("/api/models/primary/unload")
        status = await client.get("/api/status")

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["status"]["state"] == "stopped"
    assert second.json()["status"]["state"] == "stopped"
    assert "models" not in status.json()
