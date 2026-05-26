from __future__ import annotations

import builtins
from pathlib import Path
from typing import Any

import docker.errors
import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from config import RuntimeSettings
from manager import BackendManager, ContainerRuntime, RouterRuntime
from schemas import (
    AppConfig,
    BackendFamilyConfig,
    ModelPresetConfig,
    ModelSource,
    SpeculativeConfig,
)


class FakeHF:
    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir or Path("/mocked/cache")

    def resolve_source(self, source: ModelSource) -> Path:
        if source.local_path is not None:
            return source.local_path
        if source.repo_id is not None and source.filename is not None:
            return self.cache_dir / source.repo_id / source.filename
        msg = "unexpected remote source"
        raise AssertionError(msg)


class FailingHF(FakeHF):
    def resolve_source(self, source: ModelSource) -> Path:
        if source.repo_id == "broken/repo":
            msg = "download failed"
            raise ValueError(msg)
        return super().resolve_source(source)


class MockImages:
    def __init__(self) -> None:
        self.downloaded_images = {"ghcr.io/ggerganov/llama.cpp:server-rocm"}

    def get(self, name: str) -> str:
        if name in self.downloaded_images:
            return name
        raise docker.errors.ImageNotFound(f"Image {name} not found")

    def pull(self, name: str) -> str:
        self.downloaded_images.add(name)
        return name


class MockContainer:
    def __init__(
        self,
        name: str = "mock-container",
        status: str = "running",
        labels: dict[str, str] | None = None,
    ) -> None:
        self.id = "mock-id-123"
        self.name = name
        self.status = status
        self.stopped = False
        self.removed = False
        self.labels = labels or {"managed-by": "inference-server"}

    def reload(self) -> None:
        pass

    def stop(self, timeout: int = 15) -> None:
        self.stopped = True
        self.status = "exited"

    def remove(self, force: bool = False) -> None:
        self.removed = True

    def logs(self, tail: int = 500) -> bytes:
        return b"Hello from mock container\n"


class MockContainers:
    def __init__(self) -> None:
        self.active_containers: dict[str, MockContainer] = {}

    def get(self, name: str) -> MockContainer:
        if name in self.active_containers:
            return self.active_containers[name]
        raise Exception("Container not found")

    def run(self, **kwargs: Any) -> MockContainer:
        name = kwargs.get("name", "mock-container")
        labels = kwargs.get("labels")
        container = MockContainer(name=name, labels=labels)
        self.active_containers[name] = container
        return container

    def list(
        self, filters: dict[str, str] | None = None, all: bool = False
    ) -> builtins.list[MockContainer]:
        containers = list(self.active_containers.values())
        if filters and "label" in filters:
            label_filter = filters["label"]
            if "=" in label_filter:
                key, val = label_filter.split("=", 1)
                containers = [c for c in containers if c.labels.get(key) == val]
            else:
                containers = [c for c in containers if label_filter in c.labels]
        return containers


class MockDockerClient:
    def __init__(self) -> None:
        self.containers = MockContainers()
        self.images = MockImages()


class MockUpstreamApp:
    def __init__(self) -> None:
        self.loaded_models: list[str] = []
        self.unloaded_models: list[str] = []

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        path = scope["path"]
        if scope["type"] == "http":
            if path == "/models/load":
                self.loaded_models.append("loaded")
                await send(
                    {
                        "type": "http.response.start",
                        "status": 200,
                        "headers": [(b"content-type", b"application/json")],
                    }
                )
                await send(
                    {
                        "type": "http.response.body",
                        "body": b'{"success": true}',
                    }
                )
            elif path == "/models/unload":
                self.unloaded_models.append("unloaded")
                await send(
                    {
                        "type": "http.response.start",
                        "status": 200,
                        "headers": [(b"content-type", b"application/json")],
                    }
                )
                await send(
                    {
                        "type": "http.response.body",
                        "body": b'{"success": true}',
                    }
                )
            elif path == "/v1/models":
                await send(
                    {
                        "type": "http.response.start",
                        "status": 200,
                        "headers": [(b"content-type", b"application/json")],
                    }
                )
                await send(
                    {
                        "type": "http.response.body",
                        "body": b'{"models": []}',
                    }
                )


def mock_client_factory(base_url: str, app: MockUpstreamApp) -> AsyncClient:
    return AsyncClient(
        transport=ASGITransport(app=app),
        base_url=base_url,
    )


@pytest.mark.asyncio
async def test_backend_manager_load_and_unload_router(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mock_client = MockDockerClient()
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    # Use a dummy commit hash mock
    async def fake_resolve_commit_hash(*args: Any, **kwargs: Any) -> str:
        return "mock_commit_123"

    monkeypatch.setattr(
        "manager.BackendManager._resolve_commit_hash", fake_resolve_commit_hash
    )

    runtime = RuntimeSettings(
        config_path=tmp_path / "config.json",
    )
    app_config = AppConfig(
        runtime_mode="router",
        backend_families={
            "rocm": BackendFamilyConfig(
                docker_image="ghcr.io/ggerganov/llama.cpp:server-rocm",
                devices=["/dev/kfd", "/dev/dri"],
            )
        },
        models=[
            ModelPresetConfig(
                name="primary",
                backend_family="rocm",
                model=ModelSource(local_path=tmp_path / "model.gguf"),
                speculative=SpeculativeConfig(type="draft-mtp"),
            )
        ],
    )

    upstream_app = MockUpstreamApp()
    manager = BackendManager(
        runtime=runtime,
        app_config=app_config,
        hf=FakeHF(tmp_path),
        proxy_client_factory=lambda base_url: mock_client_factory(
            base_url, upstream_app
        ),
    )

    async def fake_to_thread(func: Any, /, *args: object, **kwargs: object) -> object:
        return func(*args, **kwargs)

    monkeypatch.setattr("manager.asyncio.to_thread", fake_to_thread)

    status = await manager.load("primary")

    assert status.active is True
    assert status.state == "running"
    assert status.container_id == "mock-id-123"
    assert status.model_path == str(tmp_path / "model.gguf")
    assert manager.status().active_backend == "primary"

    # Verify that it generated a preset file and launched a RouterRuntime
    assert type(manager._active_runtime) is RouterRuntime
    ini_file = tmp_path / "presets" / "rocm.ini"
    assert ini_file.exists()
    assert "[primary]" in ini_file.read_text()

    # Unload
    unloaded = await manager.unload("primary")
    assert unloaded is not None
    assert unloaded.active is False
    assert unloaded.state == "stopped"
    assert unloaded.container_id is None
    assert unloaded.model_path is None
    assert len(upstream_app.unloaded_models) == 1

    # Verify that backend statuses also report it as stopped
    statuses = manager.backend_statuses()
    assert len(statuses) == 1
    assert statuses[0].state == "stopped"
    assert statuses[0].active is False
    assert statuses[0].container_id is None
    assert statuses[0].model_path is None


@pytest.mark.asyncio
async def test_backend_manager_container_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mock_client = MockDockerClient()
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    runtime = RuntimeSettings(
        config_path=tmp_path / "config.json",
    )
    app_config = AppConfig(
        runtime_mode="container",
        backend_families={
            "rocm": BackendFamilyConfig(
                docker_image="ghcr.io/ggerganov/llama.cpp:server-rocm",
                devices=["/dev/kfd", "/dev/dri"],
            )
        },
        models=[
            ModelPresetConfig(
                name="primary",
                backend_family="rocm",
                model=ModelSource(local_path=tmp_path / "model.gguf"),
            )
        ],
    )

    upstream_app = MockUpstreamApp()
    manager = BackendManager(
        runtime=runtime,
        app_config=app_config,
        hf=FakeHF(tmp_path),
        proxy_client_factory=lambda base_url: mock_client_factory(
            base_url, upstream_app
        ),
    )

    async def fake_to_thread(func: Any, /, *args: object, **kwargs: object) -> object:
        return func(*args, **kwargs)

    monkeypatch.setattr("manager.asyncio.to_thread", fake_to_thread)

    status = await manager.load("primary")

    assert status.active is True
    assert type(manager._active_runtime) is ContainerRuntime
    container = mock_client.containers.get("inference-server-primary")
    assert container.stopped is False


@pytest.mark.asyncio
async def test_backend_manager_keeps_active_backend_when_new_model_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mock_client = MockDockerClient()
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    async def fake_resolve_commit_hash(*args: Any, **kwargs: Any) -> str:
        return "mock_commit_123"

    monkeypatch.setattr(
        "manager.BackendManager._resolve_commit_hash", fake_resolve_commit_hash
    )

    runtime = RuntimeSettings(
        config_path=tmp_path / "config.json",
    )
    app_config = AppConfig(
        runtime_mode="router",
        backend_families={
            "rocm": BackendFamilyConfig(
                docker_image="ghcr.io/ggerganov/llama.cpp:server-rocm",
                devices=["/dev/kfd", "/dev/dri"],
            )
        },
        models=[
            ModelPresetConfig(
                name="primary",
                backend_family="rocm",
                model=ModelSource(local_path=tmp_path / "model.gguf"),
            ),
            ModelPresetConfig(
                name="secondary",
                backend_family="rocm",
                model=ModelSource(repo_id="broken/repo", filename="model.gguf"),
            ),
        ],
    )

    upstream_app = MockUpstreamApp()
    manager = BackendManager(
        runtime=runtime,
        app_config=app_config,
        hf=FailingHF(tmp_path),
        proxy_client_factory=lambda base_url: mock_client_factory(
            base_url, upstream_app
        ),
    )

    async def fake_to_thread(func: Any, /, *args: object, **kwargs: object) -> object:
        return func(*args, **kwargs)

    monkeypatch.setattr("manager.asyncio.to_thread", fake_to_thread)

    await manager.load("primary")
    with pytest.raises(ValueError):
        await manager.load("secondary")

    svc_status = manager.status()
    assert svc_status.active_backend is None
    
    # primary should be stopped
    assert svc_status.backends[0].name == "primary"
    assert svc_status.backends[0].state == "stopped"
    assert svc_status.backends[0].active is False

    # secondary should be in error state
    assert svc_status.backends[1].name == "secondary"
    assert svc_status.backends[1].state == "error"
    assert svc_status.backends[1].active is False

    container = mock_client.containers.get("inference-server-router-rocm")
    assert container.stopped is False


@pytest.mark.asyncio
async def test_global_cleanup_stops_only_labeled_containers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mock_client = MockDockerClient()
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    runtime = RuntimeSettings(
        config_path=tmp_path / "config.json",
    )
    app_config = AppConfig(
        runtime_mode="container",
        backend_families={
            "rocm": BackendFamilyConfig(
                docker_image="ghcr.io/ggerganov/llama.cpp:server-rocm",
                devices=["/dev/kfd", "/dev/dri"],
            )
        },
        models=[
            ModelPresetConfig(
                name="backend1",
                backend_family="rocm",
                model=ModelSource(local_path=tmp_path / "m1.gguf"),
            )
        ],
    )

    upstream_app = MockUpstreamApp()
    manager = BackendManager(
        runtime=runtime,
        app_config=app_config,
        hf=FakeHF(tmp_path),
        proxy_client_factory=lambda base_url: mock_client_factory(
            base_url, upstream_app
        ),
    )

    async def fake_to_thread(func: Any, /, *args: object, **kwargs: object) -> object:
        return func(*args, **kwargs)

    monkeypatch.setattr("manager.asyncio.to_thread", fake_to_thread)

    await manager.load("backend1")

    unmanaged_container = MockContainer(
        name="some-other-database", labels={"managed-by": "other-app"}
    )
    mock_client.containers.active_containers["some-other-database"] = (
        unmanaged_container
    )

    await manager.cleanup()

    managed = mock_client.containers.get("inference-server-backend1")
    assert managed.stopped is True
    assert unmanaged_container.stopped is False


@pytest.mark.asyncio
async def test_open_proxy_session_closes_client_on_send_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mock_client = MockDockerClient()
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    runtime = RuntimeSettings(
        config_path=tmp_path / "config.json",
    )
    app_config = AppConfig(
        runtime_mode="container",
        backend_families={
            "rocm": BackendFamilyConfig(
                docker_image="ghcr.io/ggerganov/llama.cpp:server-rocm",
                devices=["/dev/kfd", "/dev/dri"],
            )
        },
        models=[
            ModelPresetConfig(
                name="primary",
                backend_family="rocm",
                model=ModelSource(local_path=tmp_path / "model.gguf"),
            )
        ],
    )

    class FailingProxyClient(AsyncClient):
        def __init__(self, base_url: str) -> None:
            super().__init__(base_url=base_url)
            self.closed = False

        async def send(
            self,
            request: Any,
            *,
            stream: bool = False,
            auth: object = None,
            follow_redirects: object = None,
        ) -> Any:
            raise httpx.ConnectError("refused", request=request)

        async def aclose(self) -> None:
            self.closed = True
            await super().aclose()

    clients: list[FailingProxyClient] = []

    def factory(base_url: str) -> FailingProxyClient:
        client = FailingProxyClient(base_url)
        clients.append(client)
        return client

    manager = BackendManager(
        runtime=runtime,
        app_config=app_config,
        hf=FakeHF(tmp_path),
        proxy_client_factory=factory,
    )
    # mock starting state
    manager._active_model_name = "primary"
    preset = manager._get_preset("primary")
    manager._active_runtime = ContainerRuntime(
        preset, app_config.backend_families["rocm"], manager
    )

    request = httpx.Request("POST", "http://placeholder/v1/chat/completions")

    with pytest.raises(httpx.ConnectError):
        await manager.open_proxy_session("/v1/chat/completions", request)

    assert len(clients) == 1
    assert clients[0].closed is True
