from __future__ import annotations

import asyncio
import builtins
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import docker.errors
import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from config import RuntimeSettings
from manager import ModelRuntimeManager, RuntimeContainer
from schemas import (
    AppConfig,
    ModelConfig,
    ModelSource,
    RuntimeConfig,
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
        self.downloaded_images = {"inference-server-llama-rocm:7.2.1-7e50ef7"}

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

    def exec_run(self, cmd: list[str]) -> Any:
        return SimpleNamespace(
            output=(
                b"1 Ss /app/llama-server --host 0.0.0.0 --port 8080\n"
                b"24 Sl /app/llama-server --host 127.0.0.1 --port 46951\n"
            )
        )


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
        self.readiness_probe_count = 0
        self.readiness_event: asyncio.Event | None = None

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
                        "body": b'{"object": "list", "data": []}',
                    }
                )
            elif path == "/v1/chat/completions":
                self.readiness_probe_count += 1
                if self.readiness_event is not None:
                    await self.readiness_event.wait()
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
                        "body": b'{"id":"chatcmpl-ready","choices":[{"message":{"role":"assistant","content":"ok"}}]}',
                    }
                )
            else:
                await send(
                    {
                        "type": "http.response.start",
                        "status": 404,
                        "headers": [],
                    }
                )
                await send(
                    {
                        "type": "http.response.body",
                        "body": b"Not Found",
                    }
                )


def mock_client_factory(base_url: str, app: MockUpstreamApp) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=ASGITransport(app=app),
        base_url=base_url,
    )


@pytest.mark.asyncio
async def test_model_runtime_manager_load_and_unload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mock_client = MockDockerClient()
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    async def fake_resolve_commit_hash(*args: Any, **kwargs: Any) -> str:
        return "mock_commit_123"

    monkeypatch.setattr(
        "manager.ModelRuntimeManager._resolve_commit_hash", fake_resolve_commit_hash
    )

    runtime = RuntimeSettings(
        config_path=tmp_path / "config.json",
    )
    app_config = AppConfig(
        models=[
            ModelConfig(
                name="primary",
                runtimes={
                    "rocm": RuntimeConfig(
                        docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
                        devices=["/dev/kfd", "/dev/dri"],
                        source=ModelSource(local_path=tmp_path / "model.gguf"),
                        speculative=SpeculativeConfig(type="draft-mtp"),
                    )
                },
            )
        ],
    )

    upstream_app = MockUpstreamApp()
    manager = ModelRuntimeManager(
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

    status = await manager.load("primary", "rocm")

    assert status.active is True
    assert status.state == "running"
    assert status.container_id == "mock-id-123"
    assert status.model_path == str(tmp_path / "model.gguf")
    assert manager.status().active_model == "primary"

    # Verify that it generated a llama.cpp preset INI file and launched a RuntimeContainer
    assert type(manager._active_runtime) is RuntimeContainer
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
    assert len(upstream_app.unloaded_models) == 0
    assert manager._active_runtime is None

    # Verify that model statuses also report it as stopped
    statuses = manager.model_statuses()
    assert len(statuses) == 1
    assert statuses[0].state == "stopped"
    assert statuses[0].active is False
    assert statuses[0].container_id is None
    assert statuses[0].model_path is None


@pytest.mark.asyncio
async def test_model_runtime_manager_rejects_unsupported_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mock_client = MockDockerClient()
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    runtime = RuntimeSettings(
        config_path=tmp_path / "config.json",
    )
    app_config = AppConfig(
        models=[
            ModelConfig(
                name="primary",
                runtimes={
                    "rocm": RuntimeConfig(
                        docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
                        devices=["/dev/kfd", "/dev/dri"],
                        source=ModelSource(local_path=tmp_path / "model.gguf"),
                    )
                },
            )
        ],
    )

    manager = ModelRuntimeManager(
        runtime=runtime,
        app_config=app_config,
        hf=FakeHF(tmp_path),
    )

    with pytest.raises(ValueError, match="does not support runtime"):
        await manager.load("primary", "vulkan")


@pytest.mark.asyncio
async def test_manager_load_fails_if_worker_dies_after_router_accepts_request(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mock_client = MockDockerClient()
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    runtime = RuntimeSettings(config_path=tmp_path / "config.json")
    app_config = AppConfig(
        models=[
            ModelConfig(
                name="primary",
                runtimes={
                    "rocm": RuntimeConfig(
                        docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
                        devices=["/dev/kfd", "/dev/dri"],
                        source=ModelSource(local_path=tmp_path / "model.gguf"),
                    )
                },
            )
        ],
    )

    upstream_app = MockUpstreamApp()
    manager = ModelRuntimeManager(
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

    class DefunctWorkerContainer(MockContainer):
        def exec_run(self, cmd: list[str]) -> Any:
            return SimpleNamespace(
                output=(
                    b"1 Ss /app/llama-server --host 0.0.0.0 --port 8080\n"
                    b"24 Z [llama-server] <defunct>\n"
                )
            )

    async def fake_start(self: RuntimeContainer) -> None:
        self.state = "running"
        self.container = DefunctWorkerContainer(name="inference-server-runtime-rocm")

    async def fake_stop(self: RuntimeContainer) -> None:
        self.container = None
        self.state = "stopped"

    monkeypatch.setattr(RuntimeContainer, "start", fake_start)
    monkeypatch.setattr(RuntimeContainer, "stop", fake_stop)

    with pytest.raises(RuntimeError, match="died during load"):
        await manager.load("primary", "rocm")

    assert manager._active_model_name is None
    assert manager._active_runtime is None
    assert "died during load" in (manager.status().last_error or "")


@pytest.mark.asyncio
async def test_manager_load_waits_for_readiness_probe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mock_client = MockDockerClient()
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    runtime = RuntimeSettings(config_path=tmp_path / "config.json")
    app_config = AppConfig(
        models=[
            ModelConfig(
                name="primary",
                runtimes={
                    "rocm": RuntimeConfig(
                        docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
                        devices=["/dev/kfd", "/dev/dri"],
                        source=ModelSource(local_path=tmp_path / "model.gguf"),
                    )
                },
            )
        ],
    )

    upstream_app = MockUpstreamApp()
    upstream_app.readiness_event = asyncio.Event()
    manager = ModelRuntimeManager(
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

    load_task = asyncio.create_task(manager.load("primary", "rocm"))
    await asyncio.sleep(0.05)

    assert load_task.done() is False
    assert manager._active_runtime is not None
    assert manager._active_runtime.state == "starting"
    assert manager._active_model_name is None
    assert upstream_app.readiness_probe_count >= 1

    upstream_app.readiness_event.set()
    status = await load_task

    assert status.state == "running"
    assert status.active is True
    assert manager._active_model_name == "primary"


@pytest.mark.asyncio
async def test_probe_model_readiness_uses_configured_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = RuntimeSettings(
        config_path=tmp_path / "config.json",
        model_readiness_probe_timeout_seconds=12.5,
    )
    app_config = AppConfig(models=[])
    manager = ModelRuntimeManager(
        runtime=runtime,
        app_config=app_config,
        hf=FakeHF(tmp_path),
    )
    runtime_container = RuntimeContainer(
        "rocm",
        "primary",
        RuntimeConfig(
            docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
            source=ModelSource(local_path=tmp_path / "model.gguf"),
        ),
        manager,
    )

    captured: dict[str, object] = {}

    class ProbeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        async def post(self, path: str, *args: object, **kwargs: object):
            captured["path"] = path
            captured["timeout"] = kwargs.get("timeout")
            return SimpleNamespace(status_code=200)

    monkeypatch.setattr(manager, "create_proxy_client", lambda base_url: ProbeClient())

    ready = await runtime_container._probe_model_readiness("primary")

    assert ready is True
    assert captured["path"] == "/v1/chat/completions"
    assert captured["timeout"] == 12.5


@pytest.mark.asyncio
async def test_manager_cleans_up_cancelled_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mock_client = MockDockerClient()
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    runtime = RuntimeSettings(config_path=tmp_path / "config.json")
    app_config = AppConfig(
        models=[
            ModelConfig(
                name="primary",
                runtimes={
                    "rocm": RuntimeConfig(
                        docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
                        devices=["/dev/kfd", "/dev/dri"],
                        source=ModelSource(local_path=tmp_path / "model.gguf"),
                    )
                },
            )
        ],
    )

    manager = ModelRuntimeManager(
        runtime=runtime,
        app_config=app_config,
        hf=FakeHF(tmp_path),
    )

    async def fake_start(self: RuntimeContainer) -> None:
        self.state = "running"
        self.container = MockContainer(name="inference-server-runtime-rocm")

    async def fake_load_model(self: RuntimeContainer, name: str) -> None:
        raise asyncio.CancelledError()

    async def fake_stop(self: RuntimeContainer) -> None:
        self.container = None
        self.state = "stopped"

    monkeypatch.setattr(RuntimeContainer, "start", fake_start)
    monkeypatch.setattr(RuntimeContainer, "load_model", fake_load_model)
    monkeypatch.setattr(RuntimeContainer, "stop", fake_stop)

    with pytest.raises(asyncio.CancelledError):
        await manager.load("primary", "rocm")

    assert manager._active_model_name is None
    assert manager._active_runtime is None
    assert "Load cancelled" in (manager.status().last_error or "")


@pytest.mark.asyncio
async def test_manager_keeps_active_model_when_new_model_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mock_client = MockDockerClient()
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    async def fake_resolve_commit_hash(*args: Any, **kwargs: Any) -> str:
        return "mock_commit_123"

    monkeypatch.setattr(
        "manager.ModelRuntimeManager._resolve_commit_hash", fake_resolve_commit_hash
    )

    runtime = RuntimeSettings(
        config_path=tmp_path / "config.json",
    )
    app_config = AppConfig(
        models=[
            ModelConfig(
                name="primary",
                runtimes={
                    "rocm": RuntimeConfig(
                        docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
                        devices=["/dev/kfd", "/dev/dri"],
                        source=ModelSource(local_path=tmp_path / "model.gguf"),
                    )
                },
            ),
            ModelConfig(
                name="secondary",
                runtimes={
                    "rocm": RuntimeConfig(
                        docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
                        devices=["/dev/kfd", "/dev/dri"],
                        source=ModelSource(
                            repo_id="broken/repo", filename="model.gguf"
                        ),
                    )
                },
            ),
        ],
    )

    upstream_app = MockUpstreamApp()
    manager = ModelRuntimeManager(
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

    await manager.load("primary", "rocm")
    with pytest.raises(ValueError):
        await manager.load("secondary", "rocm")

    svc_status = manager.status()
    assert svc_status.active_model == "primary"

    # primary should still be active because the secondary load failed preflight
    primary_status = manager.model_resource("primary").status
    assert primary_status.name == "primary"
    assert primary_status.state == "running"
    assert primary_status.active is True

    # secondary never became active
    secondary_status = manager.model_resource("secondary").status
    assert secondary_status.name == "secondary"
    assert secondary_status.state == "stopped"
    assert secondary_status.active is False

    container = mock_client.containers.get("inference-server-runtime-rocm")
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
        models=[
            ModelConfig(
                name="model1",
                runtimes={
                    "rocm": RuntimeConfig(
                        docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
                        devices=["/dev/kfd", "/dev/dri"],
                        source=ModelSource(local_path=tmp_path / "m1.gguf"),
                    )
                },
            )
        ],
    )

    upstream_app = MockUpstreamApp()
    manager = ModelRuntimeManager(
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

    await manager.load("model1", "rocm")

    unmanaged_container = MockContainer(
        name="some-other-database", labels={"managed-by": "other-app"}
    )
    mock_client.containers.active_containers["some-other-database"] = (
        unmanaged_container
    )

    await manager.cleanup()

    managed = mock_client.containers.get("inference-server-runtime-rocm")
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
        models=[
            ModelConfig(
                name="primary",
                runtimes={
                    "rocm": RuntimeConfig(
                        docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
                        devices=["/dev/kfd", "/dev/dri"],
                        source=ModelSource(local_path=tmp_path / "model.gguf"),
                    )
                },
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

    manager = ModelRuntimeManager(
        runtime=runtime,
        app_config=app_config,
        hf=FakeHF(tmp_path),
        proxy_client_factory=factory,
    )
    # mock ready state
    manager._active_model_name = "primary"
    model_config = manager._get_model_config("primary")
    manager._active_runtime = RuntimeContainer(
        "rocm", "primary", model_config.runtimes["rocm"], manager
    )
    manager._active_runtime.state = "running"

    request = httpx.Request("POST", "http://placeholder/v1/chat/completions")

    with pytest.raises(httpx.ConnectError):
        await manager.open_proxy_session("/v1/chat/completions", request)

    assert len(clients) == 1
    assert clients[0].closed is True


@pytest.mark.asyncio
async def test_unload_failure_preserves_runtime_state_and_honors_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mock_client = MockDockerClient()
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    async def fake_resolve_commit_hash(*args: Any, **kwargs: Any) -> str:
        return "mock_commit_123"

    monkeypatch.setattr(
        "manager.ModelRuntimeManager._resolve_commit_hash", fake_resolve_commit_hash
    )

    runtime = RuntimeSettings(config_path=tmp_path / "config.json")
    app_config = AppConfig(
        models=[
            ModelConfig(
                name="primary",
                runtimes={
                    "rocm": RuntimeConfig(
                        docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
                        source=ModelSource(local_path=tmp_path / "model.gguf"),
                    )
                },
            )
        ],
    )

    upstream_app = MockUpstreamApp()
    manager = ModelRuntimeManager(
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

    # 1. Load successfully first
    await manager.load("primary", "rocm")
    assert manager._active_model_name == "primary"
    assert manager._active_runtime is not None

    # Unload should stop and remove the active container.
    container = manager._active_runtime.container
    assert container is not None

    unloaded = await manager.unload("primary")
    assert unloaded is not None

    # 2. State should fully clear the active runtime and model.
    assert manager._active_model_name is None
    assert manager._active_runtime is None
    assert container.stopped is True
    assert container.removed is True

    # 3. status() should no longer report an active runtime.
    svc_status = manager.status()
    assert svc_status.healthy is True
    assert svc_status.active_model is None
    assert svc_status.active_runtime is None
    assert svc_status.active_container_id is None
    assert svc_status.last_error is None


@pytest.mark.asyncio
async def test_manager_no_model_loaded_error(tmp_path: Path) -> None:
    runtime = RuntimeSettings(config_path=tmp_path / "config.json")
    app_config = AppConfig(models=[])
    manager = ModelRuntimeManager(
        runtime=runtime,
        app_config=app_config,
        hf=FakeHF(tmp_path),
    )
    request = httpx.Request("POST", "http://localhost:8000/v1/chat/completions")
    with pytest.raises(RuntimeError, match="no model is loaded"):
        await manager.open_proxy_session("/v1/chat/completions", request)


@pytest.mark.asyncio
async def test_proxy_transport_failure_clears_active_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = RuntimeSettings(config_path=tmp_path / "config.json")
    app_config = AppConfig(
        models=[
            ModelConfig(
                name="primary",
                runtimes={
                    "rocm": RuntimeConfig(
                        docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
                        source=ModelSource(local_path=tmp_path / "model.gguf"),
                    )
                },
            )
        ],
    )

    upstream_app = MockUpstreamApp()
    manager = ModelRuntimeManager(
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

    await manager.load("primary", "rocm")

    async def fake_stop(self: RuntimeContainer) -> None:
        self.container = None
        self.state = "stopped"

    monkeypatch.setattr(RuntimeContainer, "stop", fake_stop)

    class FailingClient:
        def build_request(self, method, path, headers=None, content=None, params=None):
            return httpx.Request(method, f"http://runtime{path}", headers=headers, content=content, params=params)

        async def send(self, request, stream=True):
            raise httpx.ConnectError("boom", request=request)

        async def aclose(self):
            return None

    manager._proxy_client_factory = lambda base_url: FailingClient()

    request = httpx.Request("POST", "http://localhost:8000/v1/chat/completions")
    with pytest.raises(httpx.ConnectError, match="boom"):
        await manager.open_proxy_session("/v1/chat/completions", request)

    assert manager._active_runtime is None
    assert manager._active_model_name is None
    assert manager.status().last_error == "Model runtime communication failure: boom"


@pytest.mark.asyncio
async def test_runtime_timeout_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = RuntimeSettings(config_path=tmp_path / "config.json")
    app_config = AppConfig(
        models=[
            ModelConfig(
                name="gemma",
                runtimes={
                    "rocm": RuntimeConfig(
                        docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
                        source=ModelSource(local_path=tmp_path / "gemma.gguf"),
                    )
                },
            )
        ]
    )

    manager = ModelRuntimeManager(
        runtime=runtime,
        app_config=app_config,
        hf=FakeHF(tmp_path),
    )
    manager._docker_client = MockDockerClient()

    async def fake_to_thread(func: Any, /, *args: object, **kwargs: object) -> object:
        return func(*args, **kwargs)

    monkeypatch.setattr("manager.asyncio.to_thread", fake_to_thread)

    class MockLoop:
        def __init__(self) -> None:
            self.current_time = 0.0
        def time(self) -> float:
            return self.current_time

    mock_loop = MockLoop()
    monkeypatch.setattr("manager.asyncio.get_running_loop", lambda: mock_loop)

    async def fake_sleep(seconds: float) -> None:
        mock_loop.current_time += seconds

    monkeypatch.setattr("manager.asyncio.sleep", fake_sleep)

    class FailingClient:
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
        async def get(self, url):
            raise httpx.RequestError("Mock connection failure")

    monkeypatch.setattr(manager, "create_proxy_client", lambda base_url: FailingClient())

    with pytest.raises(TimeoutError, match="runtime did not become ready in time"):
        await manager.load("gemma", "rocm")


@pytest.mark.asyncio
async def test_mmproj_download_and_ini_mapping(monkeypatch, tmp_path: Path) -> None:
    from schemas import HFCachedFile

    # Setup mock file structure in cache
    repo_dir = tmp_path / "hub" / "models--org--model"
    snapshots = repo_dir / "snapshots" / "main"
    snapshots.mkdir(parents=True)
    
    model_file = snapshots / "model.gguf"
    model_file.write_text("model")
    
    mmproj_file = snapshots / "mmproj-BF16.gguf"
    mmproj_file.write_text("mmproj")

    class MockCachedHF(FakeHF):
        def cache_files(self) -> list[HFCachedFile]:
            return [
                HFCachedFile(
                    repo_id="org/model",
                    revision="main",
                    refs=["main"],
                    filename="model.gguf",
                    local_path=str(model_file),
                ),
                HFCachedFile(
                    repo_id="org/model",
                    revision="main",
                    refs=["main"],
                    filename="mmproj-BF16.gguf",
                    local_path=str(mmproj_file),
                ),
            ]

    runtime = RuntimeSettings(config_path=tmp_path / "config.json")
    app_config = AppConfig(
        models=[
            ModelConfig(
                name="m_vision",
                runtimes={
                    "rocm": RuntimeConfig(
                        docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
                        source=ModelSource(
                            repo_id="org/model", filename="model.gguf", revision="main"
                        ),
                        mmproj=ModelSource(
                            repo_id="org/model",
                            filename="mmproj-BF16.gguf",
                            revision="main",
                        ),
                    )
                },
            )
        ]
    )

    upstream_app = MockUpstreamApp()
    manager = ModelRuntimeManager(
        runtime=runtime,
        app_config=app_config,
        hf=MockCachedHF(tmp_path / "hub"),
        proxy_client_factory=lambda base_url: mock_client_factory(
            base_url, upstream_app
        ),
    )
    manager._docker_client = MockDockerClient()

    async def fake_to_thread(func: Any, /, *args: object, **kwargs: object) -> object:
        return func(*args, **kwargs)

    monkeypatch.setattr("manager.asyncio.to_thread", fake_to_thread)

    await manager.load("m_vision", "rocm")

    # Verify that the generated INI config maps the mmproj to its container path
    ini_content = await manager._generate_presets_ini("rocm")
    assert "mmproj = /huggingface/models--org--model/snapshots/main/mmproj-BF16.gguf" in ini_content


@pytest.mark.asyncio
async def test_mmproj_artifacts_resolve_before_runtime_start(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []

    class RecordingHF(FakeHF):
        def resolve_source(self, source: ModelSource) -> Path:
            events.append(f"resolve:{source.label()}")
            base = self.cache_dir / "models--org--model" / "snapshots" / "main"
            base.mkdir(parents=True, exist_ok=True)
            return base / (source.filename or "model.gguf")

    runtime = RuntimeSettings(config_path=tmp_path / "config.json")
    app_config = AppConfig(
        models=[
            ModelConfig(
                name="m_vision",
                runtimes={
                    "rocm": RuntimeConfig(
                        docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
                        source=ModelSource(
                            repo_id="org/model", filename="model.gguf", revision="main"
                        ),
                        mmproj=ModelSource(
                            repo_id="org/model",
                            filename="mmproj-BF16.gguf",
                            revision="main",
                        ),
                    )
                },
            )
        ]
    )

    async def fake_start(self: RuntimeContainer) -> None:
        events.append(
            f"start:{self.model_path.name}:{self.mmproj_path.name if self.mmproj_path else 'none'}"
        )
        self.state = "running"
        self.container = MockContainer(name="inference-server-runtime-rocm")

    async def fake_load_model(self: RuntimeContainer, name: str) -> None:
        events.append(f"load:{name}")

    async def fake_stop(self: RuntimeContainer) -> None:
        self.container = None
        self.state = "stopped"

    monkeypatch.setattr(RuntimeContainer, "start", fake_start)
    monkeypatch.setattr(RuntimeContainer, "load_model", fake_load_model)
    monkeypatch.setattr(RuntimeContainer, "stop", fake_stop)

    async def fake_to_thread(func: Any, /, *args: object, **kwargs: object) -> object:
        return func(*args, **kwargs)

    monkeypatch.setattr("manager.asyncio.to_thread", fake_to_thread)

    manager = ModelRuntimeManager(
        runtime=runtime,
        app_config=app_config,
        hf=RecordingHF(tmp_path / "hub"),
    )

    await manager.load("m_vision", "rocm")

    assert events[:2] == [
        "resolve:org/model/model.gguf",
        "resolve:org/model/mmproj-BF16.gguf",
    ]
    assert events[2].startswith("start:model.gguf:mmproj-BF16.gguf")
    assert events[3] == "load:m_vision"


@pytest.mark.asyncio
async def test_missing_mmproj_fails_before_runtime_start(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    started = False

    class MissingMmprojHF(FakeHF):
        def resolve_source(self, source: ModelSource) -> Path:
            if source.filename == "mmproj-BF16.gguf":
                raise ValueError("missing projector")
            base = self.cache_dir / "models--org--model" / "snapshots" / "main"
            base.mkdir(parents=True, exist_ok=True)
            return base / (source.filename or "model.gguf")

    runtime = RuntimeSettings(config_path=tmp_path / "config.json")
    app_config = AppConfig(
        models=[
            ModelConfig(
                name="m_vision",
                runtimes={
                    "rocm": RuntimeConfig(
                        docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
                        source=ModelSource(
                            repo_id="org/model", filename="model.gguf", revision="main"
                        ),
                        mmproj=ModelSource(
                            repo_id="org/model",
                            filename="mmproj-BF16.gguf",
                            revision="main",
                        ),
                    )
                },
            )
        ]
    )

    async def fake_start(self: RuntimeContainer) -> None:
        nonlocal started
        started = True

    monkeypatch.setattr(RuntimeContainer, "start", fake_start)

    async def fake_to_thread(func: Any, /, *args: object, **kwargs: object) -> object:
        return func(*args, **kwargs)

    monkeypatch.setattr("manager.asyncio.to_thread", fake_to_thread)

    manager = ModelRuntimeManager(
        runtime=runtime,
        app_config=app_config,
        hf=MissingMmprojHF(tmp_path / "hub"),
    )

    with pytest.raises(ValueError, match="failed to resolve multimodal projector"):
        await manager.load("m_vision", "rocm")

    assert started is False
    assert manager._active_runtime is None


@pytest.mark.asyncio
async def test_manager_load_with_pid_1_only_llama_server(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mock_client = MockDockerClient()
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    runtime = RuntimeSettings(config_path=tmp_path / "config.json")
    app_config = AppConfig(
        models=[
            ModelConfig(
                name="primary",
                runtimes={
                    "rocm": RuntimeConfig(
                        docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
                        devices=["/dev/kfd", "/dev/dri"],
                        source=ModelSource(local_path=tmp_path / "model.gguf"),
                    )
                },
            )
        ],
    )

    class PID1Container(MockContainer):
        def exec_run(self, cmd: list[str]) -> Any:
            return SimpleNamespace(
                output=b"1 S /app/llama-server --host 0.0.0.0 --port 8080\n"
            )

    async def fake_start(self: RuntimeContainer) -> None:
        self.state = "running"
        self.container = PID1Container(name="inference-server-runtime-rocm")

    async def fake_stop(self: RuntimeContainer) -> None:
        self.container = None
        self.state = "stopped"

    monkeypatch.setattr(RuntimeContainer, "start", fake_start)
    monkeypatch.setattr(RuntimeContainer, "stop", fake_stop)

    upstream_app = MockUpstreamApp()
    manager = ModelRuntimeManager(
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

    status = await manager.load("primary", "rocm")
    assert status.state == "running"
    assert status.active is True
    assert manager._active_model_name == "primary"


@pytest.mark.asyncio
async def test_manager_load_with_readiness_probe_stalled_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mock_client = MockDockerClient()
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    runtime = RuntimeSettings(
        config_path=tmp_path / "config.json",
        model_load_timeout_seconds=7,
    )
    app_config = AppConfig(
        models=[
            ModelConfig(
                name="primary",
                runtimes={
                    "rocm": RuntimeConfig(
                        docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
                        devices=["/dev/kfd", "/dev/dri"],
                        source=ModelSource(local_path=tmp_path / "model.gguf"),
                    )
                },
            )
        ],
    )

    manager = ModelRuntimeManager(
        runtime=runtime,
        app_config=app_config,
        hf=FakeHF(tmp_path),
    )
    manager._docker_client = MockDockerClient()

    async def fake_to_thread(func: Any, /, *args: object, **kwargs: object) -> object:
        return func(*args, **kwargs)

    monkeypatch.setattr("manager.asyncio.to_thread", fake_to_thread)

    class MockLoop:
        def __init__(self) -> None:
            self.current_time = 0.0
        def time(self) -> float:
            return self.current_time

    mock_loop = MockLoop()
    monkeypatch.setattr("manager.asyncio.get_running_loop", lambda: mock_loop)

    async def fake_sleep(seconds: float) -> None:
        mock_loop.current_time += seconds

    monkeypatch.setattr("manager.asyncio.sleep", fake_sleep)

    class StalledClient:
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
        async def get(self, path, *args, **kwargs):
            return SimpleNamespace(status_code=200)
        async def post(self, path, *args, **kwargs):
            if "chat/completions" in path:
                raise httpx.ReadTimeout("Read timed out", request=None)
            return SimpleNamespace(status_code=200, text='{"success": true}')

    monkeypatch.setattr(manager, "create_proxy_client", lambda base_url: StalledClient())

    with pytest.raises(
        TimeoutError,
        match=r"model 'primary' did not finish loading within 7 seconds",
    ):
        await manager.load("primary", "rocm")

    assert manager._lock.locked() is False
    assert manager._active_model_name is None
    assert manager._active_runtime is None
    assert manager.status().last_error == "model 'primary' did not finish loading within 7 seconds"


@pytest.mark.asyncio
async def test_runtime_container_enables_auto_remove(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mock_client = MockDockerClient()
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    runtime = RuntimeSettings(config_path=tmp_path / "config.json")
    app_config = AppConfig(
        models=[
            ModelConfig(
                name="primary",
                runtimes={
                    "rocm": RuntimeConfig(
                        docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
                        devices=["/dev/kfd", "/dev/dri"],
                        source=ModelSource(local_path=tmp_path / "model.gguf"),
                    )
                },
            )
        ],
    )

    captured_kwargs: dict[str, Any] = {}
    original_run = mock_client.containers.run

    def run_spy(**kwargs: Any) -> MockContainer:
        captured_kwargs.update(kwargs)
        return original_run(**kwargs)

    monkeypatch.setattr(mock_client.containers, "run", run_spy)

    async def fake_to_thread(func: Any, /, *args: object, **kwargs: object) -> object:
        return func(*args, **kwargs)

    monkeypatch.setattr("manager.asyncio.to_thread", fake_to_thread)

    upstream_app = MockUpstreamApp()
    manager = ModelRuntimeManager(
        runtime=runtime,
        app_config=app_config,
        hf=FakeHF(tmp_path),
        proxy_client_factory=lambda base_url: mock_client_factory(
            base_url, upstream_app
        ),
    )

    await manager.load("primary", "rocm")

    assert captured_kwargs["auto_remove"] is True


@pytest.mark.asyncio
async def test_stop_tolerates_auto_removed_container(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mock_client = MockDockerClient()
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    runtime = RuntimeSettings(config_path=tmp_path / "config.json")
    app_config = AppConfig(
        models=[
            ModelConfig(
                name="primary",
                runtimes={
                    "rocm": RuntimeConfig(
                        docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
                        devices=["/dev/kfd", "/dev/dri"],
                        source=ModelSource(local_path=tmp_path / "model.gguf"),
                    )
                },
            )
        ],
    )

    class AutoRemovedContainer(MockContainer):
        def stop(self, timeout: int = 15) -> None:
            self.stopped = True
            self.status = "exited"

        def remove(self, force: bool = False) -> None:
            raise docker.errors.NotFound("already removed")

    async def fake_to_thread(func: Any, /, *args: object, **kwargs: object) -> object:
        return func(*args, **kwargs)

    monkeypatch.setattr("manager.asyncio.to_thread", fake_to_thread)

    manager = ModelRuntimeManager(
        runtime=runtime,
        app_config=app_config,
        hf=FakeHF(tmp_path),
    )
    runtime_container = RuntimeContainer(
        "rocm",
        "primary",
        app_config.models[0].runtimes["rocm"],
        manager,
    )
    runtime_container.container = AutoRemovedContainer(name="inference-server-runtime-rocm")

    await runtime_container.stop()

    assert runtime_container.container is None
    assert runtime_container.state == "stopped"


@pytest.mark.asyncio
async def test_stop_tolerates_removal_already_in_progress(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mock_client = MockDockerClient()
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    runtime = RuntimeSettings(config_path=tmp_path / "config.json")
    app_config = AppConfig(
        models=[
            ModelConfig(
                name="primary",
                runtimes={
                    "rocm": RuntimeConfig(
                        docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
                        devices=["/dev/kfd", "/dev/dri"],
                        source=ModelSource(local_path=tmp_path / "model.gguf"),
                    )
                },
            )
        ],
    )

    class RemovalInProgressError(docker.errors.APIError):
        def __init__(self) -> None:
            super().__init__(
                "Conflict: removal of container mock-id-123 is already in progress"
            )
            self.response = SimpleNamespace(status_code=409)

    class RacingContainer(MockContainer):
        def remove(self, force: bool = False) -> None:
            raise RemovalInProgressError()

    async def fake_to_thread(func: Any, /, *args: object, **kwargs: object) -> object:
        return func(*args, **kwargs)

    monkeypatch.setattr("manager.asyncio.to_thread", fake_to_thread)

    manager = ModelRuntimeManager(
        runtime=runtime,
        app_config=app_config,
        hf=FakeHF(tmp_path),
    )
    runtime_container = RuntimeContainer(
        "rocm",
        "primary",
        app_config.models[0].runtimes["rocm"],
        manager,
    )
    runtime_container.container = RacingContainer(name="inference-server-runtime-rocm")

    await runtime_container.stop()

    assert runtime_container.container is None
    assert runtime_container.state == "stopped"


@pytest.mark.asyncio
async def test_failed_load_preserves_logs_for_debugging(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mock_client = MockDockerClient()
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    runtime = RuntimeSettings(
        config_path=tmp_path / "config.json",
        runtime_log_dir=tmp_path / "runtime-logs",
    )
    app_config = AppConfig(
        models=[
            ModelConfig(
                name="primary",
                runtimes={
                    "rocm": RuntimeConfig(
                        docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
                        devices=["/dev/kfd", "/dev/dri"],
                        source=ModelSource(local_path=tmp_path / "model.gguf"),
                    )
                },
            )
        ],
    )

    class ExitedContainer(MockContainer):
        def __init__(self, name: str) -> None:
            super().__init__(name=name, status="exited")

        def logs(self, tail: int = 500) -> bytes:
            return b"llama init\nCUDA out of memory\n"

    def run_exited(**kwargs: Any) -> ExitedContainer:
        container = ExitedContainer(name=kwargs.get("name", "mock-container"))
        mock_client.containers.active_containers[container.name] = container
        return container

    monkeypatch.setattr(mock_client.containers, "run", run_exited)

    async def fake_to_thread(func: Any, /, *args: object, **kwargs: object) -> object:
        return func(*args, **kwargs)

    monkeypatch.setattr("manager.asyncio.to_thread", fake_to_thread)

    manager = ModelRuntimeManager(
        runtime=runtime,
        app_config=app_config,
        hf=FakeHF(tmp_path),
    )

    with pytest.raises(RuntimeError, match="Container exited immediately"):
        await manager.load("primary", "rocm")

    logs = await manager.get_logs("primary")
    assert logs == ["llama init", "CUDA out of memory"]
    daily_log = runtime.runtime_log_dir / f"runtime-{datetime.now(tz=UTC).date().isoformat()}.log"
    assert daily_log.exists()
    persisted = daily_log.read_text()
    assert "reason=startup-exit" in persisted
    assert "model=primary" in persisted
    assert "runtime=rocm" in persisted
    assert "CUDA out of memory" in persisted


@pytest.mark.asyncio
async def test_manager_load_timeout_honors_configured_model_load_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = RuntimeSettings(
        config_path=tmp_path / "config.json",
        model_load_timeout_seconds=9,
    )
    app_config = AppConfig(
        models=[
            ModelConfig(
                name="primary",
                runtimes={
                    "rocm": RuntimeConfig(
                        docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
                        source=ModelSource(local_path=tmp_path / "model.gguf"),
                    )
                },
            )
        ]
    )

    manager = ModelRuntimeManager(
        runtime=runtime,
        app_config=app_config,
        hf=FakeHF(tmp_path),
    )
    runtime_container = RuntimeContainer(
        "rocm",
        "primary",
        app_config.models[0].runtimes["rocm"],
        manager,
    )
    runtime_container.container = MockContainer(name="inference-server-runtime-rocm")

    async def fake_to_thread(func: Any, /, *args: object, **kwargs: object) -> object:
        return func(*args, **kwargs)

    monkeypatch.setattr("manager.asyncio.to_thread", fake_to_thread)

    class MockLoop:
        def __init__(self) -> None:
            self.current_time = 0.0

        def time(self) -> float:
            return self.current_time

    mock_loop = MockLoop()
    sleep_calls: list[float] = []
    probe_calls = 0

    monkeypatch.setattr("manager.asyncio.get_running_loop", lambda: mock_loop)

    async def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)
        mock_loop.current_time += seconds

    monkeypatch.setattr("manager.asyncio.sleep", fake_sleep)

    async def fake_probe(name: str) -> bool:
        nonlocal probe_calls
        probe_calls += 1
        return False

    monkeypatch.setattr(runtime_container, "_probe_model_readiness", fake_probe)

    with pytest.raises(
        TimeoutError,
        match=r"model 'primary' did not finish loading within 9 seconds",
    ):
        await runtime_container._wait_until_model_loaded("primary")

    assert mock_loop.current_time == 9.0
    assert probe_calls == 9
    assert sleep_calls == [1] * 9


@pytest.mark.asyncio
async def test_manager_load_with_missing_ps_exit_code_127(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mock_client = MockDockerClient()
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    runtime = RuntimeSettings(config_path=tmp_path / "config.json")
    app_config = AppConfig(
        models=[
            ModelConfig(
                name="primary",
                runtimes={
                    "rocm": RuntimeConfig(
                        docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
                        devices=["/dev/kfd", "/dev/dri"],
                        source=ModelSource(local_path=tmp_path / "model.gguf"),
                    )
                },
            )
        ],
    )

    class MissingPsContainer(MockContainer):
        def exec_run(self, cmd: list[str]) -> Any:
            return SimpleNamespace(
                exit_code=127,
                output=b"sh: ps: command not found\n",
            )

    async def fake_start(self: RuntimeContainer) -> None:
        self.state = "running"
        self.container = MissingPsContainer(name="inference-server-runtime-rocm")

    async def fake_stop(self: RuntimeContainer) -> None:
        self.container = None
        self.state = "stopped"

    monkeypatch.setattr(RuntimeContainer, "start", fake_start)
    monkeypatch.setattr(RuntimeContainer, "stop", fake_stop)

    upstream_app = MockUpstreamApp()
    manager = ModelRuntimeManager(
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

    status = await manager.load("primary", "rocm")
    assert status.state == "running"
    assert status.active is True
    assert manager._active_model_name == "primary"
    assert upstream_app.readiness_probe_count >= 1


@pytest.mark.asyncio
async def test_manager_load_with_missing_ps_exec_run_raises_exception(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mock_client = MockDockerClient()
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    runtime = RuntimeSettings(config_path=tmp_path / "config.json")
    app_config = AppConfig(
        models=[
            ModelConfig(
                name="primary",
                runtimes={
                    "rocm": RuntimeConfig(
                        docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
                        devices=["/dev/kfd", "/dev/dri"],
                        source=ModelSource(local_path=tmp_path / "model.gguf"),
                    )
                },
            )
        ],
    )

    class ThrowingPsContainer(MockContainer):
        def exec_run(self, cmd: list[str]) -> Any:
            raise RuntimeError("docker exec failed")

    async def fake_start(self: RuntimeContainer) -> None:
        self.state = "running"
        self.container = ThrowingPsContainer(name="inference-server-runtime-rocm")

    async def fake_stop(self: RuntimeContainer) -> None:
        self.container = None
        self.state = "stopped"

    monkeypatch.setattr(RuntimeContainer, "start", fake_start)
    monkeypatch.setattr(RuntimeContainer, "stop", fake_stop)

    upstream_app = MockUpstreamApp()
    manager = ModelRuntimeManager(
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

    status = await manager.load("primary", "rocm")
    assert status.state == "running"
    assert status.active is True
    assert manager._active_model_name == "primary"
    assert upstream_app.readiness_probe_count >= 1
