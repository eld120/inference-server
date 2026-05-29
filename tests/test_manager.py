from __future__ import annotations

import builtins
from pathlib import Path
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
                        "body": b'{"object": "list", "data": []}',
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
                        docker_image="ghcr.io/ggerganov/llama.cpp:server-rocm",
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
    assert len(upstream_app.unloaded_models) == 1

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
                        docker_image="ghcr.io/ggerganov/llama.cpp:server-rocm",
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
                        docker_image="ghcr.io/ggerganov/llama.cpp:server-rocm",
                        devices=["/dev/kfd", "/dev/dri"],
                        source=ModelSource(local_path=tmp_path / "model.gguf"),
                    )
                },
            ),
            ModelConfig(
                name="secondary",
                runtimes={
                    "rocm": RuntimeConfig(
                        docker_image="ghcr.io/ggerganov/llama.cpp:server-rocm",
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
    assert svc_status.active_model is None

    # primary should be stopped
    assert svc_status.models[0].name == "primary"
    assert svc_status.models[0].state == "stopped"
    assert svc_status.models[0].active is False

    # secondary should be in error state
    assert svc_status.models[1].name == "secondary"
    assert svc_status.models[1].state == "error"
    assert svc_status.models[1].active is False

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
                        docker_image="ghcr.io/ggerganov/llama.cpp:server-rocm",
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
                        docker_image="ghcr.io/ggerganov/llama.cpp:server-rocm",
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
    # mock starting state
    manager._active_model_name = "primary"
    model_config = manager._get_model_config("primary")
    manager._active_runtime = RuntimeContainer(
        "rocm", "primary", model_config.runtimes["rocm"], manager
    )

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
                        docker_image="ghcr.io/ggerganov/llama.cpp:server-rocm",
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

    # Now make container.stop and container.remove fail
    container = manager._active_runtime.container
    assert container is not None

    def failing_stop(*args, **kwargs):
        raise RuntimeError("Stop failed")

    def failing_remove(*args, **kwargs):
        raise RuntimeError("Remove failed")

    container.stop = failing_stop
    container.remove = failing_remove

    # 2. Try to unload. It should fail to stop/remove and propagate the exception.
    with pytest.raises(RuntimeError, match="Remove failed"):
        await manager.unload("primary")

    # 3. State must be preserved (not cleared) and set to error state
    assert manager._active_model_name == "primary"
    assert manager._active_runtime is not None
    assert manager._active_runtime.state == "error"
    assert "Stop failed" in manager._active_runtime.last_error

    # 4. status() must reflect the error honestly
    svc_status = manager.status()
    assert svc_status.healthy is False
    assert svc_status.active_model == "primary"
    assert svc_status.active_runtime == "rocm"
    assert svc_status.active_container_id == container.id
    assert "Stop failed" in svc_status.last_error

    # 5. Successful unload after fixing container stop/remove
    # Restore container stop and remove to succeed
    container.stop = lambda *args, **kwargs: None
    container.remove = lambda *args, **kwargs: None

    unloaded = await manager.unload("primary")
    assert unloaded is not None
    assert manager._active_model_name is None
    assert manager._active_runtime is None

    svc_status2 = manager.status()
    assert svc_status2.healthy is True
    assert svc_status2.active_model is None
    assert svc_status2.active_runtime is None
    assert svc_status2.active_container_id is None


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
async def test_runtime_timeout_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = RuntimeSettings(config_path=tmp_path / "config.json")
    app_config = AppConfig(
        models=[
            ModelConfig(
                name="gemma",
                runtimes={
                    "rocm": RuntimeConfig(
                        docker_image="ghcr.io/ggerganov/llama.cpp:server-rocm",
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
                        docker_image="ghcr.io/ggerganov/llama.cpp:server-rocm",
                        source=ModelSource(
                            repo_id="org/model", filename="model.gguf", revision="main"
                        ),
                        extra_args=["--mmproj", "mmproj-BF16.gguf"],
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

