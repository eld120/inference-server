from __future__ import annotations

import builtins
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import docker.errors
import httpx
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app import create_app
from config import RuntimeSettings
from manager import BackendManager, ContainerRuntime, ProxySession
from schemas import (
    AppConfig,
    BackendFamilyConfig,
    ModelPresetConfig,
    ModelSource,
)


class FakeHF:
    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir or Path("/tmp/mocked-cache")

    def resolve_source(self, source: ModelSource) -> Path:
        return Path("/mocked/path.gguf")


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


@pytest.mark.asyncio
async def test_find_backend_for_model(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_client = MockDockerClient()
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    app_config = AppConfig(
        backend_families={
            "rocm": BackendFamilyConfig(
                docker_image="ghcr.io/ggerganov/llama.cpp:server-rocm"
            )
        },
        models=[
            ModelPresetConfig(
                name="gemma",
                backend_family="rocm",
                model=ModelSource(
                    repo_id="ggml-org/gemma-3-1b-it-GGUF",
                    filename="gemma-3-1b-it-Q4_K_M.gguf",
                ),
            )
        ],
    )
    manager = BackendManager(
        runtime=RuntimeSettings(), app_config=app_config, hf=FakeHF()
    )

    assert manager.find_backend_for_model("gemma") == "gemma"
    assert manager.find_backend_for_model("nonexistent") is None


@pytest.mark.asyncio
async def test_logs_capturing_and_get_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_client = MockDockerClient()
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    app_config = AppConfig(
        backend_families={
            "rocm": BackendFamilyConfig(
                docker_image="ghcr.io/ggerganov/llama.cpp:server-rocm"
            )
        },
        models=[
            ModelPresetConfig(
                name="gemma",
                backend_family="rocm",
                model=ModelSource(local_path=Path("mock.gguf")),
            )
        ],
    )
    manager = BackendManager(
        runtime=RuntimeSettings(), app_config=app_config, hf=FakeHF()
    )

    preset = manager._get_preset("gemma")
    runtime = ContainerRuntime(preset, app_config.backend_families["rocm"], manager)

    class CustomMockContainer:
        id = "mock-id"

        def logs(self, tail: int = 500) -> bytes:
            return b"Hello from llama-server\nAnother log line\n"

    runtime.container = CustomMockContainer()  # type: ignore
    manager._active_runtime = runtime
    manager._active_model_name = "gemma"

    logs = await manager.get_logs("gemma")
    assert logs == ["Hello from llama-server", "Another log line"]


@pytest.mark.asyncio
async def test_vram_safe_swap(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_client = MockDockerClient()
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    app_config = AppConfig(
        runtime_mode="container",
        backend_families={
            "rocm": BackendFamilyConfig(
                docker_image="ghcr.io/ggerganov/llama.cpp:server-rocm"
            )
        },
        models=[
            ModelPresetConfig(
                name="backend1",
                backend_family="rocm",
                model=ModelSource(local_path=Path("b1.gguf")),
            ),
            ModelPresetConfig(
                name="backend2",
                backend_family="rocm",
                model=ModelSource(local_path=Path("b2.gguf")),
            ),
        ],
    )
    manager = BackendManager(
        runtime=RuntimeSettings(),
        app_config=app_config,
        hf=FakeHF(),
        proxy_client_factory=lambda url: httpx.AsyncClient(base_url=url),
    )

    async def fake_wait_until_ready(_: Any) -> None:
        return None

    async def fake_to_thread(
        func: object, /, *args: object, **kwargs: object
    ) -> object:
        import typing

        return typing.cast(typing.Callable, func)(*args, **kwargs)

    async def fake_remove(*args: Any, **kwargs: Any) -> None:
        pass

    monkeypatch.setattr(manager, "_remove_conflicting_container", fake_remove)
    monkeypatch.setattr("manager.asyncio.to_thread", fake_to_thread)
    monkeypatch.setattr(ContainerRuntime, "_wait_until_ready", fake_wait_until_ready)

    # Load first backend
    await manager.load("backend1")
    assert manager._active_model_name == "backend1"
    b1_container = mock_client.containers.get("inference-server-backend1")
    assert b1_container.stopped is False

    # Load second backend (with vram_safe_swap=True,
    # b1 is terminated *before* b2 starts)
    original_run = mock_client.containers.run

    def assert_terminated_run(**kwargs: Any) -> MockContainer:
        assert b1_container.stopped is True
        return original_run(**kwargs)

    monkeypatch.setattr(mock_client.containers, "run", assert_terminated_run)
    await manager.load("backend2")
    assert manager._active_model_name == "backend2"


class FakeAppManager:
    def __init__(self, primary: ModelPresetConfig) -> None:
        self._primary = primary
        self.loaded: list[str] = []
        self._active: str | None = None

    def status(self) -> SimpleNamespace:
        return SimpleNamespace(
            healthy=True,
            api_prefix="/api",
            active_backend=self._active,
            backends=[],
        )

    def active_backend(self) -> SimpleNamespace | None:
        if self._active is None:
            return None
        return SimpleNamespace(config=self._primary)

    async def load(self, name: str) -> SimpleNamespace:
        self.loaded.append(name)
        self._active = name
        return SimpleNamespace(model_dump=lambda: {})

    def find_backend_for_model(self, model_name: str) -> str | None:
        if model_name == self._primary.name:
            return self._primary.name
        return None

    async def get_logs(self, name: str) -> list[str]:
        return ["log line"]

    async def open_proxy_session(
        self, path: str, request: httpx.Request
    ) -> ProxySession:
        backend = FastAPI()

        @backend.post("/v1/chat/completions")
        async def chat_completions() -> dict[str, str]:
            return {"status": "ok"}

        client = AsyncClient(
            transport=ASGITransport(app=backend), base_url="http://backend"
        )
        upstream_request = client.build_request("POST", path, content=request.content)
        response = await client.send(upstream_request, stream=True)
        return ProxySession(client=client, response=response)


@pytest.mark.asyncio
async def test_app_strict_mismatch_and_explicit_load() -> None:
    primary = ModelPresetConfig(
        name="gemma",
        backend_family="rocm",
        model=ModelSource(local_path=Path("mock.gguf")),
    )
    app_config = AppConfig(
        backend_families={
            "rocm": BackendFamilyConfig(
                docker_image="ghcr.io/ggerganov/llama.cpp:server-rocm"
            )
        },
        models=[primary],
    )
    fake_manager = FakeAppManager(primary)

    app = create_app(
        app_config=app_config,
        manager=fake_manager,  # type: ignore
    )

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        # 1. Initially nothing is loaded, so request should fail with 400
        response = await client.post(
            "/api/v1/chat/completions", json={"model": "gemma"}
        )
        assert response.status_code == 400
        assert "not currently loaded" in response.json()["detail"]

        # 2. Explicitly load the model via the load endpoint
        await fake_manager.load("gemma")

        # 3. Now request should succeed with 200
        response = await client.post(
            "/api/v1/chat/completions", json={"model": "gemma"}
        )
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_unknown_model_rejected() -> None:
    primary = ModelPresetConfig(
        name="gemma",
        backend_family="rocm",
        model=ModelSource(local_path=Path("mock.gguf")),
    )
    app_config = AppConfig(
        backend_families={
            "rocm": BackendFamilyConfig(
                docker_image="ghcr.io/ggerganov/llama.cpp:server-rocm"
            )
        },
        models=[primary],
    )
    fake_manager = FakeAppManager(primary)

    app = create_app(
        app_config=app_config,
        manager=fake_manager,  # type: ignore
    )

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/v1/chat/completions", json={"model": "does-not-exist"}
        )
        assert response.status_code == 400
        assert "not configured" in response.json()["detail"]



@pytest.mark.asyncio
async def test_container_spawn_failure_sets_error_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_client = MockDockerClient()
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    app_config = AppConfig(
        runtime_mode="container",
        backend_families={
            "rocm": BackendFamilyConfig(
                docker_image="ghcr.io/ggerganov/llama.cpp:server-rocm"
            )
        },
        models=[
            ModelPresetConfig(
                name="gemma",
                backend_family="rocm",
                model=ModelSource(local_path=Path("mock.gguf")),
            )
        ],
    )
    manager = BackendManager(
        runtime=RuntimeSettings(), app_config=app_config, hf=FakeHF()
    )

    def failing_run(**kwargs: Any) -> Any:
        raise RuntimeError("Docker daemon error: failed to create container")

    async def fake_to_thread(
        func: object, /, *args: object, **kwargs: object
    ) -> object:
        import typing

        return typing.cast(typing.Callable, func)(*args, **kwargs)

    monkeypatch.setattr(mock_client.containers, "run", failing_run)

    async def fake_remove(*args: Any, **kwargs: Any) -> None:
        pass

    monkeypatch.setattr(manager, "_remove_conflicting_container", fake_remove)
    monkeypatch.setattr("manager.asyncio.to_thread", fake_to_thread)

    with pytest.raises(RuntimeError, match="failed to create container"):
        await manager.load("gemma")

    status = manager.status()
    gemma_status = next(b for b in status.backends if b.name == "gemma")
    assert gemma_status.state == "error"
    assert gemma_status.last_error is not None
    assert "failed to create container" in gemma_status.last_error


@pytest.mark.asyncio
async def test_resolver_failure_sets_error_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_client = MockDockerClient()
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    class FailingHFResolver(FakeHF):
        def resolve_source(self, source: ModelSource) -> Path:
            raise ValueError("download failed")

    app_config = AppConfig(
        runtime_mode="container",
        backend_families={
            "rocm": BackendFamilyConfig(
                docker_image="ghcr.io/ggerganov/llama.cpp:server-rocm"
            )
        },
        models=[
            ModelPresetConfig(
                name="gemma",
                backend_family="rocm",
                model=ModelSource(local_path=Path("mock.gguf")),
            )
        ],
    )
    manager = BackendManager(
        runtime=RuntimeSettings(),
        app_config=app_config,
        hf=FailingHFResolver(),
    )

    async def fake_to_thread(
        func: object, /, *args: object, **kwargs: object
    ) -> object:
        import typing

        return typing.cast(typing.Callable, func)(*args, **kwargs)

    monkeypatch.setattr("manager.asyncio.to_thread", fake_to_thread)

    with pytest.raises(ValueError, match="download failed"):
        await manager.load("gemma")

    status = manager.status()
    gemma_status = next(b for b in status.backends if b.name == "gemma")
    assert gemma_status.state == "error"
    assert gemma_status.last_error is not None
    assert "download failed" in gemma_status.last_error


@pytest.mark.asyncio
async def test_container_boot_crash_sets_error_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_client = MockDockerClient()
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    app_config = AppConfig(
        runtime_mode="container",
        backend_families={
            "rocm": BackendFamilyConfig(
                docker_image="ghcr.io/ggerganov/llama.cpp:server-rocm"
            )
        },
        models=[
            ModelPresetConfig(
                name="gemma",
                backend_family="rocm",
                model=ModelSource(local_path=Path("mock.gguf")),
            )
        ],
    )
    manager = BackendManager(
        runtime=RuntimeSettings(), app_config=app_config, hf=FakeHF()
    )

    def run_exited(**kwargs: Any) -> MockContainer:
        c = MockContainer(name=kwargs.get("name", "gemma"), status="exited")
        mock_client.containers.active_containers[c.name] = c
        return c

    async def fake_to_thread(
        func: object, /, *args: object, **kwargs: object
    ) -> object:
        import typing

        return typing.cast(typing.Callable, func)(*args, **kwargs)

    monkeypatch.setattr(mock_client.containers, "run", run_exited)

    async def fake_remove(*args: Any, **kwargs: Any) -> None:
        pass

    monkeypatch.setattr(manager, "_remove_conflicting_container", fake_remove)
    monkeypatch.setattr("manager.asyncio.to_thread", fake_to_thread)

    with pytest.raises(RuntimeError, match="Container exited immediately."):
        await manager.load("gemma")

    status = manager.status()
    gemma_status = next(b for b in status.backends if b.name == "gemma")
    assert gemma_status.state == "error"
    assert gemma_status.last_error is not None
    assert "Container exited immediately" in gemma_status.last_error
