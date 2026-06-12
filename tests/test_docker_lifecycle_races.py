from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any
import pytest

import docker.errors
from config import RuntimeSettings
from manager import ModelRuntimeManager, RuntimeContainer
from schemas import (
    AppConfig,
    ModelConfig,
    ModelSource,
    RuntimeConfig,
)

class FakeHF:
    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir or Path("/mocked/cache")
    def resolve_source(self, source: ModelSource) -> Path:
        return Path("/mocked/path.gguf")

class DelayedRemovalContainers:
    def __init__(self, name_free_after_calls: int = 3, never_free: bool = False) -> None:
        self.name_free_after_calls = name_free_after_calls
        self.never_free = never_free
        self.get_calls = 0
        self.stopped = False
        self.removed = False

    def get(self, name: str) -> DelayedRemovalContainers:
        self.get_calls += 1
        if self.never_free:
            return self
        if self.get_calls > self.name_free_after_calls:
            raise docker.errors.NotFound("Container not found")
        return self

    def stop(self, timeout: int = 15) -> None:
        self.stopped = True

    def remove(self, force: bool = False) -> None:
        resp = MagicResponse(409, "removal already in progress")
        raise docker.errors.APIError("removal already in progress", response=resp)

    @property
    def name(self) -> str:
        return "inference-server-runtime-rocm"

    @property
    def labels(self) -> dict[str, str]:
        return {"managed-by": "inference-server"}

class MagicResponse:
    def __init__(self, status_code: int, text: str) -> None:
        self.status_code = status_code
        self.text = text
        self.reason = "Conflict"

class MockDockerClientWithContainers:
    def __init__(self, containers_mock: Any) -> None:
        self.containers = containers_mock
        self.images = MockImages()

class MockImages:
    def get(self, name: str) -> str:
        return name
    def pull(self, name: str) -> str:
        return name

@pytest.mark.asyncio
async def test_stop_waits_for_name_release_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mock_containers = DelayedRemovalContainers(name_free_after_calls=3)
    mock_client = MockDockerClientWithContainers(mock_containers)
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    log_dir = tmp_path / "logs"
    runtime = RuntimeSettings(runtime_log_dir=log_dir)
    app_config = AppConfig(models=[])
    manager = ModelRuntimeManager(runtime=runtime, app_config=app_config, hf=FakeHF(tmp_path))
    manager._docker_client = mock_client

    container_obj = RuntimeContainer("rocm", None, RuntimeConfig(docker_image="img", source=ModelSource(local_path=Path("mock.gguf"))), manager)
    container_obj.container = mock_containers  # type: ignore
    
    async def mock_capture(*args: Any, **kwargs: Any) -> None:
        pass
    monkeypatch.setattr(container_obj, "_capture_recent_logs", mock_capture)

    original_sleep = asyncio.sleep
    async def fast_sleep(delay: float) -> None:
        await original_sleep(0.01)
    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    await container_obj.stop()
    assert mock_containers.get_calls >= 3
    assert container_obj.container is None


@pytest.mark.asyncio
async def test_stop_fails_if_name_never_released_negative_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mock_containers = DelayedRemovalContainers(never_free=True)
    mock_client = MockDockerClientWithContainers(mock_containers)
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    log_dir = tmp_path / "logs"
    runtime = RuntimeSettings(runtime_log_dir=log_dir)
    app_config = AppConfig(models=[])
    manager = ModelRuntimeManager(runtime=runtime, app_config=app_config, hf=FakeHF(tmp_path))
    manager._docker_client = mock_client

    container_obj = RuntimeContainer("rocm", None, RuntimeConfig(docker_image="img", source=ModelSource(local_path=Path("mock.gguf"))), manager)
    container_obj.container = mock_containers  # type: ignore
    
    async def mock_capture(*args: Any, **kwargs: Any) -> None:
        pass
    monkeypatch.setattr(container_obj, "_capture_recent_logs", mock_capture)

    original_sleep = asyncio.sleep
    async def fast_sleep(delay: float) -> None:
        await original_sleep(0.001)
    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    with pytest.raises(RuntimeError, match="Failed to release container name"):
        await container_obj.stop()


class LifecycleMockContainers:
    def __init__(self) -> None:
        self.active_containers: dict[str, Any] = {}

    def get(self, name: str) -> Any:
        if name in self.active_containers:
            c = self.active_containers[name]
            if not getattr(c, "removed", False):
                return c
        raise docker.errors.NotFound("not found")

    def run(self, **kwargs: Any) -> Any:
        name = kwargs.get("name", "mock-container")
        c = MockTestContainer(name=name)
        self.active_containers[name] = c
        return c

    def list(self, **kwargs: Any) -> list[Any]:
        return [c for c in self.active_containers.values() if not getattr(c, "removed", False)]

class MockTestContainer:
    def __init__(self, name: str = "mock-container") -> None:
        self.id = "mock-id-123"
        self._name = name
        self.status = "running"
        self.stopped = False
        self.removed = False
        self.labels = {"managed-by": "inference-server"}

    @property
    def name(self) -> str:
        return self._name

    def reload(self) -> None:
        pass

    def stop(self, timeout: int = 15) -> None:
        self.stopped = True
        self.status = "exited"

    def remove(self, force: bool = False) -> None:
        self.removed = True

    def logs(self, tail: int = 500) -> bytes:
        return b"logs\n"

@pytest.mark.asyncio
async def test_quick_unload_reload_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mock_containers = LifecycleMockContainers()
    mock_client = MockDockerClientWithContainers(mock_containers)
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    log_dir = tmp_path / "logs"
    runtime = RuntimeSettings(runtime_log_dir=log_dir)

    app_config = AppConfig(
        models=[
            ModelConfig(
                name="gemma",
                runtimes={
                    "rocm": RuntimeConfig(
                        docker_image="img",
                        source=ModelSource(local_path=Path("mock.gguf")),
                    )
                },
            )
        ],
    )

    manager = ModelRuntimeManager(
        runtime=runtime, app_config=app_config, hf=FakeHF(tmp_path / "hf_cache")
    )
    manager._docker_client = mock_client

    async def mock_resolve_artifacts(*args: Any, **kwargs: Any) -> tuple[Path, Path | None, Path | None]:
        return Path("mock.gguf"), None, None
    async def mock_generate_presets(*args: Any, **kwargs: Any) -> str:
        return ""
    monkeypatch.setattr(manager, "_resolve_runtime_artifacts", mock_resolve_artifacts)
    monkeypatch.setattr(manager, "_generate_presets_ini", mock_generate_presets)

    monkeypatch.setattr(RuntimeContainer, "start", lambda self: asyncio.sleep(0))
    monkeypatch.setattr(RuntimeContainer, "load_model", lambda self, name: asyncio.sleep(0))

    await manager.load("gemma", "rocm")
    assert manager._active_model_name == "gemma"

    async def mock_unload_model(name: str) -> None:
        pass
    monkeypatch.setattr(manager._active_runtime, "unload_model", mock_unload_model)

    await manager.unload("gemma")
    assert manager._active_model_name is None
    assert manager._active_runtime is None

    await manager.load("gemma", "rocm")
    assert manager._active_model_name == "gemma"


@pytest.mark.asyncio
async def test_incompatible_runtime_replacement_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mock_containers = LifecycleMockContainers()
    mock_client = MockDockerClientWithContainers(mock_containers)
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    log_dir = tmp_path / "logs"
    runtime = RuntimeSettings(runtime_log_dir=log_dir)

    app_config = AppConfig(
        models=[
            ModelConfig(
                name="gemma",
                runtimes={
                    "rocm": RuntimeConfig(
                        docker_image="img-rocm",
                        source=ModelSource(local_path=Path("mock.gguf")),
                    )
                },
            ),
            ModelConfig(
                name="llama",
                runtimes={
                    "vulkan": RuntimeConfig(
                        docker_image="img-vulkan",
                        source=ModelSource(local_path=Path("mock.gguf")),
                    )
                },
            )
        ],
    )

    manager = ModelRuntimeManager(
        runtime=runtime, app_config=app_config, hf=FakeHF(tmp_path / "hf_cache")
    )
    manager._docker_client = mock_client

    async def mock_resolve_artifacts(*args: Any, **kwargs: Any) -> tuple[Path, Path | None, Path | None]:
        return Path("mock.gguf"), None, None
    async def mock_generate_presets(*args: Any, **kwargs: Any) -> str:
        return ""
    monkeypatch.setattr(manager, "_resolve_runtime_artifacts", mock_resolve_artifacts)
    monkeypatch.setattr(manager, "_generate_presets_ini", mock_generate_presets)

    monkeypatch.setattr(RuntimeContainer, "start", lambda self: asyncio.sleep(0))
    monkeypatch.setattr(RuntimeContainer, "load_model", lambda self, name: asyncio.sleep(0))

    await manager.load("gemma", "rocm")
    assert manager._active_model_name == "gemma"
    assert manager._active_runtime.runtime_type == "rocm"

    await manager.load("llama", "vulkan")
    assert manager._active_model_name == "llama"
    assert manager._active_runtime.runtime_type == "vulkan"
