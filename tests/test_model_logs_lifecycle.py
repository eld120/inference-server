from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any
import pytest

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

class MockContainer:
    def __init__(self, name: str = "mock-container", status: str = "running") -> None:
        self.id = "mock-container-id-999"
        self.name = name
        self.status = status
        self.stopped = False
        self.removed = False
        self.labels = {"managed-by": "inference-server"}
        self._logs_called_count = 0

    def reload(self) -> None:
        pass

    def stop(self, timeout: int = 15) -> None:
        self.stopped = True
        self.status = "exited"

    def remove(self, force: bool = False) -> None:
        self.removed = True

    def logs(self, tail: int = 500) -> bytes:
        self._logs_called_count += 1
        return f"Line 1 for {self.name}\nLine 2 for {self.name}\n".encode()

class MockContainers:
    def __init__(self) -> None:
        self.active_containers: dict[str, MockContainer] = {}

    def get(self, name: str) -> MockContainer:
        if name in self.active_containers:
            c = self.active_containers[name]
            if not getattr(c, "removed", False):
                return c
        import docker.errors
        raise docker.errors.NotFound("Container not found")

    def list(self, **kwargs: Any) -> list[MockContainer]:
        return [c for c in self.active_containers.values() if not getattr(c, "removed", False)]

    def run(self, **kwargs: Any) -> MockContainer:
        name = kwargs.get("name", "mock-container")
        c = MockContainer(name=name)
        self.active_containers[name] = c
        return c

class MockImages:
    def get(self, name: str) -> str:
        return name
    def pull(self, name: str) -> str:
        return name

class MockDockerClient:
    def __init__(self) -> None:
        self.containers = MockContainers()
        self.images = MockImages()

@pytest.mark.asyncio
async def test_logs_preserved_after_unload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mock_client = MockDockerClient()
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    log_dir = tmp_path / "logs"
    runtime = RuntimeSettings(runtime_log_dir=log_dir)

    app_config = AppConfig(
        models=[
            ModelConfig(
                name="gemma",
                runtimes={
                    "rocm": RuntimeConfig(
                        docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
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

    model_config = manager._get_model_config("gemma")
    container_obj = RuntimeContainer("rocm", "gemma", model_config.runtimes["rocm"], manager)
    container_obj.container = mock_client.containers.run(name="inference-server-rocm-gemma")
    manager._active_runtime = container_obj
    manager._active_model_name = "gemma"

    logs = await manager.get_logs("gemma")
    assert "Line 1 for inference-server-rocm-gemma" in logs[0]
    
    async def mock_unload_model(name: str) -> None:
        pass
    monkeypatch.setattr(container_obj, "unload_model", mock_unload_model)

    res = await manager.unload("gemma")
    assert res is not None

    assert manager._active_model_name is None
    retrieved_logs = await manager.get_logs("gemma")
    assert retrieved_logs == logs

    json_path = log_dir / "last_logs" / "gemma.json"
    assert json_path.exists()


@pytest.mark.asyncio
async def test_logs_preserved_after_incompatible_replacement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mock_client = MockDockerClient()
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    log_dir = tmp_path / "logs"
    runtime = RuntimeSettings(runtime_log_dir=log_dir)

    app_config = AppConfig(
        models=[
            ModelConfig(
                name="gemma",
                runtimes={
                    "rocm": RuntimeConfig(
                        docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
                        source=ModelSource(local_path=Path("mock.gguf")),
                    )
                },
            ),
            ModelConfig(
                name="llama",
                runtimes={
                    "rocm": RuntimeConfig(
                        docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
                        source=ModelSource(local_path=Path("llama.gguf")),
                    )
                },
            )
        ],
    )

    manager = ModelRuntimeManager(
        runtime=runtime, app_config=app_config, hf=FakeHF(tmp_path / "hf_cache")
    )
    manager._docker_client = mock_client

    model_config_gemma = manager._get_model_config("gemma")
    container_gemma = RuntimeContainer("rocm", "gemma", model_config_gemma.runtimes["rocm"], manager)
    container_gemma.container = mock_client.containers.run(name="inference-server-rocm-gemma")
    
    async def mock_unload_model(name: str) -> None:
        pass
    monkeypatch.setattr(container_gemma, "unload_model", mock_unload_model)

    manager._active_runtime = container_gemma
    manager._active_model_name = "gemma"

    gemma_logs = await manager.get_logs("gemma")
    assert len(gemma_logs) > 0

    async def mock_resolve_artifacts(*args: Any, **kwargs: Any) -> tuple[Path, Path | None, Path | None]:
        return Path("mock.gguf"), None, None
    
    monkeypatch.setattr(manager, "_resolve_runtime_artifacts", mock_resolve_artifacts)
    
    async def mock_start(self: RuntimeContainer) -> None:
        pass
    
    async def mock_load_model(self: RuntimeContainer, name: str) -> None:
        pass
        
    monkeypatch.setattr(RuntimeContainer, "start", mock_start)
    monkeypatch.setattr(RuntimeContainer, "load_model", mock_load_model)

    await manager.load("llama", "rocm")

    retrieved_gemma_logs = await manager.get_logs("gemma")
    assert retrieved_gemma_logs == gemma_logs


@pytest.mark.asyncio
async def test_logs_preserved_after_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mock_client = MockDockerClient()
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    log_dir = tmp_path / "logs"
    runtime = RuntimeSettings(runtime_log_dir=log_dir)

    app_config = AppConfig(
        models=[
            ModelConfig(
                name="gemma",
                runtimes={
                    "rocm": RuntimeConfig(
                        docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
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

    model_config = manager._get_model_config("gemma")
    container_obj = RuntimeContainer("rocm", "gemma", model_config.runtimes["rocm"], manager)
    container_obj.container = mock_client.containers.run(name="inference-server-rocm-gemma")
    manager._active_runtime = container_obj
    manager._active_model_name = "gemma"

    logs = await manager.get_logs("gemma")

    await manager.cleanup()

    retrieved_logs = await manager.get_logs("gemma")
    assert retrieved_logs == logs


def test_logs_loaded_from_disk_on_startup(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    last_logs_dir = log_dir / "last_logs"
    last_logs_dir.mkdir(parents=True, exist_ok=True)
    
    mock_logs = ["Gemma started", "Gemma processing", "Gemma finished"]
    with open(last_logs_dir / "gemma.json", "w", encoding="utf-8") as f:
        json.dump(mock_logs, f)

    runtime = RuntimeSettings(runtime_log_dir=log_dir)
    app_config = AppConfig(
        models=[
            ModelConfig(
                name="gemma",
                runtimes={
                    "rocm": RuntimeConfig(
                        docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
                        source=ModelSource(local_path=Path("mock.gguf")),
                    )
                },
            )
        ],
    )

    manager = ModelRuntimeManager(
        runtime=runtime, app_config=app_config, hf=FakeHF(tmp_path / "hf_cache")
    )

    loaded_logs = asyncio.run(manager.get_logs("gemma"))
    assert loaded_logs == mock_logs


@pytest.mark.asyncio
async def test_two_models_same_logs_both_persist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mock_client = MockDockerClient()
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    log_dir = tmp_path / "logs"
    runtime = RuntimeSettings(runtime_log_dir=log_dir)

    app_config = AppConfig(
        models=[
            ModelConfig(
                name="gemma",
                runtimes={
                    "rocm": RuntimeConfig(
                        docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
                        source=ModelSource(local_path=Path("mock.gguf")),
                    )
                },
            ),
            ModelConfig(
                name="llama",
                runtimes={
                    "rocm": RuntimeConfig(
                        docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
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

    model_config_gemma = manager._get_model_config("gemma")
    container_obj = RuntimeContainer("rocm", "gemma", model_config_gemma.runtimes["rocm"], manager)
    container_obj.container = mock_client.containers.run(name="inference-server-rocm-shared")
    
    async def mock_unload_model(self, name: str) -> None:
        pass
    async def mock_load_model(self, name: str) -> None:
        pass
    async def mock_start(self) -> None:
        self.container = mock_client.containers.run(name="inference-server-rocm-shared")
        self.state = "running"

    monkeypatch.setattr("manager.RuntimeContainer.unload_model", mock_unload_model)
    monkeypatch.setattr("manager.RuntimeContainer.load_model", mock_load_model)
    monkeypatch.setattr("manager.RuntimeContainer.start", mock_start)

    container_obj.state = "running"
    manager._active_runtime = container_obj
    manager._active_model_name = "gemma"
    
    gemma_logs = await manager.get_logs("gemma")
    assert len(gemma_logs) > 0

    async def mock_resolve_artifacts(*args: Any, **kwargs: Any) -> tuple[Path, Path | None, Path | None]:
        return Path("mock.gguf"), None, None
    async def mock_generate_presets(*args: Any, **kwargs: Any) -> str:
        return ""
    monkeypatch.setattr(manager, "_resolve_runtime_artifacts", mock_resolve_artifacts)
    monkeypatch.setattr(manager, "_generate_presets_ini", mock_generate_presets)

    await manager.load("llama", "rocm")
    
    llama_logs = await manager.get_logs("llama")
    assert llama_logs == gemma_logs

    gemma_json = log_dir / "last_logs" / "gemma.json"
    llama_json = log_dir / "last_logs" / "llama.json"
    assert gemma_json.exists()
    assert llama_json.exists()

    with open(gemma_json, "r", encoding="utf-8") as f:
        assert json.load(f) == gemma_logs

    with open(llama_json, "r", encoding="utf-8") as f:
        assert json.load(f) == llama_logs

