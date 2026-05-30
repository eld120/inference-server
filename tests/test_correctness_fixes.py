"""Tests for correctness fixes and edge cases."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from app import create_app
from config import RuntimeSettings
from manager import ModelRuntimeManager, RuntimeContainer
from schemas import AppConfig, ModelConfig, ModelSource, RuntimeConfig
from tests.helpers import make_app_config, make_model
from tests.test_manager import (
    FakeHF,
    MockDockerClient,
    MockUpstreamApp,
    mock_client_factory,
)


@pytest.mark.asyncio
async def test_swap_failure_rollback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """If load_model fails during a compatible swap, state must be rolled back."""
    mock_client = MockDockerClient()
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    async def fake_resolve_commit_hash(*args: Any, **kwargs: Any) -> str:
        return "mock_commit_123"

    monkeypatch.setattr(
        "manager.ModelRuntimeManager._resolve_commit_hash", fake_resolve_commit_hash
    )

    app_config = make_app_config(
        make_model("model_a", tmp_path / "model_a.gguf"),
        make_model("model_b", tmp_path / "model_b.gguf"),
    )

    upstream_app = MockUpstreamApp()
    manager = ModelRuntimeManager(
        runtime=RuntimeSettings(config_path=tmp_path / "config.json"),
        app_config=app_config,
        hf=FakeHF(tmp_path),
        proxy_client_factory=lambda base_url: mock_client_factory(
            base_url, upstream_app
        ),
    )

    async def fake_to_thread(func: Any, /, *args: object, **kwargs: object) -> object:
        return func(*args, **kwargs)

    monkeypatch.setattr("manager.asyncio.to_thread", fake_to_thread)

    # Load model_a first (succeeds)
    await manager.load("model_a", "rocm")
    assert manager._active_model_name == "model_a"
    assert manager._active_runtime is not None
    assert manager._active_runtime.model_name == "model_a"

    # Mock load_model on the runtime to fail when loading model_b
    original_load = RuntimeContainer.load_model

    async def failing_load(self, name: str) -> None:
        if name == "model_b":
            raise RuntimeError("Forced load failure")
        await original_load(self, name)

    monkeypatch.setattr(RuntimeContainer, "load_model", failing_load)

    # Attempt to swap to model_b; should raise RuntimeError and move to error state
    with pytest.raises(RuntimeError, match="Forced load failure"):
        await manager.load("model_b", "rocm")

    # Assert active model name is cleared since model_a was unloaded,
    # and runtime config points to model_b (which failed) in error state.
    assert manager._active_model_name is None
    assert manager._active_runtime is not None
    assert manager._active_runtime.model_name == "model_b"
    assert manager._active_runtime.state == "error"

    # Status checks
    statuses = {s.name: s for s in manager.model_statuses()}
    assert statuses["model_a"].state == "stopped"
    assert statuses["model_a"].active is False
    assert statuses["model_b"].state == "error"
    assert statuses["model_b"].active is False
    assert "Forced load failure" in (statuses["model_b"].last_error or "")


@pytest.mark.asyncio
async def test_stop_failure_leaves_error_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """If container stop and remove both fail, state must go to error
    and keep container reference.
    """
    mock_client = MockDockerClient()
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    # Force container stop and remove to raise exceptions
    class StubContainer:
        def __init__(self) -> None:
            self.id = "stub-id"
            self.name = "inference-server-runtime-rocm"
            self.labels = {"managed-by": "inference-server"}
            self.status = "running"

        def reload(self) -> None:
            pass

        def stop(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("stop failed")

        def remove(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("remove failed")

    stub_container = StubContainer()

    # Intercept run to return our StubContainer
    def run_spy(**kwargs: Any) -> Any:
        return stub_container

    monkeypatch.setattr(mock_client.containers, "run", run_spy)

    app_config = make_app_config(make_model("model_a", tmp_path / "model_a.gguf"))

    upstream_app = MockUpstreamApp()
    manager = ModelRuntimeManager(
        runtime=RuntimeSettings(config_path=tmp_path / "config.json"),
        app_config=app_config,
        hf=FakeHF(tmp_path),
        proxy_client_factory=lambda base_url: mock_client_factory(
            base_url, upstream_app
        ),
    )

    async def fake_to_thread(func: Any, /, *args: object, **kwargs: object) -> object:
        return func(*args, **kwargs)

    monkeypatch.setattr("manager.asyncio.to_thread", fake_to_thread)

    # Load model_a (succeeds)
    await manager.load("model_a", "rocm")
    assert manager._active_runtime is not None
    assert manager._active_runtime.state == "running"

    # Attempt stop/unload; stop and remove will fail, putting state to error
    with pytest.raises(RuntimeError, match="remove failed"):
        await manager._active_runtime.stop()

    assert manager._active_runtime is not None
    assert manager._active_runtime.state == "error"
    assert "Stop failed" in manager._active_runtime.last_error
    assert manager._active_runtime.container is not None


@pytest.mark.asyncio
async def test_v1_models_discovery(tmp_path: Path) -> None:
    """Gateway v1/models endpoints should return all configured models
    without failing when no runtime is loaded.
    """
    app_config = make_app_config(
        make_model("gemma", tmp_path / "gemma.gguf"),
        make_model("qwen", tmp_path / "qwen.gguf"),
        api_prefix="/api",
    )

    runtime = RuntimeSettings(config_path=tmp_path / "config.json")
    # ModelRuntimeManager created with no loaded runtime
    manager = ModelRuntimeManager(
        runtime=runtime,
        app_config=app_config,
        hf=FakeHF(tmp_path),
    )

    app = create_app(runtime=runtime, app_config=app_config, manager=manager)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        # 1. Test GET /api/v1/models (list)
        response = await client.get("/api/v1/models")
        assert response.status_code == 200
        res_data = response.json()
        assert res_data["object"] == "list"
        models_list = res_data["data"]
        assert len(models_list) == 2
        model_ids = {m["id"] for m in models_list}
        assert model_ids == {"gemma", "qwen"}

        # 2. Test GET /api/v1/models/gemma (retrieve detail)
        response = await client.get("/api/v1/models/gemma")
        assert response.status_code == 200
        assert response.json()["id"] == "gemma"

        # 3. Test GET /api/v1/models/unknown (retrieve missing detail)
        response = await client.get("/api/v1/models/llama")
        assert response.status_code == 404


@pytest.mark.asyncio
async def test_connect_host_incompatibility(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """If connect_host differs, models are incompatible and runtime
    container must be recreated.
    """
    mock_client = MockDockerClient()
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    run_calls = 0
    original_run = mock_client.containers.run

    def run_spy(**kwargs: Any) -> Any:
        nonlocal run_calls
        run_calls += 1
        return original_run(**kwargs)

    monkeypatch.setattr(mock_client.containers, "run", run_spy)

    async def fake_resolve_commit_hash(*args: Any, **kwargs: Any) -> str:
        return "mock_commit_123"

    monkeypatch.setattr(
        "manager.ModelRuntimeManager._resolve_commit_hash", fake_resolve_commit_hash
    )

    app_config = AppConfig(
        models=[
            ModelConfig(
                name="model_a",
                runtimes={
                    "rocm": RuntimeConfig(
                        docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
                        source=ModelSource(local_path=tmp_path / "model_a.gguf"),
                        connect_host="127.0.0.1",
                    )
                },
            ),
            ModelConfig(
                name="model_b",
                runtimes={
                    "rocm": RuntimeConfig(
                        docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
                        source=ModelSource(local_path=tmp_path / "model_b.gguf"),
                        connect_host="10.0.0.5",  # Different connect_host
                    )
                },
            ),
        ],
    )

    upstream_app = MockUpstreamApp()
    manager = ModelRuntimeManager(
        runtime=RuntimeSettings(config_path=tmp_path / "config.json"),
        app_config=app_config,
        hf=FakeHF(tmp_path),
        proxy_client_factory=lambda base_url: mock_client_factory(
            base_url, upstream_app
        ),
    )

    async def fake_to_thread(func: Any, /, *args: object, **kwargs: object) -> object:
        return func(*args, **kwargs)

    monkeypatch.setattr("manager.asyncio.to_thread", fake_to_thread)

    # 1. Load model_a
    await manager.load("model_a", "rocm")
    assert run_calls == 1

    # 2. Load model_b; since connect_host differs, they are incompatible
    # and a new container must be run.
    await manager.load("model_b", "rocm")
    assert run_calls == 2


@pytest.mark.asyncio
async def test_health_status_reflects_error_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """ServiceStatus.healthy must reflect False if active runtime has error state."""
    mock_client = MockDockerClient()
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    app_config = make_app_config(make_model("model_a", tmp_path / "model_a.gguf"))

    upstream_app = MockUpstreamApp()
    manager = ModelRuntimeManager(
        runtime=RuntimeSettings(config_path=tmp_path / "config.json"),
        app_config=app_config,
        hf=FakeHF(tmp_path),
        proxy_client_factory=lambda base_url: mock_client_factory(
            base_url, upstream_app
        ),
    )

    async def fake_to_thread(func: Any, /, *args: object, **kwargs: object) -> object:
        return func(*args, **kwargs)

    monkeypatch.setattr("manager.asyncio.to_thread", fake_to_thread)

    # Initially healthy and idle
    assert manager.status().healthy is True

    # Load model (succeeds)
    await manager.load("model_a", "rocm")
    assert manager.status().healthy is True

    assert manager._active_runtime is not None
    manager._active_runtime.state = "error"
    assert manager.status().healthy is False
