from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from config import RuntimeSettings
from manager import ModelRuntimeManager
from schemas import (
    AppConfig,
    ModelConfig,
    ModelSource,
    RuntimeConfig,
)
from tests.test_manager import (
    FakeHF,
    MockContainer,
    MockContainers,
    MockDockerClient,
    MockUpstreamApp,
    mock_client_factory,
)


class CustomMockContainer(MockContainer):
    def __init__(
        self,
        name: str = "mock-container",
        labels: dict[str, str] | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(name=name, labels=labels)
        self.kwargs = kwargs or {}


class CustomMockContainers(MockContainers):
    def __init__(self) -> None:
        super().__init__()
        self.run_calls: list[dict[str, Any]] = []

    def run(self, **kwargs: Any) -> CustomMockContainer:
        self.run_calls.append(kwargs)
        name = kwargs.get("name", "mock-container")
        labels = kwargs.get("labels")
        container = CustomMockContainer(name=name, labels=labels, kwargs=kwargs)
        self.active_containers[name] = container
        return container


class CustomMockDockerClient(MockDockerClient):
    def __init__(self) -> None:
        super().__init__()
        self.containers = CustomMockContainers()


@pytest.fixture
def setup_mocks(monkeypatch: pytest.MonkeyPatch):
    mock_client = CustomMockDockerClient()
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    async def fake_resolve_commit_hash(*args: Any, **kwargs: Any) -> str:
        return "mock_commit_123"

    monkeypatch.setattr(
        "manager.ModelRuntimeManager._resolve_commit_hash", fake_resolve_commit_hash
    )

    async def fake_to_thread(func: Any, /, *args: object, **kwargs: object) -> object:
        return func(*args, **kwargs)

    monkeypatch.setattr("manager.asyncio.to_thread", fake_to_thread)

    return mock_client


@pytest.mark.asyncio
async def test_fresh_runtime_started_even_when_configs_match(
    setup_mocks, tmp_path: Path
) -> None:
    """1. Each model switch starts a fresh runtime even for matching configs."""
    mock_client = setup_mocks

    app_config = AppConfig(
        models=[
            ModelConfig(
                name="model_a",
                runtimes={
                    "rocm": RuntimeConfig(
                        docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
                        devices=["/dev/kfd"],
                        source=ModelSource(local_path=tmp_path / "model_a.gguf"),
                        shared_args=["-c", "2048"],
                    )
                },
            ),
            ModelConfig(
                name="model_b",
                runtimes={
                    "rocm": RuntimeConfig(
                        docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
                        devices=["/dev/kfd"],
                        source=ModelSource(local_path=tmp_path / "model_b.gguf"),
                        shared_args=["-c", "2048"],  # identical
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

    # Load model_a
    await manager.load("model_a", "rocm")
    assert len(mock_client.containers.run_calls) == 1
    assert manager._active_model_name == "model_a"

    first_container = mock_client.containers.active_containers[
        "inference-server-runtime-rocm"
    ]

    # Load model_b - same runtime/config, but should still replace the container
    await manager.load("model_b", "rocm")
    assert len(mock_client.containers.run_calls) == 2
    assert manager._active_model_name == "model_b"
    assert len(upstream_app.unloaded_models) == 0
    assert first_container.stopped is True


@pytest.mark.asyncio
async def test_no_reuse_when_shared_args_differ(setup_mocks, tmp_path: Path) -> None:
    """2. No reuse when shared_args differ."""
    mock_client = setup_mocks

    app_config = AppConfig(
        models=[
            ModelConfig(
                name="model_a",
                runtimes={
                    "rocm": RuntimeConfig(
                        docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
                        devices=["/dev/kfd"],
                        source=ModelSource(local_path=tmp_path / "model_a.gguf"),
                        shared_args=["-c", "2048"],
                    )
                },
            ),
            ModelConfig(
                name="model_b",
                runtimes={
                    "rocm": RuntimeConfig(
                        docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
                        devices=["/dev/kfd"],
                        source=ModelSource(local_path=tmp_path / "model_b.gguf"),
                        shared_args=["-c", "4096"],  # different shared_args
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

    # Load model_a
    await manager.load("model_a", "rocm")
    container_a = mock_client.containers.active_containers[
        "inference-server-runtime-rocm"
    ]
    assert container_a.stopped is False

    # Load model_b - shared_args differ, should recreate container
    await manager.load("model_b", "rocm")
    assert container_a.stopped is True  # previous container stopped
    assert len(mock_client.containers.run_calls) == 2  # second container run call made
    assert manager._active_model_name == "model_b"


@pytest.mark.asyncio
async def test_runtime_switch_removes_old_container(
    setup_mocks, tmp_path: Path
) -> None:
    """3. Runtime switch from rocm to vulkan removes the old container first."""
    mock_client = setup_mocks

    app_config = AppConfig(
        models=[
            ModelConfig(
                name="model_a",
                runtimes={
                    "rocm": RuntimeConfig(
                        docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
                        source=ModelSource(local_path=tmp_path / "model_a.gguf"),
                    ),
                    "vulkan": RuntimeConfig(
                        docker_image="inference-server-llama-vulkan:26.04-7e50ef7",
                        source=ModelSource(local_path=tmp_path / "model_a.gguf"),
                    ),
                },
            )
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

    # Load model_a on rocm
    await manager.load("model_a", "rocm")
    rocm_container = mock_client.containers.active_containers[
        "inference-server-runtime-rocm"
    ]
    assert rocm_container.stopped is False

    # Load model_a on vulkan
    await manager.load("model_a", "vulkan")
    assert rocm_container.stopped is True  # Stopped ROCm first
    assert "inference-server-runtime-vulkan" in mock_client.containers.active_containers
    vulkan_container = mock_client.containers.active_containers[
        "inference-server-runtime-vulkan"
    ]
    assert vulkan_container.stopped is False


@pytest.mark.asyncio
async def test_unload_stops_active_container(setup_mocks, tmp_path: Path) -> None:
    """4. Unload tears down the active runtime container."""
    mock_client = setup_mocks

    app_config = AppConfig(
        models=[
            ModelConfig(
                name="model_a",
                runtimes={
                    "rocm": RuntimeConfig(
                        docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
                        source=ModelSource(local_path=tmp_path / "model_a.gguf"),
                    )
                },
            )
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

    await manager.load("model_a", "rocm")
    container = mock_client.containers.active_containers[
        "inference-server-runtime-rocm"
    ]
    assert container.stopped is False

    # Perform unload
    await manager.unload("model_a")
    assert container.stopped is True
    assert manager._active_runtime is None
    assert manager._active_model_name is None


@pytest.mark.asyncio
async def test_status_after_unload(setup_mocks, tmp_path: Path) -> None:
    """5. GET /api/status reports correct values after unload.

    Specifically, active_model and active runtime should both be cleared.
    """
    _ = setup_mocks

    app_config = AppConfig(
        models=[
            ModelConfig(
                name="model_a",
                runtimes={
                    "rocm": RuntimeConfig(
                        docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
                        source=ModelSource(local_path=tmp_path / "model_a.gguf"),
                    )
                },
            )
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

    # Initial check
    initial_status = manager.status()
    assert initial_status.active_model is None
    assert initial_status.active_runtime is None
    assert initial_status.active_container_id is None

    # Load model
    await manager.load("model_a", "rocm")
    loaded_status = manager.status()
    assert loaded_status.active_model == "model_a"
    assert loaded_status.active_runtime == "rocm"
    assert loaded_status.active_container_id == "mock-id-123"

    # Unload model
    await manager.unload("model_a")
    unloaded_status = manager.status()
    assert unloaded_status.active_model is None
    assert unloaded_status.active_runtime is None
    assert unloaded_status.active_container_id is None


@pytest.mark.asyncio
async def test_unload_then_load_starts_fresh_runtime(
    setup_mocks, tmp_path: Path
) -> None:
    mock_client = setup_mocks

    app_config = AppConfig(
        models=[
            ModelConfig(
                name="model_a",
                runtimes={
                    "vulkan": RuntimeConfig(
                        docker_image="inference-server-llama-vulkan:26.04-7e50ef7",
                        source=ModelSource(local_path=tmp_path / "model_a.gguf"),
                    )
                },
            ),
            ModelConfig(
                name="model_b",
                runtimes={
                    "vulkan": RuntimeConfig(
                        docker_image="inference-server-llama-vulkan:26.04-7e50ef7",
                        source=ModelSource(local_path=tmp_path / "model_b.gguf"),
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

    await manager.load("model_a", "vulkan")
    original_runtime = manager._active_runtime
    assert original_runtime is not None
    original_container = original_runtime.container
    assert original_container is not None

    await manager.unload("model_a")
    assert manager._active_runtime is None
    assert original_container.stopped is True

    await manager.load("model_b", "vulkan")
    assert manager._active_runtime is not None
    assert manager._active_runtime is not original_runtime
    assert manager._active_runtime.container is not original_container
    assert manager._active_model_name == "model_b"
    assert len(mock_client.containers.active_containers) == 1


@pytest.mark.asyncio
async def test_preset_generation_no_ngl_injection(setup_mocks, tmp_path: Path) -> None:
    """6. Preset generation does not inject ngl = 99 unless configured."""
    _ = setup_mocks

    app_config = AppConfig(
        models=[
            ModelConfig(
                name="model_a",
                runtimes={
                    "rocm": RuntimeConfig(
                        docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
                        source=ModelSource(local_path=tmp_path / "model_a.gguf"),
                    )
                },
            )
        ],
    )

    manager = ModelRuntimeManager(
        runtime=RuntimeSettings(config_path=tmp_path / "config.json"),
        app_config=app_config,
        hf=FakeHF(tmp_path),
    )

    ini_content = await manager._generate_presets_ini("rocm")
    assert (
        "ngl =" not in ini_content
    )  # No global default or preset-specific ngl injected


@pytest.mark.asyncio
async def test_shared_args_ngl_propagation(setup_mocks, tmp_path: Path) -> None:
    """7. Check ngl propagation to command line.

    A model with configured shared_args=["-ngl", "99"] still produces the
    expected runtime behavior via config, not via hidden defaults.
    """
    mock_client = setup_mocks

    app_config = AppConfig(
        models=[
            ModelConfig(
                name="model_a",
                runtimes={
                    "rocm": RuntimeConfig(
                        docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
                        source=ModelSource(local_path=tmp_path / "model_a.gguf"),
                        shared_args=["-ngl", "99"],
                    )
                },
            )
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

    await manager.load("model_a", "rocm")

    # Command line check: verify command has -ngl 99
    run_call = mock_client.containers.run_calls[0]
    command = run_call["command"]
    assert "-ngl" in command
    assert "99" in command

    # Presets check: verify preset INI does NOT contain ngl = 99
    ini_content = await manager._generate_presets_ini("rocm")
    assert "ngl = 99" not in ini_content


@pytest.mark.asyncio
async def test_cleanup_success_clears_active_state(setup_mocks, tmp_path: Path) -> None:
    """Verify cleanup() success clears active runtime/model state."""
    _ = setup_mocks

    app_config = AppConfig(
        models=[
            ModelConfig(
                name="model_a",
                runtimes={
                    "rocm": RuntimeConfig(
                        docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
                        source=ModelSource(local_path=tmp_path / "model_a.gguf"),
                    )
                },
            )
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

    await manager.load("model_a", "rocm")
    assert manager._active_runtime is not None
    assert manager._active_model_name == "model_a"

    # Run cleanup
    await manager.cleanup()

    assert manager._active_runtime is None
    assert manager._active_model_name is None

    svc_status = manager.status()
    assert svc_status.active_model is None
    assert svc_status.active_runtime is None
    assert svc_status.active_container_id is None
    assert svc_status.healthy is True


@pytest.mark.asyncio
async def test_cleanup_failure_preserves_honest_state_and_status(
    setup_mocks, tmp_path: Path
) -> None:
    """Verify cleanup() failure preserves runtime/model state honestly.

    Also verify status() remains honest.
    """
    mock_client = setup_mocks

    app_config = AppConfig(
        models=[
            ModelConfig(
                name="model_a",
                runtimes={
                    "rocm": RuntimeConfig(
                        docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
                        source=ModelSource(local_path=tmp_path / "model_a.gguf"),
                    )
                },
            )
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

    await manager.load("model_a", "rocm")
    assert manager._active_runtime is not None
    assert manager._active_model_name == "model_a"

    class FailingContainer:
        def __init__(self) -> None:
            self.id = "failing-container-id"
            self.name = "inference-server-runtime-rocm"
            self.labels = {"managed-by": "inference-server"}
            self.status = "running"

        def reload(self) -> None:
            pass

        def stop(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("mock stop failed")

        def remove(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("mock remove failed")

        def logs(self, *args: Any, **kwargs: Any) -> bytes:
            return b""

    failing_container = FailingContainer()
    manager._active_runtime.container = failing_container  # type: ignore
    mock_client.containers.active_containers["inference-server-runtime-rocm"] = (
        failing_container
    )

    # Run cleanup, which should fail and raise the remove exception
    with pytest.raises(RuntimeError, match="mock remove failed"):
        await manager.cleanup()

    # Verify that failed cleanup clears stale active runtime state.
    assert manager._active_runtime is None
    assert manager._active_model_name is None

    # Verify that status() after cleanup failure reports:
    # - healthy is False
    # - no stale active_model/runtime/container_id
    # - meaningful last_error
    svc_status = manager.status()
    assert svc_status.healthy is False
    assert svc_status.active_model is None
    assert svc_status.active_runtime is None
    assert svc_status.active_container_id is None
    assert svc_status.last_error is not None
    assert "mock remove failed" in svc_status.last_error


@pytest.mark.asyncio
async def test_unload_then_cleanup_with_logs(setup_mocks, tmp_path: Path) -> None:
    from datetime import UTC, datetime
    mock_client = setup_mocks

    app_config = AppConfig(
        models=[
            ModelConfig(
                name="model_a",
                runtimes={
                    "rocm": RuntimeConfig(
                        docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
                        source=ModelSource(local_path=tmp_path / "model_a.gguf"),
                    )
                },
            )
        ],
    )

    upstream_app = MockUpstreamApp()
    manager = ModelRuntimeManager(
        runtime=RuntimeSettings(
            config_path=tmp_path / "config.json",
            runtime_log_dir=tmp_path / "runtime-logs",
        ),
        app_config=app_config,
        hf=FakeHF(tmp_path),
        proxy_client_factory=lambda base_url: mock_client_factory(
            base_url, upstream_app
        ),
    )

    await manager.load("model_a", "rocm")
    assert manager._active_runtime is not None
    assert manager._active_model_name == "model_a"

    # Perform unload
    await manager.unload("model_a")
    assert manager._active_runtime is None
    assert manager._active_model_name is None

    # Perform cleanup
    await manager.cleanup()
    assert manager._active_runtime is None
    assert manager._active_model_name is None

    # Check logs directory contains logs from the stopped rocm container
    daily_log = manager._runtime.runtime_log_dir.expanduser() / f"runtime-{datetime.now(tz=UTC).date().isoformat()}.log"
    assert daily_log.exists()
    log_content = daily_log.read_text()
    assert "model=model_a" in log_content


@pytest.mark.asyncio
async def test_unload_then_incompatible_load_with_logs(setup_mocks, tmp_path: Path) -> None:
    from datetime import UTC, datetime
    mock_client = setup_mocks

    app_config = AppConfig(
        models=[
            ModelConfig(
                name="model_a",
                runtimes={
                    "rocm": RuntimeConfig(
                        docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
                        source=ModelSource(local_path=tmp_path / "model_a.gguf"),
                    )
                },
            ),
            ModelConfig(
                name="model_b",
                runtimes={
                    "vulkan": RuntimeConfig(
                        docker_image="inference-server-llama-vulkan:26.04-7e50ef7",
                        source=ModelSource(local_path=tmp_path / "model_b.gguf"),
                    )
                },
            ),
        ],
    )

    upstream_app = MockUpstreamApp()
    manager = ModelRuntimeManager(
        runtime=RuntimeSettings(
            config_path=tmp_path / "config.json",
            runtime_log_dir=tmp_path / "runtime-logs",
        ),
        app_config=app_config,
        hf=FakeHF(tmp_path),
        proxy_client_factory=lambda base_url: mock_client_factory(
            base_url, upstream_app
        ),
    )

    await manager.load("model_a", "rocm")
    assert manager._active_runtime is not None
    rocm_container = mock_client.containers.active_containers["inference-server-runtime-rocm"]
    assert rocm_container.stopped is False

    # Unload model_a
    await manager.unload("model_a")
    assert manager._active_runtime is None
    assert rocm_container.stopped is True

    # Load model_b on vulkan (incompatible)
    await manager.load("model_b", "vulkan")
    
    # Old container should be stopped and removed
    assert rocm_container.stopped is True
    assert rocm_container.removed is True
    
    # New container should be active
    assert "inference-server-runtime-vulkan" in mock_client.containers.active_containers
    vulkan_container = mock_client.containers.active_containers["inference-server-runtime-vulkan"]
    assert vulkan_container.stopped is False

    # Check logs directory contains logs from the stopped rocm container
    daily_log = manager._runtime.runtime_log_dir.expanduser() / f"runtime-{datetime.now(tz=UTC).date().isoformat()}.log"
    assert daily_log.exists()
    log_content = daily_log.read_text()
    assert "model=model_a" in log_content
