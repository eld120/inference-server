"""Tests for concrete LlamaRuntime settings.

Verifies that the runtime configuration produces correct docker container settings,
ports, devices, shared arguments, and that the INI file maps speculative/extra
settings correctly.
"""

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
    SpeculativeConfig,
)
from tests.test_manager import (
    FakeHF,
    MockDockerClient,
    MockUpstreamApp,
    mock_client_factory,
)


async def _capture_container_run_kwargs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    app_config: AppConfig,
    model_name: str,
) -> dict[str, Any]:
    """Helper that loads a model and captures the kwargs passed to docker run."""
    mock_client = MockDockerClient()
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    async def fake_resolve_commit_hash(*args: Any, **kwargs: Any) -> str:
        return "mock_commit_123"

    monkeypatch.setattr(
        "manager.ModelRuntimeManager._resolve_commit_hash", fake_resolve_commit_hash
    )

    captured_kwargs: dict[str, Any] = {}
    original_run = mock_client.containers.run

    def run_spy(**kwargs: Any) -> Any:
        captured_kwargs.update(kwargs)
        return original_run(**kwargs)

    monkeypatch.setattr(mock_client.containers, "run", run_spy)

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

    await manager.load(model_name, "rocm")
    return captured_kwargs


def _make_config(
    *,
    speculative: SpeculativeConfig | None = None,
    extra_args: list[str] | None = None,
    shared_args: list[str] | None = None,
    tmp_path: Path,
) -> AppConfig:
    """Create a test AppConfig with optional speculative config."""
    spec = speculative or SpeculativeConfig()
    return AppConfig(
        models=[
            ModelConfig(
                name="test_model",
                runtimes={
                    "rocm": RuntimeConfig(
                        docker_image="ghcr.io/ggerganov/llama.cpp:server-rocm",
                        devices=["/dev/kfd", "/dev/dri"],
                        shared_args=shared_args or [],
                        source=ModelSource(local_path=tmp_path / "model.gguf"),
                        speculative=spec,
                        extra_args=extra_args or [],
                    )
                },
            )
        ],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_runtime_uses_devices(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The runtime should mount the specified GPU devices."""
    config = _make_config(tmp_path=tmp_path)
    kwargs = await _capture_container_run_kwargs(
        monkeypatch, tmp_path, config, "test_model"
    )
    assert kwargs["devices"] == ["/dev/kfd:/dev/kfd", "/dev/dri:/dev/dri"]


@pytest.mark.asyncio
async def test_runtime_uses_port(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The runtime should bind to the expected port."""
    config = _make_config(tmp_path=tmp_path)
    kwargs = await _capture_container_run_kwargs(
        monkeypatch, tmp_path, config, "test_model"
    )
    assert kwargs["ports"] == {"8080/tcp": ("127.0.0.1", 39281)}


@pytest.mark.asyncio
async def test_runtime_uses_image(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The runtime should use the specified Docker image."""
    config = _make_config(tmp_path=tmp_path)
    kwargs = await _capture_container_run_kwargs(
        monkeypatch, tmp_path, config, "test_model"
    )
    assert kwargs["image"] == "ghcr.io/ggerganov/llama.cpp:server-rocm"


@pytest.mark.asyncio
async def test_runtime_includes_shared_args(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The runtime should include shared_args in its container command."""
    shared = ["--threads", "4"]
    config = _make_config(shared_args=shared, tmp_path=tmp_path)
    kwargs = await _capture_container_run_kwargs(
        monkeypatch, tmp_path, config, "test_model"
    )
    cmd = kwargs["command"]
    assert "--threads" in cmd
    assert "4" in cmd
    assert "--models-preset" in cmd


@pytest.mark.asyncio
async def test_speculative_settings_in_ini(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The runtime should include speculative type in the generated presets INI file."""
    spec = SpeculativeConfig(type="draft-mtp")
    config = _make_config(speculative=spec, tmp_path=tmp_path)

    manager = ModelRuntimeManager(
        runtime=RuntimeSettings(config_path=tmp_path / "config.json"),
        app_config=config,
        hf=FakeHF(tmp_path),
    )

    ini_content = await manager._generate_presets_ini("rocm")
    assert "spec-type = draft-mtp" in ini_content


@pytest.mark.asyncio
async def test_extra_args_in_ini(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The runtime should translate extra_args to INI key-value pairs."""
    config = _make_config(extra_args=["-c", "4096", "--temp", "0.7"], tmp_path=tmp_path)

    manager = ModelRuntimeManager(
        runtime=RuntimeSettings(config_path=tmp_path / "config.json"),
        app_config=config,
        hf=FakeHF(tmp_path),
    )

    ini_content = await manager._generate_presets_ini("rocm")
    assert "c = 4096" in ini_content
    assert "temp = 0.7" in ini_content


@pytest.mark.asyncio
async def test_runtime_labels_containers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The runtime should label containers for singleton cleanup."""
    config = _make_config(tmp_path=tmp_path)
    kwargs = await _capture_container_run_kwargs(
        monkeypatch, tmp_path, config, "test_model"
    )
    assert kwargs["labels"] == {"managed-by": "inference-server"}


@pytest.mark.asyncio
async def test_gpu_offload_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Presets INI file should not contain ngl = 99 globally by default."""
    config = _make_config(tmp_path=tmp_path)

    manager = ModelRuntimeManager(
        runtime=RuntimeSettings(config_path=tmp_path / "config.json"),
        app_config=config,
        hf=FakeHF(tmp_path),
    )

    async def fake_resolve_commit_hash(*args: Any, **kwargs: Any) -> str:
        return "mock_commit_123"

    monkeypatch.setattr(
        "manager.ModelRuntimeManager._resolve_commit_hash", fake_resolve_commit_hash
    )
    ini_content = await manager._generate_presets_ini("rocm")
    assert "ngl = 99" not in ini_content
