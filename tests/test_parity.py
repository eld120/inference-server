"""Tests for router/container parity.

Verifies that the same preset key produces equivalent effective runtime
semantics regardless of whether router or container strategy is used.
This includes GPU offload, speculative settings, extra_args, and ports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import pytest

from config import RuntimeSettings
from manager import BackendManager
from schemas import (
    AppConfig,
    BackendFamilyConfig,
    ModelPresetConfig,
    ModelSource,
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
    preset_name: str,
) -> dict[str, Any]:
    """Helper that loads a preset and captures the kwargs passed to
    docker containers.run()."""
    mock_client = MockDockerClient()
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    async def fake_resolve_commit_hash(*args: Any, **kwargs: Any) -> str:
        return "mock_commit_123"

    monkeypatch.setattr(
        "manager.BackendManager._resolve_commit_hash", fake_resolve_commit_hash
    )

    captured_kwargs: dict[str, Any] = {}
    original_run = mock_client.containers.run

    def run_spy(**kwargs: Any) -> Any:
        captured_kwargs.update(kwargs)
        return original_run(**kwargs)

    monkeypatch.setattr(mock_client.containers, "run", run_spy)

    upstream_app = MockUpstreamApp()
    manager = BackendManager(
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

    await manager.load(preset_name)
    return captured_kwargs


def _make_config(
    mode: Literal["auto", "router", "container"],
    *,
    speculative: SpeculativeConfig | None = None,
    extra_args: list[str] | None = None,
    shared_args: list[str] | None = None,
    tmp_path: Path,
) -> AppConfig:
    """Create a test AppConfig with given mode and optional speculative config."""
    spec = speculative or SpeculativeConfig()
    return AppConfig(
        runtime_mode=mode,
        backend_families={
            "rocm": BackendFamilyConfig(
                docker_image="ghcr.io/ggerganov/llama.cpp:server-rocm",
                devices=["/dev/kfd", "/dev/dri"],
                shared_args=shared_args or [],
            )
        },
        models=[
            ModelPresetConfig(
                name="test_model",
                backend_family="rocm",
                model=ModelSource(local_path=tmp_path / "model.gguf"),
                speculative=spec,
                extra_args=extra_args or [],
            )
        ],
    )


# ---------------------------------------------------------------------------
# Parity tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_both_strategies_use_same_devices(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Both strategies should mount the same GPU devices."""
    container_config = _make_config("container", tmp_path=tmp_path)
    router_config = _make_config("router", tmp_path=tmp_path)

    container_kwargs = await _capture_container_run_kwargs(
        monkeypatch, tmp_path, container_config, "test_model"
    )
    router_kwargs = await _capture_container_run_kwargs(
        monkeypatch, tmp_path, router_config, "test_model"
    )

    assert container_kwargs["devices"] == router_kwargs["devices"]
    assert container_kwargs["devices"] == ["/dev/kfd:/dev/kfd", "/dev/dri:/dev/dri"]


@pytest.mark.asyncio
async def test_both_strategies_use_same_port(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Both strategies should bind to the same host port."""
    container_config = _make_config("container", tmp_path=tmp_path)
    router_config = _make_config("router", tmp_path=tmp_path)

    container_kwargs = await _capture_container_run_kwargs(
        monkeypatch, tmp_path, container_config, "test_model"
    )
    router_kwargs = await _capture_container_run_kwargs(
        monkeypatch, tmp_path, router_config, "test_model"
    )

    assert container_kwargs["ports"] == router_kwargs["ports"]


@pytest.mark.asyncio
async def test_both_strategies_use_same_image(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Both strategies should use the same Docker image from the family."""
    container_config = _make_config("container", tmp_path=tmp_path)
    router_config = _make_config("router", tmp_path=tmp_path)

    container_kwargs = await _capture_container_run_kwargs(
        monkeypatch, tmp_path, container_config, "test_model"
    )
    router_kwargs = await _capture_container_run_kwargs(
        monkeypatch, tmp_path, router_config, "test_model"
    )

    assert container_kwargs["image"] == router_kwargs["image"]
    assert container_kwargs["image"] == "ghcr.io/ggerganov/llama.cpp:server-rocm"


@pytest.mark.asyncio
async def test_both_strategies_include_shared_args(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Both strategies should include shared_args from the backend family."""
    shared = ["-ngl", "99"]
    container_config = _make_config(
        "container", shared_args=shared, tmp_path=tmp_path
    )
    router_config = _make_config(
        "router", shared_args=shared, tmp_path=tmp_path
    )

    container_kwargs = await _capture_container_run_kwargs(
        monkeypatch, tmp_path, container_config, "test_model"
    )
    router_kwargs = await _capture_container_run_kwargs(
        monkeypatch, tmp_path, router_config, "test_model"
    )

    # Container mode appends shared_args to the command
    container_cmd = container_kwargs["command"]
    assert "-ngl" in container_cmd
    assert "99" in container_cmd

    # Router mode also appends shared_args to the command
    router_cmd = router_kwargs["command"]
    assert "-ngl" in router_cmd
    assert "99" in router_cmd


@pytest.mark.asyncio
async def test_speculative_settings_in_container_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Container strategy should include speculative type in command."""
    spec = SpeculativeConfig(type="draft-mtp")
    config = _make_config("container", speculative=spec, tmp_path=tmp_path)

    kwargs = await _capture_container_run_kwargs(
        monkeypatch, tmp_path, config, "test_model"
    )

    cmd = kwargs["command"]
    assert "--spec-type" in cmd
    spec_idx = cmd.index("--spec-type")
    assert cmd[spec_idx + 1] == "draft-mtp"


@pytest.mark.asyncio
async def test_speculative_settings_in_router_ini(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Router strategy should include speculative type in generated INI."""
    mock_client = MockDockerClient()
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    async def fake_resolve_commit_hash(*args: Any, **kwargs: Any) -> str:
        return "mock_commit_123"

    monkeypatch.setattr(
        "manager.BackendManager._resolve_commit_hash", fake_resolve_commit_hash
    )

    spec = SpeculativeConfig(type="draft-mtp")
    config = _make_config("router", speculative=spec, tmp_path=tmp_path)

    manager = BackendManager(
        runtime=RuntimeSettings(config_path=tmp_path / "config.json"),
        app_config=config,
        hf=FakeHF(tmp_path),
    )

    ini_content = await manager._generate_presets_ini("rocm")
    assert "spec-type = draft-mtp" in ini_content


@pytest.mark.asyncio
async def test_extra_args_in_container_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Container strategy should include extra_args in the command."""
    config = _make_config(
        "container", extra_args=["-c", "4096", "--temp", "0.7"], tmp_path=tmp_path
    )

    kwargs = await _capture_container_run_kwargs(
        monkeypatch, tmp_path, config, "test_model"
    )

    cmd = kwargs["command"]
    assert "-c" in cmd
    assert "4096" in cmd
    assert "--temp" in cmd
    assert "0.7" in cmd


@pytest.mark.asyncio
async def test_extra_args_in_router_ini(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Router strategy should translate extra_args to INI key-value pairs."""
    mock_client = MockDockerClient()
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    async def fake_resolve_commit_hash(*args: Any, **kwargs: Any) -> str:
        return "mock_commit_123"

    monkeypatch.setattr(
        "manager.BackendManager._resolve_commit_hash", fake_resolve_commit_hash
    )

    config = _make_config(
        "router", extra_args=["-c", "4096", "--temp", "0.7"], tmp_path=tmp_path
    )

    manager = BackendManager(
        runtime=RuntimeSettings(config_path=tmp_path / "config.json"),
        app_config=config,
        hf=FakeHF(tmp_path),
    )

    ini_content = await manager._generate_presets_ini("rocm")
    assert "c = 4096" in ini_content
    assert "temp = 0.7" in ini_content


@pytest.mark.asyncio
async def test_both_strategies_label_containers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Both strategies should label containers for singleton cleanup."""
    container_config = _make_config("container", tmp_path=tmp_path)
    router_config = _make_config("router", tmp_path=tmp_path)

    container_kwargs = await _capture_container_run_kwargs(
        monkeypatch, tmp_path, container_config, "test_model"
    )
    router_kwargs = await _capture_container_run_kwargs(
        monkeypatch, tmp_path, router_config, "test_model"
    )

    assert container_kwargs["labels"] == {"managed-by": "inference-server"}
    assert router_kwargs["labels"] == {"managed-by": "inference-server"}


@pytest.mark.asyncio
async def test_gpu_offload_defaults_parity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Both strategies should default to GPU offload (ngl=99) when not specified."""
    container_config = _make_config("container", tmp_path=tmp_path)
    router_config = _make_config("router", tmp_path=tmp_path)

    container_kwargs = await _capture_container_run_kwargs(
        monkeypatch, tmp_path, container_config, "test_model"
    )
    container_cmd = container_kwargs["command"]
    assert "-ngl" in container_cmd
    assert "99" in container_cmd

    manager = BackendManager(
        runtime=RuntimeSettings(config_path=tmp_path / "config.json"),
        app_config=router_config,
        hf=FakeHF(tmp_path),
    )
    async def fake_resolve_commit_hash(*args: Any, **kwargs: Any) -> str:
        return "mock_commit_123"
    monkeypatch.setattr(
        "manager.BackendManager._resolve_commit_hash", fake_resolve_commit_hash
    )
    ini_content = await manager._generate_presets_ini("rocm")
    assert "ngl = 99" in ini_content
