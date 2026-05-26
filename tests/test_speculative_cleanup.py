"""Tests for speculative state cleanup during model swaps.

Verifies that switching from a speculative preset to a non-speculative preset
clears stale runtime metadata (draft_model_path), and vice versa.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

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


@pytest.mark.asyncio
async def test_speculative_to_non_speculative_clears_draft_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Switching from a speculative preset to a non-speculative one
    should clear draft_model_path on the runtime."""
    mock_client = MockDockerClient()
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    async def fake_resolve_commit_hash(*args: Any, **kwargs: Any) -> str:
        return "mock_commit_123"

    monkeypatch.setattr(
        "manager.BackendManager._resolve_commit_hash", fake_resolve_commit_hash
    )

    draft_path = tmp_path / "draft.gguf"

    runtime = RuntimeSettings(config_path=tmp_path / "config.json")
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
                name="speculative_model",
                backend_family="rocm",
                model=ModelSource(local_path=tmp_path / "model_a.gguf"),
                speculative=SpeculativeConfig(
                    type="draft",
                    draft_model=ModelSource(local_path=draft_path),
                ),
            ),
            ModelPresetConfig(
                name="plain_model",
                backend_family="rocm",
                model=ModelSource(local_path=tmp_path / "model_b.gguf"),
                # No speculative config — defaults to none
            ),
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

    # 1. Load speculative model
    status = await manager.load("speculative_model")
    assert status.active is True
    assert status.speculative_type == "draft"
    assert manager._active_runtime is not None
    assert manager._active_runtime.draft_model_path is not None
    assert manager._active_runtime.draft_model_path == draft_path.resolve().absolute()

    # 2. Swap to non-speculative model (same backend family, same router)
    status = await manager.load("plain_model")
    assert status.active is True
    assert status.speculative_type == "none"

    # Key assertion: draft_model_path must be cleared
    assert manager._active_runtime is not None
    assert manager._active_runtime.draft_model_path is None

    # Status should also reflect no draft model
    statuses = {s.name: s for s in manager.backend_statuses()}
    assert statuses["plain_model"].draft_model_path is None


@pytest.mark.asyncio
async def test_non_speculative_to_speculative_sets_draft_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Switching from a non-speculative preset to a speculative one
    should correctly set draft_model_path."""
    mock_client = MockDockerClient()
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    async def fake_resolve_commit_hash(*args: Any, **kwargs: Any) -> str:
        return "mock_commit_123"

    monkeypatch.setattr(
        "manager.BackendManager._resolve_commit_hash", fake_resolve_commit_hash
    )

    draft_path = tmp_path / "draft.gguf"

    runtime = RuntimeSettings(config_path=tmp_path / "config.json")
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
                name="plain_model",
                backend_family="rocm",
                model=ModelSource(local_path=tmp_path / "model_a.gguf"),
            ),
            ModelPresetConfig(
                name="speculative_model",
                backend_family="rocm",
                model=ModelSource(local_path=tmp_path / "model_b.gguf"),
                speculative=SpeculativeConfig(
                    type="draft",
                    draft_model=ModelSource(local_path=draft_path),
                ),
            ),
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

    # 1. Load non-speculative model
    await manager.load("plain_model")
    assert manager._active_runtime is not None
    assert manager._active_runtime.draft_model_path is None

    # 2. Swap to speculative model
    status = await manager.load("speculative_model")
    assert status.active is True
    assert status.speculative_type == "draft"
    assert manager._active_runtime is not None
    assert manager._active_runtime.draft_model_path == draft_path.resolve().absolute()
