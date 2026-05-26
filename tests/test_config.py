from __future__ import annotations

from pathlib import Path

import pytest

from config import RuntimeSettings, load_app_config, save_app_config
from schemas import AppConfig, BackendFamilyConfig, ModelPresetConfig, ModelSource


def test_app_config_round_trip(tmp_path: Path) -> None:
    config = AppConfig(
        hf_cache_dir=tmp_path / "hf-cache",
        backend_families={
            "rocm": BackendFamilyConfig(
                docker_image="ghcr.io/ggerganov/llama.cpp:server-rocm",
                devices=["/dev/kfd", "/dev/dri"],
            )
        },
        models=[
            ModelPresetConfig(
                name="gemma",
                backend_family="rocm",
                model=ModelSource(local_path=tmp_path / "model.gguf"),
            )
        ],
    )

    path = tmp_path / "config.json"
    save_app_config(path, config)

    loaded = load_app_config(path)

    assert (
        loaded.backend_families["rocm"].docker_image
        == "ghcr.io/ggerganov/llama.cpp:server-rocm"
    )
    assert loaded.models[0].name == "gemma"
    assert loaded.models[0].model.local_path == tmp_path / "model.gguf"


def test_runtime_settings_hf_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INF_HF_TOKEN", "test-token")
    settings = RuntimeSettings()
    assert settings.hf_token == "test-token"
