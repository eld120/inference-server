from __future__ import annotations

from pathlib import Path

import pytest

from config import RuntimeSettings, load_app_config, save_app_config
from schemas import AppConfig, ModelConfig, ModelSource, RuntimeConfig


def test_app_config_round_trip(tmp_path: Path) -> None:
    config = AppConfig(
        hf_cache_dir=tmp_path / "hf-cache",
        models=[
            ModelConfig(
                name="gemma",
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

    path = tmp_path / "config.json"
    save_app_config(path, config)

    loaded = load_app_config(path)

    assert (
        loaded.models[0].runtimes["rocm"].docker_image
        == "ghcr.io/ggerganov/llama.cpp:server-rocm"
    )
    assert loaded.models[0].name == "gemma"
    assert (
        loaded.models[0].runtimes["rocm"].source.local_path == tmp_path / "model.gguf"
    )


def test_runtime_settings_hf_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INF_HF_TOKEN", "test-token")
    settings = RuntimeSettings()
    assert settings.hf_token == "test-token"


def test_runtime_settings_runtime_port_default() -> None:
    settings = RuntimeSettings()
    assert settings.runtime_port == 39281


def test_runtime_settings_runtime_port_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INF_RUNTIME_PORT", "12345")
    settings = RuntimeSettings()
    assert settings.runtime_port == 12345


def test_runtime_settings_backend_port_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INF_BACKEND_PORT", "54321")
    settings = RuntimeSettings()
    assert settings.runtime_port == 54321


def test_runtime_settings_port_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INF_RUNTIME_PORT", "12345")
    monkeypatch.setenv("INF_BACKEND_PORT", "54321")
    settings = RuntimeSettings()
    assert settings.runtime_port == 12345

