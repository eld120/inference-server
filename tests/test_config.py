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


def test_hf_cache_dir_expansion() -> None:
    # 1. hf_cache_dir with ~
    config = AppConfig(hf_cache_dir="~/.cache/huggingface/hub")
    expected = Path("~/.cache/huggingface/hub").expanduser().resolve()
    assert config.hf_cache_dir == expected
    assert config.hf_cache_dir.is_absolute()

    # 2. hf_cache_dir with an already absolute path
    config_abs = AppConfig(hf_cache_dir="/absolute/path/to/cache")
    assert config_abs.hf_cache_dir == Path("/absolute/path/to/cache").resolve()
    assert config_abs.hf_cache_dir.is_absolute()


def test_path_mapping_depends_on_hf_cache_dir() -> None:
    from manager import ModelRuntimeManager

    cache_dir = Path("/tmp/custom_cache_dir").resolve().absolute()
    app_config = AppConfig(hf_cache_dir=cache_dir)

    class SimpleHF:
        def __init__(self, cache_dir: Path):
            self.cache_dir = cache_dir

    runtime = RuntimeSettings()
    manager = ModelRuntimeManager(
        runtime=runtime,
        app_config=app_config,
        hf=SimpleHF(cache_dir=cache_dir),  # type: ignore
    )

    # Case A: Path is inside the cache dir
    inside_path = cache_dir / "models--org--model" / "snapshots" / "12345" / "model.gguf"
    mapped_inside = manager._map_model_path_sync(ModelSource(), inside_path)
    assert mapped_inside == "/huggingface/models--org--model/snapshots/12345/model.gguf"

    # Case B: Path is outside the cache dir
    outside_path = Path("/tmp/other_dir/some_model.gguf").resolve().absolute()
    mapped_outside = manager._map_model_path_sync(ModelSource(), outside_path)
    expected_suffix = manager._dir_hash(outside_path.parent)
    assert mapped_outside == f"/local_models_{expected_suffix}/some_model.gguf"


def test_effective_hf_token_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    from config import effective_hf_token

    # Both env and config token set
    monkeypatch.setenv("INF_HF_TOKEN", "env-token")
    settings = RuntimeSettings()
    config = AppConfig(hf_token="config-token")
    assert effective_hf_token(settings, config) == "env-token"

    # Only config token set
    monkeypatch.delenv("INF_HF_TOKEN", raising=False)
    settings_no_env = RuntimeSettings()
    assert effective_hf_token(settings_no_env, config) == "config-token"

    # No token set
    config_no_token = AppConfig()
    assert effective_hf_token(settings_no_env, config_no_token) is None



