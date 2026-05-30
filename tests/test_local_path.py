import json
import pytest
from pathlib import Path
from pydantic import ValidationError

from config import RuntimeSettings, load_app_config
from manager import ModelRuntimeManager
from schemas import AppConfig, ModelSource


class FakeHF:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir


def test_local_path_expansion() -> None:
    # 1. Primary source local_path with ~
    source = ModelSource(local_path="~/models/model.gguf")
    expected = Path("~/models/model.gguf").expanduser()
    assert source.local_path == expected
    assert source.local_path.is_absolute()

    # 2. Draft model local_path with ~
    draft = ModelSource(local_path="~/models/draft.gguf")
    expected_draft = Path("~/models/draft.gguf").expanduser()
    assert draft.local_path == expected_draft
    assert draft.local_path.is_absolute()


def test_relative_path_resolved_against_config_dir(tmp_path: Path) -> None:
    # Create a config directory that is NOT the current working directory
    config_dir = tmp_path / "custom_config_dir"
    config_dir.mkdir()
    config_path = config_dir / "config.json"

    # Save a config with relative paths
    raw_config = {
        "models": [
            {
                "name": "relative_paths_model",
                "runtimes": {
                    "rocm": {
                        "source": {"local_path": "./model.gguf"},
                        "docker_image": "ghcr.io/ggerganov/llama.cpp:server-rocm",
                        "speculative": {
                            "type": "draft",
                            "draft_model": {"local_path": "./draft.gguf"},
                        },
                    }
                },
            }
        ]
    }

    config_path.write_text(json.dumps(raw_config))

    # Load using load_app_config
    config = load_app_config(config_path)

    # Check that they are resolved relative to config_dir, not the process CWD
    rocm_runtime = config.models[0].runtimes["rocm"]

    expected_model_path = (config_dir / "model.gguf").resolve().absolute()
    expected_draft_path = (config_dir / "draft.gguf").resolve().absolute()

    assert rocm_runtime.source.local_path == expected_model_path
    assert rocm_runtime.speculative.draft_model.local_path == expected_draft_path

    # Verify they are absolute paths
    assert rocm_runtime.source.local_path.is_absolute()
    assert rocm_runtime.speculative.draft_model.local_path.is_absolute()


def test_absolute_local_path_unchanged(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"

    abs_path = "/already/absolute/path/to/model.gguf"
    raw_config = {
        "models": [
            {
                "name": "abs_path_model",
                "runtimes": {
                    "rocm": {
                        "source": {"local_path": abs_path},
                        "docker_image": "ghcr.io/ggerganov/llama.cpp:server-rocm",
                    }
                },
            }
        ]
    }
    config_path.write_text(json.dumps(raw_config))

    config = load_app_config(config_path)
    rocm_runtime = config.models[0].runtimes["rocm"]
    assert rocm_runtime.source.local_path == Path(abs_path).resolve().absolute()


def test_volume_mounting_after_normalization(tmp_path: Path) -> None:
    cache_dir = tmp_path / "hf_cache"
    cache_dir.mkdir()

    source = ModelSource(local_path="~/some_model.gguf")
    assert source.local_path == Path("~/some_model.gguf").expanduser()
    assert source.local_path.is_absolute()

    runtime = RuntimeSettings()
    app_config = AppConfig(hf_cache_dir=cache_dir)
    manager = ModelRuntimeManager(
        runtime=runtime,
        app_config=app_config,
        hf=FakeHF(cache_dir),  # type: ignore
    )

    # Verify path mapping logic maps normalized local path correctly
    mapped = manager._map_model_path_sync(source, source.local_path)
    expected_suffix = manager._dir_hash(source.local_path.parent)
    assert mapped == f"/local_models_{expected_suffix}/some_model.gguf"


def test_local_path_round_trip(tmp_path: Path) -> None:
    from config import load_app_config, save_app_config

    config_dir = tmp_path / "config_dir"
    config_dir.mkdir()
    config_path = config_dir / "config.json"

    # Save a config with relative paths
    raw_config = {
        "models": [
            {
                "name": "round_trip_model",
                "runtimes": {
                    "rocm": {
                        "source": {"local_path": "./model.gguf"},
                        "docker_image": "ghcr.io/ggerganov/llama.cpp:server-rocm",
                        "speculative": {
                            "type": "draft",
                            "draft_model": {"local_path": "./draft.gguf"},
                        },
                    }
                },
            }
        ]
    }
    config_path.write_text(json.dumps(raw_config))

    # 1. Load the config: Verify paths resolve to absolute
    config = load_app_config(config_path)
    rocm_runtime = config.models[0].runtimes["rocm"]
    assert rocm_runtime.source.local_path == (config_dir / "model.gguf").resolve().absolute()
    assert rocm_runtime.speculative.draft_model.local_path == (config_dir / "draft.gguf").resolve().absolute()

    # 2. Save the config back: Verify they are written back as relative paths
    save_app_config(config_path, config)

    saved_data = json.loads(config_path.read_text())
    saved_rocm = saved_data["models"][0]["runtimes"]["rocm"]

    # We check if they start with ./ or match exactly the original string values
    assert saved_rocm["source"]["local_path"] in ("./model.gguf", "model.gguf")
    assert saved_rocm["speculative"]["draft_model"]["local_path"] in (
        "./draft.gguf",
        "draft.gguf",
    )


def test_absolute_paths_under_config_dir_remain_absolute(tmp_path: Path) -> None:
    from config import load_app_config, save_app_config

    config_dir = tmp_path / "config_dir"
    config_dir.mkdir()
    config_path = config_dir / "config.json"

    # Save a config with absolute paths that happen to be under config_dir
    abs_model_path = str((config_dir / "model.gguf").resolve().absolute())
    abs_draft_path = str((config_dir / "draft.gguf").resolve().absolute())

    raw_config = {
        "models": [
            {
                "name": "abs_under_config_model",
                "runtimes": {
                    "rocm": {
                        "source": {"local_path": abs_model_path},
                        "docker_image": "ghcr.io/ggerganov/llama.cpp:server-rocm",
                        "speculative": {
                            "type": "draft",
                            "draft_model": {"local_path": abs_draft_path},
                        },
                    }
                },
            }
        ]
    }
    config_path.write_text(json.dumps(raw_config))

    # 1. Load the config
    config = load_app_config(config_path)
    rocm_runtime = config.models[0].runtimes["rocm"]
    assert rocm_runtime.source.local_path == Path(abs_model_path)
    assert rocm_runtime.speculative.draft_model.local_path == Path(abs_draft_path)

    # 2. Save the config back: Verify they remain absolute paths
    save_app_config(config_path, config)

    saved_data = json.loads(config_path.read_text())
    saved_rocm = saved_data["models"][0]["runtimes"]["rocm"]

    assert saved_rocm["source"]["local_path"] == abs_model_path
    assert saved_rocm["speculative"]["draft_model"]["local_path"] == abs_draft_path


def test_relative_path_escaping_config_dir_round_trip(tmp_path: Path) -> None:
    from config import load_app_config, save_app_config

    config_dir = tmp_path / "config_dir"
    config_dir.mkdir()
    config_path = config_dir / "config.json"

    # Save a config with relative paths that escape config_dir via parent (..)
    raw_config = {
        "models": [
            {
                "name": "escaping_model",
                "runtimes": {
                    "rocm": {
                        "source": {"local_path": "../models/model.gguf"},
                        "docker_image": "ghcr.io/ggerganov/llama.cpp:server-rocm",
                        "speculative": {
                            "type": "draft",
                            "draft_model": {"local_path": "../drafts/draft.gguf"},
                        },
                    }
                },
            }
        ]
    }
    config_path.write_text(json.dumps(raw_config))

    # 1. Load the config: Verify paths resolve to absolute outside config_dir
    config = load_app_config(config_path)
    rocm_runtime = config.models[0].runtimes["rocm"]

    expected_model_path = (tmp_path / "models/model.gguf").resolve().absolute()
    expected_draft_path = (tmp_path / "drafts/draft.gguf").resolve().absolute()

    assert rocm_runtime.source.local_path == expected_model_path
    assert rocm_runtime.speculative.draft_model.local_path == expected_draft_path

    # 2. Save the config back: Verify they remain written as relative paths with ..
    save_app_config(config_path, config)

    saved_data = json.loads(config_path.read_text())
    saved_rocm = saved_data["models"][0]["runtimes"]["rocm"]

    assert saved_rocm["source"]["local_path"] == "../models/model.gguf"
    assert saved_rocm["speculative"]["draft_model"]["local_path"] == "../drafts/draft.gguf"



