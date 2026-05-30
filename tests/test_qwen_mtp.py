import pytest
from pathlib import Path
from pydantic import ValidationError

from manager import ModelRuntimeManager
from config import RuntimeSettings
from schemas import AppConfig, ModelSource


class SimpleHF:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir

    def resolve_source(self, source: ModelSource) -> Path:
        return self.cache_dir / (source.filename or "model.gguf")


def test_parse_mtp_model_from_config() -> None:
    data = {
        "models": [
            {
                "name": "qwen3.6-27b-q4-mtp",
                "runtimes": {
                    "rocm": {
                        "source": {
                            "repo_id": "unsloth/Qwen3.6-27B-MTP-GGUF",
                            "filename": "Qwen3.6-27B-UD-Q4_K_XL.gguf",
                        },
                        "docker_image": "ghcr.io/ggerganov/llama.cpp:server-rocm",
                        "speculative": {"type": "draft-mtp"},
                        "extra_args": ["--spec-draft-n-max", "2"],
                    }
                },
            }
        ]
    }
    config = AppConfig.model_validate(data)
    rocm_runtime = config.models[0].runtimes["rocm"]
    assert rocm_runtime.speculative.type == "draft-mtp"
    assert rocm_runtime.extra_args == ["--spec-draft-n-max", "2"]


def test_reject_spec_type_in_extra_args() -> None:
    data = {
        "models": [
            {
                "name": "invalid-mtp",
                "runtimes": {
                    "rocm": {
                        "source": {
                            "repo_id": "unsloth/Qwen3.6-27B-MTP-GGUF",
                            "filename": "Qwen3.6-27B-UD-Q4_K_XL.gguf",
                        },
                        "docker_image": "ghcr.io/ggerganov/llama.cpp:server-rocm",
                        "extra_args": ["--spec-type", "draft-mtp"],
                    }
                },
            }
        ]
    }
    with pytest.raises(ValidationError, match="do not specify '--spec-type' in extra_args"):
        AppConfig.model_validate(data)


def test_qwen_mtp_status(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    config = AppConfig(
        hf_cache_dir=cache_dir,
        models=[
            {
                "name": "qwen3.6-27b-q4-mtp",
                "runtimes": {
                    "rocm": {
                        "source": {
                            "repo_id": "unsloth/Qwen3.6-27B-MTP-GGUF",
                            "filename": "Qwen3.6-27B-UD-Q4_K_XL.gguf",
                        },
                        "docker_image": "ghcr.io/ggerganov/llama.cpp:server-rocm",
                        "speculative": {"type": "draft-mtp"},
                        "extra_args": ["--spec-draft-n-max", "2"],
                    }
                },
            }
        ],  # type: ignore
    )
    runtime = RuntimeSettings()
    manager = ModelRuntimeManager(
        runtime=runtime, app_config=config, hf=SimpleHF(cache_dir)  # type: ignore
    )

    status = manager._to_status(config.models[0])
    assert status.name == "qwen3.6-27b-q4-mtp"
    assert status.speculative_type == "draft-mtp"


@pytest.mark.asyncio
async def test_qwen_mtp_presets_ini(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    model_dir = cache_dir / "unsloth/Qwen3.6-27B-MTP-GGUF"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file = model_dir / "Qwen3.6-27B-UD-Q4_K_XL.gguf"
    model_file.touch()

    config = AppConfig(
        hf_cache_dir=cache_dir,
        models=[
            {
                "name": "qwen3.6-27b-q4-mtp",
                "runtimes": {
                    "rocm": {
                        "source": {
                            "repo_id": "unsloth/Qwen3.6-27B-MTP-GGUF",
                            "filename": "Qwen3.6-27B-UD-Q4_K_XL.gguf",
                        },
                        "docker_image": "ghcr.io/ggerganov/llama.cpp:server-rocm",
                        "speculative": {"type": "draft-mtp"},
                        "extra_args": ["--spec-draft-n-max", "2"],
                    }
                },
            }
        ],  # type: ignore
    )
    runtime = RuntimeSettings()
    manager = ModelRuntimeManager(
        runtime=runtime, app_config=config, hf=SimpleHF(cache_dir)  # type: ignore
    )

    ini_content = await manager._generate_presets_ini(
        runtime_type="rocm",
        active_model_name="qwen3.6-27b-q4-mtp",
        active_model_resolved_path=model_file,
    )

    # Check that spec-type = draft-mtp is in the INI
    assert "spec-type = draft-mtp" in ini_content
    # Check that spec-draft-n-max = 2 is also in the INI
    assert "spec-draft-n-max = 2" in ini_content


@pytest.mark.asyncio
async def test_qwen_combined_vision_mtp_presets_ini(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    model_dir = cache_dir / "unsloth/Qwen3.6-27B-MTP-GGUF"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file = model_dir / "Qwen3.6-27B-UD-Q4_K_XL.gguf"
    model_file.touch()

    # We also mock the mmproj file being found/resolved by the manager
    mmproj_file = model_dir / "mmproj-BF16.gguf"
    mmproj_file.touch()

    config = AppConfig(
        hf_cache_dir=cache_dir,
        models=[
            {
                "name": "qwen3.6-27b-q4-vision-mtp",
                "runtimes": {
                    "rocm": {
                        "source": {
                            "repo_id": "unsloth/Qwen3.6-27B-MTP-GGUF",
                            "filename": "Qwen3.6-27B-UD-Q4_K_XL.gguf",
                        },
                        "docker_image": "ghcr.io/ggerganov/llama.cpp:server-rocm",
                        "speculative": {"type": "draft-mtp"},
                        "extra_args": [
                            "--mmproj",
                            "mmproj-BF16.gguf",
                            "--spec-draft-n-max",
                            "2",
                        ],
                    }
                },
            }
        ],  # type: ignore
    )
    runtime = RuntimeSettings()

    class MockHFWithMmproj(SimpleHF):
        def resolve_source(self, source: ModelSource) -> Path:
            if source.filename == "mmproj-BF16.gguf":
                return mmproj_file
            return super().resolve_source(source)

        def cache_files(self) -> list[Any]:
            from typing import Any
            from schemas import HFCachedFile
            return [
                HFCachedFile(
                    repo_id="unsloth/Qwen3.6-27B-MTP-GGUF",
                    revision="main",
                    refs=["main"],
                    filename="Qwen3.6-27B-UD-Q4_K_XL.gguf",
                    local_path=str(model_file),
                ),
                HFCachedFile(
                    repo_id="unsloth/Qwen3.6-27B-MTP-GGUF",
                    revision="main",
                    refs=["main"],
                    filename="mmproj-BF16.gguf",
                    local_path=str(mmproj_file),
                ),
            ]

    manager = ModelRuntimeManager(
        runtime=runtime,
        app_config=config,
        hf=MockHFWithMmproj(cache_dir),  # type: ignore
    )

    ini_content = await manager._generate_presets_ini(
        runtime_type="rocm",
        active_model_name="qwen3.6-27b-q4-vision-mtp",
        active_model_resolved_path=model_file,
    )

    # Check that spec-type = draft-mtp is in the INI
    assert "spec-type = draft-mtp" in ini_content
    # Check that spec-draft-n-max = 2 is also in the INI
    assert "spec-draft-n-max = 2" in ini_content
    # Check that mmproj path maps correctly
    assert (
        "mmproj = /huggingface/unsloth/Qwen3.6-27B-MTP-GGUF/mmproj-BF16.gguf"
        in ini_content
    )

