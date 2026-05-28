"""Tests for configuration validation.

Verifies that AppConfig and SpeculativeConfig model validators catch
invalid configurations early with clear error messages.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from schemas import (
    AppConfig,
    ModelConfig,
    ModelSource,
    RuntimeConfig,
    SpeculativeConfig,
)

# ---------------------------------------------------------------------------
# SpeculativeConfig validation
# ---------------------------------------------------------------------------


def test_speculative_none_is_default() -> None:
    config = SpeculativeConfig()
    assert config.type == "none"
    assert config.draft_model is None


def test_speculative_draft_mtp_without_draft_model_ok() -> None:
    """draft-mtp is self-contained, no draft_model needed."""
    config = SpeculativeConfig(type="draft-mtp")
    assert config.type == "draft-mtp"
    assert config.draft_model is None


def test_speculative_draft_requires_draft_model() -> None:
    with pytest.raises(ValueError, match="requires a draft_model"):
        SpeculativeConfig(type="draft")


def test_speculative_draft_simple_requires_draft_model() -> None:
    with pytest.raises(ValueError, match="requires a draft_model"):
        SpeculativeConfig(type="draft-simple")


def test_speculative_draft_with_draft_model_ok() -> None:
    config = SpeculativeConfig(
        type="draft",
        draft_model=ModelSource(local_path=Path("/models/draft.gguf")),
    )
    assert config.draft_model is not None


def test_speculative_self_contained_types_with_draft_model_rejected() -> None:
    types_val: list[Any] = ["none", "draft-mtp", "ngram-cache"]
    for type_val in types_val:
        with pytest.raises(ValueError, match=f"speculative type is '{type_val}'"):
            SpeculativeConfig(
                type=type_val,
                draft_model=ModelSource(local_path=Path("/models/draft.gguf")),
            )


def test_speculative_ngram_types_ok() -> None:
    """Ngram types should work without a draft model."""
    ngram_types: list[Any] = [
        "ngram-cache",
        "ngram-simple",
        "ngram-map-k",
        "ngram-map-k4v",
        "ngram-mod",
    ]
    for ngram_type in ngram_types:
        config = SpeculativeConfig(type=ngram_type)
        assert config.type == ngram_type


# ---------------------------------------------------------------------------
# AppConfig validation: unique model names
# ---------------------------------------------------------------------------

_ROCM_RUNTIME = RuntimeConfig(
    docker_image="ghcr.io/ggerganov/llama.cpp:server-rocm",
    source=ModelSource(local_path=Path("/models/main.gguf")),
)


def _model_cfg(name: str) -> ModelConfig:
    return ModelConfig(
        name=name,
        runtimes={"rocm": _ROCM_RUNTIME},
    )


def test_duplicate_model_names_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate model name"):
        AppConfig(
            models=[_model_cfg("gemma"), _model_cfg("gemma")],
        )


def test_unique_model_names_ok() -> None:
    config = AppConfig(
        models=[_model_cfg("gemma"), _model_cfg("qwen")],
    )
    assert len(config.models) == 2


# ---------------------------------------------------------------------------
# AppConfig validation: runtime config checks
# ---------------------------------------------------------------------------


def test_no_runtimes_rejected() -> None:
    with pytest.raises(ValidationError, match="must have at least one runtime"):
        ModelConfig(
            name="gemma",
            runtimes={},
        )


def test_unknown_runtime_key_rejected() -> None:
    with pytest.raises(ValidationError, match="should be 'rocm' or 'vulkan'"):
        ModelConfig.model_validate(
            {
                "name": "gemma",
                "runtimes": {
                    "invalid_key": {
                        "docker_image": "img",
                        "source": {"local_path": "/models/model.gguf"},
                    }
                },
            }
        )


def test_model_source_repo_id_without_filename_rejected() -> None:
    with pytest.raises(ValidationError, match="model source must specify"):
        ModelConfig(
            name="bad",
            runtimes={
                "rocm": RuntimeConfig(
                    docker_image="ghcr.io/ggerganov/llama.cpp:server-rocm",
                    source=ModelSource(repo_id="org/model"),
                )
            },
        )


def test_model_source_repo_id_with_filename_ok() -> None:
    config = ModelConfig(
        name="good",
        runtimes={
            "rocm": RuntimeConfig(
                docker_image="ghcr.io/ggerganov/llama.cpp:server-rocm",
                source=ModelSource(repo_id="org/model", filename="model.gguf"),
            )
        },
    )
    assert config.runtimes["rocm"].source.filename == "model.gguf"


def test_model_source_local_path_ok() -> None:
    config = _model_cfg("local")
    assert config.runtimes["rocm"].source.local_path is not None


def test_model_source_empty_rejected() -> None:
    with pytest.raises(ValidationError, match="model source must specify"):
        ModelConfig(
            name="empty",
            runtimes={
                "rocm": RuntimeConfig(
                    docker_image="ghcr.io/ggerganov/llama.cpp:server-rocm",
                    source=ModelSource(),
                )
            },
        )


# ---------------------------------------------------------------------------
# AppConfig validation: draft model source must also be explicit
# ---------------------------------------------------------------------------


def test_draft_model_source_must_be_explicit() -> None:
    with pytest.raises(ValidationError, match="draft_model source must"):
        ModelConfig(
            name="bad_draft",
            runtimes={
                "rocm": RuntimeConfig(
                    docker_image="ghcr.io/ggerganov/llama.cpp:server-rocm",
                    source=ModelSource(local_path=Path("/models/main.gguf")),
                    speculative=SpeculativeConfig(
                        type="draft",
                        draft_model=ModelSource(repo_id="org/draft"),
                    ),
                )
            },
        )


def test_draft_model_with_local_path_ok() -> None:
    config = ModelConfig(
        name="good_draft",
        runtimes={
            "rocm": RuntimeConfig(
                docker_image="ghcr.io/ggerganov/llama.cpp:server-rocm",
                source=ModelSource(local_path=Path("/models/main.gguf")),
                speculative=SpeculativeConfig(
                    type="draft",
                    draft_model=ModelSource(local_path=Path("/models/draft.gguf")),
                ),
            )
        },
    )
    assert config.runtimes["rocm"].speculative.draft_model is not None


# ---------------------------------------------------------------------------
# AppConfig validation: empty config is valid
# ---------------------------------------------------------------------------


def test_empty_config_valid() -> None:
    config = AppConfig()
    assert config.models == []


# ---------------------------------------------------------------------------
# AppConfig validation: nested unknown fields rejected
# ---------------------------------------------------------------------------


def test_nested_unknown_fields_rejected() -> None:
    """Unknown keys inside nested configuration objects should be rejected."""
    # 1. ModelSource unknown field
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ModelSource.model_validate({"repo_id": "foo/bar", "unknown_field": "val"})

    # 2. SpeculativeConfig unknown field
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SpeculativeConfig.model_validate({"type": "none", "unknown_field": "val"})

    # 3. RuntimeConfig unknown field
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        RuntimeConfig.model_validate(
            {
                "docker_image": "img",
                "source": {"local_path": "/ok.gguf"},
                "unknown_field": "val",
            }
        )

    # 4. ModelConfig unknown field
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ModelConfig.model_validate(
            {
                "name": "gemma",
                "runtimes": {
                    "rocm": {
                        "docker_image": "img",
                        "source": {"local_path": "/ok.gguf"},
                    }
                },
                "unknown_field": "val",
            }
        )
