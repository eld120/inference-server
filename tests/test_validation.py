"""Tests for configuration validation.

Verifies that AppConfig and SpeculativeConfig model validators catch
invalid configurations early with clear error messages.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from schemas import (
    AppConfig,
    BackendFamilyConfig,
    ModelPresetConfig,
    ModelSource,
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
# AppConfig validation: unique preset names
# ---------------------------------------------------------------------------

_ROCM_FAMILY = BackendFamilyConfig(
    docker_image="ghcr.io/ggerganov/llama.cpp:server-rocm"
)


def _preset(name: str, *, family: str = "rocm") -> ModelPresetConfig:
    return ModelPresetConfig(
        name=name,
        backend_family=family,
        model=ModelSource(local_path=Path(f"/models/{name}.gguf")),
    )


def test_duplicate_preset_names_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate preset name"):
        AppConfig(
            backend_families={"rocm": _ROCM_FAMILY},
            models=[_preset("gemma"), _preset("gemma")],
        )


def test_unique_preset_names_ok() -> None:
    config = AppConfig(
        backend_families={"rocm": _ROCM_FAMILY},
        models=[_preset("gemma"), _preset("qwen")],
    )
    assert len(config.models) == 2


# ---------------------------------------------------------------------------
# AppConfig validation: backend family references
# ---------------------------------------------------------------------------


def test_unknown_backend_family_rejected() -> None:
    with pytest.raises(ValueError, match="unknown backend family.*vulkan"):
        AppConfig(
            backend_families={"rocm": _ROCM_FAMILY},
            models=[_preset("gemma", family="vulkan")],
        )


def test_valid_backend_family_ok() -> None:
    config = AppConfig(
        backend_families={"rocm": _ROCM_FAMILY},
        models=[_preset("gemma", family="rocm")],
    )
    assert config.models[0].backend_family == "rocm"


# ---------------------------------------------------------------------------
# AppConfig validation: explicit model source
# ---------------------------------------------------------------------------


def test_model_source_repo_id_without_filename_rejected() -> None:
    with pytest.raises(ValueError, match="model source must specify"):
        AppConfig(
            backend_families={"rocm": _ROCM_FAMILY},
            models=[
                ModelPresetConfig(
                    name="bad",
                    backend_family="rocm",
                    model=ModelSource(repo_id="org/model"),
                )
            ],
        )


def test_model_source_repo_id_with_filename_ok() -> None:
    config = AppConfig(
        backend_families={"rocm": _ROCM_FAMILY},
        models=[
            ModelPresetConfig(
                name="good",
                backend_family="rocm",
                model=ModelSource(repo_id="org/model", filename="model.gguf"),
            )
        ],
    )
    assert config.models[0].model.filename == "model.gguf"


def test_model_source_local_path_ok() -> None:
    config = AppConfig(
        backend_families={"rocm": _ROCM_FAMILY},
        models=[_preset("local")],
    )
    assert config.models[0].model.local_path is not None


def test_model_source_empty_rejected() -> None:
    with pytest.raises(ValueError, match="model source must specify"):
        AppConfig(
            backend_families={"rocm": _ROCM_FAMILY},
            models=[
                ModelPresetConfig(
                    name="empty",
                    backend_family="rocm",
                    model=ModelSource(),
                )
            ],
        )


# ---------------------------------------------------------------------------
# AppConfig validation: draft model source must also be explicit
# ---------------------------------------------------------------------------


def test_draft_model_source_must_be_explicit() -> None:
    with pytest.raises(ValueError, match="draft_model source must"):
        AppConfig(
            backend_families={"rocm": _ROCM_FAMILY},
            models=[
                ModelPresetConfig(
                    name="bad_draft",
                    backend_family="rocm",
                    model=ModelSource(local_path=Path("/models/main.gguf")),
                    speculative=SpeculativeConfig(
                        type="draft",
                        draft_model=ModelSource(repo_id="org/draft"),
                    ),
                )
            ],
        )


def test_draft_model_with_local_path_ok() -> None:
    config = AppConfig(
        backend_families={"rocm": _ROCM_FAMILY},
        models=[
            ModelPresetConfig(
                name="good_draft",
                backend_family="rocm",
                model=ModelSource(local_path=Path("/models/main.gguf")),
                speculative=SpeculativeConfig(
                    type="draft",
                    draft_model=ModelSource(
                        local_path=Path("/models/draft.gguf")
                    ),
                ),
            )
        ],
    )
    assert config.models[0].speculative.draft_model is not None


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# AppConfig validation: empty config is valid
# ---------------------------------------------------------------------------


def test_empty_config_valid() -> None:
    config = AppConfig()
    assert config.models == []
    assert config.backend_families == {}


# ---------------------------------------------------------------------------
# AppConfig validation: multiple errors reported
# ---------------------------------------------------------------------------


def test_multiple_errors_reported_together() -> None:
    """Multiple validation failures should be reported in a single error."""
    with pytest.raises(ValueError, match="invalid configuration") as exc_info:
        AppConfig(
            backend_families={"rocm": _ROCM_FAMILY},
            models=[
                ModelPresetConfig(
                    name="bad1",
                    backend_family="vulkan",  # unknown family
                    model=ModelSource(repo_id="org/model"),  # missing filename
                ),
                ModelPresetConfig(
                    name="bad1",  # duplicate name
                    backend_family="rocm",
                    model=ModelSource(local_path=Path("/ok.gguf")),
                ),
            ],
        )
    error_msg = str(exc_info.value)
    assert "duplicate preset name" in error_msg
    assert "unknown backend family" in error_msg
    assert "model source must specify" in error_msg


def test_nested_unknown_fields_rejected() -> None:
    """Unknown keys inside nested configuration objects should be rejected."""
    from pydantic import ValidationError

    # 1. ModelSource unknown field
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ModelSource.model_validate({"repo_id": "foo/bar", "unknown_field": "val"})

    # 2. SpeculativeConfig unknown field
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SpeculativeConfig.model_validate({"type": "none", "unknown_field": "val"})

    # 3. BackendFamilyConfig unknown field
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        BackendFamilyConfig.model_validate(
            {"docker_image": "img", "unknown_field": "val"}
        )

    # 4. ModelPresetConfig unknown field
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ModelPresetConfig.model_validate(
            {
                "name": "gemma",
                "backend_family": "rocm",
                "model": {"local_path": "/ok.gguf"},
                "unknown_field": "val",
            }
        )
