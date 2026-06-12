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
    BackendConfig,
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
    docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
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
    with pytest.raises(
        ValidationError,
        match="source/backends usage or at least one runtime configured",
    ):
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
                    docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
                    source=ModelSource(repo_id="org/model"),
                )
            },
        )


def test_model_source_repo_id_with_filename_ok() -> None:
    config = ModelConfig(
        name="good",
        runtimes={
            "rocm": RuntimeConfig(
                docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
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
                    docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
                    source=ModelSource(),
                )
            },
        )


def test_legacy_mmproj_extra_args_translate_to_mmproj_field() -> None:
    config = AppConfig.model_validate(
        {
            "backends": {
                "rocm": {
                    "docker_image": "inference-server-llama-rocm:7.2.1-7e50ef7",
                }
            },
            "models": [
                {
                    "name": "vision",
                    "source": {
                        "repo_id": "org/model",
                        "filename": "model.gguf",
                    },
                    "extra_args": ["--mmproj", "mmproj-BF16.gguf", "--temp", "0.7"],
                }
            ],
        }
    )

    model = config.models[0]
    assert model.mmproj == ModelSource(
        repo_id="org/model",
        filename="mmproj-BF16.gguf",
        revision="main",
    )
    assert model.extra_args == ["--temp", "0.7"]


# ---------------------------------------------------------------------------
# AppConfig validation: draft model source must also be explicit
# ---------------------------------------------------------------------------


def test_draft_model_source_must_be_explicit() -> None:
    with pytest.raises(ValidationError, match="draft_model source must"):
        ModelConfig(
            name="bad_draft",
            runtimes={
                "rocm": RuntimeConfig(
                    docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
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
                docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
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


def test_shared_backend_models_require_backends() -> None:
    with pytest.raises(ValueError, match="top-level backends must be configured"):
        AppConfig(
            models=[
                ModelConfig(
                    name="gemma",
                    source=ModelSource(local_path=Path("/models/main.gguf")),
                )
            ]
        )


def test_mixed_legacy_shared_backend_models_require_backends() -> None:
    with pytest.raises(ValueError, match="top-level backends must be configured"):
        AppConfig(
            models=[
                ModelConfig(
                    name="legacy",
                    runtimes={
                        "rocm": RuntimeConfig(
                            docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
                            source=ModelSource(local_path=Path("/models/main.gguf")),
                        )
                    },
                ),
                ModelConfig(
                    name="shared",
                    source=ModelSource(local_path=Path("/models/other.gguf")),
                ),
            ]
        )


def test_shared_backend_model_valid() -> None:
    config = AppConfig(
        backends={
            "rocm": BackendConfig(
                docker_image="inference-server-llama-rocm:7.2.1-7e50ef7"
            )
        },
        models=[
            ModelConfig(
                name="gemma",
                source=ModelSource(local_path=Path("/models/main.gguf")),
            )
        ],
    )
    assert config.models[0].source is not None


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


def test_mixed_legacy_shared_backend_shape_rejected() -> None:
    # 1. runtimes + source
    with pytest.raises(ValidationError, match="cannot mix legacy and new shared-backend shapes: found 'runtimes' alongside top-level field\\(s\\) source"):
        ModelConfig.model_validate(
            {
                "name": "mixed-1",
                "runtimes": {
                    "rocm": {
                        "docker_image": "img",
                        "source": {"local_path": "/ok.gguf"},
                    }
                },
                "source": {"local_path": "/ok.gguf"},
            }
        )

    # 2. runtimes + extra_args (non-empty)
    with pytest.raises(ValidationError, match="cannot mix legacy and new shared-backend shapes: found 'runtimes' alongside top-level field\\(s\\) extra_args"):
        ModelConfig.model_validate(
            {
                "name": "mixed-2",
                "runtimes": {
                    "rocm": {
                        "docker_image": "img",
                        "source": {"local_path": "/ok.gguf"},
                    }
                },
                "extra_args": ["--foo"],
            }
        )

    # 2b. runtimes + extra_args: []
    with pytest.raises(ValidationError, match="cannot mix legacy and new shared-backend shapes: found 'runtimes' alongside top-level field\\(s\\) extra_args"):
        ModelConfig.model_validate(
            {
                "name": "mixed-2b",
                "runtimes": {
                    "rocm": {
                        "docker_image": "img",
                        "source": {"local_path": "/ok.gguf"},
                    }
                },
                "extra_args": [],
            }
        )

    # 3. runtimes + non-default speculative
    with pytest.raises(ValidationError, match="cannot mix legacy and new shared-backend shapes: found 'runtimes' alongside top-level field\\(s\\) speculative"):
        ModelConfig.model_validate(
            {
                "name": "mixed-3",
                "runtimes": {
                    "rocm": {
                        "docker_image": "img",
                        "source": {"local_path": "/ok.gguf"},
                    }
                },
                "speculative": {"type": "draft-mtp"},
            }
        )

    # 3b. runtimes + speculative: {}
    with pytest.raises(ValidationError, match="cannot mix legacy and new shared-backend shapes: found 'runtimes' alongside top-level field\\(s\\) speculative"):
        ModelConfig.model_validate(
            {
                "name": "mixed-3b",
                "runtimes": {
                    "rocm": {
                        "docker_image": "img",
                        "source": {"local_path": "/ok.gguf"},
                    }
                },
                "speculative": {},
            }
        )

    # 3c. runtimes + speculative: {"type": "none"}
    with pytest.raises(ValidationError, match="cannot mix legacy and new shared-backend shapes: found 'runtimes' alongside top-level field\\(s\\) speculative"):
        ModelConfig.model_validate(
            {
                "name": "mixed-3c",
                "runtimes": {
                    "rocm": {
                        "docker_image": "img",
                        "source": {"local_path": "/ok.gguf"},
                    }
                },
                "speculative": {"type": "none"},
            }
        )

    # 4. runtimes + multiple top-level fields
    with pytest.raises(ValidationError, match="cannot mix legacy and new shared-backend shapes: found 'runtimes' alongside top-level field\\(s\\) extra_args, source"):
        ModelConfig.model_validate(
            {
                "name": "mixed-4",
                "runtimes": {
                    "rocm": {
                        "docker_image": "img",
                        "source": {"local_path": "/ok.gguf"},
                    }
                },
                "source": {"local_path": "/ok.gguf"},
                "extra_args": ["--foo"],
            }
        )


def test_valid_legacy_and_new_shapes_pass() -> None:
    # Legacy shape
    legacy_cfg = ModelConfig.model_validate(
        {
            "name": "legacy",
            "runtimes": {
                "rocm": {
                    "docker_image": "img",
                    "source": {"local_path": "/ok.gguf"},
                }
            },
        }
    )
    assert legacy_cfg.source is None
    assert legacy_cfg.runtimes["rocm"].source.local_path == Path("/ok.gguf")

    # New shape
    new_cfg = ModelConfig.model_validate(
        {
            "name": "new",
            "source": {"local_path": "/ok.gguf"},
            "extra_args": ["--foo"],
            "speculative": {"type": "draft-mtp"},
        }
    )
    assert not new_cfg.runtimes
    assert new_cfg.source.local_path == Path("/ok.gguf")
    assert new_cfg.extra_args == ["--foo"]
    assert new_cfg.speculative.type == "draft-mtp"


# ---------------------------------------------------------------------------
# mmproj validation tests
# ---------------------------------------------------------------------------


def test_valid_structured_mmproj_local_path() -> None:
    # Top-level legacy shape
    cfg = ModelConfig.model_validate(
        {
            "name": "model_local_mmproj",
            "source": {"local_path": "/path/to/model.gguf"},
            "mmproj": {"local_path": "/path/to/mmproj.gguf"},
        }
    )
    assert cfg.mmproj is not None
    assert cfg.mmproj.local_path == Path("/path/to/mmproj.gguf")

    # Runtime-based shape
    cfg_rt = ModelConfig.model_validate(
        {
            "name": "model_local_mmproj_rt",
            "runtimes": {
                "rocm": {
                    "docker_image": "img",
                    "source": {"local_path": "/path/to/model.gguf"},
                    "mmproj": {"local_path": "/path/to/mmproj.gguf"},
                }
            },
        }
    )
    assert cfg_rt.runtimes["rocm"].mmproj is not None
    assert cfg_rt.runtimes["rocm"].mmproj.local_path == Path("/path/to/mmproj.gguf")


def test_valid_structured_mmproj_repo_and_filename() -> None:
    # Top-level legacy shape
    cfg = ModelConfig.model_validate(
        {
            "name": "model_repo_mmproj",
            "source": {"local_path": "/path/to/model.gguf"},
            "mmproj": {"repo_id": "org/model", "filename": "mmproj.gguf"},
        }
    )
    assert cfg.mmproj is not None
    assert cfg.mmproj.repo_id == "org/model"
    assert cfg.mmproj.filename == "mmproj.gguf"

    # Runtime-based shape
    cfg_rt = ModelConfig.model_validate(
        {
            "name": "model_repo_mmproj_rt",
            "runtimes": {
                "rocm": {
                    "docker_image": "img",
                    "source": {"local_path": "/path/to/model.gguf"},
                    "mmproj": {"repo_id": "org/model", "filename": "mmproj.gguf"},
                }
            },
        }
    )
    assert cfg_rt.runtimes["rocm"].mmproj is not None
    assert cfg_rt.runtimes["rocm"].mmproj.repo_id == "org/model"
    assert cfg_rt.runtimes["rocm"].mmproj.filename == "mmproj.gguf"


def test_invalid_structured_mmproj_only_repo_id() -> None:
    # Top-level legacy shape
    with pytest.raises(ValidationError) as exc_info:
        ModelConfig.model_validate(
            {
                "name": "model_invalid_repo_only",
                "source": {"local_path": "/path/to/model.gguf"},
                "mmproj": {"repo_id": "org/model"},
            }
        )
    assert "mmproj source must specify local_path" in str(exc_info.value)

    # Runtime-based shape
    with pytest.raises(ValidationError) as exc_info_rt:
        ModelConfig.model_validate(
            {
                "name": "model_invalid_repo_only_rt",
                "runtimes": {
                    "rocm": {
                        "docker_image": "img",
                        "source": {"local_path": "/path/to/model.gguf"},
                        "mmproj": {"repo_id": "org/model"},
                    }
                },
            }
        )
    assert "mmproj source must specify local_path" in str(exc_info_rt.value)


def test_invalid_structured_mmproj_only_filename() -> None:
    # Top-level legacy shape
    with pytest.raises(ValidationError) as exc_info:
        ModelConfig.model_validate(
            {
                "name": "model_invalid_file_only",
                "source": {"local_path": "/path/to/model.gguf"},
                "mmproj": {"filename": "mmproj.gguf"},
            }
        )
    assert "mmproj source must specify local_path" in str(exc_info.value)

    # Runtime-based shape
    with pytest.raises(ValidationError) as exc_info_rt:
        ModelConfig.model_validate(
            {
                "name": "model_invalid_file_only_rt",
                "runtimes": {
                    "rocm": {
                        "docker_image": "img",
                        "source": {"local_path": "/path/to/model.gguf"},
                        "mmproj": {"filename": "mmproj.gguf"},
                    }
                },
            }
        )
    assert "mmproj source must specify local_path" in str(exc_info_rt.value)


def test_legacy_mmproj_normalization() -> None:
    # With local source
    cfg_local = ModelConfig.model_validate(
        {
            "name": "model_legacy_local",
            "source": {"local_path": "/path/to/model.gguf"},
            "extra_args": ["--mmproj", "mmproj.gguf", "--other"],
        }
    )
    assert cfg_local.mmproj is not None
    assert cfg_local.mmproj.local_path == Path("/path/to/mmproj.gguf")
    assert cfg_local.extra_args == ["--other"]

    # With repo source
    cfg_repo = ModelConfig.model_validate(
        {
            "name": "model_legacy_repo",
            "source": {"repo_id": "org/model", "filename": "model.gguf"},
            "extra_args": ["--mmproj=mmproj.gguf"],
        }
    )
    assert cfg_repo.mmproj is not None
    assert cfg_repo.mmproj.repo_id == "org/model"
    assert cfg_repo.mmproj.filename == "mmproj.gguf"

