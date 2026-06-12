from __future__ import annotations

from pathlib import Path
from typing import Any

from schemas import AppConfig, BackendConfig, ModelConfig, ModelSource, SpeculativeConfig

ROCM_IMAGE = "inference-server-llama-rocm:7.2.1-7e50ef7"
VULKAN_IMAGE = "inference-server-llama-vulkan:26.04-7e50ef7"


def make_backends(
    *,
    rocm_image: str = ROCM_IMAGE,
    vulkan_image: str | None = None,
    rocm_devices: list[str] | None = None,
    vulkan_devices: list[str] | None = None,
    rocm_shared_args: list[str] | None = None,
    vulkan_shared_args: list[str] | None = None,
) -> dict[str, BackendConfig]:
    backends: dict[str, BackendConfig] = {
        "rocm": BackendConfig(
            docker_image=rocm_image,
            devices=rocm_devices or [],
            shared_args=rocm_shared_args or [],
        )
    }
    if vulkan_image is not None:
        backends["vulkan"] = BackendConfig(
            docker_image=vulkan_image,
            devices=vulkan_devices or [],
            shared_args=vulkan_shared_args or [],
        )
    return backends


def make_model(
    name: str,
    source: Path | ModelSource,
    *,
    mmproj: Path | ModelSource | None = None,
    extra_args: list[str] | None = None,
    speculative: SpeculativeConfig | dict[str, Any] | None = None,
) -> ModelConfig:
    resolved_source = (
        source if isinstance(source, ModelSource) else ModelSource(local_path=source)
    )
    resolved_mmproj = None
    if mmproj is not None:
        resolved_mmproj = (
            mmproj if isinstance(mmproj, ModelSource) else ModelSource(local_path=mmproj)
        )
    return ModelConfig(
        name=name,
        source=resolved_source,
        mmproj=resolved_mmproj,
        extra_args=extra_args or [],
        speculative=(
            speculative
            if isinstance(speculative, SpeculativeConfig)
            else SpeculativeConfig.model_validate(speculative or {})
        ),
    )


def make_app_config(
    *models: ModelConfig,
    backends: dict[str, BackendConfig] | None = None,
    hf_cache_dir: Path | None = None,
    api_prefix: str = "/api",
) -> AppConfig:
    kwargs: dict[str, object] = {
        "models": list(models),
        "backends": backends or make_backends(),
        "api_prefix": api_prefix,
    }
    if hf_cache_dir is not None:
        kwargs["hf_cache_dir"] = hf_cache_dir
    return AppConfig(**kwargs)
