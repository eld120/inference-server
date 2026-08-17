from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from schemas import AppConfig, ModelSource


def default_config_path() -> Path:
    return Path.home() / ".config" / "inference-server" / "config.json"


def default_runtime_log_dir() -> Path:
    return Path.home() / ".local" / "state" / "inference-server" / "logs"


class RuntimeSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="INF_", extra="ignore")

    config_path: Path = Field(default_factory=default_config_path)
    runtime_log_dir: Path = Field(default_factory=default_runtime_log_dir)
    host: str = "0.0.0.0"
    port: int = 8000
    model_load_timeout_seconds: int = 1800
    model_readiness_probe_timeout_seconds: float = 5.0
    runtime_port: int = Field(
        default=39281,
        validation_alias=AliasChoices(
            "INF_RUNTIME_PORT", "INF_BACKEND_PORT"
        ),
    )
    hf_token: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "INF_HF_TOKEN", "HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"
        ),
    )
    observability_db_path: Path | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "INF_OBSERVABILITY_DB_PATH", "OBSERVABILITY_DB_PATH"
        ),
    )


def _normalize_source(source: ModelSource | None, config_dir: Path) -> None:
    if source is not None and source.local_path is not None:
        lp = source.local_path
        if not lp.is_absolute():
            source._original_local_path_is_relative = True
            source.local_path = (config_dir / lp).resolve().absolute()
        else:
            source.local_path = lp.resolve().absolute()


def _serialize_source(source: ModelSource | None, orig_source: ModelSource | None, config_dir: Path) -> None:
    if source is not None and source.local_path is not None and orig_source is not None:
        is_rel = getattr(orig_source, "_original_local_path_is_relative", False)
        if is_rel:
            lp = source.local_path.resolve().absolute()
            source.local_path = Path(os.path.relpath(lp, config_dir))


def load_app_config(path: Path) -> AppConfig:
    if not path.exists():
        return AppConfig()
    data = json.loads(path.read_text())
    config = AppConfig.model_validate(data)

    config_dir = path.parent.resolve().absolute()
    for m in config.models:
        _normalize_source(m.source, config_dir)
        _normalize_source(m.mmproj, config_dir)
        _normalize_source(m.vae, config_dir)
        _normalize_source(m.clip_l, config_dir)
        _normalize_source(m.t5xxl, config_dir)
        _normalize_source(m.clip_vision, config_dir)
        _normalize_source(m.control_net, config_dir)
        _normalize_source(m.speculative.draft_model, config_dir)

        for rt in m.runtimes.values():
            _normalize_source(rt.source, config_dir)
            _normalize_source(rt.mmproj, config_dir)
            _normalize_source(rt.vae, config_dir)
            _normalize_source(rt.clip_l, config_dir)
            _normalize_source(rt.t5xxl, config_dir)
            _normalize_source(rt.clip_vision, config_dir)
            _normalize_source(rt.control_net, config_dir)
            _normalize_source(rt.speculative.draft_model, config_dir)
    return config


def save_app_config(path: Path, config: AppConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    config_copy = config.model_copy(deep=True)
    config_dir = path.parent.resolve().absolute()
    for m_idx, m in enumerate(config_copy.models):
        orig_model = config.models[m_idx]
        _serialize_source(m.source, orig_model.source, config_dir)
        _serialize_source(m.mmproj, orig_model.mmproj, config_dir)
        _serialize_source(m.vae, orig_model.vae, config_dir)
        _serialize_source(m.clip_l, orig_model.clip_l, config_dir)
        _serialize_source(m.t5xxl, orig_model.t5xxl, config_dir)
        _serialize_source(m.clip_vision, orig_model.clip_vision, config_dir)
        _serialize_source(m.control_net, orig_model.control_net, config_dir)
        _serialize_source(m.speculative.draft_model, orig_model.speculative.draft_model, config_dir)

        for rt_name, rt in m.runtimes.items():
            orig_rt = orig_model.runtimes[rt_name]
            _serialize_source(rt.source, orig_rt.source, config_dir)
            _serialize_source(rt.mmproj, orig_rt.mmproj, config_dir)
            _serialize_source(rt.vae, orig_rt.vae, config_dir)
            _serialize_source(rt.clip_l, orig_rt.clip_l, config_dir)
            _serialize_source(rt.t5xxl, orig_rt.t5xxl, config_dir)
            _serialize_source(rt.clip_vision, orig_rt.clip_vision, config_dir)
            _serialize_source(rt.control_net, orig_rt.control_net, config_dir)
            _serialize_source(rt.speculative.draft_model, orig_rt.speculative.draft_model, config_dir)

    path.write_text(config_copy.model_dump_json(indent=2, exclude_unset=True))


def effective_hf_token(runtime: RuntimeSettings, app_config: AppConfig) -> str | None:
    return runtime.hf_token or app_config.hf_token
