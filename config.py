from __future__ import annotations

import json
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from schemas import AppConfig


def default_config_path() -> Path:
    return Path.home() / ".config" / "inference-server" / "config.json"


class RuntimeSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="INF_", extra="ignore")

    config_path: Path = Field(default_factory=default_config_path)
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


def load_app_config(path: Path) -> AppConfig:
    if not path.exists():
        return AppConfig()
    data = json.loads(path.read_text())
    config = AppConfig.model_validate(data)

    config_dir = path.parent.resolve().absolute()
    for m in config.models:
        if m.source is not None and m.source.local_path is not None:
            lp = m.source.local_path
            if not lp.is_absolute():
                m.source._original_local_path_is_relative = True
                m.source.local_path = (config_dir / lp).resolve().absolute()
            else:
                m.source.local_path = lp.resolve().absolute()
        if m.speculative.draft_model is not None and m.speculative.draft_model.local_path is not None:
            lp = m.speculative.draft_model.local_path
            if not lp.is_absolute():
                m.speculative.draft_model._original_local_path_is_relative = True
                m.speculative.draft_model.local_path = (config_dir / lp).resolve().absolute()
            else:
                m.speculative.draft_model.local_path = lp.resolve().absolute()

        for rt in m.runtimes.values():
            if rt.source.local_path is not None:
                lp = rt.source.local_path
                if not lp.is_absolute():
                    rt.source._original_local_path_is_relative = True
                    rt.source.local_path = (config_dir / lp).resolve().absolute()
                else:
                    rt.source.local_path = lp.resolve().absolute()
            if (
                rt.speculative.draft_model is not None
                and rt.speculative.draft_model.local_path is not None
            ):
                lp = rt.speculative.draft_model.local_path
                if not lp.is_absolute():
                    rt.speculative.draft_model._original_local_path_is_relative = True
                    rt.speculative.draft_model.local_path = (
                        config_dir / lp
                    ).resolve().absolute()
                else:
                    rt.speculative.draft_model.local_path = lp.resolve().absolute()
    return config


def save_app_config(path: Path, config: AppConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    config_copy = config.model_copy(deep=True)
    config_dir = path.parent.resolve().absolute()
    for m_idx, m in enumerate(config_copy.models):
        orig_model = config.models[m_idx]
        if m.source is not None and m.source.local_path is not None:
            is_rel = getattr(orig_model.source, "_original_local_path_is_relative", False)
            if is_rel:
                lp = m.source.local_path.resolve().absolute()
                import os
                m.source.local_path = Path(os.path.relpath(lp, config_dir))
        if m.speculative.draft_model is not None and m.speculative.draft_model.local_path is not None:
            is_rel = getattr(
                orig_model.speculative.draft_model,
                "_original_local_path_is_relative",
                False,
            )
            if is_rel:
                lp = m.speculative.draft_model.local_path.resolve().absolute()
                import os
                m.speculative.draft_model.local_path = Path(os.path.relpath(lp, config_dir))
        for rt_name, rt in m.runtimes.items():
            orig_rt = orig_model.runtimes[rt_name]
            if rt.source.local_path is not None:
                is_rel = getattr(orig_rt.source, "_original_local_path_is_relative", False)
                if is_rel:
                    lp = rt.source.local_path.resolve().absolute()
                    import os
                    rt.source.local_path = Path(os.path.relpath(lp, config_dir))
            if (
                rt.speculative.draft_model is not None
                and rt.speculative.draft_model.local_path is not None
            ):
                is_rel = getattr(
                    orig_rt.speculative.draft_model,
                    "_original_local_path_is_relative",
                    False,
                )
                if is_rel:
                    lp = rt.speculative.draft_model.local_path.resolve().absolute()
                    import os
                    rt.speculative.draft_model.local_path = Path(
                        os.path.relpath(lp, config_dir)
                    )
    path.write_text(config_copy.model_dump_json(indent=2, exclude_unset=True))


def effective_hf_token(runtime: RuntimeSettings, app_config: AppConfig) -> str | None:
    return runtime.hf_token or app_config.hf_token
