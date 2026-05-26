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
    backend_port: int = 39281
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
    return AppConfig.model_validate(data)


def save_app_config(path: Path, config: AppConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(config.model_dump_json(indent=2))


def effective_hf_token(runtime: RuntimeSettings, app_config: AppConfig) -> str | None:
    return runtime.hf_token or app_config.hf_token
