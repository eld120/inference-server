from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SpeculativeType = Literal[
    "none",
    "draft",
    "draft-simple",
    "draft-mtp",
    "ngram-cache",
    "ngram-simple",
    "ngram-map-k",
    "ngram-map-k4v",
    "ngram-mod",
]
BackendState = Literal["stopped", "starting", "pulling", "running", "stopping", "error"]


class ModelSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repo_id: str | None = None
    filename: str | None = None
    local_path: Path | None = None
    revision: str = "main"

    def is_local(self) -> bool:
        return self.local_path is not None

    def label(self) -> str:
        if self.local_path is not None:
            return str(self.local_path)
        if self.repo_id and self.filename:
            return f"{self.repo_id}/{self.filename}"
        if self.repo_id:
            return self.repo_id
        return "unconfigured"


# Speculative types that require a separate draft model
_DRAFT_MODEL_TYPES: set[SpeculativeType] = {"draft", "draft-simple"}

# Speculative types that are self-contained (no draft model needed)
_SELF_CONTAINED_TYPES: set[SpeculativeType] = {"draft-mtp"}


class SpeculativeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: SpeculativeType = "none"
    draft_model: ModelSource | None = None

    @model_validator(mode="after")
    def _validate_draft_model_consistency(self) -> SpeculativeConfig:
        if self.type in _DRAFT_MODEL_TYPES and self.draft_model is None:
            msg = (
                f"speculative type '{self.type}' requires a draft_model "
                f"to be configured"
            )
            raise ValueError(msg)
        if self.type not in _DRAFT_MODEL_TYPES and self.draft_model is not None:
            msg = (
                f"draft_model is configured but speculative type is '{self.type}'; "
                f"set a speculative type that uses a draft model or remove draft_model"
            )
            raise ValueError(msg)
        return self


class BackendFamilyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    docker_image: str
    devices: list[str] = Field(default_factory=list)
    volumes: dict[str, str] = Field(default_factory=dict)
    shared_args: list[str] = Field(default_factory=list)
    router_supported: bool = True


class ModelPresetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    model: ModelSource
    backend_family: str
    speculative: SpeculativeConfig = Field(default_factory=SpeculativeConfig)
    extra_args: list[str] = Field(default_factory=list)
    bind_host: str = "127.0.0.1"
    connect_host: str = "127.0.0.1"


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    runtime_mode: Literal["auto", "router", "container"] = "auto"
    backend_families: dict[str, BackendFamilyConfig] = Field(default_factory=dict)
    models: list[ModelPresetConfig] = Field(default_factory=list)
    hf_cache_dir: Path = Field(
        default_factory=lambda: Path.home() / ".cache" / "huggingface" / "hub"
    )
    hf_token: str | None = None
    api_prefix: str = "/api"

    @model_validator(mode="after")
    def _validate_config(self) -> AppConfig:
        errors: list[str] = []

        # 1. Preset names must be unique
        names = [m.name for m in self.models]
        seen: set[str] = set()
        for name in names:
            if name in seen:
                errors.append(f"duplicate preset name: '{name}'")
            seen.add(name)

        # 2. Each preset must reference a valid backend family
        for m in self.models:
            if m.backend_family not in self.backend_families:
                errors.append(
                    f"preset '{m.name}' references unknown backend family "
                    f"'{m.backend_family}'"
                )

        # 3. Model source must be explicit
        for m in self.models:
            src = m.model
            if src.local_path is None:
                if src.repo_id is None or src.filename is None:
                    errors.append(
                        f"preset '{m.name}': model source must specify "
                        f"local_path, or both repo_id and filename"
                    )

        # 4. Speculative draft model source must also be explicit
        for m in self.models:
            if m.speculative.draft_model is not None:
                dm = m.speculative.draft_model
                if dm.local_path is None:
                    if dm.repo_id is None or dm.filename is None:
                        errors.append(
                            f"preset '{m.name}': draft_model source must "
                            f"specify local_path, or both repo_id and "
                            f"filename"
                        )



        if errors:
            msg = "invalid configuration:\n  - " + "\n  - ".join(errors)
            raise ValueError(msg)
        return self


class BackendStatus(BaseModel):
    name: str
    state: BackendState
    model: str
    speculative_type: SpeculativeType
    host: str
    port: int
    container_id: str | None = None
    last_error: str | None = None
    model_path: str | None = None
    draft_model_path: str | None = None
    base_url: str
    active: bool = False


class ServiceStatus(BaseModel):
    healthy: bool
    api_prefix: str
    active_backend: str | None
    backends: list[BackendStatus]


class HFSearchResult(BaseModel):
    repo_id: str
    likes: int | None = None
    downloads: int | None = None
    pipeline_tag: str | None = None
    tags: list[str] = Field(default_factory=list)
    author: str | None = None


class HFRepoFile(BaseModel):
    repo_id: str
    revision: str
    path: str


class HFDownloadRequest(BaseModel):
    repo_id: str
    filename: str | None = None
    revision: str = "main"


class HFDownloadResponse(BaseModel):
    repo_id: str
    filename: str
    revision: str
    local_path: str


class HFCachedFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repo_id: str
    revision: str | None = None
    refs: list[str] = Field(default_factory=list)
    filename: str
    local_path: str
    size_on_disk: int | None = None
