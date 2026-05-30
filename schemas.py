from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator, field_validator, PrivateAttr

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
RuntimeState = Literal["stopped", "starting", "pulling", "running", "stopping", "error"]


class ModelSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repo_id: str | None = None
    filename: str | None = None
    local_path: Path | None = None
    revision: str = "main"

    _original_local_path_is_relative: bool = PrivateAttr(default=False)

    @field_validator("local_path", mode="after")
    @classmethod
    def _validate_local_path(cls, v: Path | None) -> Path | None:
        if v is not None:
            return v.expanduser()
        return v

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


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: ModelSource
    docker_image: str
    devices: list[str] = Field(default_factory=list)
    volumes: dict[str, str] = Field(default_factory=dict)
    shared_args: list[str] = Field(default_factory=list)
    extra_args: list[str] = Field(default_factory=list)
    speculative: SpeculativeConfig = Field(default_factory=SpeculativeConfig)
    bind_host: str = "127.0.0.1"
    connect_host: str = "127.0.0.1"

    @model_validator(mode="after")
    def _validate_speculative_args(self) -> RuntimeConfig:
        for arg in self.extra_args:
            if arg == "--spec-type" or arg.startswith("--spec-type="):
                raise ValueError(
                    "do not specify '--spec-type' in extra_args; "
                    "use the structured 'speculative' configuration field instead"
                )
        return self


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    runtimes: dict[Literal["rocm", "vulkan"], RuntimeConfig]

    @model_validator(mode="after")
    def _validate_runtimes(self) -> ModelConfig:
        if not self.runtimes:
            raise ValueError("model must have at least one runtime configured")

        for rt_name, rt_cfg in self.runtimes.items():
            src = rt_cfg.source
            if src.local_path is None:
                if src.repo_id is None or src.filename is None:
                    raise ValueError(
                        f"model '{self.name}' runtime '{rt_name}': "
                        "model source must specify local_path, "
                        "or both repo_id and filename"
                    )

            if rt_cfg.speculative.draft_model is not None:
                dm = rt_cfg.speculative.draft_model
                if dm.local_path is None:
                    if dm.repo_id is None or dm.filename is None:
                        raise ValueError(
                            f"model '{self.name}' runtime '{rt_name}': "
                            "draft_model source must specify local_path, "
                            "or both repo_id and filename"
                        )
        return self


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True


class ModelStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    state: RuntimeState
    active: bool = False
    active_runtime: str | None = None
    model: str | None = None
    speculative_type: SpeculativeType | None = None
    host: str
    port: int
    container_id: str | None = None
    last_error: str | None = None
    model_path: str | None = None
    draft_model_path: str | None = None
    base_url: str


RuntimeStatus = ModelStatus


class ModelResource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    config: ModelConfig
    status: ModelStatus

    @property
    def state(self) -> RuntimeState:
        return self.status.state

    @property
    def active(self) -> bool:
        return self.status.active

    @property
    def container_id(self) -> str | None:
        return self.status.container_id

    @property
    def last_error(self) -> str | None:
        return self.status.last_error

    @property
    def model_path(self) -> str | None:
        return self.status.model_path

    @property
    def draft_model_path(self) -> str | None:
        return self.status.draft_model_path

    @property
    def base_url(self) -> str:
        return self.status.base_url

    @property
    def host(self) -> str:
        return self.status.host

    @property
    def port(self) -> int:
        return self.status.port

    @property
    def model(self) -> str:
        active_rt = self.status.active_runtime
        if active_rt and active_rt in self.config.runtimes:
            return self.config.runtimes[active_rt].source.label()
        if self.config.runtimes:
            first_rt = list(self.config.runtimes.keys())[0]
            return self.config.runtimes[first_rt].source.label()
        return "unconfigured"

    @property
    def speculative_type(self) -> SpeculativeType:
        active_rt = self.status.active_runtime
        if active_rt and active_rt in self.config.runtimes:
            return self.config.runtimes[active_rt].speculative.type
        if self.config.runtimes:
            first_rt = list(self.config.runtimes.keys())[0]
            return self.config.runtimes[first_rt].speculative.type
        return "none"


class ModelsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    models: list[ModelResource]


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    models: list[ModelConfig] = Field(default_factory=list)
    hf_cache_dir: Path = Field(
        default_factory=lambda: Path.home() / ".cache" / "huggingface" / "hub"
    )
    hf_token: str | None = None
    api_prefix: str = "/api"

    @field_validator("hf_cache_dir", mode="after")
    @classmethod
    def _validate_hf_cache_dir(cls, v: Path) -> Path:
        return v.expanduser().resolve()

    @model_validator(mode="after")
    def _validate_config(self) -> AppConfig:
        errors: list[str] = []

        names = [m.name for m in self.models]
        seen: set[str] = set()
        for name in names:
            if name in seen:
                errors.append(f"duplicate model name: '{name}'")
            seen.add(name)

        if errors:
            msg = "invalid configuration:\n  - " + "\n  - ".join(errors)
            raise ValueError(msg)
        return self


class ServiceStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    healthy: bool
    api_prefix: str
    config_path: str
    active_model: str | None
    active_runtime: str | None = None
    active_container_id: str | None = None
    last_error: str | None = None
    models: list[ModelResource]


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


class LogsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    logs: list[str]


class OpenAIModelSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str


class OpenAIModelListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    object: Literal["list"] = "list"
    data: list[OpenAIModelSummary]


class HFCachedFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repo_id: str
    revision: str | None = None
    refs: list[str] = Field(default_factory=list)
    filename: str
    local_path: str
    size_on_disk: int | None = None


class LoadModelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    runtime: Literal["rocm", "vulkan"]
