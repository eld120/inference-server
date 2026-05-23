from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

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
BackendState = Literal["stopped", "starting", "running", "stopping", "error"]


class ModelSource(BaseModel):
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


class SpeculativeConfig(BaseModel):
    type: SpeculativeType = "none"
    draft_model: ModelSource | None = None


class BackendConfig(BaseModel):
    name: str
    model: ModelSource
    speculative: SpeculativeConfig = Field(default_factory=SpeculativeConfig)
    host: str = "127.0.0.1"
    port: int = 8080
    llama_server_bin: str | None = None
    extra_args: list[str] = Field(default_factory=list)


class AppConfig(BaseModel):
    backends: list[BackendConfig] = Field(default_factory=list)
    default_backend: str | None = None
    hf_cache_dir: Path = Field(
        default_factory=lambda: Path.home() / ".cache" / "huggingface" / "hub"
    )
    hf_token: str | None = None
    api_prefix: str = "/api"


class BackendStatus(BaseModel):
    name: str
    state: BackendState
    model: str
    speculative_type: SpeculativeType
    host: str
    port: int
    pid: int | None = None
    returncode: int | None = None
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
    repo_id: str
    revision: str | None = None
    filename: str
    local_path: str
    size_on_disk: int | None = None
