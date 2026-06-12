from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Protocol

import httpx

from schemas import (
    HFCachedFile,
    HFDownloadResponse,
    HFRepoFile,
    HFSearchResult,
    ModelConfig,
    ModelResource,
    ModelSource,
    ModelStatus,
    ServiceStatus,
)


class HFModelInfoProtocol(Protocol):
    modelId: str
    likes: int | None
    downloads: int | None
    pipeline_tag: str | None
    tags: Sequence[str] | None
    author: str | None


class HFApiProtocol(Protocol):
    def list_models(
        self, search: str | None = None, limit: int = 20
    ) -> Iterable[HFModelInfoProtocol]: ...


class ModelResolverProtocol(Protocol):
    cache_dir: Path

    def resolve_source(self, source: ModelSource) -> Path: ...


class HuggingFaceServiceProtocol(ModelResolverProtocol, Protocol):
    def search_models(
        self, query: str | None, limit: int = 20
    ) -> list[HFSearchResult]: ...

    def repo_files(self, repo_id: str, revision: str = "main") -> list[HFRepoFile]: ...

    def cache_files(self) -> list[HFCachedFile]: ...

    def download(self, source: ModelSource) -> HFDownloadResponse: ...


class ProxySessionProtocol(Protocol):
    client: httpx.AsyncClient
    response: httpx.Response


class ActiveModelProtocol(Protocol):
    config: ModelConfig


class ModelRuntimeManagerProtocol(Protocol):
    def status(self) -> ServiceStatus: ...

    def model_statuses(self) -> list[ModelStatus]: ...

    def model_resources(self) -> list[ModelResource]: ...

    def model_resource(self, name: str) -> ModelResource: ...

    async def load(self, name: str, runtime: str) -> ModelResource: ...

    async def unload(self, name: str | None = None) -> ModelResource | None: ...

    def active_model(self) -> ActiveModelProtocol | None: ...

    async def open_proxy_session(
        self, path: str, request: httpx.Request
    ) -> ProxySessionProtocol: ...

    async def get_logs(self, name: str) -> list[str]: ...

    async def cleanup(self) -> None: ...

    async def handle_runtime_communication_failure(self, exc: Exception) -> None: ...

    def models(self) -> list[ModelConfig]: ...

    def find_model_for_name(self, model_name: str) -> str | None: ...
