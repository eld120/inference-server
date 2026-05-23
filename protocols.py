from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Protocol

import httpx

from schemas import (
    BackendConfig,
    BackendStatus,
    HFCachedFile,
    HFDownloadResponse,
    HFRepoFile,
    HFSearchResult,
    ModelSource,
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


class ActiveBackendProtocol(Protocol):
    config: BackendConfig


class BackendManagerProtocol(Protocol):
    def status(self) -> ServiceStatus: ...

    def backend_statuses(self) -> list[BackendStatus]: ...

    async def load(self, name: str) -> BackendStatus: ...

    async def unload(self, name: str | None = None) -> BackendStatus | None: ...

    def active_backend(self) -> ActiveBackendProtocol | None: ...

    async def open_proxy_session(
        self, path: str, request: httpx.Request
    ) -> ProxySessionProtocol: ...
