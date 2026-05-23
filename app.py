from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from config import RuntimeSettings, effective_hf_token, load_app_config
from hf import HuggingFaceService
from manager import BackendManager
from protocols import BackendManagerProtocol, HuggingFaceServiceProtocol
from schemas import AppConfig, HFDownloadRequest, ModelSource

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "host",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "content-length",
}


def _filtered_headers(headers: httpx.Headers) -> dict[str, str]:
    return {
        key: value
        for key, value in headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS and key.lower() != "content-length"
    }


def _filtered_request_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {
        key: value
        for key, value in headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS
    }


def create_app(
    runtime: RuntimeSettings | None = None,
    app_config: AppConfig | None = None,
    hf: HuggingFaceServiceProtocol | None = None,
    manager: BackendManagerProtocol | None = None,
) -> FastAPI:
    runtime = runtime or RuntimeSettings()
    app_config = app_config or load_app_config(runtime.config_path)
    api_prefix = app_config.api_prefix.rstrip("/") or "/api"
    hf = hf or HuggingFaceService(
        cache_dir=app_config.hf_cache_dir,
        token=effective_hf_token(runtime, app_config),
    )
    manager = manager or BackendManager(runtime=runtime, app_config=app_config, hf=hf)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if app_config.default_backend is not None:
            await manager.load(app_config.default_backend)
        yield
        active = manager.active_backend()
        if active is not None:
            await manager.unload(active.config.name)

    app = FastAPI(title="inference-server", lifespan=lifespan)
    app.state.manager = manager
    app.state.hf = hf

    @app.get(f"{api_prefix}/healthz")
    async def healthz() -> dict[str, bool]:
        return {"ok": True}

    @app.get(f"{api_prefix}/status")
    async def status() -> JSONResponse:
        return JSONResponse(manager.status().model_dump())

    @app.get(f"{api_prefix}/backends")
    async def backends() -> JSONResponse:
        backends = [backend.model_dump() for backend in manager.backend_statuses()]
        return JSONResponse({"backends": backends})

    @app.post(f"{api_prefix}/backends/{{name}}/load")
    async def load_backend(name: str) -> JSONResponse:
        try:
            status = await manager.load(name)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return JSONResponse(status.model_dump())

    @app.post(f"{api_prefix}/backends/{{name}}/unload")
    async def unload_backend(name: str) -> JSONResponse:
        try:
            status = await manager.unload(name)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if status is None:
            return JSONResponse({"ok": True})
        return JSONResponse(status.model_dump())

    @app.get(f"{api_prefix}/hf/models")
    async def hf_models(q: str | None = None, limit: int = 20) -> JSONResponse:
        models = [
            item.model_dump()
            for item in await asyncio.to_thread(hf.search_models, q, limit)
        ]
        return JSONResponse({"models": models})

    @app.get(f"{api_prefix}/hf/repos/{{repo_id:path}}/files")
    async def hf_repo_files(repo_id: str, revision: str = "main") -> JSONResponse:
        files = [
            item.model_dump()
            for item in await asyncio.to_thread(hf.repo_files, repo_id, revision)
        ]
        return JSONResponse({"files": files})

    @app.get(f"{api_prefix}/hf/cache")
    async def hf_cache() -> JSONResponse:
        files = [item.model_dump() for item in await asyncio.to_thread(hf.cache_files)]
        return JSONResponse({"files": files})

    @app.post(f"{api_prefix}/hf/download")
    async def hf_download(payload: HFDownloadRequest) -> JSONResponse:
        try:
            result = await asyncio.to_thread(
                hf.download,
                ModelSource(
                    repo_id=payload.repo_id,
                    filename=payload.filename,
                    revision=payload.revision,
                ),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse(result.model_dump())

    async def _proxy(request: Request, path: str) -> Response:
        body = await request.body()
        upstream_url = httpx.URL(
            f"http://placeholder/{path}",
            params=list(request.query_params.multi_items()),
        )
        upstream_request = httpx.Request(
            method=request.method,
            url=upstream_url,
            headers=_filtered_request_headers(request.headers),
            content=body,
        )
        try:
            session = await manager.open_proxy_session(f"/{path}", upstream_request)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        upstream_response = session.response
        headers = _filtered_headers(upstream_response.headers)
        content_type = upstream_response.headers.get("content-type", "")
        if "text/event-stream" in content_type:

            async def iterator() -> AsyncIterator[bytes]:
                try:
                    async for chunk in upstream_response.aiter_raw():
                        yield chunk
                finally:
                    await upstream_response.aclose()
                    await session.client.aclose()

            return StreamingResponse(
                iterator(),
                status_code=upstream_response.status_code,
                headers=headers,
                media_type=content_type,
            )

        try:
            content = await upstream_response.aread()
            return Response(
                content=content,
                status_code=upstream_response.status_code,
                headers=headers,
                media_type=content_type or None,
            )
        finally:
            await upstream_response.aclose()
            await session.client.aclose()

    @app.api_route(
        f"{api_prefix}/v1",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    )
    async def api_v1_root(request: Request) -> Response:
        return await _proxy(request, "v1")

    @app.api_route(
        f"{api_prefix}/v1/",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    )
    async def api_v1_root_slash(request: Request) -> Response:
        return await _proxy(request, "v1")

    @app.api_route(
        f"{api_prefix}/v1/{{path:path}}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    )
    async def api_v1_proxy(request: Request, path: str) -> Response:
        return await _proxy(request, f"v1/{path}")

    return app
