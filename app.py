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
from schemas import AppConfig

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
        yield
        await manager.cleanup()

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

    @app.get(f"{api_prefix}/backends/{{name}}/logs")
    async def backend_logs(name: str) -> JSONResponse:
        try:
            logs = await manager.get_logs(name)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JSONResponse({"name": name, "logs": logs})



    async def _proxy(request: Request, path: str) -> Response:
        body = await request.body()

        # Extract model from JSON body
        model_name: str | None = None
        is_json = body and "application/json" in request.headers.get("content-type", "")
        data = None
        if is_json:
            try:
                import json

                data = json.loads(body)
                if isinstance(data, dict):
                    model_name = data.get("model")
            except Exception:
                pass

        target_backend: str | None = None
        active = manager.active_backend()

        if model_name:
            # Verify model is configured
            target_backend = manager.find_backend_for_model(model_name)
            if target_backend is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Requested model '{model_name}' is not configured.",
                )

            # Verify configured model matches currently loaded model
            if active is None or active.config.name != target_backend:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Model '{model_name}' is not currently loaded. "
                        "Please call the load endpoint first."
                    ),
                )
        else:
            # No model specified: check if any model is loaded
            if active is None:
                raise HTTPException(
                    status_code=400,
                    detail="No model is currently loaded.",
                )
            target_backend = active.config.name

        # Rewrite model name in the JSON body so that the router routes it correctly
        if target_backend and is_json and data is not None:
            try:
                import json

                data["model"] = target_backend
                body = json.dumps(data).encode("utf-8")
            except Exception:
                pass

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
        except (RuntimeError, httpx.HTTPError) as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Backend communication failure: {exc}",
            ) from exc

        upstream_response = session.response
        headers = _filtered_headers(upstream_response.headers)
        content_type = upstream_response.headers.get("content-type", "")
        if "text/event-stream" in content_type:

            async def iterator() -> AsyncIterator[bytes]:
                try:
                    async for chunk in upstream_response.aiter_raw():
                        yield chunk
                finally:
                    try:
                        await asyncio.shield(upstream_response.aclose())
                    finally:
                        await asyncio.shield(session.client.aclose())

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
            try:
                await asyncio.shield(upstream_response.aclose())
            finally:
                await asyncio.shield(session.client.aclose())

    @app.get(f"{api_prefix}/v1/models")
    @app.get(f"{api_prefix}/v1/models/")
    async def list_v1_models() -> JSONResponse:
        import time
        created_time = int(time.time())
        data = [
            {
                "id": m.name,
                "object": "model",
                "created": created_time,
                "owned_by": "inference-server",
            }
            for m in manager.backends()
        ]
        return JSONResponse({"object": "list", "data": data})

    @app.get(f"{api_prefix}/v1/models/{{model_name}}")
    async def retrieve_v1_model(model_name: str) -> JSONResponse:
        import time
        found = None
        for m in manager.backends():
            if m.name == model_name:
                found = m
                break
        if found is None:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' not found in configuration.",
            )
        return JSONResponse({
            "id": found.name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "inference-server",
        })

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
