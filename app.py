from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from config import RuntimeSettings, effective_hf_token, load_app_config
from hf import HuggingFaceService
from manager import ModelRuntimeManager
from protocols import ModelRuntimeManagerProtocol, HuggingFaceServiceProtocol
from schemas import (
    AppConfig,
    HealthResponse,
    LoadModelRequest,
    LogsResponse,
    ModelResource,
    ModelsResponse,
    OpenAIModelListResponse,
    OpenAIModelSummary,
    ServiceStatus,
)

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


def _manager_active_model(manager: ModelRuntimeManagerProtocol):
    return manager.active_model()


def _manager_find_model(manager: ModelRuntimeManagerProtocol, model_name: str) -> str | None:
    return manager.find_model_for_name(model_name)


def create_app(
    runtime: RuntimeSettings | None = None,
    app_config: AppConfig | None = None,
    hf: HuggingFaceServiceProtocol | None = None,
    manager: ModelRuntimeManagerProtocol | None = None,
) -> FastAPI:
    runtime = runtime or RuntimeSettings()
    app_config = app_config or load_app_config(runtime.config_path)
    api_prefix = app_config.api_prefix.rstrip("/") or "/api"
    hf = hf or HuggingFaceService(
        cache_dir=app_config.hf_cache_dir,
        token=effective_hf_token(runtime, app_config),
    )
    manager = manager or ModelRuntimeManager(runtime=runtime, app_config=app_config, hf=hf)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        await manager.cleanup()

    app = FastAPI(title="inference-server", lifespan=lifespan)
    app.state.manager = manager
    app.state.hf = hf

    @app.get(f"{api_prefix}/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse()

    @app.get(f"{api_prefix}/status", response_model=ServiceStatus)
    async def status() -> ServiceStatus:
        return manager.status()

    @app.get(f"{api_prefix}/models", response_model=ModelsResponse)
    async def models() -> ModelsResponse:
        return ModelsResponse(models=manager.model_resources())

    @app.get(f"{api_prefix}/models/{{name}}", response_model=ModelResource)
    async def model(name: str) -> ModelResource:
        try:
            return manager.model_resource(name)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post(f"{api_prefix}/models/{{name}}/load", response_model=ModelResource)
    async def load_model(name: str, body: LoadModelRequest) -> ModelResource:
        try:
            status = await manager.load(name, body.runtime)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return status

    @app.post(f"{api_prefix}/models/{{name}}/unload", response_model=ModelResource)
    async def unload_model(name: str) -> ModelResource:
        try:
            status = await manager.unload(name)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if status is None:
            raise HTTPException(
                status_code=409,
                detail=f"Model '{name}' is not currently active.",
            )
        return status

    @app.get(f"{api_prefix}/models/{{name}}/logs", response_model=LogsResponse)
    async def model_logs(name: str) -> LogsResponse:
        try:
            logs = await manager.get_logs(name)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return LogsResponse(name=name, logs=logs)

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

        target_model: str | None = None
        active = _manager_active_model(manager)

        if model_name:
            # Verify model is configured
            target_model = _manager_find_model(manager, model_name)
            if target_model is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Requested model '{model_name}' is not configured.",
                )

            # Verify configured model matches currently loaded model
            if active is None or active.config.name != target_model:
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
            target_model = active.config.name

        # Rewrite model name in the JSON body so that the runtime container
        # processes it correctly
        if target_model and is_json and data is not None:
            try:
                import json

                data["model"] = target_model
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
                detail=f"Model runtime communication failure: {exc}",
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

    @app.get(f"{api_prefix}/v1/models", response_model=OpenAIModelListResponse)
    async def list_v1_models() -> OpenAIModelListResponse:
        import time

        created_time = int(time.time())
        data = [
            OpenAIModelSummary(
                id=m.name,
                created=created_time,
                owned_by="inference-server",
            )
            for m in manager.models()
        ]
        return OpenAIModelListResponse(data=data)

    @app.get(
        f"{api_prefix}/v1/models/{{model_name}}", response_model=OpenAIModelSummary
    )
    async def retrieve_v1_model(model_name: str) -> OpenAIModelSummary:
        import time

        found = None
        for m in manager.models():
            if m.name == model_name:
                found = m
                break
        if found is None:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' not found in configuration.",
            )
        return OpenAIModelSummary(
            id=found.name,
            created=int(time.time()),
            owned_by="inference-server",
        )

    @app.api_route(
        f"{api_prefix}/v1",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        include_in_schema=False,
    )
    async def api_v1_root(request: Request) -> Response:
        return await _proxy(request, "v1")

    @app.api_route(
        f"{api_prefix}/v1/{{path:path}}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        include_in_schema=False,
    )
    async def api_v1_proxy(request: Request, path: str) -> Response:
        return await _proxy(request, f"v1/{path}")

    return app
