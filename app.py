from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from datetime import datetime, timezone

logger = logging.getLogger("inference-server.app")
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


async def increment_inflight_requests(app: FastAPI) -> None:
    start_time = datetime.now(timezone.utc)
    async with app.state.observability_lock:
        app.state.inflight_requests += 1
        if app.state.inflight_requests == 1:
            db = getattr(app.state, "observability_db", None)
            if db:
                fut = asyncio.Future()
                app.state.current_serving_session_id_future = fut
                
                async def run_start(f: asyncio.Future):
                    try:
                        session_id = await asyncio.to_thread(db.start_serving_session, start_time)
                        f.set_result(session_id)
                    except Exception as exc:
                        f.set_exception(exc)
                        
                task = asyncio.create_task(run_start(fut))
                app.state.observability_tasks.add(task)
                task.add_done_callback(app.state.observability_tasks.discard)
                
        # Proactively trigger a retry of any failed closes in the background
        db = getattr(app.state, "observability_db", None)
        if db:
            db.trigger_retry(app.state.observability_tasks)


async def decrement_inflight_requests(app: FastAPI) -> None:
    end_time = datetime.now(timezone.utc)
    async with app.state.observability_lock:
        app.state.inflight_requests -= 1
        if app.state.inflight_requests < 0:
            app.state.inflight_requests = 0
        if app.state.inflight_requests == 0:
            fut = getattr(app.state, "current_serving_session_id_future", None)
            if fut is not None:
                app.state.current_serving_session_id_future = None
                db = getattr(app.state, "observability_db", None)
                if db:
                    async def run_end(f: asyncio.Future):
                        try:
                            session_id = await f
                            await asyncio.to_thread(db.end_serving_session, session_id, end_time)
                        except Exception as exc:
                            logger.exception("Failed to end serving session in DB: %s", exc)
                            
                    task = asyncio.create_task(run_end(fut))
                    app.state.observability_tasks.add(task)
                    task.add_done_callback(app.state.observability_tasks.discard)
                    
        # Proactively trigger a retry of any failed closes in the background
        db = getattr(app.state, "observability_db", None)
        if db:
            db.trigger_retry(app.state.observability_tasks)


def create_app(
    runtime: RuntimeSettings | None = None,
    app_config: AppConfig | None = None,
    hf: HuggingFaceServiceProtocol | None = None,
    manager: ModelRuntimeManagerProtocol | None = None,
    observability_db_path: str | Path | None = None,
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
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        from observability import ObservabilityDB, start_telemetry_logger
        db = ObservabilityDB(db_path=observability_db_path)
        app.state.observability_db = db
        polling_task = asyncio.create_task(
            start_telemetry_logger(db, app.state.observability_tasks, interval_seconds=30)
        )
        try:
            yield
        finally:
            polling_task.cancel()
            try:
                await polling_task
            except asyncio.CancelledError:
                pass
            
            # Wait for any pending observability background tasks to finish
            tasks = getattr(app.state, "observability_tasks", set())
            if tasks:
                try:
                    await asyncio.wait_for(asyncio.gather(*list(tasks), return_exceptions=True), timeout=5.0)
                except asyncio.TimeoutError:
                    logger.warning("Timeout waiting for pending observability tasks to finish on shutdown.")
                except Exception as exc:
                    logger.exception("Error waiting for pending observability tasks on shutdown: %s", exc)
                    
            db.close()
            await manager.cleanup()

    app = FastAPI(title="inference-server", lifespan=lifespan)
    app.state.manager = manager
    app.state.hf = hf
    app.state.inflight_requests = 0
    app.state.observability_lock = asyncio.Lock()
    app.state.current_serving_session_id_future = None
    app.state.observability_tasks = set()

    @app.get(f"{api_prefix}/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse()

    @app.get(f"{api_prefix}/status", response_model=ServiceStatus)
    async def status() -> ServiceStatus:
        return manager.status()

    @app.get(f"{api_prefix}/observability/metrics")
    async def observability_metrics():
        from observability import HardwareMonitor
        monitor = HardwareMonitor()
        return await asyncio.to_thread(monitor.collect)

    @app.get(f"{api_prefix}/observability/history")
    async def observability_history(window_hours: int = 24, summary_only: bool = False):
        from observability import ObservabilityDB
        db: ObservabilityDB = app.state.observability_db
        summary = await asyncio.to_thread(db.get_summary, window_hours=window_hours)
        if summary_only:
            return {"summary": summary}
        active_sessions = await asyncio.to_thread(db.get_active_sessions, window_hours=window_hours, limit=10)
        serving_sessions = await asyncio.to_thread(db.get_serving_sessions, window_hours=window_hours)
        return {
            "summary": summary,
            "active_sessions": active_sessions,
            "serving_sessions": serving_sessions,
            "sessions": active_sessions,  # alias for backward compatibility
        }

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
        is_streaming = False
        incremented = False
        try:
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

                target_status = manager.model_resource(target_model).status
                if target_status.state in {"pulling", "starting"}:
                    raise HTTPException(
                        status_code=503,
                        detail=f"Model '{target_model}' is still loading.",
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
                starting_models = [
                    status.name
                    for status in manager.model_statuses()
                    if status.state in {"pulling", "starting"} and status.name is not None
                ]
                if starting_models:
                    raise HTTPException(
                        status_code=503,
                        detail=f"Model '{starting_models[0]}' is still loading.",
                    )
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

            # Validation & Upstream Connection passed successfully -> start serving session
            await increment_inflight_requests(app)
            incremented = True

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
                        await asyncio.shield(decrement_inflight_requests(app))

                is_streaming = True
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
        finally:
            if incremented and not is_streaming:
                await asyncio.shield(decrement_inflight_requests(app))

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
