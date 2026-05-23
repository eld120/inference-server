from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

from config import RuntimeSettings
from protocols import (
    ModelResolverProtocol,
    ProxySessionProtocol,
)
from schemas import (
    AppConfig,
    BackendConfig,
    BackendState,
    BackendStatus,
    ServiceStatus,
)


@dataclass(slots=True)
class BackendRuntime:
    config: BackendConfig
    state: BackendState = "stopped"
    process: asyncio.subprocess.Process | None = None
    last_error: str | None = None
    model_path: Path | None = None
    draft_model_path: Path | None = None
    started_at: datetime | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.config.host}:{self.config.port}"


@dataclass(slots=True)
class ProxySession:
    client: httpx.AsyncClient
    response: httpx.Response


class BackendManager:
    def __init__(
        self,
        runtime: RuntimeSettings,
        app_config: AppConfig,
        hf: ModelResolverProtocol,
        proxy_client_factory: Callable[[str], httpx.AsyncClient] | None = None,
    ) -> None:
        self._runtime = runtime
        self._config = app_config
        self._hf = hf
        self._proxy_client_factory = (
            proxy_client_factory or self._default_client_factory
        )
        self._lock = asyncio.Lock()
        self._active_backend_name: str | None = None
        self._runtimes = {
            backend.name: BackendRuntime(backend) for backend in app_config.backends
        }

    @staticmethod
    def _default_client_factory(base_url: str) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(60.0, read=None),
        )

    def backends(self) -> list[BackendConfig]:
        return list(self._config.backends)

    def backend_statuses(self) -> list[BackendStatus]:
        return [
            self._to_status(runtime, runtime.config.name == self._active_backend_name)
            for runtime in self._runtimes.values()
        ]

    def _to_status(self, runtime: BackendRuntime, active: bool) -> BackendStatus:
        process = runtime.process
        return BackendStatus(
            name=runtime.config.name,
            state=runtime.state,
            model=runtime.config.model.label(),
            speculative_type=runtime.config.speculative.type,
            host=runtime.config.host,
            port=runtime.config.port,
            pid=None if process is None else process.pid,
            returncode=None if process is None else process.returncode,
            last_error=runtime.last_error,
            model_path=None if runtime.model_path is None else str(runtime.model_path),
            draft_model_path=None
            if runtime.draft_model_path is None
            else str(runtime.draft_model_path),
            base_url=runtime.base_url,
            active=active,
        )

    def status(self) -> ServiceStatus:
        return ServiceStatus(
            healthy=True,
            api_prefix=self._config.api_prefix,
            active_backend=self._active_backend_name,
            backends=self.backend_statuses(),
        )

    def _get_runtime(self, name: str) -> BackendRuntime:
        try:
            return self._runtimes[name]
        except KeyError as exc:
            msg = f"unknown backend: {name}"
            raise KeyError(msg) from exc

    async def load(self, name: str) -> BackendStatus:
        async with self._lock:
            runtime = self._get_runtime(name)
            if (
                runtime.state == "running"
                and runtime.process is not None
                and runtime.process.returncode is None
            ):
                self._active_backend_name = name
                return self._to_status(runtime, True)

            model_path = await asyncio.to_thread(
                self._hf.resolve_source, runtime.config.model
            )
            draft_model_path = None
            if runtime.config.speculative.draft_model is not None:
                draft_model_path = await asyncio.to_thread(
                    self._hf.resolve_source,
                    runtime.config.speculative.draft_model,
                )

            runtime.model_path = model_path
            runtime.draft_model_path = draft_model_path

            current_name = self._active_backend_name
            current_runtime = None
            if current_name is not None:
                current_runtime = self._runtimes[current_name]

            same_endpoint = (
                current_runtime is not None
                and current_name != name
                and current_runtime.config.host == runtime.config.host
                and current_runtime.config.port == runtime.config.port
            )

            if same_endpoint and current_name is not None:
                await self._unload_locked(current_name)

            runtime.state = "starting"
            runtime.last_error = None
            command = self._build_command(runtime, model_path, draft_model_path)
            runtime.process = await asyncio.create_subprocess_exec(*command)
            runtime.started_at = datetime.now(tz=UTC)

            try:
                await self._wait_until_ready(runtime.base_url)
            except Exception as exc:  # noqa: BLE001
                runtime.state = "error"
                runtime.last_error = str(exc)
                await self._stop_process(runtime)
                runtime.state = "error"
                raise

            runtime.state = "running"
            self._active_backend_name = name
            if not same_endpoint and current_name is not None and current_name != name:
                await self._unload_locked(current_name)
            return self._to_status(runtime, True)

    async def unload(self, name: str | None = None) -> BackendStatus | None:
        async with self._lock:
            return await self._unload_locked(name)

    async def _unload_locked(self, name: str | None = None) -> BackendStatus | None:
        target_name = name or self._active_backend_name
        if target_name is None:
            return None

        runtime = self._get_runtime(target_name)
        await self._stop_process(runtime)
        runtime.state = "stopped"
        runtime.model_path = None
        runtime.draft_model_path = None
        if self._active_backend_name == target_name:
            self._active_backend_name = None
        return self._to_status(runtime, False)

    async def _stop_process(self, runtime: BackendRuntime) -> None:
        process = runtime.process
        if process is None:
            return
        runtime.state = "stopping"
        process.terminate()
        with suppress(asyncio.TimeoutError):
            await asyncio.wait_for(process.wait(), timeout=15)
        if process.returncode is None:
            process.kill()
            await process.wait()
        runtime.process = None

    async def _wait_until_ready(self, base_url: str) -> None:
        deadline = asyncio.get_running_loop().time() + 90
        last_error: Exception | None = None
        while asyncio.get_running_loop().time() < deadline:
            try:
                async with self._proxy_client_factory(base_url) as client:
                    response = await client.get("/v1/models")
                    if response.status_code == 200:
                        return
            except Exception as exc:  # noqa: BLE001
                last_error = exc
            await asyncio.sleep(1)
        msg = "backend did not become ready in time"
        if last_error is not None:
            raise TimeoutError(msg) from last_error
        raise TimeoutError(msg)

    def _build_command(
        self,
        runtime: BackendRuntime,
        model_path: Path,
        draft_model_path: Path | None,
    ) -> list[str]:
        binary = runtime.config.llama_server_bin or self._runtime.llama_server_bin
        command = [
            binary,
            "--host",
            runtime.config.host,
            "--port",
            str(runtime.config.port),
            "-m",
            str(model_path),
        ]
        if runtime.config.speculative.type != "none":
            command.extend(["--spec-type", runtime.config.speculative.type])
        if draft_model_path is not None:
            command.extend(["-md", str(draft_model_path)])
        command.extend(runtime.config.extra_args)
        return command

    def create_proxy_client(self, base_url: str) -> httpx.AsyncClient:
        return self._proxy_client_factory(base_url)

    async def open_proxy_session(
        self,
        path: str,
        request: httpx.Request,
    ) -> ProxySessionProtocol:
        if self._active_backend_name is None:
            msg = "no backend is loaded"
            raise RuntimeError(msg)
        runtime = self._get_runtime(self._active_backend_name)
        client = self._proxy_client_factory(runtime.base_url)
        try:
            upstream = client.build_request(
                request.method,
                path,
                headers=request.headers,
                content=request.content,
                params=request.url.params,
            )
            response = await client.send(upstream, stream=True)
        except Exception:  # noqa: BLE001
            await client.aclose()
            raise
        return ProxySession(client=client, response=response)

    def active_backend(self) -> BackendRuntime | None:
        if self._active_backend_name is None:
            return None
        return self._runtimes[self._active_backend_name]
