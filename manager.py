from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
import re

import docker.errors
import docker.models.containers
import httpx
import huggingface_hub

import docker
from config import RuntimeSettings
from protocols import (
    ActiveModelProtocol,
    ModelResolverProtocol,
    ProxySessionProtocol,
)
from schemas import (
    AppConfig,
    BackendConfig,
    ModelConfig,
    ModelResource,
    ModelSource,
    ModelStatus,
    RuntimeConfig,
    RuntimeState,
    ServiceStatus,
    SpeculativeConfig,
)

logger = logging.getLogger(__name__)


class RuntimeContainer:
    def __init__(
        self,
        runtime_type: str,
        model_name: str | None,
        rt_cfg: RuntimeConfig,
        manager: ModelRuntimeManager,
    ) -> None:
        self.runtime_type = runtime_type
        self._model_name = model_name
        self.last_model_name = model_name
        self.config = rt_cfg
        self.manager = manager
        self.state: RuntimeState = "stopped"
        self.container: docker.models.containers.Container | None = None
        self.last_error: str | None = None
        self.model_path: Path | None = None
        self.draft_model_path: Path | None = None
        self.mmproj_path: Path | None = None
        self._recent_logs: list[str] = []
        self._last_persisted_log_blob: str | None = None
        self._last_persisted_log_blob_by_model: dict[str, str] = {}
        self.started_at: datetime | None = None

    @property
    def model_name(self) -> str | None:
        return self._model_name

    @model_name.setter
    def model_name(self, value: str | None) -> None:
        self._model_name = value
        if value is not None:
            self.last_model_name = value

    def clear_loaded_model_state(self) -> None:
        self.model_name = None
        self.model_path = None
        self.draft_model_path = None
        self.mmproj_path = None
        self.last_error = None

    @property
    def base_url(self) -> str:
        return f"http://{self.config.connect_host}:{self.manager.runtime_port}"

    @property
    def docker_image(self) -> str:
        return self.config.docker_image

    @property
    def devices(self) -> list[str]:
        return self.config.devices

    @property
    def volumes(self) -> dict[str, str]:
        return self.config.volumes

    @property
    def bind_host(self) -> str:
        return self.config.bind_host

    @property
    def connect_host(self) -> str:
        return self.config.connect_host

    @property
    def shared_args(self) -> list[str]:
        return self.config.shared_args

    @property
    def extra_args(self) -> list[str]:
        return self.config.extra_args

    @property
    def speculative(self) -> SpeculativeConfig:
        return self.config.speculative

    async def start(self) -> None:
        self.state = "starting"
        self.last_error = None
        try:
            # 1. Generate models preset INI file on host
            ini_content = await self.manager._generate_presets_ini(
                self.runtime_type,
                active_model_name=self.model_name,
                active_model_resolved_path=self.model_path,
                active_model_draft_resolved_path=self.draft_model_path,
                active_model_mmproj_resolved_path=self.mmproj_path,
            )
            ini_dir = self.manager._hf.cache_dir / "presets"
            await asyncio.to_thread(ini_dir.mkdir, parents=True, exist_ok=True)
            ini_path = ini_dir / f"{self.runtime_type}.ini"
            await asyncio.to_thread(ini_path.write_text, ini_content)

            await self._ensure_image_pulled()

            # 2. Gather volume bindings (always mount HF cache and local parents)
            volumes = {
                str(self.manager._hf.cache_dir.resolve().absolute()): {
                    "bind": "/huggingface",
                    "mode": "ro",
                }
            }
            # Mount host directory containing presets
            volumes[str(ini_dir.resolve().absolute())] = {
                "bind": "/config",
                "mode": "ro",
            }

            for m in self.manager._config.models:
                try:
                    rt_cfg = self.manager._resolve_runtime_config(m, self.runtime_type)
                except ValueError:
                    continue
                if rt_cfg.source.local_path is not None:
                    local_p = Path(rt_cfg.source.local_path).resolve().absolute()
                    cache_dir = self.manager._hf.cache_dir.resolve().absolute()
                    if not local_p.is_relative_to(cache_dir):
                        parent = local_p.parent
                        suffix = self.manager._dir_hash(parent)
                        bind_path = f"/local_models_{suffix}"
                        volumes[str(parent)] = {
                            "bind": bind_path,
                            "mode": "ro",
                        }
                if (
                    rt_cfg.speculative.draft_model is not None
                    and rt_cfg.speculative.draft_model.local_path is not None
                ):
                    local_p = (
                        Path(rt_cfg.speculative.draft_model.local_path)
                        .resolve()
                        .absolute()
                    )
                    cache_dir = self.manager._hf.cache_dir.resolve().absolute()
                    if not local_p.is_relative_to(cache_dir):
                        parent = local_p.parent
                        suffix = self.manager._dir_hash(parent)
                        bind_path = f"/local_models_{suffix}"
                        volumes[str(parent)] = {
                            "bind": bind_path,
                            "mode": "ro",
                        }
                if rt_cfg.mmproj is not None and rt_cfg.mmproj.local_path is not None:
                    local_p = Path(rt_cfg.mmproj.local_path).resolve().absolute()
                    cache_dir = self.manager._hf.cache_dir.resolve().absolute()
                    if not local_p.is_relative_to(cache_dir):
                        parent = local_p.parent
                        suffix = self.manager._dir_hash(parent)
                        bind_path = f"/local_models_{suffix}"
                        volumes[str(parent)] = {
                            "bind": bind_path,
                            "mode": "ro",
                        }

            # Apply runtime specific custom mounts
            for src, dst in self.volumes.items():
                abs_src = str(Path(src).resolve().absolute())
                volumes[abs_src] = {"bind": dst, "mode": "ro"}

            # Port & GPU setups
            ports = {"8080/tcp": (self.bind_host, self.manager.runtime_port)}
            devices = [f"{d}:{d}" for d in self.devices]

            # Start llama-server: load via --models-preset
            command = [
                "--host",
                "0.0.0.0",
                "--port",
                "8080",
                "--models-preset",
                f"/config/{self.runtime_type}.ini",
            ]
            command.extend(self.shared_args)

            container_name = f"inference-server-runtime-{self.runtime_type}"
            await self.manager._remove_conflicting_container(container_name)

            self.container = await asyncio.to_thread(
                self.manager.docker_client.containers.run,
                image=self.docker_image,
                command=command,
                devices=devices,
                ports=ports,
                volumes=volumes,
                detach=True,
                name=container_name,
                auto_remove=True,
                labels={"managed-by": "inference-server"},
            )
            self.started_at = datetime.now(tz=UTC)
            await self._wait_until_ready()
            self.state = "running"
        except Exception as exc:
            self.state = "error"
            self.last_error = str(exc)
            await self.stop()
            self.state = "error"
            raise


    async def stop(self) -> None:
        container = self.container
        if container is None:
            self.state = "stopped"
            return
        container_id = getattr(container, "id", None)
        try:
            container_name = container.name
        except Exception:
            container_name = None
        self.state = "stopping"
        try:
            logger.info("Stopping container: %s", container_name or "unknown")
            await self._capture_recent_logs(container, "stop")
            try:
                await asyncio.to_thread(container.stop, timeout=10)
            except Exception as stop_exc:
                if not (
                    self.manager._is_container_not_found_error(stop_exc)
                    or self.manager._is_removal_in_progress_error(stop_exc)
                ):
                    raise
            try:
                await asyncio.to_thread(container.remove, force=True)
            except Exception as stop_exc:
                if not (
                    self.manager._is_container_not_found_error(stop_exc)
                    or self.manager._is_removal_in_progress_error(stop_exc)
                ):
                    raise
            self.container = None
        except Exception as stop_exc:
            logger.warning("Graceful stop failed: %s. Force removing.", stop_exc)
            try:
                await self._capture_recent_logs(container, "force-remove")
                try:
                    await asyncio.to_thread(container.remove, force=True)
                except Exception as remove_exc:
                    if not (
                        self.manager._is_container_not_found_error(remove_exc)
                        or self.manager._is_removal_in_progress_error(remove_exc)
                    ):
                        raise
                self.container = None
            except Exception as rm_exc:
                logger.error("Force remove also failed: %s", rm_exc)
                self.state = "error"
                self.last_error = f"Stop failed: {stop_exc}. Remove failed: {rm_exc}."
                raise rm_exc
        if container_name:
            await self.manager._wait_for_container_release(
                container_name,
                known_container_id=container_id,
            )
        self.state = "stopped"

    async def load_model(self, name: str) -> None:
        async with self.manager.create_proxy_client(self.base_url) as client:
            response = await client.post("/models/load", json={"model": name})
            if response.status_code != 200:
                raise RuntimeError(
                    f"Failed to load model inside runtime container: {response.text}"
                )
        await self._wait_until_model_loaded(name)

    async def unload_model(self, name: str) -> None:
        async with self.manager.create_proxy_client(self.base_url) as client:
            response = await client.post("/models/unload", json={"model": name})
            if response.status_code != 200:
                raise RuntimeError(
                    f"Failed to unload model inside runtime container: {response.text}"
                )

    async def get_logs(self) -> list[str]:
        container = self.container
        if container is None:
            return list(self._recent_logs)
        try:
            logs_bytes = await asyncio.to_thread(container.logs, tail=500)
            lines = logs_bytes.decode("utf-8", errors="replace").splitlines()
            self._recent_logs = lines
            await self._persist_log_snapshot("live-fetch", lines)
            return lines
        except Exception:
            return list(self._recent_logs)

    async def _ensure_image_pulled(self) -> None:
        image = self.docker_image
        try:
            await asyncio.to_thread(self.manager.docker_client.images.get, image)
        except docker.errors.ImageNotFound:
            self.state = "pulling"
            logger.info("Pulling Docker image: %s", image)
            await asyncio.to_thread(self.manager.docker_client.images.pull, image)
            self.state = "starting"

    async def _wait_until_ready(self) -> None:
        deadline = asyncio.get_running_loop().time() + 90
        last_error: Exception | None = None
        while asyncio.get_running_loop().time() < deadline:
            container = self.container
            if container is not None:
                try:
                    await asyncio.to_thread(container.reload)
                    status = container.status
                except Exception:
                    status = "exited"

                if status == "exited":
                    try:
                        logs_bytes = await asyncio.to_thread(container.logs, tail=50)
                        last_logs = logs_bytes.decode("utf-8", errors="replace").strip()
                        lines = last_logs.splitlines()
                        self._recent_logs = lines
                        await self._persist_log_snapshot("startup-exit", lines)
                    except Exception:
                        last_logs = ""
                    msg = "Container exited immediately."
                    if last_logs:
                        msg += f"\nLast logs:\n{last_logs}"
                    raise RuntimeError(msg)

            try:
                async with self.manager.create_proxy_client(self.base_url) as client:
                    response = await client.get("/v1/models")
                    if response.status_code == 200:
                        return
            except Exception as exc:  # noqa: BLE001
                last_error = exc
            await asyncio.sleep(1)
        msg = "runtime did not become ready in time"
        if last_error is not None:
            raise TimeoutError(msg) from last_error
        raise TimeoutError(msg)

    async def _wait_until_model_loaded(self, name: str) -> None:
        timeout_seconds = self.manager._runtime.model_load_timeout_seconds
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        while asyncio.get_running_loop().time() < deadline:
            container = self.container
            if container is None:
                raise RuntimeError(
                    f"Runtime container disappeared while loading model '{name}'."
                )

            try:
                await asyncio.to_thread(container.reload)
                status = container.status
            except Exception:
                status = "exited"

            if status == "exited":
                try:
                    logs_bytes = await asyncio.to_thread(container.logs, tail=100)
                    last_logs = logs_bytes.decode("utf-8", errors="replace").strip()
                    lines = last_logs.splitlines()
                    self._recent_logs = lines
                    await self._persist_log_snapshot("load-exit", lines)
                except Exception:
                    last_logs = ""
                msg = f"Runtime container exited while loading model '{name}'."
                if last_logs:
                    msg += f"\nLast logs:\n{last_logs}"
                raise RuntimeError(msg)

            worker_state = await self._inspect_worker_process_state()
            if worker_state == "defunct":
                try:
                    logs_bytes = await asyncio.to_thread(container.logs, tail=100)
                    last_logs = logs_bytes.decode("utf-8", errors="replace").strip()
                    lines = last_logs.splitlines()
                    self._recent_logs = lines
                    await self._persist_log_snapshot("worker-defunct", lines)
                except Exception:
                    last_logs = ""
                msg = f"Model worker for '{name}' died during load."
                if last_logs:
                    msg += f"\nLast logs:\n{last_logs}"
                raise RuntimeError(msg)
            if worker_state in {"running", "unknown"} and await self._probe_model_readiness(
                name
            ):
                return

            await asyncio.sleep(1)

        raise TimeoutError(
            f"model '{name}' did not finish loading within {timeout_seconds} seconds"
        )

    async def _probe_model_readiness(self, name: str) -> bool:
        try:
            async with self.manager.create_proxy_client(self.base_url) as client:
                response = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": name,
                        "messages": [{"role": "user", "content": "."}],
                        "max_tokens": 1,
                        "temperature": 0,
                        "stream": False,
                    },
                    timeout=self.manager._runtime.model_readiness_probe_timeout_seconds,
                )
        except (httpx.TimeoutException, httpx.HTTPError):
            return False

        if response.status_code != 200:
            return False
        return True

    async def _inspect_worker_process_state(self) -> str:
        container = self.container
        if container is None or not hasattr(container, "exec_run"):
            return "unknown"

        try:
            result = await asyncio.to_thread(
                container.exec_run,
                ["ps", "-eo", "pid=,stat=,cmd="],
            )
        except Exception:
            return "unknown"

        exit_code = getattr(result, "exit_code", None)
        if exit_code is None and isinstance(result, tuple) and len(result) >= 1:
            exit_code = result[0]

        if exit_code is not None and exit_code != 0:
            return "unknown"

        output = getattr(result, "output", None)
        if output is None and isinstance(result, tuple) and len(result) >= 2:
            output = result[1]
        if output is None:
            output = b""

        if isinstance(output, bytes):
            text = output.decode("utf-8", errors="replace")
        else:
            text = str(output)

        pid1_entry = None
        other_entries = []
        any_llama_server_exists = False

        for line in text.splitlines():
            stripped = line.strip()
            if "llama-server" not in stripped:
                continue
            parts = stripped.split(None, 2)
            if len(parts) < 3:
                continue
            pid, stat, cmd = parts
            any_llama_server_exists = True

            is_defunct = "Z" in stat or "<defunct>" in cmd
            if pid == "1":
                pid1_entry = (pid, stat, cmd, is_defunct)
            else:
                other_entries.append((pid, stat, cmd, is_defunct))

        if not any_llama_server_exists:
            return "missing"

        if (pid1_entry and pid1_entry[3]) or any(entry[3] for entry in other_entries):
            return "defunct"

        if other_entries:
            return "running"

        if pid1_entry:
            return "running"

        return "missing"

    async def _capture_recent_logs(
        self,
        container: docker.models.containers.Container,
        reason: str = "snapshot",
    ) -> None:
        try:
            logs_bytes = await asyncio.to_thread(container.logs, tail=500)
        except Exception:
            return
        lines = logs_bytes.decode("utf-8", errors="replace").splitlines()
        self._recent_logs = lines
        await self._persist_log_snapshot(reason, lines)

    async def _persist_log_snapshot(self, reason: str, lines: list[str]) -> None:
        if not lines:
            return

        model_name = self.model_name or self.last_model_name
        if model_name:
            self.manager._logs_cache[model_name] = lines

        blob = "\n".join(lines)
        if blob != self._last_persisted_log_blob:
            self._last_persisted_log_blob = blob
            await asyncio.to_thread(self._append_logs_to_disk, reason, blob)

        if model_name:
            last_model_blob = self._last_persisted_log_blob_by_model.get(model_name)
            if blob != last_model_blob:
                self._last_persisted_log_blob_by_model[model_name] = blob
                await asyncio.to_thread(self._save_last_logs_to_disk, model_name, lines)

    def _append_logs_to_disk(self, reason: str, blob: str) -> None:
        log_dir = self.manager._runtime.runtime_log_dir.expanduser().resolve()
        log_dir.mkdir(parents=True, exist_ok=True)

        stamp = datetime.now(tz=UTC)
        filename = f"runtime-{stamp.date().isoformat()}.log"
        log_path = log_dir / filename

        model_slug = self._slugify(self.model_name or self.last_model_name)
        runtime_slug = self._slugify(self.runtime_type)
        container_id = "-"
        if self.container is not None and getattr(self.container, "id", None):
            container_id = str(self.container.id)

        started_at = (
            self.started_at.isoformat() if self.started_at is not None else "-"
        )
        header = (
            f"[{stamp.isoformat()}] "
            f"model={model_slug} runtime={runtime_slug} state={self.state} "
            f"reason={reason} container_id={container_id} started_at={started_at}"
        )
        entry = f"{header}\n{blob}\n\n"
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(entry)

    def _save_last_logs_to_disk(self, model_name: str, lines: list[str]) -> None:
        import json
        try:
            log_dir = self.manager._runtime.runtime_log_dir.expanduser().resolve()
            last_logs_dir = log_dir / "last_logs"
            last_logs_dir.mkdir(parents=True, exist_ok=True)
            log_path = last_logs_dir / f"{self._slugify(model_name)}.json"
            with log_path.open("w", encoding="utf-8") as fh:
                json.dump(lines, fh)
        except Exception as e:
            logger.warning("Failed to save last logs for model %s to disk: %s", model_name, e)

    @staticmethod
    def _slugify(value: str | None) -> str:
        if not value:
            return "unknown"
        slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
        return slug or "unknown"


@dataclass(slots=True)
class ProxySession:
    client: httpx.AsyncClient
    response: httpx.Response


class ActiveModel:
    def __init__(
        self,
        name: str,
        model_config: ModelConfig,
        runtime: str,
        manager: ModelRuntimeManager,
    ) -> None:
        self.name = name
        self.model_config = model_config
        self.runtime = runtime
        self.manager = manager

    @property
    def config(self):
        rt_cfg = self.manager._resolve_runtime_config(self.model_config, self.runtime)
        return SimpleNamespace(
            name=self.model_config.name,
            model=rt_cfg.source,
            mmproj=rt_cfg.mmproj,
            speculative=rt_cfg.speculative,
            extra_args=rt_cfg.extra_args,
            bind_host=rt_cfg.bind_host,
            connect_host=rt_cfg.connect_host,
        )


class ModelRuntimeManager:
    _container_release_poll_attempts = 30
    _container_release_poll_interval_seconds = 0.5

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
        self._active_model_name: str | None = None
        self._active_runtime: RuntimeContainer | None = None
        self._docker_client: docker.DockerClient | None = None
        self._logs_cache: dict[str, list[str]] = {}
        self._last_error: str | None = None

    @staticmethod
    def _dir_hash(parent: Path) -> str:
        import hashlib

        return hashlib.md5(str(parent).encode()).hexdigest()[:8]

    @staticmethod
    def _exception_message_parts(exc: BaseException) -> list[str]:
        parts: list[str] = []
        for arg in getattr(exc, "args", ()):
            if arg:
                parts.append(repr(arg) if isinstance(arg, BaseException) else str(arg))

        response = getattr(exc, "response", None)
        explanation = getattr(exc, "explanation", None)
        if explanation:
            parts.append(str(explanation))
        if response is not None:
            for attr in ("text", "reason", "status", "status_code"):
                value = getattr(response, attr, None)
                if value:
                    parts.append(str(value))
        return parts

    @staticmethod
    def _is_removal_in_progress_error(exc: BaseException) -> bool:
        if not isinstance(exc, docker.errors.APIError):
            return False

        status_code = getattr(exc, "status_code", None)
        response = getattr(exc, "response", None)
        if status_code is None and response is not None:
            status_code = getattr(response, "status_code", None)
        if status_code != 409:
            return False

        message = " ".join(ModelRuntimeManager._exception_message_parts(exc)).lower()
        return (
            "removal already in progress" in message
            or "removal of container" in message and "already in progress" in message
        )

    @staticmethod
    def _is_container_not_found_error(exc: BaseException) -> bool:
        if isinstance(exc, docker.errors.NotFound):
            return True
        return "not found" in " ".join(
            ModelRuntimeManager._exception_message_parts(exc)
        ).lower()

    @property
    def docker_client(self) -> docker.DockerClient:
        if self._docker_client is None:
            self._docker_client = docker.from_env()
        return self._docker_client

    @property
    def runtime_port(self) -> int:
        return self._runtime.runtime_port

    @staticmethod
    def _default_client_factory(base_url: str) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(60.0, read=None),
        )

    def models(self) -> list[ModelConfig]:
        return list(self._config.models)

    def model_statuses(self) -> list[ModelStatus]:
        return [self._to_status(model_config) for model_config in self._config.models]

    def _to_status(self, model_config: ModelConfig) -> ModelStatus:
        is_current = (
            self._active_runtime is not None
            and self._active_runtime.model_name == model_config.name
        )
        active = self._active_model_name == model_config.name
        state: RuntimeState = "stopped"
        container_id: str | None = None
        last_error: str | None = None
        model_path: str | None = None
        draft_model_path: str | None = None
        active_runtime: str | None = None

        if self._active_runtime is not None:
            active_runtime = self._active_runtime.runtime_type
            if is_current:
                if self._active_runtime.state == "running" and not active:
                    state = "stopped"
                else:
                    state = self._active_runtime.state
                    last_error = self._active_runtime.last_error
                    if active:
                        container_id = (
                            None
                            if self._active_runtime.container is None
                            else self._active_runtime.container.id
                        )
                        model_path = (
                            None
                            if self._active_runtime.model_path is None
                            else str(self._active_runtime.model_path)
                        )
                        draft_model_path = (
                            None
                            if self._active_runtime.draft_model_path is None
                            else str(self._active_runtime.draft_model_path)
                        )

        active_rt = active_runtime if active else None
        first_rt_name = self._first_runtime_name_for_model(model_config)
        rt_cfg: RuntimeConfig | None = None
        if first_rt_name is not None:
            rt_cfg = self._resolve_runtime_config(model_config, first_rt_name)
        if active_rt is not None:
            rt_cfg = self._resolve_runtime_config(model_config, active_rt)

        model_label = "unconfigured"
        spec_type = "none"
        if rt_cfg is not None:
            model_label = rt_cfg.source.label()
            spec_type = rt_cfg.speculative.type
        elif model_config.source is not None:
            model_label = model_config.source.label()
            spec_type = model_config.speculative.type

        host = "127.0.0.1"
        base_url = f"http://127.0.0.1:{self.runtime_port}"
        if rt_cfg is not None:
            host = rt_cfg.connect_host
            base_url = f"http://{rt_cfg.connect_host}:{self.runtime_port}"

        return ModelStatus(
            name=model_config.name,
            state=state,
            active=active,
            active_runtime=active_rt,
            model=model_label,
            speculative_type=spec_type,
            host=host,
            port=self.runtime_port,
            container_id=container_id,
            last_error=last_error,
            model_path=model_path,
            draft_model_path=draft_model_path,
            base_url=base_url,
        )

    def _to_resource(self, model_config: ModelConfig) -> ModelResource:
        return ModelResource(
            name=model_config.name,
            config=model_config,
            status=self._to_status(model_config),
        )

    def model_resources(self) -> list[ModelResource]:
        return [self._to_resource(model_config) for model_config in self._config.models]

    def model_resource(self, name: str) -> ModelResource:
        return self._to_resource(self._get_model_config(name))

    def status(self) -> ServiceStatus:
        healthy = self._last_error is None
        active_container_id = None
        last_error = self._last_error
        active_runtime = None
        if self._active_runtime is not None:
            if self._active_runtime.state == "error":
                healthy = False
            last_error = self._active_runtime.last_error or self._last_error
            active_runtime = self._active_runtime.runtime_type
            if self._active_runtime.container is not None:
                active_container_id = self._active_runtime.container.id

        return ServiceStatus(
            healthy=healthy,
            api_prefix=self._config.api_prefix,
            config_path=str(self._runtime.config_path),
            active_model=self._active_model_name,
            active_runtime=active_runtime,
            active_container_id=active_container_id,
            last_error=last_error,
        )

    def _get_model_config(self, name: str) -> ModelConfig:
        for m in self._config.models:
            if m.name == name:
                return m
        msg = f"unknown model: {name}"
        raise KeyError(msg)

    def _resolve_runtime_config(
        self,
        model_config: ModelConfig,
        runtime: str,
    ) -> RuntimeConfig:
        if model_config.runtimes:
            if runtime not in model_config.runtimes:
                raise ValueError(
                    f"model '{model_config.name}' does not support runtime '{runtime}'"
                )
            return model_config.runtimes[runtime]

        if runtime not in self._config.backends:
            raise ValueError(
                f"runtime '{runtime}' is not configured in top-level backends"
            )
        if model_config.source is None:
            raise ValueError(f"model '{model_config.name}' has no configured source")

        backend_cfg: BackendConfig = self._config.backends[runtime]
        return RuntimeConfig(
            source=model_config.source,
            docker_image=backend_cfg.docker_image,
            devices=backend_cfg.devices,
            volumes=backend_cfg.volumes,
            shared_args=backend_cfg.shared_args,
            mmproj=model_config.mmproj,
            extra_args=model_config.extra_args,
            speculative=model_config.speculative,
            bind_host=backend_cfg.bind_host,
            connect_host=backend_cfg.connect_host,
        )

    def _first_runtime_name_for_model(self, model_config: ModelConfig) -> str | None:
        if model_config.runtimes:
            return next(iter(model_config.runtimes))
        if self._config.backends:
            return next(iter(self._config.backends))
        return None

    async def _resolve_runtime_artifacts(
        self,
        rt_cfg: RuntimeConfig,
    ) -> tuple[Path, Path | None, Path | None]:
        try:
            model_path = await asyncio.to_thread(self._hf.resolve_source, rt_cfg.source)
        except Exception as exc:
            raise ValueError(
                f"failed to resolve model artifact '{rt_cfg.source.label()}': {exc}"
            ) from exc

        draft_model_path: Path | None = None
        if rt_cfg.speculative.draft_model is not None:
            try:
                draft_model_path = await asyncio.to_thread(
                    self._hf.resolve_source,
                    rt_cfg.speculative.draft_model,
                )
            except Exception as exc:
                raise ValueError(
                    f"failed to resolve draft model artifact "
                    f"'{rt_cfg.speculative.draft_model.label()}': {exc}"
                ) from exc

        mmproj_path: Path | None = None
        if rt_cfg.mmproj is not None:
            try:
                mmproj_path = await asyncio.to_thread(
                    self._hf.resolve_source, rt_cfg.mmproj
                )
            except Exception as exc:
                repo_label = rt_cfg.mmproj.repo_id or rt_cfg.source.repo_id or "local"
                raise ValueError(
                    f"failed to resolve multimodal projector for repo "
                    f"'{repo_label}' ({rt_cfg.mmproj.label()}): {exc}"
                ) from exc

        return (
            Path(model_path).resolve().absolute(),
            None if draft_model_path is None else Path(draft_model_path).resolve().absolute(),
            None if mmproj_path is None else Path(mmproj_path).resolve().absolute(),
        )

    async def load(self, name: str, runtime: str) -> ModelResource:
        async with self._lock:
            model_cfg = self._get_model_config(name)
            rt_cfg = self._resolve_runtime_config(model_cfg, runtime)

            model_path, draft_model_path, mmproj_path = (
                await self._resolve_runtime_artifacts(rt_cfg)
            )

            if self._active_runtime is not None:
                try:
                    await self._stop_runtime(self._active_runtime)
                except Exception as exc:
                    self._last_error = str(exc)
                    raise

            await self._cleanup_all_managed_containers()

            runtime_obj = RuntimeContainer(runtime, name, rt_cfg, self)
            runtime_obj.model_path = model_path
            runtime_obj.draft_model_path = draft_model_path
            runtime_obj.mmproj_path = mmproj_path

            self._active_runtime = runtime_obj
            self._active_model_name = None
            self._last_error = None

            try:
                await runtime_obj.start()
                runtime_obj.state = "starting"
                await runtime_obj.load_model(name)

                runtime_obj.state = "running"
                runtime_obj.last_error = None
                self._active_model_name = name
                self._last_error = None
                return self._to_resource(model_cfg)
            except asyncio.CancelledError:
                logger.warning("Load cancelled for model %s", name)
                error_message = f"Load cancelled for model '{name}'."
                runtime_obj.state = "error"
                runtime_obj.last_error = error_message
                self._last_error = error_message
                await self._discard_runtime(runtime_obj)
                raise
            except Exception as exc:
                logger.error(
                    "Failed to start/load runtime for model %s: %s", name, exc
                )
                runtime_obj.state = "error"
                runtime_obj.last_error = str(exc)
                self._last_error = str(exc)
                await self._discard_runtime(runtime_obj)
                raise

    async def unload(self, name: str | None = None) -> ModelResource | None:
        async with self._lock:
            if name is None:
                target_name = self._active_model_name
                if target_name is None:
                    return None
            else:
                target_name = name

            model_cfg = self._get_model_config(target_name)

            active_runtime = self._active_runtime
            if active_runtime is None:
                self._active_model_name = None
                self._last_error = None
                return self._to_resource(model_cfg)

            if target_name != self._active_model_name:
                return self._to_resource(model_cfg)

            try:
                await self._stop_runtime(active_runtime)
            except Exception as exc:
                self._last_error = str(exc)
                raise

            self._last_error = None
            return self._to_resource(model_cfg)

    async def _wait_for_container_release(
        self,
        name: str,
        known_container_id: str | None = None,
    ) -> None:
        for _ in range(self._container_release_poll_attempts):
            try:
                container = await asyncio.to_thread(self.docker_client.containers.get, name)
            except Exception as exc:
                if self._is_container_not_found_error(exc):
                    return
                raise

            current_id = getattr(container, "id", None)
            if known_container_id is not None and current_id != known_container_id:
                return

            if getattr(container, "removed", False):
                return

            await asyncio.sleep(self._container_release_poll_interval_seconds)

        raise RuntimeError(
            f"Failed to release container name '{name}' within wait budget."
        )

    async def _remove_conflicting_container(self, name: str) -> None:
        try:
            old = await asyncio.to_thread(self.docker_client.containers.get, name)
        except Exception as exc:
            if self._is_container_not_found_error(exc):
                return None
            raise

        if old.labels.get("managed-by") != "inference-server":
            return

        logger.info("Removing conflicting container: %s", name)
        old_id = getattr(old, "id", None)
        try:
            await asyncio.to_thread(old.stop, timeout=5)
        except Exception as stop_exc:
            if not (
                self._is_container_not_found_error(stop_exc)
                or self._is_removal_in_progress_error(stop_exc)
            ):
                raise

        try:
            await asyncio.to_thread(old.remove, force=True)
        except Exception as remove_exc:
            if not (
                self._is_container_not_found_error(remove_exc)
                or self._is_removal_in_progress_error(remove_exc)
            ):
                raise

        await self._wait_for_container_release(name, known_container_id=old_id)

    async def _resolve_commit_hash(self, repo_id: str, revision: str = "main") -> str:
        token = getattr(self._hf, "_token", None)
        try:
            info = await asyncio.to_thread(
                huggingface_hub.model_info, repo_id, revision=revision, token=token
            )
            if info.sha:
                return info.sha
        except Exception:
            pass

        try:
            repo_escaped = "models--" + repo_id.replace("/", "--")
            snapshots_dir = self._hf.cache_dir / repo_escaped / "snapshots"
            if snapshots_dir.exists():
                snapshots = list(snapshots_dir.iterdir())
                if snapshots:
                    snapshots.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    return snapshots[0].name
        except Exception:
            pass

        return "unknown_commit"

    def _find_cached_path(self, source: ModelSource) -> Path | None:
        if source.local_path is not None:
            return Path(source.local_path).resolve().absolute()
        if source.repo_id is not None and hasattr(self._hf, "cache_files"):
            filename = source.filename
            revision = source.revision or "main"
            try:
                from typing import Any, cast

                cached = cast(Any, self._hf).cache_files()
                for cf in cached:
                    if cf.repo_id == source.repo_id:
                        filename_match = False
                        if filename is not None and cf.filename == filename:
                            filename_match = True
                        elif filename is None and cf.filename.endswith(".gguf"):
                            filename_match = True

                        if filename_match:
                            cf_refs = cf.refs if hasattr(cf, "refs") else []
                            if cf.revision == revision or revision in cf_refs:
                                return Path(cf.local_path).resolve().absolute()
            except Exception:
                pass
        return None

    def _map_model_path_sync(self, source: ModelSource, resolved_path: Path) -> str:
        local_p = resolved_path.resolve().absolute()
        cache_dir = self._hf.cache_dir.resolve().absolute()
        if local_p.is_relative_to(cache_dir):
            rel = local_p.relative_to(cache_dir)
            return f"/huggingface/{rel}"
        else:
            parent = local_p.parent
            suffix = self._dir_hash(parent)
            bind_path = f"/local_models_{suffix}"
            return f"{bind_path}/{local_p.name}"

    async def _generate_presets_ini(
        self,
        runtime_type: str,
        active_model_name: str | None = None,
        active_model_resolved_path: Path | None = None,
        active_model_draft_resolved_path: Path | None = None,
        active_model_mmproj_resolved_path: Path | None = None,
    ) -> str:
        lines = []

        for m in self._config.models:
            try:
                rt_cfg = self._resolve_runtime_config(m, runtime_type)
            except ValueError:
                continue

            resolved_model_path = None
            resolved_draft_path = None
            resolved_mmproj_path = None

            if m.name == active_model_name:
                resolved_model_path = active_model_resolved_path
                resolved_draft_path = active_model_draft_resolved_path
                resolved_mmproj_path = active_model_mmproj_resolved_path
                if resolved_mmproj_path is None and rt_cfg.mmproj is not None:
                    resolved_mmproj_path = self._find_cached_path(rt_cfg.mmproj)
            else:
                resolved_model_path = self._find_cached_path(rt_cfg.source)
                if rt_cfg.speculative.draft_model is not None:
                    resolved_draft_path = self._find_cached_path(
                        rt_cfg.speculative.draft_model
                    )
                if rt_cfg.mmproj is not None:
                    resolved_mmproj_path = self._find_cached_path(rt_cfg.mmproj)

            if resolved_model_path is None:
                continue

            lines.append(f"[{m.name}]")
            container_path = self._map_model_path_sync(
                rt_cfg.source, resolved_model_path
            )
            lines.append(f"model = {container_path}")

            if rt_cfg.speculative.type != "none":
                lines.append(f"spec-type = {rt_cfg.speculative.type}")
                if (
                    rt_cfg.speculative.draft_model is not None
                    and resolved_draft_path is not None
                ):
                    draft_p = self._map_model_path_sync(
                        rt_cfg.speculative.draft_model, resolved_draft_path
                    )
                    lines.append(f"model-draft = {draft_p}")

            if rt_cfg.mmproj is not None and resolved_mmproj_path is not None:
                mmproj_p = self._map_model_path_sync(rt_cfg.mmproj, resolved_mmproj_path)
                lines.append(f"mmproj = {mmproj_p}")

            i = 0
            while i < len(rt_cfg.extra_args):
                arg = rt_cfg.extra_args[i]
                if not (arg.startswith("-") and len(arg) > 1):
                    i += 1
                    continue

                if "=" in arg:
                    parts = arg.split("=", 1)
                    key_part = parts[0]
                    val = parts[1]
                    key = key_part[2:] if key_part.startswith("--") else key_part[1:]
                    lines.append(f"{key} = {val}")
                    i += 1
                    continue

                key = arg[2:] if arg.startswith("--") else arg[1:]
                if i + 1 < len(rt_cfg.extra_args) and not rt_cfg.extra_args[i + 1].startswith("-"):
                    val = rt_cfg.extra_args[i + 1]
                    lines.append(f"{key} = {val}")
                    i += 2
                else:
                    lines.append(f"{key} = true")
                    i += 1
            lines.append("")
        return "\n".join(lines)

    def create_proxy_client(self, base_url: str) -> httpx.AsyncClient:
        return self._proxy_client_factory(base_url)

    async def open_proxy_session(
        self,
        path: str,
        request: httpx.Request,
    ) -> ProxySessionProtocol:
        if self._active_runtime is None:
            msg = "no model is loaded"
            raise RuntimeError(msg)
        if self._active_runtime.state != "running" or self._active_model_name is None:
            msg = (
                f"model '{self._active_runtime.model_name}' is still loading "
                f"(runtime state: {self._active_runtime.state})"
            )
            raise RuntimeError(msg)
        client = self._proxy_client_factory(self._active_runtime.base_url)
        try:
            upstream = client.build_request(
                request.method,
                path,
                headers=request.headers,
                content=request.content,
                params=request.url.params,
            )
            response = await client.send(upstream, stream=True)
        except Exception as exc:
            await client.aclose()
            await self.handle_runtime_communication_failure(exc)
            raise
        return ProxySession(client=client, response=response)

    def active_model(self) -> ActiveModelProtocol | None:
        if self._active_model_name is None or self._active_runtime is None:
            return None
        model_cfg = self._get_model_config(self._active_model_name)
        return ActiveModel(
            self._active_model_name,
            model_cfg,
            self._active_runtime.runtime_type,
            self,
        )

    def _load_last_logs_from_disk(self, model_name: str) -> list[str]:
        import json
        try:
            log_dir = self._runtime.runtime_log_dir.expanduser().resolve()
            log_path = log_dir / "last_logs" / f"{RuntimeContainer._slugify(model_name)}.json"
            if log_path.exists():
                with log_path.open("r", encoding="utf-8") as fh:
                    return json.load(fh)
        except Exception as e:
            logger.warning("Failed to load last logs for model %s from disk: %s", model_name, e)
        return []

    async def get_logs(self, name: str) -> list[str]:
        if (
            self._active_runtime is not None
            and self._active_runtime.model_name == name
        ):
            lines = await self._active_runtime.get_logs()
            self._logs_cache[name] = lines
            return lines
        if name in self._logs_cache:
            return self._logs_cache[name]
        lines = self._load_last_logs_from_disk(name)
        if lines:
            self._logs_cache[name] = lines
            return lines
        return []

    def find_model_for_name(self, model_name: str) -> str | None:
        for m in self._config.models:
            if m.name == model_name:
                return m.name
        return None

    async def cleanup(self) -> None:
        async with self._lock:
            stop_exc = None
            if self._active_runtime is not None:
                try:
                    await self._stop_runtime(self._active_runtime)
                    self._last_error = None
                except Exception as exc:
                    logger.error(
                        "Failed to stop active runtime during cleanup: %s", exc
                    )
                    self._last_error = str(exc)
                    stop_exc = exc

            await self._cleanup_all_managed_containers()

            if stop_exc is not None:
                raise stop_exc

    async def handle_runtime_communication_failure(self, exc: Exception) -> None:
        error_message = f"Model runtime communication failure: {exc}"
        async with self._lock:
            self._last_error = error_message
            active_runtime = self._active_runtime
            if active_runtime is None:
                return

            active_runtime.state = "error"
            active_runtime.last_error = error_message
            await self._discard_runtime(active_runtime)

    async def _stop_runtime(self, runtime_obj: RuntimeContainer) -> None:
        self._clear_active_runtime(runtime_obj)
        try:
            await runtime_obj.stop()
        finally:
            self._clear_active_runtime(runtime_obj)

    async def _discard_runtime(self, runtime_obj: RuntimeContainer) -> None:
        self._clear_active_runtime(runtime_obj)
        try:
            await runtime_obj.stop()
        except Exception as stop_exc:
            logger.error("Failed to stop runtime during discard: %s", stop_exc)
            existing_error = runtime_obj.last_error or self._last_error or "unknown error"
            self._last_error = f"{existing_error}; cleanup failed: {stop_exc}"
        finally:
            self._clear_active_runtime(runtime_obj)

    def _clear_active_runtime(self, runtime_obj: RuntimeContainer) -> None:
        if self._active_runtime is runtime_obj:
            self._active_runtime = None
        if self._active_model_name == runtime_obj.model_name:
            self._active_model_name = None
        elif self._active_runtime is None:
            self._active_model_name = None

    async def _cleanup_all_managed_containers(self) -> None:
        try:
            client = self.docker_client
        except Exception as exc:
            logger.warning(
                "Docker daemon is not running or accessible. "
                "Skipping container cleanup: %s",
                exc,
            )
            return

        try:
            containers = await asyncio.to_thread(
                client.containers.list,
                filters={"label": "managed-by=inference-server"},
                all=True,
            )
            for container in containers:
                if self._active_runtime is not None:
                    active_container = self._active_runtime.container
                    if active_container is not None and (
                        container.id == active_container.id
                        or container.name == active_container.name
                    ):
                        logger.info(
                            "Skipping active container %s from general cleanup",
                            container.name,
                        )
                        continue
                    active_name = (
                        f"inference-server-runtime-{self._active_runtime.runtime_type}"
                    )
                    if container.name == active_name:
                        logger.info(
                            "Skipping active container by name %s from general cleanup",
                            container.name,
                        )
                        continue

                try:
                    logger.info("Cleaning up container: %s", container.name)
                    try:
                        await asyncio.to_thread(container.stop, timeout=5)
                    except docker.errors.NotFound:
                        pass
                    except Exception as stop_exc:
                        if not self._is_removal_in_progress_error(stop_exc):
                            logger.warning(
                                "Stop failed for container %s: %s",
                                container.name,
                                stop_exc,
                            )
                    try:
                        await asyncio.to_thread(container.remove, force=True)
                    except docker.errors.NotFound:
                        pass
                    except Exception as remove_exc:
                        if not self._is_removal_in_progress_error(remove_exc):
                            raise
                except Exception as exc:
                    logger.warning(
                        "Failed to clean up container %s: %s",
                        container.name,
                        exc,
                    )
        except Exception as exc:
            logger.error("Failed to list containers for cleanup: %s", exc)
