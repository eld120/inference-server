from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import docker
import docker.errors
import docker.models.containers
import httpx
import huggingface_hub

from config import RuntimeSettings
from protocols import (
    ActiveModelProtocol,
    ModelResolverProtocol,
    ProxySessionProtocol,
)
from schemas import (
    AppConfig,
    RuntimeState,
    ModelConfig,
    ModelResource,
    ModelSource,
    ModelStatus,
    RuntimeConfig,
    ServiceStatus,
    SpeculativeConfig,
)

logger = logging.getLogger(__name__)


class RuntimeContainer:
    def __init__(
        self,
        runtime_type: str,
        model_name: str,
        rt_cfg: RuntimeConfig,
        manager: ModelRuntimeManager,
    ) -> None:
        self.runtime_type = runtime_type
        self.model_name = model_name
        self.config = rt_cfg
        self.manager = manager
        self.state: RuntimeState = "stopped"
        self.container: docker.models.containers.Container | None = None
        self.last_error: str | None = None
        self.model_path: Path | None = None
        self.draft_model_path: Path | None = None
        self.started_at: datetime | None = None

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
                if self.runtime_type in m.runtimes:
                    rt_cfg = m.runtimes[self.runtime_type]
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
            return
        self.state = "stopping"
        try:
            logger.info("Stopping container: %s", container.name)
            await asyncio.to_thread(container.stop, timeout=10)
            self.container = None
            self.state = "stopped"
        except Exception as stop_exc:
            logger.warning("Graceful stop failed: %s. Force removing.", stop_exc)
            try:
                await asyncio.to_thread(container.remove, force=True)
                self.container = None
                self.state = "stopped"
            except Exception as rm_exc:
                logger.error("Force remove also failed: %s", rm_exc)
                self.state = "error"
                self.last_error = f"Stop failed: {stop_exc}. Remove failed: {rm_exc}."
                raise rm_exc

    async def load_model(self, name: str) -> None:
        async with self.manager.create_proxy_client(self.base_url) as client:
            response = await client.post("/models/load", json={"model": name})
            if response.status_code != 200:
                raise RuntimeError(
                    f"Failed to load model inside runtime container: {response.text}"
                )

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
            return []
        try:
            logs_bytes = await asyncio.to_thread(container.logs, tail=500)
            return logs_bytes.decode("utf-8", errors="replace").splitlines()
        except Exception:
            return []

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


@dataclass(slots=True)
class ProxySession:
    client: httpx.AsyncClient
    response: httpx.Response


class ActiveModel:
    def __init__(self, name: str, model_config: ModelConfig, runtime: str) -> None:
        self.model_config = model_config
        self.runtime = runtime

    @property
    def config(self):
        rt_cfg = self.model_config.runtimes[self.runtime]
        return SimpleNamespace(
            name=self.model_config.name,
            model=rt_cfg.source,
            speculative=rt_cfg.speculative,
            extra_args=rt_cfg.extra_args,
            bind_host=rt_cfg.bind_host,
            connect_host=rt_cfg.connect_host,
        )


class ModelRuntimeManager:
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

    @staticmethod
    def _dir_hash(parent: Path) -> str:
        import hashlib

        return hashlib.md5(str(parent).encode()).hexdigest()[:8]

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
        first_rt_name = list(model_config.runtimes.keys())[0]
        rt_cfg = model_config.runtimes[first_rt_name]
        if active_rt and active_rt in model_config.runtimes:
            rt_cfg = model_config.runtimes[active_rt]

        # Get model label:
        model_label = "unconfigured"
        if active_rt and active_rt in model_config.runtimes:
            model_label = model_config.runtimes[active_rt].source.label()
        elif model_config.runtimes:
            model_label = model_config.runtimes[first_rt_name].source.label()

        # Get speculative type:
        spec_type = "none"
        if active_rt and active_rt in model_config.runtimes:
            spec_type = model_config.runtimes[active_rt].speculative.type
        elif model_config.runtimes:
            spec_type = model_config.runtimes[first_rt_name].speculative.type

        return ModelStatus(
            name=model_config.name,
            state=state,
            active=active,
            active_runtime=active_rt,
            model=model_label,
            speculative_type=spec_type,
            host=rt_cfg.connect_host,
            port=self.runtime_port,
            container_id=container_id,
            last_error=last_error,
            model_path=model_path,
            draft_model_path=draft_model_path,
            base_url=f"http://{rt_cfg.connect_host}:{self.runtime_port}",
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
        healthy = True
        active_container_id = None
        last_error = None
        active_runtime = None
        if self._active_runtime is not None:
            if self._active_runtime.state == "error":
                healthy = False
            last_error = self._active_runtime.last_error
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
            models=self.model_resources(),
        )

    def _get_model_config(self, name: str) -> ModelConfig:
        for m in self._config.models:
            if m.name == name:
                return m
        msg = f"unknown model: {name}"
        raise KeyError(msg)

    async def load(self, name: str, runtime: str) -> ModelResource:
        async with self._lock:
            # 1. Resolve target model config
            model_cfg = self._get_model_config(name)
            if runtime not in model_cfg.runtimes:
                raise ValueError(f"model '{name}' does not support runtime '{runtime}'")
            rt_cfg = model_cfg.runtimes[runtime]

            # 2. Check if running container is already active and compatible
            is_compatible = (
                self._active_runtime is not None
                and self._active_runtime.runtime_type == runtime
                and self._active_runtime.docker_image == rt_cfg.docker_image
                and self._active_runtime.devices == rt_cfg.devices
                and self._active_runtime.volumes == rt_cfg.volumes
                and self._active_runtime.bind_host == rt_cfg.bind_host
                and self._active_runtime.connect_host == rt_cfg.connect_host
                and self._active_runtime.shared_args == rt_cfg.shared_args
            )

            if is_compatible and self._active_runtime is not None:
                if self._active_model_name == name:
                    return self._to_resource(model_cfg)

                # Swap the model on the existing runtime container
                # 1. Always unload the old model first
                if self._active_model_name:
                    try:
                        await self._active_runtime.unload_model(self._active_model_name)
                    except Exception as exc:
                        self._active_runtime.state = "error"
                        self._active_runtime.last_error = str(exc)
                        self._active_model_name = None
                        raise
                    self._active_model_name = None

                # 2. Update config to target model
                self._active_runtime.model_name = name
                self._active_runtime.config = rt_cfg

                try:
                    # 3. Resolve and download the new model and draft model
                    model_path = await asyncio.to_thread(
                        self._hf.resolve_source, rt_cfg.source
                    )
                    draft_model_path = None
                    if rt_cfg.speculative.draft_model is not None:
                        draft_model_path = await asyncio.to_thread(
                            self._hf.resolve_source,
                            rt_cfg.speculative.draft_model,
                        )

                    # Check for --mmproj in extra_args and pre-download it
                    for idx in range(len(rt_cfg.extra_args)):
                        arg = rt_cfg.extra_args[idx]
                        mmproj_filename = None
                        if arg == "--mmproj" and idx + 1 < len(rt_cfg.extra_args):
                            mmproj_filename = rt_cfg.extra_args[idx + 1]
                        elif arg.startswith("--mmproj="):
                            parts = arg.split("=", 1)
                            if len(parts) > 1:
                                mmproj_filename = parts[1]

                        if mmproj_filename is not None:
                            mmproj_source = ModelSource(
                                repo_id=rt_cfg.source.repo_id,
                                filename=mmproj_filename,
                                revision=rt_cfg.source.revision,
                            )
                            await asyncio.to_thread(
                                self._hf.resolve_source, mmproj_source
                            )

                    self._active_runtime.model_path = (
                        Path(model_path).resolve().absolute()
                    )
                    if draft_model_path is not None:
                        self._active_runtime.draft_model_path = (
                            Path(draft_model_path).resolve().absolute()
                        )
                    else:
                        self._active_runtime.draft_model_path = None

                    # 4. Regenerate presets INI file with target model path
                    ini_content = await self._generate_presets_ini(
                        runtime_type=runtime,
                        active_model_name=name,
                        active_model_resolved_path=(self._active_runtime.model_path),
                        active_model_draft_resolved_path=(
                            self._active_runtime.draft_model_path
                        ),
                    )
                    ini_dir = self._hf.cache_dir / "presets"
                    await asyncio.to_thread(ini_dir.mkdir, parents=True, exist_ok=True)
                    ini_path = ini_dir / f"{runtime}.ini"
                    await asyncio.to_thread(ini_path.write_text, ini_content)

                    # 5. Load the model on the runtime
                    await self._active_runtime.load_model(name)
                    self._active_model_name = name
                except Exception as exc:
                    self._active_runtime.state = "error"
                    self._active_runtime.last_error = str(exc)
                    self._active_model_name = None
                    raise

                return self._to_resource(model_cfg)

            # 3. Not compatible. Stop current active runtime if running
            if self._active_runtime is not None:
                if self._active_model_name:
                    try:
                        await self._active_runtime.unload_model(self._active_model_name)
                    except Exception:
                        pass
                await self._active_runtime.stop()
                self._active_runtime = None
                self._active_model_name = None

            # Clean up any other running managed containers
            # (e.g. orphans from previous crashes)
            await self._cleanup_all_managed_containers()

            # 4. Initialize and start new runtime
            runtime_obj = RuntimeContainer(runtime, name, rt_cfg, self)

            self._active_runtime = runtime_obj

            try:
                # Resolve paths before starting the runtime
                model_path = await asyncio.to_thread(
                    self._hf.resolve_source, rt_cfg.source
                )
                draft_model_path = None
                if rt_cfg.speculative.draft_model is not None:
                    draft_model_path = await asyncio.to_thread(
                        self._hf.resolve_source, rt_cfg.speculative.draft_model
                    )

                runtime_obj.model_path = Path(model_path).resolve().absolute()
                if draft_model_path is not None:
                    runtime_obj.draft_model_path = (
                        Path(draft_model_path).resolve().absolute()
                    )

                await runtime_obj.start()
                await runtime_obj.load_model(name)

                self._active_model_name = name
                return self._to_resource(model_cfg)
            except Exception as exc:
                logger.error(
                    "Failed to start/load runtime for model %s: %s", name, exc
                )
                try:
                    await runtime_obj.stop()
                except Exception as stop_exc:
                    logger.error(
                        "Failed to stop runtime after start/load failure: %s",
                        stop_exc,
                    )

                runtime_obj.state = "error"
                runtime_obj.last_error = str(exc)
                self._active_model_name = None
                raise

    async def unload(self, name: str | None = None) -> ModelResource | None:
        async with self._lock:
            target_name = name or self._active_model_name
            if target_name is None or self._active_model_name != target_name:
                return None

            model_cfg = self._get_model_config(target_name)
            if self._active_runtime is not None:
                try:
                    await self._active_runtime.unload_model(target_name)
                except Exception as unload_exc:
                    try:
                        await self._active_runtime.stop()
                        self._active_runtime = None
                        self._active_model_name = None
                    except Exception:
                        pass
                    raise unload_exc

                await self._active_runtime.stop()
                self._active_runtime = None
                self._active_model_name = None
            else:
                self._active_model_name = None

            return self._to_resource(model_cfg)

    async def _remove_conflicting_container(self, name: str) -> None:
        try:
            old = await asyncio.to_thread(self.docker_client.containers.get, name)
            if old.labels.get("managed-by") == "inference-server":
                logger.info("Removing conflicting container: %s", name)
                await asyncio.to_thread(old.stop, timeout=5)
                await asyncio.to_thread(old.remove, force=True)
        except Exception:
            pass

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
    ) -> str:
        lines = []

        for m in self._config.models:
            if runtime_type in m.runtimes:
                rt_cfg = m.runtimes[runtime_type]
                resolved_model_path = None
                resolved_draft_path = None

                if m.name == active_model_name:
                    resolved_model_path = active_model_resolved_path
                    resolved_draft_path = active_model_draft_resolved_path
                else:
                    resolved_model_path = self._find_cached_path(rt_cfg.source)
                    if rt_cfg.speculative.draft_model is not None:
                        resolved_draft_path = self._find_cached_path(
                            rt_cfg.speculative.draft_model
                        )

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

                # Extra args mapping
                i = 0
                while i < len(rt_cfg.extra_args):
                    arg = rt_cfg.extra_args[i]
                    if arg.startswith("-") and len(arg) > 1:
                        if "=" in arg:
                            parts = arg.split("=", 1)
                            key_part = parts[0]
                            val = parts[1]
                            key = (
                                key_part[2:]
                                if key_part.startswith("--")
                                else key_part[1:]
                            )
                            if key == "mmproj":
                                mmproj_source = ModelSource(
                                    repo_id=rt_cfg.source.repo_id,
                                    filename=val,
                                    revision=rt_cfg.source.revision,
                                )
                                resolved_mmproj = self._find_cached_path(mmproj_source)
                                if resolved_mmproj is not None:
                                    val = self._map_model_path_sync(
                                        mmproj_source, resolved_mmproj
                                    )
                            lines.append(f"{key} = {val}")
                            i += 1
                        else:
                            key = arg[2:] if arg.startswith("--") else arg[1:]
                            if i + 1 < len(rt_cfg.extra_args) and not rt_cfg.extra_args[
                                i + 1
                            ].startswith("-"):
                                val = rt_cfg.extra_args[i + 1]
                                if key == "mmproj":
                                    mmproj_source = ModelSource(
                                        repo_id=rt_cfg.source.repo_id,
                                        filename=val,
                                        revision=rt_cfg.source.revision,
                                    )
                                    resolved_mmproj = self._find_cached_path(
                                        mmproj_source
                                    )
                                    if resolved_mmproj is not None:
                                        val = self._map_model_path_sync(
                                            mmproj_source, resolved_mmproj
                                        )
                                lines.append(f"{key} = {val}")
                                i += 2
                            else:
                                lines.append(f"{key} = true")
                                i += 1
                    else:
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
        except Exception:
            await client.aclose()
            raise
        return ProxySession(client=client, response=response)

    def active_model(self) -> ActiveModelProtocol | None:
        if self._active_model_name is None or self._active_runtime is None:
            return None
        model_cfg = self._get_model_config(self._active_model_name)
        return ActiveModel(
            self._active_model_name, model_cfg, self._active_runtime.runtime_type
        )

    async def get_logs(self, name: str) -> list[str]:
        if self._active_model_name == name and self._active_runtime is not None:
            return await self._active_runtime.get_logs()
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
                    await self._active_runtime.stop()
                    self._active_runtime = None
                    self._active_model_name = None
                except Exception as exc:
                    logger.error(
                        "Failed to stop active runtime during cleanup: %s", exc
                    )
                    stop_exc = exc

            await self._cleanup_all_managed_containers()

            if stop_exc is not None:
                raise stop_exc

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
                    except Exception as stop_exc:
                        logger.warning(
                            "Stop failed for container %s: %s",
                            container.name,
                            stop_exc,
                        )
                    await asyncio.to_thread(container.remove, force=True)
                except Exception as exc:
                    logger.warning(
                        "Failed to clean up container %s: %s",
                        container.name,
                        exc,
                    )
        except Exception as exc:
            logger.error("Failed to list containers for cleanup: %s", exc)
