from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import docker
import docker.errors
import docker.models.containers
import httpx
import huggingface_hub

from config import RuntimeSettings
from protocols import (
    ActiveBackendProtocol,
    ModelResolverProtocol,
    ProxySessionProtocol,
)
from schemas import (
    AppConfig,
    BackendFamilyConfig,
    BackendState,
    BackendStatus,
    ModelPresetConfig,
    ModelSource,
    ServiceStatus,
)

logger = logging.getLogger(__name__)


class BackendRuntime:
    def __init__(
        self,
        preset: ModelPresetConfig,
        family: BackendFamilyConfig,
        manager: BackendManager,
    ) -> None:
        self.config = preset
        self.family_config = family
        self.manager = manager
        self.state: BackendState = "stopped"
        self.container: docker.models.containers.Container | None = None
        self.last_error: str | None = None
        self.model_path: Path | None = None
        self.draft_model_path: Path | None = None
        self.started_at: datetime | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.config.connect_host}:{self.manager.backend_port}"

    async def start(self) -> None:
        raise NotImplementedError

    async def stop(self) -> None:
        container = self.container
        if container is None:
            return
        self.state = "stopping"
        try:
            # Authoritative stop semantics:
            # Try a graceful stop, fallback to force remove if it fails/times out
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
        raise NotImplementedError

    async def unload_model(self, name: str) -> None:
        raise NotImplementedError

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
        image = self.family_config.docker_image
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
        msg = "backend did not become ready in time"
        if last_error is not None:
            raise TimeoutError(msg) from last_error
        raise TimeoutError(msg)


class ContainerRuntime(BackendRuntime):
    async def start(self) -> None:
        self.state = "starting"
        self.last_error = None
        try:
            # 1. Resolve model sources on host
            model_path = await asyncio.to_thread(
                self.manager._hf.resolve_source, self.config.model
            )
            draft_model_path = None
            if self.config.speculative.draft_model is not None:
                draft_model_path = await asyncio.to_thread(
                    self.manager._hf.resolve_source,
                    self.config.speculative.draft_model,
                )

            self.model_path = Path(model_path).resolve().absolute()
            if draft_model_path is not None:
                self.draft_model_path = Path(draft_model_path).resolve().absolute()

            await self._ensure_image_pulled()

            # 2. Setup volume binds
            host_model_dir = str(self.model_path.parent)
            container_model_path = f"/models/{self.model_path.name}"
            volumes = {host_model_dir: {"bind": "/models", "mode": "ro"}}

            container_draft_model_path = None
            if self.draft_model_path is not None:
                if self.draft_model_path.parent == self.model_path.parent:
                    container_draft_model_path = f"/models/{self.draft_model_path.name}"
                else:
                    volumes[str(self.draft_model_path.parent)] = {
                        "bind": "/draft_models",
                        "mode": "ro",
                    }
                    container_draft_model_path = (
                        f"/draft_models/{self.draft_model_path.name}"
                    )

            # Apply family specific custom mounts
            for src, dst in self.family_config.volumes.items():
                abs_src = str(Path(src).resolve().absolute())
                volumes[abs_src] = {"bind": dst, "mode": "ro"}

            # Setup Port mappings & Devices
            ports = {"8080/tcp": (self.config.bind_host, self.manager.backend_port)}
            devices = [f"{d}:{d}" for d in self.family_config.devices]

            # Build direct model launch command
            command = [
                "--host",
                "0.0.0.0",
                "--port",
                "8080",
                "-m",
                container_model_path,
            ]
            if self.config.speculative.type != "none":
                command.extend(["--spec-type", self.config.speculative.type])
            if container_draft_model_path is not None:
                command.extend(["-md", container_draft_model_path])

            # Ensure GPU offload defaults match between strategies
            # (default to -ngl 99 if not specified)
            has_gpu_offload = any(
                arg.split("=")[0] in ["-ngl", "--n-gpu-layers", "--gpu-layers"]
                for arg in (self.family_config.shared_args + self.config.extra_args)
            )
            if not has_gpu_offload:
                command.extend(["-ngl", "99"])

            command.extend(self.family_config.shared_args)
            command.extend(self.config.extra_args)

            container_name = f"inference-server-{self.config.name}"
            await self.manager._remove_conflicting_container(container_name)

            self.container = await asyncio.to_thread(
                self.manager.docker_client.containers.run,
                image=self.family_config.docker_image,
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

    async def load_model(self, name: str) -> None:
        # For Container strategy, starting the container loads the model
        pass

    async def unload_model(self, name: str) -> None:
        # Unloading the model stops the container
        await self.stop()


class RouterRuntime(BackendRuntime):
    def __init__(
        self,
        preset: ModelPresetConfig,
        family: BackendFamilyConfig,
        manager: BackendManager,
        family_name: str,
    ) -> None:
        super().__init__(preset, family, manager)
        self.backend_family = family_name

    async def start(self) -> None:
        self.state = "starting"
        self.last_error = None
        try:
            # 1. Generate models preset INI file on host
            ini_content = await self.manager._generate_presets_ini(
                self.backend_family,
                active_preset_name=self.config.name,
                active_preset_resolved_path=self.model_path,
                active_preset_draft_resolved_path=self.draft_model_path,
            )
            ini_dir = self.manager._hf.cache_dir / "presets"
            await asyncio.to_thread(ini_dir.mkdir, parents=True, exist_ok=True)
            ini_path = ini_dir / f"{self.backend_family}.ini"
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
                if m.backend_family == self.backend_family:
                    if m.model.local_path is not None:
                        local_p = Path(m.model.local_path).resolve().absolute()
                        cache_dir = self.manager._hf.cache_dir.resolve().absolute()
                        if not local_p.is_relative_to(cache_dir):
                            parent = local_p.parent
                            suffix = self.manager._dir_hash(parent)
                            bind_path = f"/local_models_{suffix}"
                            volumes[str(parent)] = {
                                "bind": bind_path,
                                "mode": "ro",
                            }
                    # Resolve speculative draft model local parent directories
                    if (
                        m.speculative.draft_model is not None
                        and m.speculative.draft_model.local_path is not None
                    ):
                        local_p = (
                            Path(m.speculative.draft_model.local_path)
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

            # Apply family specific custom mounts
            for src, dst in self.family_config.volumes.items():
                abs_src = str(Path(src).resolve().absolute())
                volumes[abs_src] = {"bind": dst, "mode": "ro"}

            # Port & GPU setups
            ports = {"8080/tcp": (self.config.bind_host, self.manager.backend_port)}
            devices = [f"{d}:{d}" for d in self.family_config.devices]

            # Start router mode: load via --models-preset and do not specify -m
            command = [
                "--host",
                "0.0.0.0",
                "--port",
                "8080",
                "--models-preset",
                f"/config/{self.backend_family}.ini",
            ]
            command.extend(self.family_config.shared_args)

            container_name = f"inference-server-router-{self.backend_family}"
            await self.manager._remove_conflicting_container(container_name)

            self.container = await asyncio.to_thread(
                self.manager.docker_client.containers.run,
                image=self.family_config.docker_image,
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

    async def load_model(self, name: str) -> None:
        # Explicitly post load to the router container
        async with self.manager.create_proxy_client(self.base_url) as client:
            response = await client.post("/models/load", json={"model": name})
            if response.status_code != 200:
                raise RuntimeError(
                    f"Failed to load model inside router: {response.text}"
                )

    async def unload_model(self, name: str) -> None:
        # Explicitly post unload to the router container
        async with self.manager.create_proxy_client(self.base_url) as client:
            response = await client.post("/models/unload", json={"model": name})
            if response.status_code != 200:
                raise RuntimeError(
                    f"Failed to unload model inside router: {response.text}"
                )


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
        self._active_model_name: str | None = None
        self._active_family: str | None = None
        self._active_runtime: BackendRuntime | None = None
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
    def backend_port(self) -> int:
        return self._runtime.backend_port

    @staticmethod
    def _default_client_factory(base_url: str) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(60.0, read=None),
        )

    def backends(self) -> list[ModelPresetConfig]:
        return list(self._config.models)

    def backend_statuses(self) -> list[BackendStatus]:
        return [self._to_status(model_config) for model_config in self._config.models]

    def _to_status(self, model_config: ModelPresetConfig) -> BackendStatus:
        is_current = (
            self._active_runtime is not None
            and self._active_runtime.config.name == model_config.name
        )
        active = self._active_model_name == model_config.name
        state: BackendState = "stopped"
        container_id: str | None = None
        last_error: str | None = None
        model_path: str | None = None
        draft_model_path: str | None = None

        if is_current and self._active_runtime is not None:
            # In Router mode, the container might be running, but if this preset
            # is not the active model, then the preset is stopped/unloaded.
            if (
                isinstance(self._active_runtime, RouterRuntime)
                and self._active_runtime.state == "running"
                and not active
            ):
                state = "stopped"
            else:
                state = self._active_runtime.state
                last_error = self._active_runtime.last_error
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

        return BackendStatus(
            name=model_config.name,
            state=state,
            model=model_config.model.label(),
            speculative_type=model_config.speculative.type,
            host=model_config.connect_host,
            port=self.backend_port,
            container_id=container_id,
            last_error=last_error,
            model_path=model_path,
            draft_model_path=draft_model_path,
            base_url=f"http://{model_config.connect_host}:{self.backend_port}",
            active=active,
        )

    def status(self) -> ServiceStatus:
        healthy = True
        if self._active_runtime is not None and self._active_runtime.state == "error":
            healthy = False
        return ServiceStatus(
            healthy=healthy,
            api_prefix=self._config.api_prefix,
            active_backend=self._active_model_name,
            backends=self.backend_statuses(),
        )

    def _get_preset(self, name: str) -> ModelPresetConfig:
        for m in self._config.models:
            if m.name == name:
                return m
        msg = f"unknown backend: {name}"
        raise KeyError(msg)

    async def load(self, name: str) -> BackendStatus:
        async with self._lock:
            # 1. Resolve target preset config
            preset = self._get_preset(name)
            family_name = preset.backend_family
            family_config = self._config.backend_families.get(family_name)
            if family_config is None:
                msg = f"unknown backend family: {family_name}"
                raise ValueError(msg)

            # Determine runtime strategy class
            mode = self._config.runtime_mode
            if mode == "router":
                runtime_class: type[BackendRuntime] = RouterRuntime
            elif mode == "container":
                runtime_class = ContainerRuntime
            else:
                # auto mode
                if family_config.router_supported:
                    runtime_class = RouterRuntime
                else:
                    runtime_class = ContainerRuntime

            # 2. Check if running container is already active and compatible
            is_compatible = (
                self._active_runtime is not None
                and self._active_family == family_name
                and type(self._active_runtime) is runtime_class
                and self._active_runtime.config.bind_host == preset.bind_host
                and self._active_runtime.config.connect_host == preset.connect_host
            )

            if is_compatible and self._active_runtime is not None:
                if self._active_model_name == name:
                    return self._to_status(preset)

                # Swap the model on the existing router container
                if runtime_class is RouterRuntime:
                    # 1. Always unload the old model first
                    if self._active_model_name:
                        try:
                            await self._active_runtime.unload_model(
                                self._active_model_name
                            )
                        except Exception as exc:
                            self._active_runtime.state = "error"
                            self._active_runtime.last_error = str(exc)
                            self._active_model_name = None
                            raise
                        self._active_model_name = None

                    # 2. Update config to the target preset so that status()
                    # correctly associates errors with the target preset
                    self._active_runtime.config = preset

                    try:
                        # 3. Resolve and download the new model and draft model
                        model_path = await asyncio.to_thread(
                            self._hf.resolve_source, preset.model
                        )
                        draft_model_path = None
                        if preset.speculative.draft_model is not None:
                            draft_model_path = await asyncio.to_thread(
                                self._hf.resolve_source,
                                preset.speculative.draft_model,
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

                        # 4. Regenerate presets INI file with target preset path
                        ini_content = await self._generate_presets_ini(
                            family_name=family_name,
                            active_preset_name=name,
                            active_preset_resolved_path=(
                                self._active_runtime.model_path
                            ),
                            active_preset_draft_resolved_path=(
                                self._active_runtime.draft_model_path
                            ),
                        )
                        ini_dir = self._hf.cache_dir / "presets"
                        await asyncio.to_thread(
                            ini_dir.mkdir, parents=True, exist_ok=True
                        )
                        ini_path = ini_dir / f"{family_name}.ini"
                        await asyncio.to_thread(ini_path.write_text, ini_content)

                        # 5. Load the model on the router
                        await self._active_runtime.load_model(name)
                        self._active_model_name = name
                    except Exception as exc:
                        self._active_runtime.state = "error"
                        self._active_runtime.last_error = str(exc)
                        self._active_model_name = None
                        raise

                    return self._to_status(preset)

            # 3. Not compatible. Stop current active runtime if running
            if self._active_runtime is not None:
                await self._active_runtime.stop()
                self._active_runtime = None
                self._active_model_name = None
                self._active_family = None

            # Clean up any other running managed containers
            # (e.g. orphans from previous crashes)
            await self._cleanup_all_managed_containers()

            # 4. Initialize and start new runtime
            if runtime_class is RouterRuntime:
                runtime = RouterRuntime(preset, family_config, self, family_name)
            else:
                runtime = ContainerRuntime(preset, family_config, self)

            self._active_runtime = runtime
            self._active_family = family_name

            try:
                # For Router mode, resolve paths *before* starting the runtime
                # so that the models are already downloaded and cached
                # before generating the INI file
                if runtime_class is RouterRuntime:
                    model_path = await asyncio.to_thread(
                        self._hf.resolve_source, preset.model
                    )
                    draft_model_path = None
                    if preset.speculative.draft_model is not None:
                        draft_model_path = await asyncio.to_thread(
                            self._hf.resolve_source, preset.speculative.draft_model
                        )

                    runtime.model_path = Path(model_path).resolve().absolute()
                    if draft_model_path is not None:
                        runtime.draft_model_path = (
                            Path(draft_model_path).resolve().absolute()
                        )
                    
                    pass

                await runtime.start()

                if runtime_class is RouterRuntime:
                    await runtime.load_model(name)

                self._active_model_name = name
                return self._to_status(preset)
            except Exception as exc:
                logger.error(
                    "Failed to start/load runtime for preset %s: %s", name, exc
                )
                try:
                    await runtime.stop()
                except Exception as stop_exc:
                    logger.error(
                        "Failed to stop runtime after start/load failure: %s",
                        stop_exc,
                    )
                
                # Keep active runtime reference to expose the error state via status()
                runtime.state = "error"
                runtime.last_error = str(exc)
                self._active_model_name = None
                raise

    async def unload(self, name: str | None = None) -> BackendStatus | None:
        async with self._lock:
            target_name = name or self._active_model_name
            if target_name is None or self._active_model_name != target_name:
                return None

            preset = self._get_preset(target_name)
            if self._active_runtime is not None:
                await self._active_runtime.unload_model(target_name)

            # If Container mode, stopping is handled during unload.
            # In Router mode, we keep the router container running but
            # unset active state.
            if type(self._active_runtime) is ContainerRuntime:
                self._active_runtime = None
                self._active_family = None

            self._active_model_name = None
            return self._to_status(preset)

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
        # Query commit hash (revision sha) from HF hub
        token = getattr(self._hf, "_token", None)
        try:
            info = await asyncio.to_thread(
                huggingface_hub.model_info, repo_id, revision=revision, token=token
            )
            if info.sha:
                return info.sha
        except Exception:
            pass

        # Local cache scan fallback
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

    def _map_preset_path_sync(self, source: ModelSource, resolved_path: Path) -> str:
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
        family_name: str,
        active_preset_name: str | None = None,
        active_preset_resolved_path: Path | None = None,
        active_preset_draft_resolved_path: Path | None = None,
    ) -> str:
        lines = []
        lines.append("[*]")
        lines.append("ngl = 99")
        lines.append("")

        for m in self._config.models:
            if m.backend_family == family_name:
                resolved_model_path = None
                resolved_draft_path = None

                if m.name == active_preset_name:
                    resolved_model_path = active_preset_resolved_path
                    resolved_draft_path = active_preset_draft_resolved_path
                else:
                    resolved_model_path = self._find_cached_path(m.model)
                    if m.speculative.draft_model is not None:
                        resolved_draft_path = self._find_cached_path(
                            m.speculative.draft_model
                        )

                if resolved_model_path is None:
                    continue

                lines.append(f"[{m.name}]")
                container_path = self._map_preset_path_sync(
                    m.model, resolved_model_path
                )
                lines.append(f"model = {container_path}")

                if m.speculative.type != "none":
                    lines.append(f"spec-type = {m.speculative.type}")
                    if (
                        m.speculative.draft_model is not None
                        and resolved_draft_path is not None
                    ):
                        draft_p = self._map_preset_path_sync(
                            m.speculative.draft_model, resolved_draft_path
                        )
                        lines.append(f"model-draft = {draft_p}")

                # Extra args mapping
                i = 0
                while i < len(m.extra_args):
                    arg = m.extra_args[i]
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
                            lines.append(f"{key} = {val}")
                            i += 1
                        else:
                            key = arg[2:] if arg.startswith("--") else arg[1:]
                            if i + 1 < len(m.extra_args) and not m.extra_args[
                                i + 1
                            ].startswith("-"):
                                val = m.extra_args[i + 1]
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
            msg = "no backend is loaded"
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
        except Exception:  # noqa: BLE001
            await client.aclose()
            raise
        return ProxySession(client=client, response=response)

    def active_backend(self) -> ActiveBackendProtocol | None:
        if self._active_model_name is None:
            return None
        # Return a namespace containing the active ModelPresetConfig for compatibility
        preset = self._get_preset(self._active_model_name)
        return SimpleNamespace(config=preset)

    async def get_logs(self, name: str) -> list[str]:
        if self._active_model_name == name and self._active_runtime is not None:
            return await self._active_runtime.get_logs()
        return []

    def find_backend_for_model(self, model_name: str) -> str | None:
        for m in self._config.models:
            if m.name == model_name:
                return m.name
        return None

    async def cleanup(self) -> None:
        async with self._lock:
            # 1. Stop current runtime
            if self._active_runtime is not None:
                try:
                    await self._active_runtime.stop()
                except Exception as exc:
                    logger.error(
                        "Failed to stop active runtime during cleanup: %s", exc
                    )
                finally:
                    self._active_runtime = None
                    self._active_model_name = None
                    self._active_family = None

            # 2. Aggressively clean up all containers matching our label
            await self._cleanup_all_managed_containers()

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


from types import SimpleNamespace  # noqa: E402
