from pathlib import Path
from typing import Any

import httpx
import pytest

from config import RuntimeSettings
from manager import BackendManager
from schemas import AppConfig, BackendConfig, ModelSource, SpeculativeConfig


class FakeHF:
    def resolve_source(self, source: ModelSource) -> Path:
        if source.local_path is not None:
            return source.local_path
        msg = "unexpected remote source"
        raise AssertionError(msg)


class FailingHF(FakeHF):
    def resolve_source(self, source: ModelSource) -> Path:
        if source.repo_id == "broken/repo":
            msg = "download failed"
            raise ValueError(msg)
        return super().resolve_source(source)


class FakeProcess:
    def __init__(self) -> None:
        self.pid = 1234
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    async def wait(self) -> int:
        self.returncode = 0
        return 0


class FailingProxyClient(httpx.AsyncClient):
    def __init__(self, base_url: str) -> None:
        super().__init__(base_url=base_url)
        self.closed = False

    async def send(
        self,
        request: httpx.Request,
        *,
        stream: bool = False,
        auth: object = httpx.USE_CLIENT_DEFAULT,
        follow_redirects: object = httpx.USE_CLIENT_DEFAULT,
    ) -> httpx.Response:
        msg = f"backend unavailable for {request.url}"
        raise httpx.ConnectError(msg, request=request)

    async def aclose(self) -> None:
        self.closed = True
        await super().aclose()


@pytest.mark.asyncio
async def test_backend_manager_load_and_unload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = RuntimeSettings(
        config_path=tmp_path / "config.json",
        llama_server_bin="llama-server",
    )
    app_config = AppConfig(
        backends=[
            BackendConfig(
                name="primary",
                model=ModelSource(local_path=tmp_path / "model.gguf"),
                speculative=SpeculativeConfig(type="draft-mtp"),
            )
        ]
    )
    manager = BackendManager(
        runtime=runtime,
        app_config=app_config,
        hf=FakeHF(),
        proxy_client_factory=lambda base_url: httpx.AsyncClient(base_url=base_url),
    )

    fake_process = FakeProcess()

    async def fake_create_subprocess_exec(*args: str, **kwargs: object) -> FakeProcess:
        return fake_process

    async def fake_wait_until_ready(_: str) -> None:
        return None

    async def fake_to_thread(func: Any, /, *args: object, **kwargs: object) -> object:
        return func(*args, **kwargs)

    monkeypatch.setattr(
        "manager.asyncio.create_subprocess_exec",
        fake_create_subprocess_exec,
    )
    monkeypatch.setattr(manager, "_wait_until_ready", fake_wait_until_ready)
    monkeypatch.setattr("manager.asyncio.to_thread", fake_to_thread)

    status = await manager.load("primary")

    assert status.active is True
    assert status.state == "running"
    assert status.pid == 1234
    assert status.model_path == str(tmp_path / "model.gguf")
    assert manager.status().active_backend == "primary"
    assert "draft-mtp" in manager._build_command(
        manager._get_runtime("primary"),
        tmp_path / "model.gguf",
        None,
    )

    unloaded = await manager.unload("primary")

    assert unloaded is not None
    assert unloaded.state == "stopped"
    assert fake_process.terminated is True


@pytest.mark.asyncio
async def test_backend_manager_keeps_active_backend_when_new_model_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = RuntimeSettings(
        config_path=tmp_path / "config.json",
        llama_server_bin="llama-server",
    )
    app_config = AppConfig(
        backends=[
            BackendConfig(
                name="primary",
                model=ModelSource(local_path=tmp_path / "model.gguf"),
            ),
            BackendConfig(
                name="secondary",
                model=ModelSource(repo_id="broken/repo"),
            ),
        ]
    )
    manager = BackendManager(
        runtime=runtime,
        app_config=app_config,
        hf=FailingHF(),
        proxy_client_factory=lambda base_url: httpx.AsyncClient(base_url=base_url),
    )

    fake_process = FakeProcess()

    async def fake_create_subprocess_exec(*args: str, **kwargs: object) -> FakeProcess:
        return fake_process

    async def fake_wait_until_ready(_: str) -> None:
        return None

    async def fake_to_thread(func: Any, /, *args: object, **kwargs: object) -> object:
        return func(*args, **kwargs)

    monkeypatch.setattr(
        "manager.asyncio.create_subprocess_exec",
        fake_create_subprocess_exec,
    )
    monkeypatch.setattr(manager, "_wait_until_ready", fake_wait_until_ready)
    monkeypatch.setattr("manager.asyncio.to_thread", fake_to_thread)

    await manager.load("primary")
    with pytest.raises(ValueError):
        await manager.load("secondary")

    assert manager.status().active_backend == "primary"
    assert manager.status().backends[0].state == "running"
    assert fake_process.terminated is False


@pytest.mark.asyncio
async def test_backend_manager_adds_draft_model_flag_for_draft_simple(
    tmp_path: Path,
) -> None:
    runtime = RuntimeSettings(
        config_path=tmp_path / "config.json",
        llama_server_bin="llama-server",
    )
    app_config = AppConfig(
        backends=[
            BackendConfig(
                name="primary",
                model=ModelSource(local_path=tmp_path / "model.gguf"),
                speculative=SpeculativeConfig(
                    type="draft-simple",
                    draft_model=ModelSource(local_path=tmp_path / "draft.gguf"),
                ),
            )
        ]
    )
    manager = BackendManager(runtime=runtime, app_config=app_config, hf=FakeHF())

    command = manager._build_command(
        manager._get_runtime("primary"),
        tmp_path / "model.gguf",
        tmp_path / "draft.gguf",
    )

    assert "--spec-type" in command
    assert "draft-simple" in command
    assert "-md" in command
    assert str(tmp_path / "draft.gguf") in command


@pytest.mark.asyncio
async def test_open_proxy_session_closes_client_on_send_failure(
    tmp_path: Path,
) -> None:
    runtime = RuntimeSettings(
        config_path=tmp_path / "config.json",
        llama_server_bin="llama-server",
    )
    app_config = AppConfig(
        backends=[
            BackendConfig(
                name="primary",
                model=ModelSource(local_path=tmp_path / "model.gguf"),
            )
        ]
    )
    clients: list[FailingProxyClient] = []

    def factory(base_url: str) -> FailingProxyClient:
        client = FailingProxyClient(base_url)
        clients.append(client)
        return client

    manager = BackendManager(
        runtime=runtime,
        app_config=app_config,
        hf=FakeHF(),
        proxy_client_factory=factory,
    )
    manager._active_backend_name = "primary"

    request = httpx.Request("POST", "http://placeholder/v1/chat/completions")

    with pytest.raises(httpx.ConnectError):
        await manager.open_proxy_session("/v1/chat/completions", request)

    assert len(clients) == 1
    assert clients[0].closed is True
