from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from app import create_app
from config import RuntimeSettings
from manager import ModelRuntimeManager, RuntimeContainer
from schemas import (
    AppConfig,
    BackendConfig,
    ModelConfig,
    ModelSource,
    RuntimeConfig,
)
from tests.test_manager import (
    FakeHF,
    MockContainer,
    MockDockerClient,
    MockUpstreamApp,
    mock_client_factory,
)


@pytest.mark.asyncio
async def test_runtime_startup_path_and_volumes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mock_client = MockDockerClient()
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    captured_kwargs: dict[str, Any] = {}
    original_run = mock_client.containers.run

    def run_spy(**kwargs: Any) -> Any:
        captured_kwargs.update(kwargs)
        return original_run(**kwargs)

    monkeypatch.setattr(mock_client.containers, "run", run_spy)

    async def fake_resolve_commit_hash(*args: Any, **kwargs: Any) -> str:
        return "mock_commit_123"

    monkeypatch.setattr(
        "manager.ModelRuntimeManager._resolve_commit_hash", fake_resolve_commit_hash
    )

    runtime = RuntimeSettings(
        config_path=tmp_path / "config.json",
    )
    app_config = AppConfig(
        models=[
            ModelConfig(
                name="primary",
                runtimes={
                    "rocm": RuntimeConfig(
                        docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
                        devices=["/dev/kfd", "/dev/dri"],
                        source=ModelSource(local_path=tmp_path / "model.gguf"),
                    )
                },
            )
        ],
    )

    upstream_app = MockUpstreamApp()
    manager = ModelRuntimeManager(
        runtime=runtime,
        app_config=app_config,
        hf=FakeHF(tmp_path),
        proxy_client_factory=lambda base_url: mock_client_factory(
            base_url, upstream_app
        ),
    )

    async def fake_to_thread(func: Any, /, *args: object, **kwargs: object) -> object:
        return func(*args, **kwargs)

    monkeypatch.setattr("manager.asyncio.to_thread", fake_to_thread)

    await manager.load("primary", "rocm")

    assert captured_kwargs.get("command") == [
        "--host",
        "0.0.0.0",
        "--port",
        "8080",
        "--models-preset",
        "/config/rocm.ini",
    ]

    volumes = captured_kwargs.get("volumes", {})
    presets_dir_str = str((tmp_path / "presets").resolve().absolute())
    assert presets_dir_str in volumes
    assert volumes[presets_dir_str] == {"bind": "/config", "mode": "ro"}


@pytest.mark.asyncio
async def test_shared_backend_config_loads_model_on_selected_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mock_client = MockDockerClient()
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    runtime = RuntimeSettings(config_path=tmp_path / "config.json")
    app_config = AppConfig(
        backends={
            "rocm": BackendConfig(
                docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
                devices=["/dev/kfd", "/dev/dri"],
            )
        },
        models=[
            ModelConfig(
                name="primary",
                source=ModelSource(local_path=tmp_path / "model.gguf"),
            )
        ],
    )

    upstream_app = MockUpstreamApp()
    manager = ModelRuntimeManager(
        runtime=runtime,
        app_config=app_config,
        hf=FakeHF(tmp_path),
        proxy_client_factory=lambda base_url: mock_client_factory(
            base_url, upstream_app
        ),
    )

    async def fake_to_thread(func: Any, /, *args: object, **kwargs: object) -> object:
        return func(*args, **kwargs)

    monkeypatch.setattr("manager.asyncio.to_thread", fake_to_thread)

    status = await manager.load("primary", "rocm")

    assert status.active is True
    assert status.status.active_runtime == "rocm"
    assert manager._active_runtime is not None
    assert manager._active_runtime.docker_image == "inference-server-llama-rocm:7.2.1-7e50ef7"


@pytest.mark.asyncio
async def test_extra_args_short_and_long_translation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app_config = AppConfig(
        models=[
            ModelConfig(
                name="primary",
                runtimes={
                    "rocm": RuntimeConfig(
                        docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
                        source=ModelSource(local_path=tmp_path / "model.gguf"),
                        extra_args=[
                            "-ngl",
                            "32",
                            "-c",
                            "2048",
                            "--ctx-size",
                            "4096",
                            "--temp",
                            "0.7",
                            "-v",
                            "--top-p=0.9",
                        ],
                    )
                },
            )
        ],
    )

    async def fake_resolve_commit_hash(*args: Any, **kwargs: Any) -> str:
        return "mock_commit_123"

    manager = ModelRuntimeManager(
        runtime=RuntimeSettings(config_path=tmp_path / "config.json"),
        app_config=app_config,
        hf=FakeHF(tmp_path),
    )
    monkeypatch.setattr(
        "manager.ModelRuntimeManager._resolve_commit_hash", fake_resolve_commit_hash
    )

    ini_content = await manager._generate_presets_ini("rocm")

    assert "ngl = 32" in ini_content
    assert "c = 2048" in ini_content
    assert "ctx-size = 4096" in ini_content
    assert "temp = 0.7" in ini_content
    assert "v = true" in ini_content
    assert "top-p = 0.9" in ini_content


@pytest.mark.asyncio
async def test_status_correctness_after_runtime_swap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mock_client = MockDockerClient()
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    async def fake_resolve_commit_hash(*args: Any, **kwargs: Any) -> str:
        return "mock_commit_123"

    monkeypatch.setattr(
        "manager.ModelRuntimeManager._resolve_commit_hash", fake_resolve_commit_hash
    )

    runtime = RuntimeSettings(
        config_path=tmp_path / "config.json",
    )
    app_config = AppConfig(
        models=[
            ModelConfig(
                name="model_a",
                runtimes={
                    "rocm": RuntimeConfig(
                        docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
                        devices=["/dev/kfd", "/dev/dri"],
                        source=ModelSource(local_path=tmp_path / "model_a.gguf"),
                    )
                },
            ),
            ModelConfig(
                name="model_b",
                runtimes={
                    "rocm": RuntimeConfig(
                        docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
                        devices=["/dev/kfd", "/dev/dri"],
                        source=ModelSource(local_path=tmp_path / "model_b.gguf"),
                    )
                },
            ),
        ],
    )

    upstream_app = MockUpstreamApp()
    manager = ModelRuntimeManager(
        runtime=runtime,
        app_config=app_config,
        hf=FakeHF(tmp_path),
        proxy_client_factory=lambda base_url: mock_client_factory(
            base_url, upstream_app
        ),
    )

    async def fake_to_thread(func: Any, /, *args: object, **kwargs: object) -> object:
        return func(*args, **kwargs)

    monkeypatch.setattr("manager.asyncio.to_thread", fake_to_thread)

    await manager.load("model_a", "rocm")
    assert manager._active_model_name == "model_a"
    assert manager.status().active_model == "model_a"

    statuses = {s.name: s for s in manager.model_statuses()}
    assert statuses["model_a"].state == "running"
    assert statuses["model_a"].active is True
    assert statuses["model_b"].state == "stopped"
    assert statuses["model_b"].active is False

    await manager.load("model_b", "rocm")
    assert manager._active_model_name == "model_b"
    assert manager.status().active_model == "model_b"

    statuses = {s.name: s for s in manager.model_statuses()}
    assert statuses["model_a"].state == "stopped"
    assert statuses["model_a"].active is False
    assert statuses["model_b"].state == "running"
    assert statuses["model_b"].active is True


@pytest.mark.asyncio
async def test_load_retries_by_restarting_after_failed_runtime_start(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mock_client = MockDockerClient()
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    runtime = RuntimeSettings(config_path=tmp_path / "config.json")
    app_config = AppConfig(
        models=[
            ModelConfig(
                name="primary",
                runtimes={
                    "rocm": RuntimeConfig(
                        docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
                        source=ModelSource(local_path=tmp_path / "model.gguf"),
                    )
                },
            )
        ],
    )

    manager = ModelRuntimeManager(
        runtime=runtime,
        app_config=app_config,
        hf=FakeHF(tmp_path),
    )

    start_attempts = 0
    load_attempts = 0

    async def fake_start(self: RuntimeContainer) -> None:
        nonlocal start_attempts
        start_attempts += 1
        if start_attempts == 1:
            self.state = "error"
            self.container = None
            self.last_error = "initial startup failed"
            raise RuntimeError("initial startup failed")
        self.state = "running"
        self.container = MockContainer(name="inference-server-runtime-rocm")

    async def fake_load_model(self: RuntimeContainer, name: str) -> None:
        nonlocal load_attempts
        load_attempts += 1

    async def fake_stop(self: RuntimeContainer) -> None:
        self.container = None
        self.state = "stopped"

    monkeypatch.setattr(RuntimeContainer, "start", fake_start)
    monkeypatch.setattr(RuntimeContainer, "load_model", fake_load_model)
    monkeypatch.setattr(RuntimeContainer, "stop", fake_stop)

    with pytest.raises(RuntimeError, match="initial startup failed"):
        await manager.load("primary", "rocm")

    assert manager._active_runtime is not None
    assert manager._active_runtime.state == "error"

    status = await manager.load("primary", "rocm")

    assert status.active is True
    assert status.state == "running"
    assert start_attempts == 2
    assert load_attempts == 1


@pytest.mark.asyncio
async def test_proxy_canonical_model_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mock_client = MockDockerClient()
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    async def fake_resolve_commit_hash(*args: Any, **kwargs: Any) -> str:
        return "mock_commit_123"

    monkeypatch.setattr(
        "manager.ModelRuntimeManager._resolve_commit_hash", fake_resolve_commit_hash
    )

    app_config = AppConfig(
        models=[
            ModelConfig(
                name="gemma",
                runtimes={
                    "rocm": RuntimeConfig(
                        docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
                        source=ModelSource(local_path=tmp_path / "gemma.gguf"),
                    )
                },
            ),
        ],
    )

    captured_bodies: list[bytes] = []

    class ProxySpyApp:
        async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
            if scope["type"] == "http":
                body = b""
                more_body = True
                while more_body:
                    message = await receive()
                    body += message.get("body", b"")
                    more_body = message.get("more_body", False)

                captured_bodies.append(body)

                await send(
                    {
                        "type": "http.response.start",
                        "status": 200,
                        "headers": [(b"content-type", b"application/json")],
                    }
                )
                await send(
                    {
                        "type": "http.response.body",
                        "body": b'{"choices": []}',
                    }
                )

    spy_app = ProxySpyApp()

    manager = ModelRuntimeManager(
        runtime=RuntimeSettings(config_path=tmp_path / "config.json"),
        app_config=app_config,
        hf=FakeHF(tmp_path),
        proxy_client_factory=lambda base_url: AsyncClient(
            transport=ASGITransport(app=spy_app),
            base_url=base_url,
        ),
    )

    async def fake_to_thread(func: Any, /, *args: object, **kwargs: object) -> object:
        return func(*args, **kwargs)

    monkeypatch.setattr("manager.asyncio.to_thread", fake_to_thread)

    app = create_app(
        app_config=app_config,
        manager=manager,
    )

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        # Requesting when nothing is loaded should fail
        res = await client.post("/api/v1/chat/completions", json={"model": "gemma"})
        assert res.status_code == 400

        # Load model
        await manager.load("gemma", "rocm")

        # Now proxy request should succeed
        res = await client.post(
            "/api/v1/chat/completions", json={"model": "gemma", "messages": []}
        )
        assert res.status_code == 200
        assert len(captured_bodies) >= 1
        parsed_captured = json.loads(captured_bodies[-1])
        assert parsed_captured["model"] == "gemma"


@pytest.mark.asyncio
async def test_runtime_model_path_matches_resolved_hf_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mock_client = MockDockerClient()
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    cache_dir = tmp_path / "hf_cache"
    repo_id = "org/remote-model"
    filename = "model.gguf"
    commit_hash = "real_commit_456"

    repo_escaped = "models--" + repo_id.replace("/", "--")
    snapshot_dir = cache_dir / repo_escaped / "snapshots" / commit_hash
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    model_file = snapshot_dir / filename
    model_file.write_text("dummy model content")

    captured_ini_content: str = ""
    original_run = mock_client.containers.run

    def run_spy(**kwargs: Any) -> Any:
        volumes = kwargs.get("volumes", {})
        presets_dir_str = next(
            (k for k, v in volumes.items() if v.get("bind") == "/config"), None
        )
        if presets_dir_str:
            ini_path = Path(presets_dir_str) / "rocm.ini"
            if ini_path.exists():
                nonlocal captured_ini_content
                captured_ini_content = ini_path.read_text()
        return original_run(**kwargs)

    monkeypatch.setattr(mock_client.containers, "run", run_spy)

    async def fake_resolve_commit_hash(*args: Any, **kwargs: Any) -> str:
        return commit_hash

    monkeypatch.setattr(
        "manager.ModelRuntimeManager._resolve_commit_hash", fake_resolve_commit_hash
    )

    class RealishHF:
        def __init__(self, cache_dir: Path) -> None:
            self.cache_dir = cache_dir

        def resolve_source(self, source: ModelSource) -> Path:
            return model_file

        def cache_files(self) -> list[Any]:
            from schemas import HFCachedFile

            return [
                HFCachedFile(
                    repo_id=repo_id,
                    revision=commit_hash,
                    refs=["main"],
                    filename=filename,
                    local_path=str(model_file),
                )
            ]

    app_config = AppConfig(
        models=[
            ModelConfig(
                name="remote_model",
                runtimes={
                    "rocm": RuntimeConfig(
                        docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
                        source=ModelSource(repo_id=repo_id, filename=filename),
                    )
                },
            )
        ],
    )

    upstream_app = MockUpstreamApp()
    manager = ModelRuntimeManager(
        runtime=RuntimeSettings(config_path=tmp_path / "config.json"),
        app_config=app_config,
        hf=RealishHF(cache_dir),
        proxy_client_factory=lambda base_url: mock_client_factory(
            base_url, upstream_app
        ),
    )

    async def fake_to_thread(func: Any, /, *args: object, **kwargs: object) -> object:
        return func(*args, **kwargs)

    monkeypatch.setattr("manager.asyncio.to_thread", fake_to_thread)

    await manager.load("remote_model", "rocm")

    assert (
        f"/huggingface/models--org--remote-model/snapshots/{commit_hash}/{filename}"
        in captured_ini_content
    )


@pytest.mark.asyncio
async def test_fresh_manager_cleanup_orphans(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mock_client = MockDockerClient()
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    orphan = MockContainer(name="inference-server-runtime-rocm")
    mock_client.containers.active_containers["inference-server-runtime-rocm"] = orphan

    manager = ModelRuntimeManager(
        runtime=RuntimeSettings(config_path=tmp_path / "config.json"),
        app_config=AppConfig(),
        hf=FakeHF(tmp_path),
    )

    await manager.cleanup()
    assert orphan.stopped is True


@pytest.mark.asyncio
async def test_cleanup_robustness_on_stop_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mock_client = MockDockerClient()
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    class StubContainer:
        def __init__(self) -> None:
            self.id = "stub-id"
            self.name = "inference-server-runtime-rocm"
            self.labels = {"managed-by": "inference-server"}
            self.status = "running"
            self.stopped = False
            self.removed = False

        def reload(self) -> None:
            pass

        def stop(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("stop failed")

        def remove(self, *args: Any, **kwargs: Any) -> None:
            self.removed = True

    stub = StubContainer()
    mock_client.containers.active_containers["inference-server-runtime-rocm"] = stub  # type: ignore

    manager = ModelRuntimeManager(
        runtime=RuntimeSettings(config_path=tmp_path / "config.json"),
        app_config=AppConfig(),
        hf=FakeHF(tmp_path),
    )

    await manager.cleanup()
    assert stub.removed is True


@pytest.mark.asyncio
async def test_runtime_load_failure_state_correctness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mock_client = MockDockerClient()
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    async def fake_resolve_commit_hash(*args: Any, **kwargs: Any) -> str:
        return "mock_commit_123"

    monkeypatch.setattr(
        "manager.ModelRuntimeManager._resolve_commit_hash", fake_resolve_commit_hash
    )

    app_config = AppConfig(
        models=[
            ModelConfig(
                name="gemma",
                runtimes={
                    "rocm": RuntimeConfig(
                        docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
                        source=ModelSource(local_path=tmp_path / "gemma.gguf"),
                    )
                },
            ),
        ],
    )

    upstream_app = MockUpstreamApp()
    manager = ModelRuntimeManager(
        runtime=RuntimeSettings(config_path=tmp_path / "config.json"),
        app_config=app_config,
        hf=FakeHF(tmp_path),
        proxy_client_factory=lambda base_url: mock_client_factory(
            base_url, upstream_app
        ),
    )

    async def fake_to_thread(func: Any, /, *args: object, **kwargs: object) -> object:
        return func(*args, **kwargs)

    monkeypatch.setattr("manager.asyncio.to_thread", fake_to_thread)

    async def failing_load(self, name: str) -> None:
        raise RuntimeError("Load crash")

    monkeypatch.setattr(RuntimeContainer, "load_model", failing_load)

    with pytest.raises(RuntimeError, match="Load crash"):
        await manager.load("gemma", "rocm")

    assert manager._active_model_name is None
    assert manager._active_runtime is not None
    assert manager._active_runtime.state == "error"


@pytest.mark.asyncio
async def test_runtime_swap_failure_leaves_honest_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mock_client = MockDockerClient()
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    async def fake_resolve_commit_hash(*args: Any, **kwargs: Any) -> str:
        return "mock_commit_123"

    monkeypatch.setattr(
        "manager.ModelRuntimeManager._resolve_commit_hash", fake_resolve_commit_hash
    )

    app_config = AppConfig(
        models=[
            ModelConfig(
                name="model_a",
                runtimes={
                    "rocm": RuntimeConfig(
                        docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
                        source=ModelSource(local_path=tmp_path / "model_a.gguf"),
                    )
                },
            ),
            ModelConfig(
                name="model_b",
                runtimes={
                    "rocm": RuntimeConfig(
                        docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
                        source=ModelSource(local_path=tmp_path / "model_b.gguf"),
                    )
                },
            ),
        ],
    )

    upstream_app = MockUpstreamApp()
    manager = ModelRuntimeManager(
        runtime=RuntimeSettings(config_path=tmp_path / "config.json"),
        app_config=app_config,
        hf=FakeHF(tmp_path),
        proxy_client_factory=lambda base_url: mock_client_factory(
            base_url, upstream_app
        ),
    )

    async def fake_to_thread(func: Any, /, *args: object, **kwargs: object) -> object:
        return func(*args, **kwargs)

    monkeypatch.setattr("manager.asyncio.to_thread", fake_to_thread)

    await manager.load("model_a", "rocm")

    async def failing_load(self, name: str) -> None:
        if name == "model_b":
            raise RuntimeError("Swap failure")

    monkeypatch.setattr(RuntimeContainer, "load_model", failing_load)

    with pytest.raises(RuntimeError, match="Swap failure"):
        await manager.load("model_b", "rocm")

    assert manager._active_model_name is None
    assert manager._active_runtime is not None
    assert manager._active_runtime.state == "error"


@pytest.mark.asyncio
async def test_runtime_dynamic_model_regeneration_on_swap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mock_client = MockDockerClient()
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    async def fake_resolve_commit_hash(*args: Any, **kwargs: Any) -> str:
        return "mock_commit_123"

    monkeypatch.setattr(
        "manager.ModelRuntimeManager._resolve_commit_hash", fake_resolve_commit_hash
    )

    app_config = AppConfig(
        models=[
            ModelConfig(
                name="model_a",
                runtimes={
                    "rocm": RuntimeConfig(
                        docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
                        source=ModelSource(local_path=tmp_path / "model_a.gguf"),
                    )
                },
            ),
            ModelConfig(
                name="model_b",
                runtimes={
                    "rocm": RuntimeConfig(
                        docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
                        source=ModelSource(local_path=tmp_path / "model_b.gguf"),
                    )
                },
            ),
        ],
    )

    upstream_app = MockUpstreamApp()
    manager = ModelRuntimeManager(
        runtime=RuntimeSettings(config_path=tmp_path / "config.json"),
        app_config=app_config,
        hf=FakeHF(tmp_path),
        proxy_client_factory=lambda base_url: mock_client_factory(
            base_url, upstream_app
        ),
    )

    async def fake_to_thread(func: Any, /, *args: object, **kwargs: object) -> object:
        return func(*args, **kwargs)

    monkeypatch.setattr("manager.asyncio.to_thread", fake_to_thread)

    await manager.load("model_a", "rocm")
    ini_content = (tmp_path / "presets" / "rocm.ini").read_text()
    assert "[model_a]" in ini_content

    await manager.load("model_b", "rocm")
    ini_content = (tmp_path / "presets" / "rocm.ini").read_text()
    assert "[model_b]" in ini_content


@pytest.mark.asyncio
async def test_runtime_cache_lookup_respects_revision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "hf_cache"
    model_dir = cache_dir / "models--org--model"
    snapshots = model_dir / "snapshots"
    snapshots.mkdir(parents=True)

    commit1_dir = snapshots / "commit1"
    commit1_dir.mkdir()
    (commit1_dir / "model.gguf").write_text("commit1")

    commit2_dir = snapshots / "commit2"
    commit2_dir.mkdir()
    (commit2_dir / "model.gguf").write_text("commit2")

    from schemas import HFCachedFile

    class MockCachedHF(FakeHF):
        def cache_files(self) -> list[HFCachedFile]:
            return [
                HFCachedFile(
                    repo_id="org/model",
                    revision="commit1",
                    refs=["main"],
                    filename="model.gguf",
                    local_path=str(commit1_dir / "model.gguf"),
                ),
                HFCachedFile(
                    repo_id="org/model",
                    revision="commit2",
                    refs=["other"],
                    filename="model.gguf",
                    local_path=str(commit2_dir / "model.gguf"),
                ),
            ]

    app_config = AppConfig(
        models=[
            ModelConfig(
                name="m1",
                runtimes={
                    "rocm": RuntimeConfig(
                        docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
                        source=ModelSource(
                            repo_id="org/model", filename="model.gguf", revision="main"
                        ),
                    )
                },
            ),
            ModelConfig(
                name="m2",
                runtimes={
                    "rocm": RuntimeConfig(
                        docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
                        source=ModelSource(
                            repo_id="org/model", filename="model.gguf", revision="other"
                        ),
                    )
                },
            ),
        ],
    )

    manager = ModelRuntimeManager(
        runtime=RuntimeSettings(config_path=tmp_path / "config.json"),
        app_config=app_config,
        hf=MockCachedHF(cache_dir),
    )

    path1 = manager._find_cached_path(app_config.models[0].runtimes["rocm"].source)
    path2 = manager._find_cached_path(app_config.models[1].runtimes["rocm"].source)

    assert path1 == commit1_dir / "model.gguf"
    assert path2 == commit2_dir / "model.gguf"


@pytest.mark.asyncio
async def test_cleanup_removes_even_if_stop_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mock_client = MockDockerClient()
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    class StubContainer:
        def __init__(self) -> None:
            self.id = "stub-id"
            self.name = "inference-server-runtime-rocm"
            self.labels = {"managed-by": "inference-server"}
            self.status = "running"
            self.removed = False

        def reload(self) -> None:
            pass

        def stop(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("stop failed")

        def remove(self, *args: Any, **kwargs: Any) -> None:
            self.removed = True

    stub = StubContainer()
    mock_client.containers.active_containers["inference-server-runtime-rocm"] = stub  # type: ignore

    manager = ModelRuntimeManager(
        runtime=RuntimeSettings(config_path=tmp_path / "config.json"),
        app_config=AppConfig(),
        hf=FakeHF(tmp_path),
    )

    await manager.cleanup()
    assert stub.removed is True


@pytest.mark.asyncio
async def test_runtime_unload_failure_state_handling(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mock_client = MockDockerClient()
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    cache_dir = tmp_path / "hf_cache"
    repo_id = "org/model"
    filename = "model.gguf"
    old_commit = "old_commit_123"
    old_file = cache_dir / "models--org--model" / "snapshots" / old_commit / filename
    old_file.parent.mkdir(parents=True, exist_ok=True)
    old_file.write_text("old model")

    new_commit = "new_commit_456"
    new_file = cache_dir / "models--org--model" / "snapshots" / new_commit / filename
    new_file.parent.mkdir(parents=True, exist_ok=True)
    new_file.write_text("new model")

    class LocalHF:
        def __init__(self) -> None:
            self.cache_dir = cache_dir

        def resolve_source(self, source: ModelSource) -> Path:
            if source.revision == "oldrev":
                return old_file
            return new_file

        def cache_files(self) -> list[Any]:
            from schemas import HFCachedFile

            return [
                HFCachedFile(
                    repo_id=repo_id,
                    revision=old_commit,
                    refs=["oldrev", "main"],
                    filename=filename,
                    local_path=str(old_file),
                ),
                HFCachedFile(
                    repo_id=repo_id,
                    revision=new_commit,
                    refs=["newrev"],
                    filename=filename,
                    local_path=str(new_file),
                ),
            ]

    app_config = AppConfig(
        models=[
            ModelConfig(
                name="model_old",
                runtimes={
                    "rocm": RuntimeConfig(
                        docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
                        source=ModelSource(
                            repo_id=repo_id, filename=filename, revision="oldrev"
                        ),
                    )
                },
            ),
            ModelConfig(
                name="model_new",
                runtimes={
                    "rocm": RuntimeConfig(
                        docker_image="inference-server-llama-rocm:7.2.1-7e50ef7",
                        source=ModelSource(
                            repo_id=repo_id, filename=filename, revision="newrev"
                        ),
                    )
                },
            ),
        ],
    )

    class FailingUnloadUpstreamApp(MockUpstreamApp):
        async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
            path = scope["path"]
            if scope["type"] == "http" and path == "/models/unload":
                await send(
                    {
                        "type": "http.response.start",
                        "status": 500,
                        "headers": [(b"content-type", b"application/json")],
                    }
                )
                await send(
                    {
                        "type": "http.response.body",
                        "body": b'{"error": "unload failed"}',
                    }
                )
            else:
                await super().__call__(scope, receive, send)

    upstream_app = FailingUnloadUpstreamApp()
    manager = ModelRuntimeManager(
        runtime=RuntimeSettings(config_path=tmp_path / "config.json"),
        app_config=app_config,
        hf=LocalHF(),
        proxy_client_factory=lambda base_url: mock_client_factory(
            base_url, upstream_app
        ),
    )

    async def fake_to_thread(func: Any, /, *args: object, **kwargs: object) -> object:
        return func(*args, **kwargs)

    monkeypatch.setattr("manager.asyncio.to_thread", fake_to_thread)

    async def fake_resolve_commit_hash(*args: Any, **kwargs: Any) -> str:
        return "mock_commit_123"

    monkeypatch.setattr(
        "manager.ModelRuntimeManager._resolve_commit_hash", fake_resolve_commit_hash
    )

    await manager.load("model_old", "rocm")
    assert manager.status().active_model == "model_old"

    with pytest.raises(RuntimeError) as exc_info:
        await manager.load("model_new", "rocm")
    assert "Failed to unload model inside" in str(exc_info.value)

    svc_status = manager.status()
    assert svc_status.healthy is False
    assert svc_status.active_model is None

    statuses = {s.name: s for s in svc_status.models}
    assert statuses["model_old"].state == "error"
    assert statuses["model_old"].last_error is not None
    assert "Failed to unload model inside" in statuses["model_old"].last_error
