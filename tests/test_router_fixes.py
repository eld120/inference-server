from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from httpx import ASGITransport, AsyncClient

from app import create_app
from config import RuntimeSettings
from manager import BackendManager, ContainerRuntime
from schemas import AppConfig, BackendFamilyConfig, ModelPresetConfig, ModelSource
from tests.test_manager import (
    FakeHF,
    MockDockerClient,
    MockUpstreamApp,
    mock_client_factory,
)


@pytest.mark.asyncio
async def test_router_startup_path_and_volumes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mock_client = MockDockerClient()
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    # Capture run arguments using a spy
    captured_kwargs: dict[str, Any] = {}
    original_run = mock_client.containers.run

    def run_spy(**kwargs: Any) -> Any:
        captured_kwargs.update(kwargs)
        return original_run(**kwargs)

    monkeypatch.setattr(mock_client.containers, "run", run_spy)

    async def fake_resolve_commit_hash(*args: Any, **kwargs: Any) -> str:
        return "mock_commit_123"

    monkeypatch.setattr(
        "manager.BackendManager._resolve_commit_hash", fake_resolve_commit_hash
    )

    runtime = RuntimeSettings(
        config_path=tmp_path / "config.json",
    )
    app_config = AppConfig(
        runtime_mode="router",
        backend_families={
            "rocm": BackendFamilyConfig(
                docker_image="ghcr.io/ggerganov/llama.cpp:server-rocm",
                devices=["/dev/kfd", "/dev/dri"],
            )
        },
        models=[
            ModelPresetConfig(
                name="primary",
                backend_family="rocm",
                model=ModelSource(local_path=tmp_path / "model.gguf"),
            )
        ],
    )

    upstream_app = MockUpstreamApp()
    manager = BackendManager(
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

    await manager.load("primary")

    # Assert that command correctly points to f"/config/{backend_family}.ini"
    assert captured_kwargs.get("command") == [
        "--host",
        "0.0.0.0",
        "--port",
        "8080",
        "--models-preset",
        "/config/rocm.ini",
    ]

    # Assert that volumes include the mount from presets directory to /config
    volumes = captured_kwargs.get("volumes", {})
    presets_dir_str = str((tmp_path / "presets").resolve().absolute())
    assert presets_dir_str in volumes
    assert volumes[presets_dir_str] == {"bind": "/config", "mode": "ro"}


@pytest.mark.asyncio
async def test_extra_args_short_and_long_translation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # We want to test that short-form flags like -ngl and -c, and long flags
    # like --ctx-size are correctly translated to standard INI keys without
    # leading hyphens.
    app_config = AppConfig(
        runtime_mode="router",
        backend_families={
            "rocm": BackendFamilyConfig(
                docker_image="ghcr.io/ggerganov/llama.cpp:server-rocm",
            )
        },
        models=[
            ModelPresetConfig(
                name="primary",
                backend_family="rocm",
                model=ModelSource(local_path=tmp_path / "model.gguf"),
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
        ],
    )

    async def fake_resolve_commit_hash(*args: Any, **kwargs: Any) -> str:
        return "mock_commit_123"

    manager = BackendManager(
        runtime=RuntimeSettings(config_path=tmp_path / "config.json"),
        app_config=app_config,
        hf=FakeHF(tmp_path),
    )
    monkeypatch.setattr(
        "manager.BackendManager._resolve_commit_hash", fake_resolve_commit_hash
    )

    ini_content = await manager._generate_presets_ini("rocm")

    # Assert generated keys
    assert "ngl = 32" in ini_content
    assert "c = 2048" in ini_content
    assert "ctx-size = 4096" in ini_content
    assert "temp = 0.7" in ini_content
    assert "v = true" in ini_content
    assert "top-p = 0.9" in ini_content


@pytest.mark.asyncio
async def test_status_correctness_after_router_swap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mock_client = MockDockerClient()
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    async def fake_resolve_commit_hash(*args: Any, **kwargs: Any) -> str:
        return "mock_commit_123"

    monkeypatch.setattr(
        "manager.BackendManager._resolve_commit_hash", fake_resolve_commit_hash
    )

    runtime = RuntimeSettings(
        config_path=tmp_path / "config.json",
    )
    app_config = AppConfig(
        runtime_mode="router",
        backend_families={
            "rocm": BackendFamilyConfig(
                docker_image="ghcr.io/ggerganov/llama.cpp:server-rocm",
                devices=["/dev/kfd", "/dev/dri"],
            )
        },
        models=[
            ModelPresetConfig(
                name="model_a",
                backend_family="rocm",
                model=ModelSource(local_path=tmp_path / "model_a.gguf"),
            ),
            ModelPresetConfig(
                name="model_b",
                backend_family="rocm",
                model=ModelSource(local_path=tmp_path / "model_b.gguf"),
            ),
        ],
    )

    upstream_app = MockUpstreamApp()
    manager = BackendManager(
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

    # 1. Load model_a
    await manager.load("model_a")

    # Check status of both before swap
    statuses = {s.name: s for s in manager.backend_statuses()}
    assert statuses["model_a"].active is True
    assert statuses["model_a"].state == "running"
    assert statuses["model_b"].active is False
    assert statuses["model_b"].state == "stopped"

    # 2. Swap to model_b (compatible swap)
    await manager.load("model_b")

    # Check status of both after swap
    statuses = {s.name: s for s in manager.backend_statuses()}
    assert statuses["model_a"].active is False
    assert statuses["model_a"].state == "stopped"  # Should look stopped now!
    assert statuses["model_b"].active is True
    assert statuses["model_b"].state == "running"  # Should look running now!
    assert statuses["model_b"].container_id == "mock-id-123"


@pytest.mark.asyncio
async def test_proxy_canonical_preset_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mock_client = MockDockerClient()
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    async def fake_resolve_commit_hash(*args: Any, **kwargs: Any) -> str:
        return "mock_commit_123"

    monkeypatch.setattr(
        "manager.BackendManager._resolve_commit_hash", fake_resolve_commit_hash
    )

    runtime = RuntimeSettings(
        config_path=tmp_path / "config.json",
    )

    app_config = AppConfig(
        runtime_mode="router",
        backend_families={
            "rocm": BackendFamilyConfig(
                docker_image="ghcr.io/ggerganov/llama.cpp:server-rocm",
            )
        },
        models=[
            ModelPresetConfig(
                name="gemma",
                backend_family="rocm",
                model=ModelSource(
                    repo_id="ggml-org/gemma-2-9b-it-Q4_K_M", filename="gemma.gguf"
                ),
            )
        ],
    )

    # Capture request bodies sent to upstream
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

    manager = BackendManager(
        runtime=runtime,
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

    app = create_app(runtime=runtime, app_config=app_config, manager=manager)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        await manager.load("gemma")

        # 1. Canonical preset key should work and pass through
        response = await client.post(
            "/api/v1/chat/completions",
            json={"model": "gemma", "messages": []},
        )
        assert response.status_code == 200
        assert len(captured_bodies) >= 1
        parsed_captured = json.loads(captured_bodies[-1])
        assert parsed_captured["model"] == "gemma"

        # 2. Raw repo_id should be rejected (not a canonical preset key)
        response = await client.post(
            "/api/v1/chat/completions",
            json={"model": "ggml-org/gemma-2-9b-it-Q4_K_M", "messages": []},
        )
        assert response.status_code == 400
        assert "not configured" in response.json()["detail"]


@pytest.mark.asyncio
async def test_router_preset_path_matches_resolved_hf_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mock_client = MockDockerClient()
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    # Set up a real-looking HF cache structure under tmp_path
    cache_dir = tmp_path / "hf_cache"
    repo_id = "org/remote-model"
    filename = "model.gguf"
    commit_hash = "real_commit_456"

    # Create the snapshot directory and file
    repo_escaped = "models--" + repo_id.replace("/", "--")
    snapshot_dir = cache_dir / repo_escaped / "snapshots" / commit_hash
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    model_file = snapshot_dir / filename
    model_file.write_text("dummy model content")

    # Capture the generated INI content
    captured_ini_content: str = ""
    original_run = mock_client.containers.run

    def run_spy(**kwargs: Any) -> Any:
        # Read the generated INI file from the presets directory
        # The presets directory path is mounted as /config
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

    # Mock huggingface_hub.model_info to verify token is passed and return commit
    captured_token = None
    def mock_model_info(repo, revision=None, token=None):
        nonlocal captured_token
        captured_token = token
        class Info:
            sha = commit_hash
        return Info()

    monkeypatch.setattr("huggingface_hub.model_info", mock_model_info)

    # Mock hf_hub_download to return our local model file path
    class RealishHF:
        def __init__(self, cache_dir: Path) -> None:
            self.cache_dir = cache_dir
            self._token = "mock_hf_token_xyz"

        def resolve_source(self, source: ModelSource) -> Path:
            # Emulate resolving/downloading
            return model_file

    app_config = AppConfig(
        runtime_mode="router",
        backend_families={
            "rocm": BackendFamilyConfig(
                docker_image="ghcr.io/ggerganov/llama.cpp:server-rocm",
            )
        },
        models=[
            ModelPresetConfig(
                name="remote_model",
                backend_family="rocm",
                model=ModelSource(repo_id=repo_id, filename=filename),
            )
        ],
    )

    manager = BackendManager(
        runtime=RuntimeSettings(config_path=tmp_path / "config.json"),
        app_config=app_config,
        hf=RealishHF(cache_dir),
        proxy_client_factory=lambda base_url: mock_client_factory(
            base_url, MockUpstreamApp()
        ),
    )

    async def fake_to_thread(func: Any, /, *args: object, **kwargs: object) -> object:
        return func(*args, **kwargs)

    monkeypatch.setattr("manager.asyncio.to_thread", fake_to_thread)

    # Load the remote model preset
    await manager.load("remote_model")

    # Assert that the path in the INI file matches the actual resolved cache path
    # In the container, the HF cache is mounted to /huggingface
    expected_container_path = (
        f"/huggingface/{repo_escaped}/snapshots/{commit_hash}/{filename}"
    )
    assert f"model = {expected_container_path}" in captured_ini_content


@pytest.mark.asyncio
async def test_fresh_manager_cleanup_orphans(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mock_client = MockDockerClient()
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    orphan_container = mock_client.containers.run(
        name="inference-server-orphan",
        labels={"managed-by": "inference-server"},
    )

    app_config = AppConfig(
        runtime_mode="container",
        backend_families={
            "rocm": BackendFamilyConfig(
                docker_image="ghcr.io/ggerganov/llama.cpp:server-rocm",
            )
        },
        models=[
            ModelPresetConfig(
                name="primary",
                backend_family="rocm",
                model=ModelSource(local_path=tmp_path / "model.gguf"),
            )
        ],
    )

    manager = BackendManager(
        runtime=RuntimeSettings(config_path=tmp_path / "config.json"),
        app_config=app_config,
        hf=FakeHF(tmp_path),
    )

    async def fake_to_thread(func: Any, /, *args: object, **kwargs: object) -> object:
        return func(*args, **kwargs)

    monkeypatch.setattr("manager.asyncio.to_thread", fake_to_thread)

    # Calling cleanup on a fresh manager should still sweep and remove the orphan
    await manager.cleanup()

    assert orphan_container.stopped is True
    assert orphan_container.removed is True


@pytest.mark.asyncio
async def test_cleanup_robustness_on_stop_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mock_client = MockDockerClient()
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    # Create two mock containers in the docker client
    active_container = mock_client.containers.run(
        name="inference-server-active",
        labels={"managed-by": "inference-server"},
    )
    orphan_container = mock_client.containers.run(
        name="inference-server-orphan",
        labels={"managed-by": "inference-server"},
    )

    # Make active_container.stop/remove raise an exception to simulate stop failure
    def failing_stop(*args, **kwargs):
        raise RuntimeError("docker stop failed")
    monkeypatch.setattr(active_container, "stop", failing_stop)
    monkeypatch.setattr(active_container, "remove", failing_stop)

    app_config = AppConfig(
        runtime_mode="container",
        backend_families={
            "rocm": BackendFamilyConfig(
                docker_image="ghcr.io/ggerganov/llama.cpp:server-rocm",
            )
        },
        models=[
            ModelPresetConfig(
                name="primary",
                backend_family="rocm",
                model=ModelSource(local_path=tmp_path / "model.gguf"),
            )
        ],
    )

    manager = BackendManager(
        runtime=RuntimeSettings(config_path=tmp_path / "config.json"),
        app_config=app_config,
        hf=FakeHF(tmp_path),
    )

    # Inject mock active runtime manually
    runtime = ContainerRuntime(
        app_config.models[0], app_config.backend_families["rocm"], manager
    )
    runtime.container = cast(Any, active_container)
    runtime.state = "running"
    
    manager._active_runtime = runtime
    manager._active_family = "rocm"
    manager._active_model_name = "primary"

    async def fake_to_thread(func: Any, /, *args: object, **kwargs: object) -> object:
        return func(*args, **kwargs)

    monkeypatch.setattr("manager.asyncio.to_thread", fake_to_thread)

    # Call cleanup. It should catch the stop exception
    # and continue to clean up the orphan container!
    await manager.cleanup()

    # The active runtime should be cleared
    assert manager._active_runtime is None
    assert manager._active_model_name is None
    assert manager._active_family is None

    # The orphan container should have been stopped and removed
    assert orphan_container.stopped is True
    assert orphan_container.removed is True


@pytest.mark.asyncio
async def test_router_load_failure_state_correctness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mock_client = MockDockerClient()
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    # Setup a failing HF resolver that raises ValueError during load
    class FailingHF:
        def __init__(self) -> None:
            self.cache_dir = tmp_path

        def resolve_source(self, source: ModelSource) -> Path:
            raise ValueError("download failed")

    app_config = AppConfig(
        runtime_mode="router",
        backend_families={
            "rocm": BackendFamilyConfig(
                docker_image="ghcr.io/ggerganov/llama.cpp:server-rocm",
            )
        },
        models=[
            ModelPresetConfig(
                name="failed_model",
                backend_family="rocm",
                model=ModelSource(repo_id="org/repo", filename="file.gguf"),
            )
        ],
    )

    manager = BackendManager(
        runtime=RuntimeSettings(config_path=tmp_path / "config.json"),
        app_config=app_config,
        hf=FailingHF(),
        proxy_client_factory=lambda base_url: mock_client_factory(
            base_url, MockUpstreamApp()
        ),
    )

    async def fake_to_thread(func: Any, /, *args: object, **kwargs: object) -> object:
        return func(*args, **kwargs)

    monkeypatch.setattr("manager.asyncio.to_thread", fake_to_thread)

    # Try loading the model. It should raise ValueError
    with pytest.raises(ValueError, match="download failed"):
        await manager.load("failed_model")

    # Assert manager status reports unhealthy and backend in error state
    svc_status = manager.status()
    assert svc_status.healthy is False
    assert svc_status.active_backend is None

    backend_status = next(s for s in svc_status.backends if s.name == "failed_model")
    assert backend_status.state == "error"
    assert backend_status.last_error is not None
    assert "download failed" in backend_status.last_error
    assert backend_status.container_id is None


@pytest.mark.asyncio
async def test_router_swap_failure_leaves_honest_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mock_client = MockDockerClient()
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    # We want to fail resolve_source only when resolving "B"
    class DynamicHF:
        def __init__(self) -> None:
            self.cache_dir = tmp_path

        def resolve_source(self, source: ModelSource) -> Path:
            if source.local_path is not None and "model_b" in str(source.local_path):
                raise ValueError("resolver failed for B")
            # Otherwise return a dummy file path
            p = tmp_path / "model.gguf"
            p.write_text("dummy")
            return p

    app_config = AppConfig(
        runtime_mode="router",
        backend_families={
            "rocm": BackendFamilyConfig(
                docker_image="ghcr.io/ggerganov/llama.cpp:server-rocm",
            )
        },
        models=[
            ModelPresetConfig(
                name="A",
                backend_family="rocm",
                model=ModelSource(local_path=tmp_path / "model_a.gguf"),
            ),
            ModelPresetConfig(
                name="B",
                backend_family="rocm",
                model=ModelSource(local_path=tmp_path / "model_b.gguf"),
            ),
        ],
    )

    upstream_app = MockUpstreamApp()
    manager = BackendManager(
        runtime=RuntimeSettings(config_path=tmp_path / "config.json"),
        app_config=app_config,
        hf=DynamicHF(),
        proxy_client_factory=lambda base_url: mock_client_factory(
            base_url, upstream_app
        ),
    )

    async def fake_to_thread(func: Any, /, *args: object, **kwargs: object) -> object:
        return func(*args, **kwargs)

    monkeypatch.setattr("manager.asyncio.to_thread", fake_to_thread)

    # 1. Load preset A successfully
    status_a = await manager.load("A")
    assert status_a.state == "running"
    assert status_a.active is True
    assert manager.status().active_backend == "A"

    # 2. Swap to B (resolver fails)
    with pytest.raises(ValueError, match="resolver failed for B"):
        await manager.load("B")

    # 3. Assert honest state
    svc_status = manager.status()
    assert svc_status.active_backend is None

    # Check status of preset A (should be stopped, not running)
    status_a_post = next(s for s in svc_status.backends if s.name == "A")
    assert status_a_post.state == "stopped"
    assert status_a_post.active is False

    # Check status of preset B (should be error)
    status_b_post = next(s for s in svc_status.backends if s.name == "B")
    assert status_b_post.state == "error"
    assert status_b_post.active is False
    assert "resolver failed for B" in (status_b_post.last_error or "")


@pytest.mark.asyncio
async def test_router_dynamic_preset_regeneration_on_swap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mock_client = MockDockerClient()
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    # Create dummy models in cache directories
    cache_dir = tmp_path / "hf_cache"
    model_a_file = cache_dir / "models--org--model-a/snapshots/commit_a/model.gguf"
    model_a_file.parent.mkdir(parents=True, exist_ok=True)
    model_a_file.write_text("model A content")

    model_b_file = cache_dir / "models--org--model-b/snapshots/commit_b/model.gguf"
    model_b_file.parent.mkdir(parents=True, exist_ok=True)
    model_b_file.write_text("model B content")

    class TestHF:
        def __init__(self) -> None:
            self.cache_dir = cache_dir

        def resolve_source(self, source: ModelSource) -> Path:
            if source.repo_id == "org/model-a":
                return model_a_file
            return model_b_file

        def cache_files(self) -> list[Any]:
            # Emulate scanned cache files
            from schemas import HFCachedFile
            return [
                HFCachedFile(
                    repo_id="org/model-a",
                    revision="commit_a",
                    filename="model.gguf",
                    local_path=str(model_a_file),
                    size_on_disk=100,
                ),
                HFCachedFile(
                    repo_id="org/model-b",
                    revision="commit_b",
                    filename="model.gguf",
                    local_path=str(model_b_file),
                    size_on_disk=200,
                ),
            ]

    app_config = AppConfig(
        runtime_mode="router",
        backend_families={
            "rocm": BackendFamilyConfig(
                docker_image="ghcr.io/ggerganov/llama.cpp:server-rocm",
            )
        },
        models=[
            ModelPresetConfig(
                name="A",
                backend_family="rocm",
                model=ModelSource(repo_id="org/model-a", filename="model.gguf"),
            ),
            ModelPresetConfig(
                name="B",
                backend_family="rocm",
                model=ModelSource(repo_id="org/model-b", filename="model.gguf"),
            ),
        ],
    )

    captured_ini_contents: list[str] = []

    # Spy on write_text to see the updated INI contents
    original_write = Path.write_text
    def spy_write(self_path: Any, text: str, *args: Any, **kwargs: Any) -> Any:
        if self_path.name.endswith(".ini"):
            captured_ini_contents.append(text)
        return original_write(self_path, text, *args, **kwargs)
    monkeypatch.setattr("pathlib.Path.write_text", spy_write)

    upstream_app = MockUpstreamApp()
    manager = BackendManager(
        runtime=RuntimeSettings(config_path=tmp_path / "config.json"),
        app_config=app_config,
        hf=TestHF(),
        proxy_client_factory=lambda base_url: mock_client_factory(
            base_url, upstream_app
        ),
    )

    async def fake_to_thread(func: Any, /, *args: object, **kwargs: object) -> object:
        return func(*args, **kwargs)

    monkeypatch.setattr("manager.asyncio.to_thread", fake_to_thread)

    # 1. Load A
    await manager.load("A")
    assert len(captured_ini_contents) >= 1
    ini_1 = captured_ini_contents[-1]
    assert "[A]" in ini_1
    expected_a = (
        "model = /huggingface/models--org--model-a/snapshots/commit_a/model.gguf"
    )
    assert expected_a in ini_1

    # 2. Swap to B (router is running, should rewrite INI file with B resolved path)
    await manager.load("B")
    assert len(captured_ini_contents) >= 2
    ini_2 = captured_ini_contents[-1]
    assert "[B]" in ini_2
    expected_b = (
        "model = /huggingface/models--org--model-b/snapshots/commit_b/model.gguf"
    )
    assert expected_b in ini_2


@pytest.mark.asyncio
async def test_router_cache_lookup_respects_revision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mock_client = MockDockerClient()
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    # Set up two revisions of the same repo/file in hf_cache
    cache_dir = tmp_path / "hf_cache"
    repo_id = "org/model"
    filename = "model.gguf"

    # Old revision
    old_commit = "old_commit_123"
    old_file = cache_dir / "models--org--model/snapshots" / old_commit / filename
    old_file.parent.mkdir(parents=True, exist_ok=True)
    old_file.write_text("old model")

    # New revision
    new_commit = "new_commit_456"
    new_file = cache_dir / "models--org--model/snapshots" / new_commit / filename
    new_file.parent.mkdir(parents=True, exist_ok=True)
    new_file.write_text("new model")

    class RevisionTestHF:
        def __init__(self) -> None:
            self.cache_dir = cache_dir

        def resolve_source(self, source: ModelSource) -> Path:
            if source.revision == "oldrev" or source.revision == old_commit:
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
                    size_on_disk=100,
                ),
                HFCachedFile(
                    repo_id=repo_id,
                    revision=new_commit,
                    refs=["newrev"],
                    filename=filename,
                    local_path=str(new_file),
                    size_on_disk=200,
                ),
            ]

    app_config = AppConfig(
        runtime_mode="router",
        backend_families={
            "rocm": BackendFamilyConfig(
                docker_image="ghcr.io/ggerganov/llama.cpp:server-rocm",
            )
        },
        models=[
            ModelPresetConfig(
                name="model_old",
                backend_family="rocm",
                model=ModelSource(
                    repo_id=repo_id, filename=filename, revision="oldrev"
                ),
            ),
            ModelPresetConfig(
                name="model_new",
                backend_family="rocm",
                model=ModelSource(
                    repo_id=repo_id, filename=filename, revision="newrev"
                ),
            ),
        ],
    )

    captured_ini_contents: list[str] = []
    original_write = Path.write_text
    def spy_write(self_path: Any, text: str, *args: Any, **kwargs: Any) -> Any:
        if self_path.name.endswith(".ini"):
            captured_ini_contents.append(text)
        return original_write(self_path, text, *args, **kwargs)
    monkeypatch.setattr("pathlib.Path.write_text", spy_write)

    upstream_app = MockUpstreamApp()
    manager = BackendManager(
        runtime=RuntimeSettings(config_path=tmp_path / "config.json"),
        app_config=app_config,
        hf=RevisionTestHF(),
        proxy_client_factory=lambda base_url: mock_client_factory(
            base_url, upstream_app
        ),
    )

    async def fake_to_thread(func: Any, /, *args: object, **kwargs: object) -> object:
        return func(*args, **kwargs)

    monkeypatch.setattr("manager.asyncio.to_thread", fake_to_thread)

    # 1. Load the new revision preset
    await manager.load("model_new")
    assert len(captured_ini_contents) >= 1
    ini_new = captured_ini_contents[-1]
    assert "[model_new]" in ini_new
    expected_new = (
        f"model = /huggingface/models--org--model/snapshots/{new_commit}/{filename}"
    )
    assert expected_new in ini_new

    # 2. Swap to the old revision preset (re-generates INI with the old snapshot)
    await manager.load("model_old")
    assert len(captured_ini_contents) >= 2
    ini_old = captured_ini_contents[-1]
    assert "[model_old]" in ini_old
    expected_old = (
        f"model = /huggingface/models--org--model/snapshots/{old_commit}/{filename}"
    )
    assert expected_old in ini_old


@pytest.mark.asyncio
async def test_cleanup_removes_even_if_stop_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mock_client = MockDockerClient()
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    # Put a managed container in there
    container = mock_client.containers.run(
        name="orphan-container", labels={"managed-by": "inference-server"}
    )

    # Force stop to raise an exception
    def failing_stop(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("already stopped")

    monkeypatch.setattr(container, "stop", failing_stop)

    app_config = AppConfig(
        runtime_mode="container",
        backend_families={
            "rocm": BackendFamilyConfig(
                docker_image="ghcr.io/ggerganov/llama.cpp:server-rocm"
            )
        },
        models=[],
    )

    manager = BackendManager(
        runtime=RuntimeSettings(config_path=tmp_path / "config.json"),
        app_config=app_config,
        hf=FakeHF(),
    )

    async def fake_to_thread(func: Any, /, *args: object, **kwargs: object) -> object:
        return func(*args, **kwargs)

    monkeypatch.setattr("manager.asyncio.to_thread", fake_to_thread)

    await manager.cleanup()

    # Container should have remove() called on it, so removed=True
    assert container.removed is True


@pytest.mark.asyncio
async def test_router_unload_failure_state_handling(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mock_client = MockDockerClient()
    monkeypatch.setattr("docker.from_env", lambda: mock_client)

    cache_dir = tmp_path / "hf_cache"
    repo_id = "org/model"
    filename = "model.gguf"

    # Old revision
    old_commit = "old_commit_123"
    old_file = cache_dir / "models--org--model/snapshots" / old_commit / filename
    old_file.parent.mkdir(parents=True, exist_ok=True)
    old_file.write_text("old model")

    # New revision
    new_commit = "new_commit_456"
    new_file = cache_dir / "models--org--model/snapshots" / new_commit / filename
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
                    size_on_disk=100,
                ),
                HFCachedFile(
                    repo_id=repo_id,
                    revision=new_commit,
                    refs=["newrev"],
                    filename=filename,
                    local_path=str(new_file),
                    size_on_disk=200,
                ),
            ]

    app_config = AppConfig(
        runtime_mode="router",
        backend_families={
            "rocm": BackendFamilyConfig(
                docker_image="ghcr.io/ggerganov/llama.cpp:server-rocm"
            )
        },
        models=[
            ModelPresetConfig(
                name="model_old",
                backend_family="rocm",
                model=ModelSource(
                    repo_id=repo_id, filename=filename, revision="oldrev"
                ),
            ),
            ModelPresetConfig(
                name="model_new",
                backend_family="rocm",
                model=ModelSource(
                    repo_id=repo_id, filename=filename, revision="newrev"
                ),
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
    manager = BackendManager(
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
        "manager.BackendManager._resolve_commit_hash", fake_resolve_commit_hash
    )

    # 1. Load the first model model_old successfully
    await manager.load("model_old")
    assert manager.status().active_backend == "model_old"

    # 2. Swap to model_new, but unload fails!
    with pytest.raises(RuntimeError) as exc_info:
        await manager.load("model_new")
    assert "Failed to unload model inside router" in str(exc_info.value)

    # 3. Check status transitions
    svc_status = manager.status()
    assert svc_status.healthy is False
    assert svc_status.active_backend is None

    # Check that model_old reports error
    statuses = {s.name: s for s in svc_status.backends}
    assert statuses["model_old"].state == "error"
    assert statuses["model_old"].last_error is not None
    assert "Failed to unload model inside router" in statuses["model_old"].last_error



