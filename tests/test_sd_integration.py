from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app import _preferred_runtime_for_model, create_app
from config import RuntimeSettings, load_app_config, save_app_config
from manager import RuntimeContainer
from schemas import (
    AppConfig,
    BackendConfig,
    ModelConfig,
    ModelSource,
    RuntimeConfig,
)


def test_app_config_with_sd_backends():
    cfg = AppConfig(
        backends={
            "rocm": BackendConfig(docker_image="llama-rocm:latest"),
            "sd-rocm": BackendConfig(
                docker_image="inference-server-sd-rocm:7.2.4",
                devices=["/dev/kfd", "/dev/dri"],
            ),
        },
        models=[
            ModelConfig(
                name="qwen-image",
                task="image_generation",
                source=ModelSource(
                    repo_id="Qwen/Qwen-Image", filename="qwen_image_q4.gguf"
                ),
                vae=ModelSource(repo_id="stabilityai/sdxl-vae", filename="vae.gguf"),
                clip_l=ModelSource(
                    repo_id="openai/clip-vit-large", filename="clip_l.gguf"
                ),
            ),
            ModelConfig(
                name="minimax-h3",
                task="video_generation",
                source=ModelSource(
                    repo_id="molbal/MiniMax-H3-GGUF",
                    filename="MiniMax-H3-Q4_K_M.gguf",
                ),
            ),
        ],
    )
    assert len(cfg.models) == 2
    assert cfg.models[0].task == "image_generation"
    assert cfg.models[0].vae is not None
    assert cfg.models[0].vae.filename == "vae.gguf"
    assert cfg.models[1].task == "video_generation"


def test_config_path_normalization(tmp_path: Path):
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(
        """{
        "backends": {
            "sd-rocm": {
                "docker_image": "sd-rocm:latest"
            }
        },
        "models": [
            {
                "name": "local-diffusion",
                "task": "image_generation",
                "source": {
                    "local_path": "models/model.gguf"
                },
                "vae": {
                    "local_path": "models/vae.gguf"
                }
            }
        ]
    }"""
    )
    app_cfg = load_app_config(cfg_file)
    expected_src = (tmp_path / "models" / "model.gguf").resolve()
    expected_vae = (tmp_path / "models" / "vae.gguf").resolve()
    assert app_cfg.models[0].source.local_path == expected_src
    assert app_cfg.models[0].vae.local_path == expected_vae

    save_app_config(cfg_file, app_cfg)
    app_cfg_reloaded = load_app_config(cfg_file)
    assert app_cfg_reloaded.models[0].source.local_path == expected_src


def test_preferred_runtime_for_diffusion():
    app_cfg = AppConfig(
        backends={
            "rocm": BackendConfig(docker_image="llama-rocm:latest"),
            "vulkan": BackendConfig(docker_image="llama-vulkan:latest"),
            "sd-rocm": BackendConfig(docker_image="sd-rocm:latest"),
            "sd-vulkan": BackendConfig(docker_image="sd-vulkan:latest"),
        },
        models=[
            ModelConfig(
                name="chat-model",
                task="chat",
                source=ModelSource(repo_id="test/chat", filename="model.gguf"),
            ),
            ModelConfig(
                name="image-model",
                task="image_generation",
                source=ModelSource(repo_id="test/image", filename="model.gguf"),
            ),
        ],
    )
    mock_mgr = MagicMock()
    mock_mgr.models.return_value = app_cfg.models

    # Chat model should prefer vulkan then rocm
    pref_chat = _preferred_runtime_for_model(mock_mgr, app_cfg, "chat-model")
    assert pref_chat == "vulkan"

    # Image model should prefer sd-vulkan then sd-rocm
    pref_image = _preferred_runtime_for_model(mock_mgr, app_cfg, "image-model")
    assert pref_image == "sd-vulkan"



@pytest.mark.asyncio
async def test_runtime_container_sd_command_generation(tmp_path: Path):
    rt_cfg = RuntimeConfig(
        source=ModelSource(local_path=tmp_path / "model.gguf"),
        vae=ModelSource(local_path=tmp_path / "vae.gguf"),
        clip_l=ModelSource(local_path=tmp_path / "clip_l.gguf"),
        t5xxl=ModelSource(local_path=tmp_path / "t5xxl.gguf"),
        task="image_generation",
        docker_image="inference-server-sd-rocm:7.2.4",
        devices=["/dev/kfd", "/dev/dri"],
        shared_args=["--steps", "30"],
        extra_args=["--cfg-scale", "7.0"],
    )

    mock_mgr = MagicMock()
    mock_mgr._hf.cache_dir = tmp_path / "hf_cache"
    mock_mgr._dir_hash.return_value = "abc12345"
    mock_mgr.runtime_port = 39281
    mock_mgr.docker_client.containers.run = MagicMock()
    mock_mgr._remove_conflicting_container = AsyncMock()

    container = RuntimeContainer("sd-rocm", "qwen-image", rt_cfg, mock_mgr)
    container.model_path = tmp_path / "model.gguf"
    container.vae_path = tmp_path / "vae.gguf"
    container.clip_l_path = tmp_path / "clip_l.gguf"
    container.t5xxl_path = tmp_path / "t5xxl.gguf"

    assert container.is_sd_runtime() is True

    with (
        patch.object(container, "_ensure_image_pulled", new_callable=AsyncMock),
        patch.object(container, "_wait_until_ready", new_callable=AsyncMock),
    ):
        await container.start()

    run_args = mock_mgr.docker_client.containers.run.call_args
    assert run_args is not None
    cmd = run_args.kwargs["command"]

    assert "--host" in cmd
    assert "0.0.0.0" in cmd
    assert "-m" in cmd
    assert "--vae" in cmd
    assert "--clip_l" in cmd
    assert "--t5xxl" in cmd
    assert "--steps" in cmd
    assert "30" in cmd
    assert "--cfg-scale" in cmd
    assert "7.0" in cmd


def test_app_image_generation_route_proxy(tmp_path: Path):
    runtime_settings = RuntimeSettings(
        config_path=tmp_path / "config.json",
        runtime_log_dir=tmp_path / "logs",
    )
    app_cfg = AppConfig(
        backends={
            "sd-rocm": BackendConfig(docker_image="sd-rocm:latest"),
        },
        models=[
            ModelConfig(
                name="qwen-image",
                task="image_generation",
                source=ModelSource(repo_id="Qwen/Qwen-Image", filename="model.gguf"),
            ),
        ],
    )

    mock_mgr = MagicMock()
    mock_mgr.models.return_value = app_cfg.models
    mock_mgr.model_statuses.return_value = []
    mock_mgr.find_model_for_name.return_value = "qwen-image"
    mock_mgr.active_model.return_value = None
    mock_resource = MagicMock()
    mock_resource.status.state = "running"
    mock_mgr.model_resource.return_value = mock_resource
    mock_mgr.load = AsyncMock()

    # Mock proxy response
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}
    mock_response.aread = AsyncMock(
        return_value=b'{"created": 12345, "data": [{"b64_json": "AAAA"}]}'
    )
    mock_response.aclose = AsyncMock()
    mock_response.aiter_raw.return_value = [
        b'{"created": 12345, "data": [{"b64_json": "AAAA"}]}'
    ]
    mock_client.send.return_value = mock_response
    mock_client.aclose = AsyncMock()


    mock_mgr.create_proxy_client.return_value = mock_client
    mock_session = MagicMock()
    mock_session.client = mock_client
    mock_session.response = mock_response
    mock_mgr.open_proxy_session = AsyncMock(return_value=mock_session)

    mock_hf = MagicMock()

    app = create_app(
        runtime=runtime_settings,
        app_config=app_cfg,
        manager=mock_mgr,
        hf=mock_hf,
    )

    client = TestClient(app)

    # Test /api/v1/images/generations
    response = client.post(
        "/api/v1/images/generations",
        json={"model": "qwen-image", "prompt": "a beautiful forest at sunrise"},
    )
    assert response.status_code == 200
    assert response.json()["created"] == 12345

    # Test /api/sdapi/v1/txt2img
    response_sdapi = client.post(
        "/api/sdapi/v1/txt2img",
        json={"model": "qwen-image", "prompt": "a cyberpunk city"},
    )
    assert response_sdapi.status_code == 200
