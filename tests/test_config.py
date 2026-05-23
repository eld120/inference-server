from pathlib import Path

from config import load_app_config, save_app_config
from schemas import AppConfig, BackendConfig, ModelSource


def test_app_config_round_trip(tmp_path: Path) -> None:
    config = AppConfig(
        default_backend="primary",
        hf_cache_dir=tmp_path / "hf-cache",
        backends=[
            BackendConfig(
                name="primary",
                model=ModelSource(local_path=tmp_path / "model.gguf"),
            )
        ],
    )

    path = tmp_path / "config.json"
    save_app_config(path, config)

    loaded = load_app_config(path)

    assert loaded.default_backend == "primary"
    assert loaded.backends[0].name == "primary"
    assert loaded.backends[0].model.local_path == tmp_path / "model.gguf"
