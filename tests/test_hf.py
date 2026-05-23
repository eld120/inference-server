from pathlib import Path

import pytest

import hf as hf_module
from hf import HuggingFaceService
from schemas import ModelSource


class FakeModelInfo:
    def __init__(self, model_id: str) -> None:
        self.modelId = model_id
        self.likes = 10
        self.downloads = 20
        self.pipeline_tag = "text-generation"
        self.tags = ["gguf"]
        self.author = "author"


class FakeApi:
    def list_models(self, search: str | None, limit: int):
        return [FakeModelInfo(f"{search or 'model'}-{index}") for index in range(limit)]


def test_search_models_uses_hf_api(tmp_path: Path) -> None:
    service = HuggingFaceService(cache_dir=tmp_path, api=FakeApi())

    results = service.search_models("gemma", limit=2)

    assert [item.repo_id for item in results] == ["gemma-0", "gemma-1"]


def test_resolve_source_downloads_repo_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    service = HuggingFaceService(cache_dir=tmp_path, token="token", api=FakeApi())
    calls: list[tuple[object, ...]] = []

    def fake_list_repo_files(
        repo_id: str,
        repo_type: str,
        revision: str,
        token: str | None,
    ) -> list[str]:
        calls.append((repo_id, repo_type, revision, token))
        return ["README.md", "model.gguf"]

    def fake_download(**kwargs: object) -> str:
        calls.append(tuple(sorted(kwargs.items())))
        return str(tmp_path / "model.gguf")

    monkeypatch.setattr(hf_module, "list_repo_files", fake_list_repo_files)
    monkeypatch.setattr(hf_module, "hf_hub_download", fake_download)

    path = service.resolve_source(ModelSource(repo_id="repo/model"))

    assert path == tmp_path / "model.gguf"
    assert calls[0] == ("repo/model", "model", "main", "token")
