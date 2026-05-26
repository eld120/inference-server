from __future__ import annotations

from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download, list_repo_files, scan_cache_dir

from schemas import (
    HFCachedFile,
    HFDownloadResponse,
    HFRepoFile,
    HFSearchResult,
    ModelSource,
)


class HuggingFaceService:
    def __init__(
        self,
        cache_dir: Path,
        token: str | None = None,
        api: Any = None,
    ):
        self._cache_dir = cache_dir
        self._token = token
        self._api = api or HfApi(token=self._token)

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    def search_models(self, query: str | None, limit: int = 20) -> list[HFSearchResult]:
        models = self._api.list_models(search=query, limit=limit)
        return [
            HFSearchResult(
                repo_id=model.modelId,
                likes=model.likes,
                downloads=model.downloads,
                pipeline_tag=model.pipeline_tag,
                tags=list(model.tags or []),
                author=model.author,
            )
            for model in models
        ]

    def repo_files(self, repo_id: str, revision: str = "main") -> list[HFRepoFile]:
        files = list_repo_files(
            repo_id,
            repo_type="model",
            revision=revision,
            token=self._token,
        )
        return [
            HFRepoFile(repo_id=repo_id, revision=revision, path=file_path)
            for file_path in files
        ]

    def cache_files(self) -> list[HFCachedFile]:
        if not self._cache_dir.exists():
            return []
        cached = scan_cache_dir(self._cache_dir)
        files: list[HFCachedFile] = []
        for repo in cached.repos:
            for file_info in repo.revisions:
                # Gather all refs pointing to this commit hash
                refs = []
                for ref_name, rev_obj in repo.refs.items():
                    val = getattr(rev_obj, "commit_hash", rev_obj)
                    if val == file_info.commit_hash:
                        refs.append(ref_name)

                for sibling in file_info.files:
                    files.append(
                        HFCachedFile(
                            repo_id=repo.repo_id,
                            revision=file_info.commit_hash,
                            refs=refs,
                            filename=sibling.file_name,
                            local_path=str(sibling.file_path),
                            size_on_disk=sibling.size_on_disk,
                        )
                    )
        return files

    def resolve_source(self, source: ModelSource) -> Path:
        if source.local_path is not None:
            return source.local_path
        if source.repo_id is None:
            msg = "model source requires either local_path or repo_id"
            raise ValueError(msg)

        filename = source.filename
        if filename is None:
            files = self.repo_files(source.repo_id, source.revision)
            gguf_files = [item.path for item in files if item.path.endswith(".gguf")]
            if not gguf_files:
                msg = f"no GGUF file found in {source.repo_id}@{source.revision}"
                raise ValueError(msg)
            filename = gguf_files[0]

        local_path = hf_hub_download(
            repo_id=source.repo_id,
            filename=filename,
            revision=source.revision,
            token=self._token,
            cache_dir=self._cache_dir,
            repo_type="model",
        )
        return Path(local_path)

    def download(self, source: ModelSource) -> HFDownloadResponse:
        if source.local_path is not None:
            return HFDownloadResponse(
                repo_id="local",
                filename=source.local_path.name,
                revision="local",
                local_path=str(source.local_path),
            )
        if source.repo_id is None:
            msg = "model source requires either local_path or repo_id"
            raise ValueError(msg)
        path = self.resolve_source(source)
        filename = source.filename or path.name
        return HFDownloadResponse(
            repo_id=source.repo_id,
            filename=filename,
            revision=source.revision,
            local_path=str(path),
        )
