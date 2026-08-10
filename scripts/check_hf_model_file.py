from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe Hugging Face model repos for candidate GGUF filenames."
    )
    parser.add_argument("repo_id", help="Model repo, for example unsloth/Qwen3.6-35B-A3B-MTP-GGUF")
    parser.add_argument(
        "--revision",
        default="main",
        help="Repo revision to inspect. Defaults to main.",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Candidate filename to test. Repeat for multiple candidates.",
    )
    parser.add_argument(
        "--contains",
        action="append",
        default=[],
        help="Substring filter for listing repo files. Repeat for multiple filters.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Attempt hf_hub_download for each candidate.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face cache dir override.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Validate every Hugging Face-backed file reference in a config JSON file.",
    )
    return parser.parse_args()


def _files_for_repo(
    api: HfApi, cache: dict[tuple[str, str], list[str]], repo_id: str, revision: str
) -> list[str]:
    key = (repo_id, revision)
    if key not in cache:
        cache[key] = sorted(api.list_repo_files(repo_id, revision=revision, repo_type="model"))
    return cache[key]


def _iter_config_refs(config_data: dict[str, Any]) -> list[dict[str, str]]:
    refs: list[dict[str, str]] = []

    def add_source(
        owner: str, repo_id: str | None, filename: str | None, revision: str | None
    ) -> None:
        if repo_id is None or filename is None:
            return
        refs.append(
            {
                "owner": owner,
                "repo_id": repo_id,
                "filename": filename,
                "revision": revision or "main",
            }
        )

    def add_mmproj(
        owner: str,
        source: dict[str, Any],
        mmproj: dict[str, Any] | None,
        extra_args: list[str],
    ) -> None:
        if mmproj:
            add_source(
                f"{owner} [mmproj]",
                mmproj.get("repo_id"),
                mmproj.get("filename"),
                mmproj.get("revision", "main"),
            )
            return

        repo_id = source.get("repo_id")
        revision = source.get("revision", "main")
        for idx, arg in enumerate(extra_args):
            mmproj_filename = None
            if arg == "--mmproj" and idx + 1 < len(extra_args):
                mmproj_filename = extra_args[idx + 1]
            elif arg.startswith("--mmproj="):
                mmproj_filename = arg.split("=", 1)[1]
            if mmproj_filename:
                add_source(f"{owner} [mmproj]", repo_id, mmproj_filename, revision)

    for model in config_data.get("models", []):
        model_name = model["name"]
        runtimes = model.get("runtimes", {})
        if runtimes:
            for runtime_name, runtime_cfg in runtimes.items():
                owner = f"{model_name}.{runtime_name}"
                source = runtime_cfg.get("source", {})
                add_source(
                    owner,
                    source.get("repo_id"),
                    source.get("filename"),
                    source.get("revision", "main"),
                )
                add_mmproj(
                    owner,
                    source,
                    runtime_cfg.get("mmproj"),
                    runtime_cfg.get("extra_args", []),
                )
                draft = runtime_cfg.get("speculative", {}).get("draft_model")
                if draft:
                    add_source(
                        f"{owner} [draft]",
                        draft.get("repo_id"),
                        draft.get("filename"),
                        draft.get("revision", "main"),
                    )
        else:
            owner = model_name
            source = model.get("source", {})
            add_source(
                owner,
                source.get("repo_id"),
                source.get("filename"),
                source.get("revision", "main"),
            )
            add_mmproj(
                owner,
                source,
                model.get("mmproj"),
                model.get("extra_args", []),
            )
            draft = model.get("speculative", {}).get("draft_model")
            if draft:
                add_source(
                    f"{owner} [draft]",
                    draft.get("repo_id"),
                    draft.get("filename"),
                    draft.get("revision", "main"),
                )

    return refs


def validate_config(api: HfApi, config_path: Path) -> int:
    config_data = json.loads(config_path.read_text())
    refs = _iter_config_refs(config_data)
    cache: dict[tuple[str, str], list[str]] = {}
    failures: list[dict[str, str]] = []

    print(f"config={config_path}")
    print(f"references={len(refs)}")

    for ref in refs:
        files = _files_for_repo(api, cache, ref["repo_id"], ref["revision"])
        if ref["filename"] not in set(files):
            failures.append(ref)

    print(f"repos_checked={len(cache)}")
    print(f"missing={len(failures)}")
    if failures:
        print("\nmissing references:")
        for ref in failures:
            print(
                f"  {ref['owner']}: {ref['repo_id']}/{ref['filename']}@{ref['revision']}"
            )
        return 1

    print("\nall Hugging Face-backed references exist")
    return 0


def main() -> int:
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    if args.config is not None:
        return validate_config(api, Path(args.config))

    files = sorted(api.list_repo_files(args.repo_id, revision=args.revision, repo_type="model"))
    print(f"repo={args.repo_id} revision={args.revision}")
    print(f"files={len(files)}")

    filtered = files
    for needle in args.contains:
        filtered = [path for path in filtered if needle in path]

    if args.contains:
        print("\nmatching files:")
        if filtered:
            for path in filtered:
                print(f"  {path}")
        else:
            print("  <none>")

    if args.candidate:
        print("\ncandidate checks:")
        file_set = set(files)
        for candidate in args.candidate:
            exists = candidate in file_set
            print(f"  {candidate}: {'present' if exists else 'missing'}")

            if not args.download:
                continue

            try:
                local_path = hf_hub_download(
                    repo_id=args.repo_id,
                    filename=candidate,
                    revision=args.revision,
                    token=token,
                    cache_dir=args.cache_dir,
                    repo_type="model",
                )
                print(f"    download: ok -> {Path(local_path)}")
            except Exception as exc:
                print(f"    download: error -> {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
