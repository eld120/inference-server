# Development

This project uses `uv` for dependency management and task execution.

## Runtime

```bash
uv sync --no-dev
uv run --no-dev python main.py
```

## Development

```bash
uv sync
uv run pytest
uv run ruff check .
uv run ruff format .
uv run ty check
```

## Backend Images

The runtime containers are built from vendored Dockerfiles in [`docker/`](../docker).
Both recipes fetch `llama.cpp` from the pinned upstream commit
`ebc10770ac5a9331824c53ef0c6adad780904dc3` during the Docker build.

Build the ROCm server image:

```bash
docker build \
  -f docker/llama-rocm.Dockerfile \
  --target server \
  -t inference-server-llama-rocm:7.2.1-7e50ef7 \
  .
```

Build the Vulkan server image:

```bash
docker build \
  -f docker/llama-vulkan.Dockerfile \
  --target server \
  -t inference-server-llama-vulkan:26.04-ebc1077 \
  .
```

Those image tags are the intended backend images for this repo. Build them first, then use the same tags in the single shared `backends` section of [config.example.json](../config.example.json).

## Notes

- The `dev` dependency group contains the test and lint tooling.
- The project is intentionally marked as a virtual project, so uv does not build or install it as a package.
- The application is run from source with `uv run --no-dev python main.py`.
