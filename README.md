# inference-server

Local LLM orchestration for a Fedora host.

## Runtime

- Proxy OpenAI-compatible traffic under `/api/v1`
- Keep health and management routes under `/api`
- Manage `llama-server` as a subprocess so models can be loaded and unloaded on demand
- Use the standard Hugging Face cache for downloads and cache inspection

## Configuration

Copy `config.example.json` to your local config path and point the service at it with `--config` or `INF_CONFIG_PATH`.

The config controls:

- the default backend to load on startup
- Hugging Face cache location and token
- backend model source, `llama-server` binary path, and speculative decoding mode

## `llama.cpp`

Build `ggml-org/llama.cpp` from `master` with Vulkan enabled, then point each backend entry at that `llama-server` binary.

## Tooling

Use `uv` for everything in this repo.

```bash
uv run pytest
uv run ruff check .
uv run ruff format .
uv run ty check
```

## Entry point

```bash
uv run inference-server
```
