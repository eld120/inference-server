# inference-server

Local LLM gateway and orchestrator for Linux. It fronts Docker-based inference runtimes behind an OpenAI-compatible API, keeps only one workload active at a time, and starts a fresh runtime container for each model activation.

## Getting Started

1. Install Docker and `uv`.
2. Copy `config.example.json` to `~/.config/inference-server/config.json`, or point `INF_CONFIG_PATH` at a different file.
3. Sync dependencies with `uv sync --no-dev`.
4. Start the service with `uv run --no-dev python main.py`.
5. Point your client at `http://localhost:8000/api/v1`.

Proxy requests that include a configured `model` name will automatically load that model if it is not already active. `POST /api/models/{name}/load` is still available if you want to pre-warm a specific runtime explicitly.

`POST /api/models/{name}/unload` is idempotent for configured model names. It returns `200` whether the runtime was active, already stopped, or still finishing Docker teardown.

`GET /api/status` returns a compact operator summary only. Use `GET /api/models` or `GET /api/models/{name}` for detailed model configuration and per-model runtime state.

Runtime/container log snapshots are persisted by default under `~/.local/state/inference-server/logs`.

## Documentation

- [Architecture and runtime behavior](docs/architecture.md)
- [Configuration reference](docs/configuration.md)
- [API reference](docs/api.md)
- [Client integration examples](docs/client-integration.md)
- [Development workflow](docs/development.md)
