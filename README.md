# inference-server

Local LLM gateway and orchestrator for Linux. It fronts Docker-based inference runtimes behind an OpenAI-compatible API and keeps only one workload active at a time.

## Getting Started

1. Install Docker and `uv`.
2. Copy `config.example.json` to `~/.config/inference-server/config.json`, or point `INF_CONFIG_PATH` at a different file.
3. Sync dependencies with `uv sync --no-dev`.
4. Start the service with `uv run --no-dev python main.py`.
5. Load a model with `POST /api/models/{name}/load`.
6. Point your client at `http://localhost:8000/api/v1`.

## Documentation

- [Architecture and runtime behavior](docs/architecture.md)
- [Configuration reference](docs/configuration.md)
- [API reference](docs/api.md)
- [Client integration examples](docs/client-integration.md)
- [Development workflow](docs/development.md)
