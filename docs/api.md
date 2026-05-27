# API Reference

All routes are rooted at `api_prefix`, which defaults to `/api`.

## Health and Status

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Minimal health check. |
| `GET` | `/api/status` | Rich operator summary with active model and per-model state. |

## Model Management

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/models` | List all configured model presets with runtime status. |
| `GET` | `/api/models/{name}` | Return one configured model preset with runtime status. |
| `POST` | `/api/models/{name}/load` | Load a preset by name. |
| `POST` | `/api/models/{name}/unload` | Unload the active preset by name. |
| `GET` | `/api/models/{name}/logs` | Fetch recent container logs for the preset. |

## OpenAI-Compatible Proxy

The proxy forwards standard OpenAI-style requests under `/api/v1/*` to the active model runtime.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/models` | List configured preset ids. |
| `GET` | `/api/v1/models/{model_name}` | Return one configured preset id. |
| `GET, POST` | `/api/v1/{path}` | Forward OpenAI-compatible requests to the loaded runtime. |

Notes:

- The request body `model` value must match the loaded preset key.
- If no model is loaded, the proxy returns `400`.
- SSE streaming responses are passed through unchanged.
