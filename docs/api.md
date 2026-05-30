# API Reference

All routes are rooted at `api_prefix`, which defaults to `/api`.

## Health and Status

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Minimal health check. Returns `{"ok": true}`. |
| `GET` | `/api/status` | Rich operator summary showing the configuration path, active model/runtime state, and per-model resources. |

---

## Model Management

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/models` | List all configured models with their runtime status. |
| `GET` | `/api/models/{name}` | Return one configured model with its runtime status. |
| `POST` | `/api/models/{name}/load` | Load a model by name. Requires a JSON body choosing the runtime: `{"runtime": "rocm" | "vulkan"}`. |
| `POST` | `/api/models/{name}/unload` | Unload the active model. |
| `GET` | `/api/models/{name}/logs` | Fetch recent container logs for the model's active runtime. |

### Loading a Model

`POST /api/models/{name}/load` expects a JSON body specifying the target runtime. In the preferred config shape, models are backend-agnostic and the runtime selects the shared backend definition:

```json
{
  "runtime": "rocm"
}
```

Or:

```json
{
  "runtime": "vulkan"
}
```

#### Response Example
```json
{
  "name": "gemma",
  "config": {
    "name": "gemma",
    "source": { ... },
    "extra_args": [],
    "speculative": { "type": "none" }
  },
  "status": {
    "name": "gemma",
    "state": "running",
    "active": true,
    "active_runtime": "rocm",
    "host": "127.0.0.1",
    "port": 39281,
    "container_id": "...",
    "base_url": "http://127.0.0.1:39281"
  }
}
```

Legacy note:
- Older configs may still return `config.runtimes` instead of top-level `source`/`extra_args`/`speculative`.

---

## OpenAI-Compatible Proxy

The proxy forwards standard OpenAI-style requests under `/api/v1/*` to the active model runtime.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/models` | List configured model ids. |
| `GET` | `/api/v1/models/{model_name}` | Return one configured model id. |
| `GET, POST` | `/api/v1/{path}` | Forward OpenAI-compatible requests to the loaded runtime. |

Notes:
- The request body `model` value must match the loaded model name (e.g. `gemma`).
- If no model is loaded, the proxy returns `400`.
- SSE streaming responses are passed through unchanged.
