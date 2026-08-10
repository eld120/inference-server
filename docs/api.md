# API Reference

All routes are rooted at `api_prefix`, which defaults to `/api`.

## Health and Status

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Minimal health check. Returns `{"ok": true}`. |
| `GET` | `/api/status` | Compact operator summary showing configuration path, active model/runtime state, and the last service/runtime error. |

---

## Model Management

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/models` | List all configured models with their runtime status. |
| `GET` | `/api/models/{name}` | Return one configured model with its runtime status. |
| `POST` | `/api/models/{name}/load` | Load a model by name. Requires a JSON body choosing the runtime: `{"runtime": "rocm" | "vulkan"}`. The call blocks until the model is ready to serve requests. |
| `POST` | `/api/models/{name}/unload` | Unload the active model and tear down its runtime container. |
| `GET` | `/api/models/{name}/logs` | Fetch recent runtime logs for the model, including the latest persisted failure logs after teardown. |

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

Behavior:
- The orchestrator always starts a fresh runtime container for the activation attempt.
- If another model is currently active, its runtime container is stopped and removed before the new one starts.
- A successful response means the model is ready to accept proxied `/api/v1/*` requests immediately.
- You do not need to call this endpoint before every inference request if the request already names a configured model. The proxy will activate it on demand.
- On explicit `POST /load`, the client chooses the runtime. On implicit proxy auto-load, the server picks a default runtime for that model and now prefers `vulkan` when both `vulkan` and `rocm` are configured.

#### Response Example
```json
{
  "name": "gemma",
  "config": {
    "name": "gemma",
    "source": { ... },
    "mmproj": null,
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
- Older configs may still return `config.runtimes` instead of top-level `source`/`mmproj`/`extra_args`/`speculative`.

### Unloading a Model

`POST /api/models/{name}/unload` stops and removes the active runtime container.

Behavior:
- Returns `200` for any configured model name, even if no model is currently active.
- If a runtime teardown is already underway in Docker, the API still treats the unload as accepted once the container name is observed to release.
- Returns `404` only when `{name}` is not a configured model.

After unload:
- `active_model` becomes `null`.
- `active_runtime` becomes `null`.
- `active_container_id` becomes `null`.
- Requests that omit `model` still return `400` until a model is loaded again.

#### Status Example
```json
{
  "healthy": true,
  "api_prefix": "/api",
  "config_path": "/home/user/.config/inference-server/config.json",
  "active_model": null,
  "active_runtime": null,
  "active_container_id": null,
  "last_error": null
}
```

Use `GET /api/models` or `GET /api/models/{name}` for detailed model configuration and per-model runtime state.

---

## OpenAI-Compatible Proxy

The proxy forwards standard OpenAI-style requests under `/api/v1/*` to the active model runtime.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/models` | List configured model ids. |
| `GET` | `/api/v1/models/{model_name}` | Return one configured model id. |
| `GET, POST` | `/api/v1/{path}` | Forward OpenAI-compatible requests to the loaded runtime. |

Notes:
- The request body `model` value must match a configured model name (e.g. `gemma`).
- If the requested model is configured but not active, the proxy loads it automatically before forwarding the request.
- If no model is active and the request does not name a model, the proxy returns `400`.
- If runtime communication fails while proxying, the gateway clears the active runtime state and returns `503`. A new `load` is required before retrying inference.
- SSE streaming responses are passed through unchanged.
