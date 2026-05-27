# Configuration

Configuration is split between environment variables and the JSON config file.

## Environment Variables

Create a `.env` file in the project root if you want to override runtime defaults.

| Variable | Description |
|---|---|
| `INF_CONFIG_PATH` | Path to `config.json`. Default: `~/.config/inference-server/config.json` |
| `INF_HOST` | Gateway bind address. Default: `0.0.0.0` |
| `INF_PORT` | Gateway bind port. Default: `8000` |
| `INF_BACKEND_PORT` | Host port mapped to the active model container. Default: `39281` |
| `INF_HF_TOKEN` | Hugging Face token for gated or private models. Also reads `HF_TOKEN` and `HUGGINGFACE_HUB_TOKEN`. |

## Config File

The server reads JSON from `config.json`. See [config.example.json](../config.example.json) for a complete sample.

Top-level fields:

| Field | Type | Default | Notes |
|---|---|---|---|
| `runtime_mode` | `"auto"` \| `"router"` \| `"container"` | `"auto"` | Selects the runtime strategy. |
| `api_prefix` | `string` | `"/api"` | Prefix for all server routes. |
| `hf_cache_dir` | `string` | `~/.cache/huggingface/hub` | Local Hugging Face cache directory. |
| `hf_token` | `string \| null` | `null` | Used when no env token is present. |
| `backend_families` | `object` | `{}` | Backend family definitions. |
| `models` | `array` | `[]` | Preset definitions. |

## Backend Families

Each backend family defines the runtime image and shared container settings.

| Field | Type | Notes |
|---|---|---|
| `docker_image` | `string` | Container image used for this family. |
| `devices` | `string[]` | Device nodes passed through to the container. |
| `volumes` | `object` | Additional host-to-container mounts. |
| `shared_args` | `string[]` | CLI arguments applied to every model in the family. |
| `router_supported` | `boolean` | Whether router mode can be used. |

## Model Presets

Each preset must have a unique `name` and reference a known backend family.

| Field | Type | Notes |
|---|---|---|
| `name` | `string` | Canonical preset key used by clients. |
| `backend_family` | `string` | References `backend_families`. |
| `bind_host` | `string` | Container bind host. Default: `127.0.0.1`. |
| `connect_host` | `string` | Host used by the proxy to reach the model runtime. Default: `127.0.0.1`. |
| `model` | `object` | Requires either `local_path` or `repo_id` + `filename`. |
| `speculative` | `object` | Optional speculative decoding config. |
| `extra_args` | `string[]` | Preset-specific CLI flags. |

### Model Source

Use one of these forms:

- `local_path`
- `repo_id` + `filename`

`revision` defaults to `main`.

### Speculative Decoding

Supported speculative types include:

- `none`
- `draft`
- `draft-simple`
- `draft-mtp`
- `ngram-cache`
- `ngram-simple`
- `ngram-map-k`
- `ngram-map-k4v`
- `ngram-mod`

Rules:

- `draft` and `draft-simple` require a `draft_model`.
- Other types reject `draft_model`.
- `draft-mtp` is self-contained and does not need a separate draft model.

## Example

Use [config.example.json](../config.example.json) as the starting point for a real config file.
