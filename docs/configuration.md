# Configuration

Configuration is split between environment variables and the JSON config file.

## Environment Variables

Create a `.env` file in the project root if you want to override runtime defaults.

| Variable | Description |
|---|---|
| `INF_CONFIG_PATH` | Path to `config.json`. Default: `~/.config/inference-server/config.json` |
| `INF_HOST` | Gateway bind address. Default: `0.0.0.0` |
| `INF_PORT` | Gateway bind port. Default: `8000` |
| `INF_MODEL_LOAD_TIMEOUT_SECONDS` | Maximum time to wait for a model to become inference-ready after `/models/load`. Default: `1800` |
| `INF_MODEL_READINESS_PROBE_TIMEOUT_SECONDS` | Timeout for each readiness probe request sent during model load. Default: `5.0` |
| `INF_RUNTIME_PORT` | Host port mapped to the active runtime container. Default: `39281` (deprecated alias: `INF_BACKEND_PORT`) |
| `INF_HF_TOKEN` | Hugging Face token for gated or private models. Also reads `HF_TOKEN` and `HUGGINGFACE_HUB_TOKEN`. |

## Config File

The server reads JSON from `config.json`. See [config.example.json](../config.example.json) for a complete sample.

Top-level fields:

| Field | Type | Default | Notes |
|---|---|---|---|
| `backends` | `object` | `{}` | Shared backend definitions keyed by `"rocm"` and/or `"vulkan"`. |
| `api_prefix` | `string` | `"/api"` | Prefix for all server routes. |
| `hf_cache_dir` | `string` | `~/.cache/huggingface/hub` | Local Hugging Face cache directory. |
| `hf_token` | `string \| null` | `null` | (Discouraged) Fallback if no env token is present. Using environment variables like `HF_TOKEN` or `INF_HF_TOKEN` is strongly preferred for secret hygiene. |
| `models` | `array` | `[]` | Configured models. |

---

## Configured Models

Each model must have a unique `name`. In the preferred format, models define only model-specific settings and the selected runtime (`"rocm"` or `"vulkan"`) maps through shared top-level backend definitions.

| Field | Type | Notes |
|---|---|---|
| `name` | `string` | Canonical model name/key used by clients (e.g. `gemma`, `qwen`). |
| `source` | `object` | Model source info (see below). |
| `extra_args` | `string[]` | Model-specific llama.cpp arguments. |
| `speculative` | `object` | Optional speculative decoding configuration. |

Legacy compatibility:
- `runtimes` is still accepted, but it is deprecated. It duplicates backend image and device configuration per model.

---

## Backend Configurations

Each backend block specifies the shared container/runtime details for a hardware backend.

| Field | Type | Default | Notes |
|---|---|---|---|
| `docker_image` | `string` | Required | Container image to run (e.g. `inference-server-llama-rocm:7.2.1-7e50ef7`). |
| `devices` | `string[]` | `[]` | Device nodes passed through to the container. |
| `volumes` | `object` | `{}` | Host-to-container mount paths. |
| `shared_args` | `string[]` | `[]` | Global llama.cpp options. |
| `bind_host` | `string` | `"127.0.0.1"` | Bind host for the runtime. |
| `connect_host` | `string` | `"127.0.0.1"` | Target host the gateway uses to forward proxy requests. |

### Model Source

Specify either a local filepath or Hugging Face Hub attributes:

- `local_path` (string): Absolute or relative path to a local GGUF model file. Relative paths are resolved relative to the directory containing the loaded `config.json` file, and are preserved as relative paths when saved. Tilde `~` paths expand to the user's home directory.
- `repo_id` (string): The Hugging Face repo name (e.g., `ggml-org/gemma-3-1b-it-GGUF`).
- `filename` (string): The GGUF filename inside the repo (e.g., `gemma-3-1b-it-Q4_K_M.gguf`).
- `revision` (string): The repo branch or revision (defaults to `"main"`).

> [!NOTE]
> If using `repo_id`, both `repo_id` and `filename` must be supplied.

### Speculative Decoding

Configured via the `speculative` field on the model:

| Field | Type | Default | Notes |
|---|---|---|---|
| `type` | `string` | `"none"` | One of the speculative types listed below. |
| `draft_model` | `object \| null` | `null` | Optional draft Model Source (same structure as `source`). |

Supported speculative types:
- `none`
- `draft` (Requires `draft_model`, typically used for separate assistant models like Gemma 4 Assistant)
- `draft-simple` (Requires `draft_model`)
- `draft-mtp` (Self-contained Multi-Token Prediction, e.g., for Qwen 3.6 MTP models)
- `ngram-cache`
- `ngram-simple`
- `ngram-map-k`
- `ngram-map-k4v`
- `ngram-mod`

### Multimodal / Vision Models

When running multimodal models that require a vision projection adapter (like Qwen 3.6 VL), configure the main model as the `source` and add the `--mmproj` flag to `extra_args`:

* Format: `["--mmproj", "mmproj-BF16.gguf"]` (or target projector filename).
* The orchestrator will automatically resolve and download the projector file from Hugging Face (assuming it resides in the same repository as the base model), and map its location into the container path correctly.

### Disk I/O & Memory Optimizations (HDD / VRAM Balance)

* **Prompt Caching**: `llama-server` manages prompt caching entirely in memory (RAM/VRAM) within active slots. No disk writes are performed for prompt caching.
* **HDD Bottlenecking (`--no-mmap`)**: If running on a slow HDD, add `--no-mmap` to `shared_args`. This forces llama.cpp to load the entire model sequentially into memory at startup, preventing slow on-demand page faults during token generation.
* **KV Cache Quantization**: Reduce KV memory footprints using `--cache-type-k q8_0 --cache-type-v q8_0` (or `q4_0` for extreme savings) along with `-fa on` (Flash Attention).
* **VRAM Limits vs Context Size**: For a 32GB VRAM GPU running 27B parameter models:
  * **Q4 model quants**: Use context size `-c 96000` (fits with Q8 KV Cache).
  * **Q5 model quants**: Use context size `-c 64000` (fits with Q8 KV Cache).
  * **Q6 model quants**: Use context size `-c 48000` (fits with Q8 KV Cache).

---

## Example

Refer to [config.example.json](../config.example.json) as the base template. It assumes you build the backend images from [`docker/`](../docker) first, then reference those local image tags from the single shared `backends` section.
