# API Server Subsystem

> **Code:** `src/llama_video/server.py`, `src/llama_video/client.py`
> **Tests:** `tests/unit/test_client.py`, `tests/integration/test_server.py`
> **Last verified:** 2026-04-24 (against `src/` at HEAD)

## Purpose

FastAPI service that accepts video file paths and returns captions. Wraps the full pipeline — extraction → preprocessing → `llama-server` inference — behind a small HTTP surface.

## Endpoints

Three endpoints. No batch endpoint and no WebSocket — batching is exposed as a Python-library function (`llama_video.batch.batch_caption`).

### `POST /v1/caption`

Caption one video. Backed by `caption_video` in `server.py`.

**Request body** (`CaptionRequest` in `types.py`; only `video_path` is required):

```json
{
  "video_path": "/path/to/clip.mp4",
  "prompt": "Describe what happens in this video.",
  "fps": 2.0,
  "max_frames": 64,
  "model_profile": "qwen3.5",
  "max_tokens": 2048,
  "preset": "default",
  "temperature": null
}
```

- `fps`: `> 0, ≤ 30`. Extraction FPS.
- `max_frames`: `> 0, ≤ 512`. Hard cap on extracted frames.
- `preset`: name of an `InferencePreset` (`default` or `precise`).
- `temperature`: `null` uses the preset's value; pass a float to override.

**Response** (`CaptionResponse`):

```json
{
  "caption": "A person walks across a park and sits on a bench.",
  "metadata": {
    "frames_extracted": 8,
    "super_frames": 4,
    "grid_thw": [4, 16, 16],
    "processing_time_ms": 2340.5,
    "model_tokens_used": null
  }
}
```

**Errors**:

| Status | Condition | Source exception |
|--------|-----------|------------------|
| `422` | Extraction or preprocessing failed | `ExtractionError`, `PreprocessingError` |
| `502` | `llama-server` unreachable / errored | `LlamaVideoError` subclass |

### `GET /v1/health`

```json
{
  "status": "ok",
  "llama_server_reachable": true,
  "llama_server_url": "http://localhost:8080",
  "version": "0.1.3"
}
```

`status` is `"ok"` when `llama-server` answers a health check, `"degraded"` otherwise.

### `GET /v1/debug/last-request`

Returns a `DebugInfo` for the most recent caption request. Fields are all optional and populated opportunistically:

```json
{
  "request": { "...": "original CaptionRequest dict" },
  "frames_extracted": 8,
  "super_frame_shapes": [[6, 448, 448], [6, 448, 448]],
  "grid_thw": [4, 16, 16],
  "temporal_positions": [0, 1, 2, 3],
  "llama_server_request": null,
  "llama_server_response": null,
  "timing": {
    "timing_extract_ms": 120.3,
    "timing_preprocess_ms": 45.2,
    "timing_inference_ms": 2174.9
  },
  "error": null
}
```

If the last request failed, `error` holds the stringified exception and the other fields may be partial.

---

## llama-server client — `LlamaServerClient`

Low-level async HTTP client. Lives in `llama_video.client`; not re-exported from the top-level package (callers typically use the service, not the client directly).

### Constructor

```python
LlamaServerClient(config: ServerConfig | None = None)
```

Falls back to `ServerConfig()` (which reads `LLAMA_SERVER_*` env vars) if no config is passed.

### Methods

| Method | Signature | Notes |
|--------|-----------|-------|
| `caption_video` | `(video_input: VideoInput, prompt: str, max_tokens: int = 2048, temperature: float \| None = None, preset: InferencePreset \| None = None, cache_prompt: bool = True) -> str` | Sends video + `mm_processor_kwargs` |
| `caption_image` | `(image_path: str, prompt: str, max_tokens: int = 2048, temperature: float \| None = None, preset: InferencePreset \| None = None, cache_prompt: bool = True) -> str` | Takes a **path**, not an `Image` object; no `mm_processor_kwargs` |
| `health_check` | `() -> bool` | `GET /health` on the llama-server |
| `close` | `() -> None` | Close the underlying `httpx.AsyncClient`; always call in a `finally` / `async with` |

Retry behavior: `max_retries` attempts with exponential backoff (`retry_delay × 2^attempt`), configured on `ServerConfig`.

### Request shape to `llama-server`

Video payload (abridged — the actual payload also includes all sampler params from the preset):

```json
{
  "messages": [{
    "role": "user",
    "content": [
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
      "... one image_url per source FRAME (each super-frame decomposes into 2)",
      {"type": "text", "text": "Describe what happens in this video."}
    ]
  }],
  "max_tokens": 2048,
  "temperature": 1.0,
  "top_p": 0.95,
  "top_k": 20,
  "min_p": 0.0,
  "presence_penalty": 1.5,
  "cache_prompt": true,
  "mm_processor_kwargs": {
    "fps": 2.0,
    "is_video": true,
    "grid_thw": [4, 16, 16],
    "temporal_positions": [0, 1, 2, 3]
  }
}
```

Key facts:

- **Standard OpenAI content type** (`image_url`). There is no `video_url` or `video_frames` content type — the C patch does not add one.
- **`mm_processor_kwargs` is the video-mode signal.** Without `is_video: true`, the patched server treats the request as multi-image and does not apply temporal M-RoPE.
- **One `image_url` per source frame**, not per super-frame — super-frames are decomposed back to their two JPEGs on the client side so the server receives frames in their natural count.

Image payload (no `mm_processor_kwargs`):

```json
{
  "messages": [{
    "role": "user",
    "content": [
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
      {"type": "text", "text": "Describe this image."}
    ]
  }],
  "max_tokens": 2048,
  "cache_prompt": true
}
```

### Streaming and reasoning-mode output

`send_completion` awaits the full response and returns a `CompletionResult(content, reasoning)`. The `reasoning` field captures the model's thinking output when the server separates it (e.g., Gemma4 via `reasoning_content` in the JSON response). For models where thinking is embedded inline (e.g., Qwen3.5's thinking tags), `adapter.parse_response()` extracts it from the content text.

---

## Configuration

Environment variables that affect this subsystem:

| Variable | Default | Config class | Purpose |
|----------|---------|--------------|---------|
| `LLAMA_SERVER_URL` | `http://localhost:8080` | `ServerConfig` | Upstream `llama-server` URL |
| `LLAMA_SERVER_MODEL_NAME` | *(empty)* | `ServerConfig` | Optional `model` field (router mode) |
| `LLAMA_SERVER_TIMEOUT` | `600` | `ServerConfig` | HTTP request timeout (seconds) |
| `LLAMA_SERVER_MAX_RETRIES` | `3` | `ServerConfig` | Retry attempts |
| `LLAMA_SERVER_RETRY_DELAY` | `1.0` | `ServerConfig` | Base retry delay (seconds) |
| `LLAMA_VIDEO_HOST` | `0.0.0.0` | `ServiceConfig` | Bind host for the Python service |
| `LLAMA_VIDEO_PORT` | `9000` | `ServiceConfig` | Bind port for the Python service |
| `LLAMA_VIDEO_WORKERS` | `1` | `ServiceConfig` | Uvicorn workers |
| `LLAMA_VIDEO_LOG_LEVEL` | `INFO` | `ServiceConfig` | Python logging level |

The service does **not** have its own HTTP timeout knob — per-request timeout is the `LLAMA_SERVER_TIMEOUT` that the client uses to call upstream.
