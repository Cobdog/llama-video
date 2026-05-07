# Architecture and Design Decisions

## Overview

llama-video is a two-layer system:

1. **C patch** — surgical modifications to llama.cpp's `tools/mtmd/` and `tools/server/` that teach it a new `VIDEO` input chunk type, per-super-frame temporal M-RoPE positions, and a `mm_processor_kwargs` passthrough on the OpenAI-compat `/v1/chat/completions` endpoint.
2. **Python orchestration** — frame extraction, preprocessing, HTTP service, Gradio WebUI, CLIs. Everything that doesn't need to live in C.

The C layer is intentionally tiny. Most of the complexity is in Python where iteration, testing, and maintenance are cheap.

## Design principles

### 1. Minimal patch surface

Every line added to llama.cpp is a line that must be rebased onto a moving upstream. The patch changes exactly what's needed for Qwen3.5 video decoding (Gemma4 and other stock models do not need this patch):

- Add an `is_video` flag to `clip_image_f32_batch`.
- Route 6-channel super-frames through the existing Conv3D decomposition in `qwen3vl.cpp`.
- Add `MTMD_INPUT_CHUNK_TYPE_VIDEO` + `MTMD_POS_TYPE_VIDEO` + `mtmd_tokenize_video()` + temporal-aware `get_decoder_pos` case.
- Accept `mm_processor_kwargs` in the server's `/v1/chat/completions` handler.

Everything else — frame extraction, pairing, resizing, token budgeting, templates, batching — lives in Python.

### 2. Python does the heavy lifting

ffmpeg, retry logic, timeouts, error types, logging, the FastAPI service, the Gradio WebUI — all Python. The C layer is a dumb pipe that processes correctly-formatted video tensors.

### 3. Model-agnostic where possible

`ModelConfig` centralizes `temporal_patch_size`, `spatial_patch_size`, `merge_size`, pixel bounds, and CLIP normalization constants. Adding a future model family is a new config profile plus (if its encoder differs) a new `tools/mtmd/models/*.cpp` branch. The `ModelAdapter` base class (`src/llama_video/adapters/base.py`) provides a per-family interface for preprocessing, payload construction, response parsing, and token estimation, with concrete implementations for Qwen3.5 and Gemma4.

### 4. Fail loudly

No silent fallback from video → multi-image mode. If the preprocessor can't build super-frames it raises; if the client can't reach `llama-server` it raises. The user asked for video understanding; pretending to provide it is worse than crashing.

---

## Component architecture

### Frame extractor — `src/llama_video/extractor.py`

- Subprocess-based ffmpeg via `asyncio.subprocess.create_subprocess_exec`.
- Default FPS 2.0, configurable via `LLAMA_VIDEO_DEFAULT_FPS` or `ExtractorConfig(fps=...)`.
- Hard frame cap (default 64, `LLAMA_VIDEO_MAX_FRAMES`) to prevent OOM on long videos.
- Output: `list[Frame]` where each `Frame.data` is an `(H, W, 3)` uint8 numpy RGB array.
- `SamplingStrategy` enum declares `UNIFORM`, `KEYFRAME`, `SCENE_CHANGE` — **only `UNIFORM` is currently wired through ffmpeg**; the other values are reserved for future work.

### Preprocessor — `src/llama_video/preprocessor.py`

> **Note:** This is the Qwen3.5-specific pipeline. Other model families use the adapter system (`src/llama_video/adapters/`) which has its own preprocessing logic.

Takes extracted `Frame`s plus a `ModelConfig`, produces a `VideoInput` ready for the vision encoder:

- Resize every frame so the pixel count is within `[min_pixels, max_pixels]` and divisible by `grid_unit = spatial_patch_size × merge_size` (28 for Qwen3.5). Interpolation is BICUBIC (matches HF reference).
- Normalize with CLIP mean/std.
- Pair consecutive frames into 6-channel `(6, H, W)` float32 super-frames. `OddFrameStrategy.PAD` (default) duplicates the last frame; `OddFrameStrategy.DROP` drops it.
- Compute `grid_thw = (T, H/grid_unit, W/grid_unit)`.
- Compute `temporal_positions[i] = round(i × temporal_patch_size / fps)` for `i ∈ [0, T)`.

Output is the `VideoInput` passed to `LlamaServerClient.caption_video()`.

### llama-server client — `src/llama_video/client.py`

`LlamaServerClient`:

- `send_completion(payload) -> CompletionResult(content, reasoning)` — sends a chat completion payload and returns both the content and reasoning fields.
- `stream_completion(payload) -> AsyncGenerator[tuple[str, bool], None]` — streaming variant that yields `(token, is_reasoning)` tuples.
- `health_check() -> bool`

The adapter layer builds the payload (model-specific content, `mm_processor_kwargs` for Qwen3.5, etc.) and calls `send_completion`. The `CompletionResult` captures both `content` and `reasoning_content` from the llama-server response — the latter is used by models like Gemma4 that separate thinking output from the final answer.

- Retry with exponential backoff (`max_retries`, `retry_delay` from `ServerConfig`).
- HTTP timeout from `LLAMA_SERVER_TIMEOUT` (default 600s — thinking mode can be slow).

### HTTP service — `src/llama_video/server.py`

FastAPI app, three endpoints:

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/v1/caption` | Caption a video file by path |
| `GET` | `/v1/health` | Service + `llama-server` reachability |
| `GET` | `/v1/debug/last-request` | Diagnostics on the most recent caption call |

That's all. There's no `/v1/caption/batch` endpoint and no WebSocket streaming — those remain in Python-library territory (`llama_video.batch`) and haven't been exposed over HTTP.

### Configuration — `src/llama_video/config.py`

Four config classes, all Pydantic `BaseSettings` so fields load from env vars:

- `ModelConfig` — vision encoder params (not env-configurable; use `ModelConfig.qwen35()` or instantiate directly)
- `ServerConfig` — `LLAMA_SERVER_*` prefix
- `ExtractorSettings` — `LLAMA_VIDEO_*` prefix
- `ServiceConfig` — `LLAMA_VIDEO_*` prefix (bind host/port/workers/log level)

`Settings` bundles all four under a single root. See the README's env-var table for the full list.

### Model adapters — `src/llama_video/adapters/`

The adapter system provides a per-model-family interface for the full preprocessing-to-response pipeline:

- `ModelAdapter` (abstract base) — defines `preprocess()`, `build_payload()`, `parse_response()`, `estimate_tokens()`, and adapter metadata (name, max duration, etc.).
- `AdapterPreset` — dataclass carrying sampler parameters (temperature, top_p, top_k, min_p, presence_penalty).
- `register_adapter(name, cls)` / `get_adapter(name)` — registry with case-insensitive lookup.
- `detect_adapter(server_url)` — auto-detects model family from llama-server's `/v1/models` endpoint.

**Implemented adapters:**

| Adapter | Preprocessing | Payload | Response parsing | C patch |
|---------|--------------|---------|-----------------|---------|
| Qwen3.5 | Super-frame pairing, grid THW, temporal M-RoPE | `image_url` per frame + `mm_processor_kwargs` | Extract `<think >` tags from content | Required |
| Gemma4 | Frames as individual images with integer timestamps | Standard `image_url` per frame | Content is the caption; reasoning from `CompletionResult.reasoning` | Not needed |

Adding a new model family means subclassing `ModelAdapter`, calling `register_adapter()` at module level, and importing the module in `adapters/__init__.py`.

---

## C patch architecture (current, post-rebase 0adede8)

The patch aligns with upstream's `mtmd_image_tokens_get_decoder_pos()` abstraction rather than introducing a parallel code path. See [`subsystems/c-patch.md`](subsystems/c-patch.md) for the file-by-file walkthrough.

### What changes in `tools/mtmd/`

- **`clip-impl.h` / `clip.cpp`** — `clip_image_f32_batch` gains an `is_video` boolean.
- **`models/qwen3vl.cpp`** — Conv3D decomposition path handles both 3-channel (single frame, weight_t0 only) and 6-channel (super-frame, weight_t0 + weight_t1) inputs.
- **`mtmd.h`** — new enum value `MTMD_INPUT_CHUNK_TYPE_VIDEO`; new accessors `mtmd_image_tokens_get_nt()` and `mtmd_image_tokens_get_temporal_positions()`; new entry point `mtmd_tokenize_video()`.
- **`mtmd.cpp`** —
  - New `MTMD_POS_TYPE_VIDEO` in the `mtmd_pos_type` enum.
  - `struct mtmd_image_tokens` gains `nt` and `temporal_positions`.
  - `n_tokens()` returns `nx × ny × nt` (unchanged for images where `nt=1`).
  - `mtmd_image_tokens_get_decoder_pos()` gains a `VIDEO` case returning `{t = pos_0 + temporal_positions[frame_idx], x, y, 0}` per token.
  - `mtmd_image_tokens_get_n_pos()` gains a `VIDEO` case returning `max({nt, nx, ny})`.
- **`mtmd-helper.cpp`** — `mtmd_helper_decode_image_chunk` widens its M-RoPE branch to accept both `IMAGE` and `VIDEO` chunks (both now flow through the same `set_position_mrope_2d(rel_pos, seq_id)` call; `get_decoder_pos` produces the correct per-token positions for both).

### What changes in `tools/server/`

- **`server-common.cpp` / `.h`** — recognize `mm_processor_kwargs` on `/v1/chat/completions` and pass the relevant fields (`fps`, `is_video`, `grid_thw`, `temporal_positions`) through to the mtmd layer.
- **`server-context.cpp`** — plumb the kwargs through the completions handler.

### What does NOT change

- No new chat-completions content type (no `video_url`, no `video_frames`). Video is signaled by `mm_processor_kwargs.is_video` while frames travel as standard `image_url` entries.
- No new CLI flags on `llama-server`.
- No changes to the GGUF format or mmproj loader.

---

## Data flow

```
Video file (MP4)
    │
    ▼  [ffmpeg @ 2 fps]
Frames: [F0, F1, F2, F3, F4, F5, F6, F7]            (8 frames from a 4s clip)
    │
    ▼  [resize to (H, W) on grid_unit=28; normalize with CLIP mean/std]
Normalized (6, H, W) super-frames:
  [SF0(F0,F1), SF1(F2,F3), SF2(F4,F5), SF3(F6,F7)]   (4 super-frames)
    │
    ▼  [compute grid_thw and temporal_positions]
grid_thw = (4, H/28, W/28),  temporal_positions = [0, 1, 2, 3]
    │
    ▼  [POST /v1/chat/completions]
Payload:
  - content: [image_url × 8, text]   (one image_url per FRAME, not per SF)
  - mm_processor_kwargs: {fps, is_video=true, grid_thw, temporal_positions}
    │
    ▼  [patched server accepts mm_processor_kwargs → mtmd_tokenize_video]
mtmd_image_tokens { nx, ny, nt=4, pos=MTMD_POS_TYPE_VIDEO, temporal_positions }
    │
    ▼  [qwen3vl.cpp Conv3D — frame_a × weight_t0 + frame_b × weight_t1]
    ▼  [get_decoder_pos produces per-token (t, x, y, 0)]
    ▼  [set_position_mrope_2d writes positions into the batch]
    │
Vision embeddings with temporal M-RoPE → LLM → caption
```

---

## Error handling

See [`DEBUGGING.md`](DEBUGGING.md) for the full debugging guide.

### Error categories

1. **Extraction** — ffmpeg not found, corrupt video, unsupported codec → `ExtractionError` subclasses
2. **Preprocessing** — inconsistent frame sizes, resolution out of bounds → `PreprocessingError` subclasses
3. **HTTP** — `llama-server` unreachable, model not loaded, server 5xx → `LlamaVideoError` subclasses (`ServerUnreachableError`, `ServerResponseError`, `ModelNotLoadedError`)
4. **C patch runtime** — `GGML_ASSERT` / `GGML_ABORT` in llama.cpp (these kill the server process)

Every Python exception class carries a `context: dict` for diagnostics. The `/v1/debug/last-request` endpoint exposes the full extraction/preprocessing trace for the most recent request — useful when the client sees a generic 5xx.

### Strategy

- Trust boundaries: validate inputs at the HTTP boundary, trust in-process Python types thereafter.
- No silent fallback. Video request fails loudly rather than degrade to multi-image mode.
- Every log line includes the module name; `LLAMA_VIDEO_LOG_LEVEL=DEBUG` turns on per-stage timing logs.

---

## Future extensibility

### Other model families

The `ModelAdapter` architecture supports any GGUF + mmproj model with multimodal capabilities. The work per family:

1. A new `ModelAdapter` subclass (preprocessing, payload shape, response parsing, token estimation).
2. Register it via `register_adapter()` in the adapters package.
3. If the encoder requires special C support (like Qwen3.5's Conv3D decomposition), a new file under `tools/mtmd/models/` and a corresponding patch hunk. Models using stock llama.cpp (like Gemma4) skip this step entirely.

Currently implemented: Qwen3.5 (patched) and Gemma4 (stock llama.cpp).

### Audio + video

Qwen3.5 also supports audio. The Python preprocessor would need an audio path (feature extraction via ffmpeg) and the C patch would need audio M-RoPE positions analogous to temporal M-RoPE. Out of scope for v1.

### Upstream convergence

When llama.cpp ships native video support (tracking: upstream issue #18389), we plan to:

1. Drop the C patch entirely.
2. Simplify the Python client to use the native API.
3. Keep the public `llama_video` surface unchanged so callers don't notice.
