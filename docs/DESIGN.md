# Architecture & Design Decisions

## Overview

llama-video is a two-layer system:
1. **C patch layer** — minimal changes to llama.cpp's `clip.cpp`/`mtmd` to accept video frames with temporal encoding
2. **Python orchestration layer** — frame extraction, preprocessing, API service

## Design Principles

### 1. Minimal Patch Surface
The C patch must be as small as possible to minimize rebase friction when upstream llama.cpp updates. We change only what's necessary to make the vision encoder treat input as temporal video frames rather than independent images.

### 2. Python Does the Heavy Lifting
Frame extraction (ffmpeg), frame pairing, FPS management, API serving, error handling, logging — all in Python. The C layer is a dumb pipe that processes correctly-formatted video tensors.

### 3. Model-Agnostic Where Possible
The Python library abstracts model-specific details (temporal_patch_size, merge_size) behind configuration. Adding support for future models (e.g., if InternVL adds video support in GGUF) should only require a new config profile.

### 4. Fail Loudly
No silent fallbacks. If temporal encoding fails, we error — we don't silently fall back to image-only mode. The user asked for video understanding; pretending to provide it is worse than crashing.

## Component Architecture

### Frame Extractor (`extractor.py`)
- Uses ffmpeg via `asyncio.subprocess`
- Configurable FPS (default 2.0, matching Qwen3.5's default)
- Configurable max frames (prevents OOM on long videos)
- Outputs frames as PIL Images or raw bytes
- Handles: MP4, MKV, AVI, MOV, WebM
- Frame selection strategies: uniform, keyframe-biased, scene-change-aware

### Video Preprocessor (`preprocessor.py`)
- Takes extracted frames + model config
- Constructs super-frames (pairs consecutive frames into 6-channel tensors)
- Handles odd frame counts (last frame duplicated or dropped, configurable)
- Computes `video_grid_thw` [T, H, W] tensor
- Computes M-RoPE temporal indices
- Resizes frames to model's expected resolution
- Outputs: preprocessed image data ready for llama-server API

### llama-server Client (`client.py`)
- HTTP client for patched llama-server's `/chat/completions` endpoint
- Sends preprocessed video frames as base64 image_url entries
- Includes video metadata headers (fps, temporal_patch_size, is_video flag)
- Handles streaming responses
- Retry logic with exponential backoff
- Connection pooling

### API Server (`server.py`)
- FastAPI application
- `POST /v1/caption` — accept video file/path + prompt, return caption
- `POST /v1/caption/batch` — batch multiple clips
- `GET /v1/health` — server + llama-server health check
- WebSocket support for streaming captions
- Configurable via environment variables or config file

### Configuration (`config.py`)
- Model profiles: Qwen3.5 family defaults (temporal_patch_size=2, merge_size=2, etc.)
- Server settings: llama-server URL, timeout, retry config
- Extraction settings: default FPS, max frames, resolution limits
- Loaded from: env vars → config file → defaults (priority order)

## C Patch Design

### What Changes in clip.cpp

1. **Super-frame acceptance:** When `is_video=true`, expect 6-channel input tensors instead of 3-channel
2. **Conv3D path selection:** Route video super-frames through the existing Conv3D decomposition (2×Conv2D + sum) instead of the single-image Conv2D path
3. **Grid THW propagation:** Accept `video_grid_thw` with T>1 and propagate to position embedding computation

### What Changes in mtmd.cpp

1. **Video input struct:** New `mtmd_video_input` alongside existing `mtmd_image_input`
2. **Temporal M-RoPE:** When processing video, compute temporal indices from grid THW and pass to the existing M-RoPE function with temporal>1
3. **Token counting:** Video tokens = T × H × W (not just H × W as for images)

### What Changes in server.cpp

1. **Video URL parsing:** Recognize `video_url` content type in chat completions
2. **Frame relay:** Pass video frames through to mtmd with the video flag set
3. **FPS parameter:** Accept `mm_processor_kwargs.fps` to control frame extraction

## Data Flow

```
Video File (MP4)
    │
    ▼ [ffmpeg @ 2fps]
Frames: [F0, F1, F2, F3, F4, F5, F6, F7]  (8 frames from 4s clip)
    │
    ▼ [pair consecutive frames]
Super-frames: [SF0(F0+F1), SF1(F2+F3), SF2(F4+F5), SF3(F6+F7)]  (4 super-frames, 6ch each)
    │
    ▼ [resize to model resolution]
Resized super-frames: same count, model-appropriate dimensions
    │
    ▼ [compute grid_thw]
grid_thw = [4, H/14/2, W/14/2]  (T=4 temporal positions)
    │
    ▼ [POST to llama-server /chat/completions]
    │   Content: super-frames as base64, grid_thw metadata, is_video=true
    │
    ▼ [clip.cpp Conv3D path]
Vision tokens with temporal M-RoPE positions
    │
    ▼ [LLM generates caption]
"A person picks up a red ball and throws it to another person standing near a tree."
```

## Error Handling Strategy

See `docs/DEBUGGING.md` for the full debugging guide.

### Error Categories
1. **Extraction errors** — ffmpeg not found, corrupt video, unsupported codec
2. **Preprocessing errors** — wrong frame dimensions, insufficient frames, memory limits
3. **Patch errors** — tensor shape mismatch, Conv3D failure, M-RoPE computation error
4. **Server errors** — llama-server unreachable, model not loaded, OOM
5. **Integration errors** — API contract mismatch between Python service and client

### Strategy
- Every error category has a specific exception class
- All errors include diagnostic context (tensor shapes, frame counts, model config)
- The C patch logs tensor dimensions at debug level before every operation
- The Python service exposes `/v1/debug/last-request` for post-mortem analysis

## Future Extensibility

### Other Model Families
The architecture supports adding non-Qwen models if they:
1. Use GGUF + mmproj format
2. Have a temporal dimension in their vision encoder
3. Are supported by llama.cpp for image input

### Audio + Video
Qwen3.5 also supports audio. The same architecture extends — the preprocessor would handle audio feature extraction, and the C patch would need audio M-RoPE positions. Out of scope for v1.

### Upstream Convergence
When llama.cpp ships native video support (Issue #18389), we:
1. Drop the C patch entirely
2. Simplify the Python library to use the native API
3. Keep the same `llama_video` interface — callers don't change
