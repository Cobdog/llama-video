# llama-video

Temporal video captioning for llama.cpp — frame extraction, super-frame preprocessing, and M-RoPE temporal encoding for Qwen3.5 GGUF models.

> **Why a patch?** Sending multiple images to llama.cpp gives zero temporal understanding. The model needs 6-channel super-frames (Conv3D) and temporal M-RoPE positions to reason about motion, sequence, and change. This patch adds that.

---

## Contents

- [At a glance](#at-a-glance)
- [Supported models](#supported-models)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Full setup](#full-setup)
  - [1. Install llama-video](#1-install-llama-video)
  - [2. Clone and patch llama.cpp](#2-clone-and-patch-llamacpp)
  - [3. Build llama.cpp](#3-build-llamacpp)
  - [4. Download a model](#4-download-a-model)
  - [5. Start llama-server](#5-start-llama-server)
- [Using llama-video](#using-llama-video)
  - [Python library](#python-library)
  - [HTTP service](#http-service)
  - [Gradio WebUI](#gradio-webui)
  - [Debug CLI](#debug-cli)
- [Configuration](#configuration)
  - [Environment variables](#environment-variables)
  - [Inference presets](#inference-presets)
  - [Prompt templates](#prompt-templates)
- [Caveats and limitations](#caveats-and-limitations)
- [Troubleshooting](#troubleshooting)
- [Project layout](#project-layout)
- [Further reading](#further-reading)
- [License](#license)

---

## At a glance

```
┌────────────┐   ffmpeg    ┌──────────────┐   6-channel   ┌──────────────────┐
│ video file │ ──────────▶ │ super-frames │ ───────────▶ │ patched          │
│  (.mp4)    │  2 fps      │ + M-RoPE t   │  image_url    │ llama-server     │
└────────────┘             └──────────────┘  + metadata   │ (Qwen3.5 GGUF)   │
                                                          └──────────────────┘
                                                                  │
                                                                  ▼
                                                           caption text
```

Three moving parts:

1. A **C patch** against llama.cpp's `tools/mtmd/` and `tools/server/` that adds a `VIDEO` chunk type, per-super-frame temporal M-RoPE positions, and a `mm_processor_kwargs` passthrough on `/v1/chat/completions`.
2. A **Python library** that extracts frames with ffmpeg, pairs them into 6-channel super-frames, computes grid THW + temporal indices, and POSTs to the patched `llama-server`.
3. **User surfaces** on top of that library: a FastAPI HTTP service, a Gradio WebUI, and a debug CLI.

## Supported models

All Qwen3.5 vision models share the same vision encoder (`temporal_patch_size=2`, `spatial_patch_size=14`, `merge_size=2`), so one patch covers every size:

| Model | Active / total params | Notes |
|-------|----------------------|-------|
| Qwen3.5-0.8B | 0.8B | Smallest, fast iteration |
| Qwen3.5-3B | 3B | Good balance |
| Qwen3.5-35B-A3B | 3B / 35B | MoE — recommended starting point |
| Qwen3.5-122B-A10B | 10B / 122B | MoE — best quality/VRAM ratio |
| Qwen3.5-397B-A17B | 17B / 397B | Largest MoE |

Each model requires **two files**: the GGUF model and the mmproj (vision projector) GGUF.

## Requirements

| Dependency | Version | Purpose |
|------------|---------|---------|
| Python | 3.11+ | The library and services |
| ffmpeg | any recent | Frame extraction (must be on PATH) |
| CMake | 3.21+ | Building patched llama.cpp |
| git | any recent | Applying the patch |
| CUDA toolkit | 11.7+ | *(optional)* GPU inference; CPU works but is very slow |

**Windows:** Visual Studio 2019+ with the C++ workload (for MSVC), or MinGW-w64. See the [Full setup](#full-setup) section for MSVC-specific build commands.

## Quick start

For users who already have a patched llama.cpp build and a running `llama-server`:

```bash
pip install "llama-video[ui]"
llama-video-ui   # → http://localhost:7860
```

Point the WebUI at a video, pick a preset, get a caption. Everything else in this README is for getting to that point from scratch.

---

## Full setup

### 1. Install llama-video

From PyPI:

```bash
pip install llama-video              # library + HTTP service + CLIs
pip install "llama-video[ui]"        # also pulls in Gradio for the WebUI
```

From source (needed if you want the build scripts):

```bash
git clone https://github.com/Cobdog/llama-video.git
cd llama-video
pip install ".[ui]"
# or, with uv (dev setup):
uv sync --dev
```

### 2. Clone and patch llama.cpp

The patch is pinned to a specific upstream commit. If master has moved forward and the patch fails, pin to this commit to reproduce a known-good state:

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git checkout 0adede8
cd ..
llama-video-patch ./llama.cpp
```

Or apply the patch manually:

```bash
cd llama.cpp
git checkout 0adede8
git apply /path/to/llama-video/patches/video-support-20260424.patch
```

> The `llama-video-patch` CLI is a thin wrapper around `git apply`. Pass it the directory of your llama.cpp checkout; it discovers the bundled patch file and applies every `.patch` in `patches/`.

### 3. Build llama.cpp

#### Interactive (recommended — source install only)

```bash
./scripts/build.sh ./llama.cpp
```

Prompts for GPU backend (CUDA / HIP / Vulkan / Metal / CPU), build type, and parallel jobs. Auto-detects `nvcc` in `/opt/cuda/bin`, `/usr/local/cuda/bin`, and `/usr/local/cuda-*/bin` if it isn't on PATH.

#### Manual — Linux / macOS

```bash
cd llama.cpp

# CUDA (NVIDIA)
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j$(nproc)

# HIP (AMD)
cmake -B build -DGGML_HIP=ON -DGPU_TARGETS=gfx1030   # or your arch
cmake --build build --config Release -j$(nproc)

# Vulkan (cross-platform)
cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release -j$(nproc)

# Metal (macOS — default)
cmake -B build
cmake --build build --config Release -j$(sysctl -n hw.ncpu)

# CPU only (slow)
cmake -B build
cmake --build build --config Release -j$(nproc)
```

#### Manual — Windows (MSVC)

Open the **x64 Native Tools Command Prompt** (installed with Visual Studio):

```cmd
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release
```

The server binary lands at `build\bin\Release\llama-server.exe`.

#### Manual — Windows (MinGW)

```bash
cd llama.cpp
cmake -B build -G "MinGW Makefiles" -DGGML_CUDA=ON
cmake --build build -j%NUMBER_OF_PROCESSORS%
```

> **Verify your GPU backend was picked up.** After `cmake -B build`, look in `build/CMakeCache.txt` for the backend flag (`GGML_CUDA:BOOL=ON`, etc.). If it's `OFF` or missing, the toolchain wasn't found — fix that before building, not after.

### 4. Download a model

Grab the GGUF model **and** its mmproj from HuggingFace. Example for Qwen3.5-35B-A3B:

- `qwen3.5-35b-a3b-q4_k_m.gguf` — quantized weights
- `mmproj-Qwen3.5-35B-A3B-F16.gguf` — vision projector

Any community quant works; the mmproj must match the model family.

### 5. Start llama-server

#### Linux / macOS

```bash
./llama.cpp/build/bin/llama-server \
    -m /path/to/model.gguf \
    --mmproj /path/to/mmproj.gguf \
    --host 0.0.0.0 --port 8080 \
    --ctx-size 65536 \
    --jinja
```

#### Windows

```cmd
llama.cpp\build\bin\Release\llama-server.exe ^
    -m C:\models\model.gguf ^
    --mmproj C:\models\mmproj.gguf ^
    --host 0.0.0.0 --port 8080 ^
    --ctx-size 65536 ^
    --jinja
```

Required and notable flags:

- `--mmproj` — the vision projector GGUF. Without it, the server has no vision encoder.
- `--jinja` — required for Qwen3.5's chat template.
- `--ctx-size` — context window. More frames × higher resolution = more vision tokens = needs a bigger window. 65536 is a sensible default.
- `--port 8080` — the default port llama-video expects. Override via `LLAMA_SERVER_URL`.

Wait for `llama server listening` before proceeding.

> **One server at a time.** Each `llama-server` instance loads the full model into VRAM (35GB+ for the flagship quant). The bundled `scripts/run-server.sh` checks for a running process and refuses to start a duplicate; do the same in your own wrapper scripts.

---

## Using llama-video

### Python library

Minimal end-to-end example (requires a running `llama-server` from step 5):

```python
import asyncio
from llama_video import Extractor, Preprocessor, Settings, get_preset
from llama_video.client import LlamaServerClient

async def caption(video_path: str) -> str:
    settings = Settings()
    extractor = Extractor(settings.extractor)
    preprocessor = Preprocessor(settings.model)
    client = LlamaServerClient(settings.server)
    try:
        frames = await extractor.extract_frames_async(video_path)
        video_input = preprocessor.process(frames, fps=2.0)
        return await client.caption_video(
            video_input,
            prompt="Describe what happens in this video.",
            preset=get_preset("default"),
        )
    finally:
        await client.close()

print(asyncio.run(caption("my_video.mp4")))
```

Re-exported from the top-level package:

- Config: `Settings`, `ModelConfig`, `ServerConfig`, `PRESETS`, `InferencePreset`, `get_preset`
- Pipeline: `Extractor`, `ExtractorConfig`, `Preprocessor`, `VideoInput`
- Templates: `PromptTemplate`, `BUILT_IN_TEMPLATES`, `get_template`, `get_templates`, `render_template`
- Tokens: `TokenBudget`, `TokenEstimator`
- History: `CaptionHistory`
- Types: `Frame`, `SuperFrame`, `CaptionResult`
- Images (single-image mode): `load_image`, `build_image_message`
- Batching: `batch_caption`, `detect_mode`, `validate_batch_mode`

`LlamaServerClient` (HTTP client) lives in `llama_video.client`.

### HTTP service

Start it:

```bash
llama-video-server
# Service at http://0.0.0.0:9000
```

Three endpoints:

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/v1/caption` | Caption a video file on disk |
| `GET` | `/v1/health` | Service + llama-server reachability |
| `GET` | `/v1/debug/last-request` | Diagnostics on the most recent caption call |

Request body (`POST /v1/caption`):

```json
{
  "video_path": "/path/to/video.mp4",
  "prompt": "Describe what happens in this video.",
  "fps": 2.0,
  "max_frames": 64,
  "model_profile": "qwen3.5",
  "max_tokens": 2048,
  "preset": "default",
  "temperature": null
}
```

Only `video_path` is required; every other field has the default shown above.

Response:

```json
{
  "caption": "A person walks across a park and sits on a bench.",
  "metadata": {
    "frames_extracted": 8,
    "super_frames": 4,
    "grid_thw": [4, 16, 16],
    "processing_time_ms": 2340,
    "model_tokens_used": null
  }
}
```

Example call:

```bash
curl -X POST http://localhost:9000/v1/caption \
  -H "Content-Type: application/json" \
  -d '{"video_path": "/path/to/video.mp4"}'
```

### Gradio WebUI

```bash
llama-video-ui
# Opens at http://localhost:7860
```

The WebUI shows a live token-budget bar as you change FPS / max-frames / resolution, so you can keep vision tokens under your context window before running inference. Requires the `[ui]` extra (`pip install "llama-video[ui]"`).

### Debug CLI

`llama-video-debug` has three subcommands:

```bash
# Extract and inspect frames (optionally dump to disk)
llama-video-debug extract <video> [--fps N] [--max-frames N] [--output-dir DIR]

# Preprocess a video and print grid THW + temporal positions
llama-video-debug preprocess <video> [--fps N] [--max-frames N] [--model PROFILE]

# Verify the patch is active on a running llama-server
llama-video-debug validate-patch [--server-url http://localhost:8080]
```

---

## Configuration

### Environment variables

Every `Settings` field can be overridden with an env var. Prefixes:
- `LLAMA_SERVER_` → `llama-server` connection (`ServerConfig`)
- `LLAMA_VIDEO_` → everything else (extractor + Python service)

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_SERVER_URL` | `http://localhost:8080` | Patched `llama-server` base URL |
| `LLAMA_SERVER_MODEL_NAME` | *(empty)* | Optional — sent as `model` field in router mode |
| `LLAMA_SERVER_TIMEOUT` | `600` | HTTP request timeout (seconds) |
| `LLAMA_SERVER_MAX_RETRIES` | `3` | Retry attempts on transient errors |
| `LLAMA_SERVER_RETRY_DELAY` | `1.0` | Base retry delay (seconds, exponential) |
| `LLAMA_VIDEO_FFMPEG_PATH` | `ffmpeg` | Path to ffmpeg binary |
| `LLAMA_VIDEO_DEFAULT_FPS` | `2.0` | Default extraction FPS |
| `LLAMA_VIDEO_MAX_FRAMES` | `64` | Default frame cap per video |
| `LLAMA_VIDEO_EXTRACTION_TIMEOUT` | `60.0` | ffmpeg subprocess timeout (seconds) |
| `LLAMA_VIDEO_HOST` | `0.0.0.0` | `llama-video-server` bind host |
| `LLAMA_VIDEO_PORT` | `9000` | `llama-video-server` bind port |
| `LLAMA_VIDEO_WORKERS` | `1` | Uvicorn workers |
| `LLAMA_VIDEO_LOG_LEVEL` | `INFO` | Python logging level |
| `LLAMA_VIDEO_DEBUG` | `false` | Toggle verbose debug behavior |

### Inference presets

Two built-in presets (based on official Qwen team recommendations):

| Preset | Temperature | Top P | Top K | Min P | Presence penalty | Use case |
|--------|-------------|-------|-------|-------|------------------|----------|
| `default` | 1.0 | 0.95 | 20 | 0.0 | 1.5 | General captioning (thinking mode) |
| `precise` | 0.6 | 0.95 | 20 | 0.0 | 0.0 | Factual / reproducible descriptions |

Accessed via `from llama_video import get_preset; get_preset("default")` and passed to `LlamaServerClient.caption_video(..., preset=...)` or to `POST /v1/caption`'s `preset` field.

### Prompt templates

Built-in templates with `{variable}` substitution (see `llama_video.templates`):

| Template | Mode | Template text (abbreviated) |
|----------|------|-----------------------------|
| `general` | image + video | "Describe what happens in this {media_type}." |
| `detailed` | image + video | Characters, setting, actions, atmosphere |
| `motion` | video only | Movement and action focus |
| `composition` | image + video | Framing, camera, lighting, color |
| `character` | image + video | Character focus (accepts `{character_name}`) |
| `narrative` | image + video | Screenplay-style narration |

Access via `from llama_video import get_template, render_template`.

---

## Caveats and limitations

### Patch compatibility

The patch was built against llama.cpp `master` as of 2026-04-24 (commit `0adede8`). It modifies:

- `tools/mtmd/` — `clip-impl.h`, `clip.cpp`, `clip.h`, `models/qwen3vl.cpp`, `mtmd.cpp`, `mtmd.h`, `mtmd-helper.cpp`
- `tools/server/` — `server-common.cpp`, `server-common.h`, `server-context.cpp`

If upstream has refactored these files, the patch may not apply cleanly.

**If the patch fails to apply:**

```bash
cd llama.cpp
git log --oneline -1          # note your current commit
git apply --3way /path/to/patches/video-support-20260424.patch
# 3-way merge auto-resolves hunks where context shifted;
# only real collisions are left as conflict markers.
```

See `docs/subsystems/c-patch.md` for the rebase recipe.

### Context window vs. VRAM

More frames × higher resolution = more vision tokens = more VRAM. Rough guide:

| Settings | Vision tokens (approx) | Notes |
|----------|------------------------|-------|
| 4 frames, 280×280 | ~200 | Minimal, fast |
| 8 frames, 560×560 | ~3,200 | Good balance |
| 16 frames, 1120×1120 | ~50,000 | High quality, needs 65K+ context |
| 64 frames, 1920×1080 | ~200,000+ | Will exceed most context windows |

The WebUI has a live token-budget bar so you can tune before running inference.

### Known limitations

- **One `llama-server` at a time.** Each instance loads the full model into VRAM. `scripts/run-server.sh` guards against accidental duplicates.
- **ffmpeg must be on PATH** — or set `LLAMA_VIDEO_FFMPEG_PATH` to the binary's full path.
- **Only uniform sampling is implemented.** `SamplingStrategy` declares `KEYFRAME` and `SCENE_CHANGE`, but only `UNIFORM` is actually wired through ffmpeg.
- **Thinking mode is verbose.** With the `default` preset Qwen3.5 emits internal reasoning before the final caption; the library extracts the answer automatically but the round-trip can take 30-120s for longer videos. Set timeouts accordingly.
- **No audio processing.** Only visual frames are extracted; audio tracks are ignored.
- **Super-frame pairing.** Frames are paired sequentially (`0+1`, `2+3`, …). Odd counts default to duplicating the last frame (`odd_strategy=PAD`); pass `OddFrameStrategy.DROP` to drop it instead. Extracting an even number of frames avoids the question.

### Windows notes

- Use forward slashes or raw strings in Python paths: `r"C:\videos\clip.mp4"` or `"C:/videos/clip.mp4"`.
- PowerShell env var: `$env:LLAMA_SERVER_URL = "http://localhost:8080"`.
- `llama-video-patch` shells out to `git apply` — make sure `git` is on PATH.
- CUDA builds want MSVC, not MinGW, for best compatibility.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `ffmpeg: command not found` | Not installed or not on PATH | Install ffmpeg; or set `LLAMA_VIDEO_FFMPEG_PATH` |
| `Extracted 0 frames` | FPS × clip length rounded to 0 | Lower FPS, or use a longer clip |
| `llama-server has no model loaded` (503) | Server running but `--mmproj` missing or model path wrong | Re-check the server command line |
| Patch apply fails | Upstream drifted from `0adede8` | `git checkout 0adede8` in llama.cpp, or retry with `git apply --3way` |
| Caption arrives but describes isolated frames ("I see several images…") | Patch not active, or model treated input as multi-image | Run `llama-video-debug validate-patch` to confirm; rebuild llama.cpp |
| Timeout on long videos | Default HTTP timeout too low | Raise `LLAMA_SERVER_TIMEOUT` |
| `GGML_CUDA:BOOL=OFF` in CMakeCache | `nvcc` not discovered at cmake time | Put CUDA bin on PATH and re-run `cmake -B build -DGGML_CUDA=ON` |

For layer-by-layer debugging across frame extraction → preprocessing → C patch → HTTP, see [`docs/DEBUGGING.md`](docs/DEBUGGING.md).

---

## Project layout

```
llama-video/
├── src/llama_video/         # Python library
│   ├── __init__.py          # Top-level re-exports
│   ├── config.py            # Settings + InferencePreset + ModelConfig
│   ├── extractor.py         # ffmpeg frame extraction
│   ├── preprocessor.py      # Super-frame construction, grid THW, M-RoPE positions
│   ├── types.py             # Frame, SuperFrame, CaptionRequest/Response, etc.
│   ├── errors.py            # Exception hierarchy
│   ├── client.py            # LlamaServerClient (OpenAI-compatible HTTP)
│   ├── server.py            # FastAPI captioning service
│   ├── webui.py             # Gradio experimentation UI
│   ├── templates.py         # Built-in prompt templates
│   ├── tokens.py            # TokenBudget / TokenEstimator
│   ├── history.py           # SQLite caption history
│   ├── batch.py             # Batch captioning
│   ├── image.py             # Single-image captioning helpers
│   ├── patch_cli.py         # llama-video-patch entry point
│   └── debug_cli.py         # llama-video-debug entry point
├── patches/                 # C patches for llama.cpp (one unified file)
├── scripts/                 # setup.sh, build.sh, apply-patches.sh, extract-patches.sh, run-server.sh
├── tests/                   # unit / integration / smoke
│   ├── unit/                # Fast, no external deps
│   ├── integration/         # Need a running llama-server or real video files
│   └── smoke/               # End-to-end against a real model
├── docs/                    # Design, debugging, subsystem and reference docs
└── testvid/                 # Studio Ghibli clips with reference captions
```

## Further reading

- [`docs/DESIGN.md`](docs/DESIGN.md) — architecture and design decisions.
- [`docs/DEBUGGING.md`](docs/DEBUGGING.md) — layer-by-layer debugging guide.
- [`docs/subsystems/c-patch.md`](docs/subsystems/c-patch.md) — what the C patch does and how to rebase it.
- [`docs/subsystems/frame-extraction.md`](docs/subsystems/frame-extraction.md) — Extractor API and ffmpeg details.
- [`docs/subsystems/preprocessing.md`](docs/subsystems/preprocessing.md) — super-frame construction, grid THW, M-RoPE positions.
- [`docs/subsystems/api-server.md`](docs/subsystems/api-server.md) — HTTP service and llama-server client.
- [`docs/references/qwen35-vision-architecture.md`](docs/references/qwen35-vision-architecture.md) — Qwen3.5 vision encoder details.
- [`docs/references/llamacpp-multimodal-internals.md`](docs/references/llamacpp-multimodal-internals.md) — `libmtmd` + `clip.cpp` internals.
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — what's planned next.

## License

MIT. See [LICENSE](LICENSE).
