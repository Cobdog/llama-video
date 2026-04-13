# llama-video

Temporal video captioning for llama.cpp — frame extraction, super-frame preprocessing, and M-RoPE temporal encoding for Qwen3.5 GGUF models.

> **Why a patch?** Sending multiple images to llama.cpp gives zero temporal understanding. The model needs 6-channel super-frames (Conv3D) and temporal M-RoPE positions to reason about motion, sequence, and change. This patch adds that.

## Supported Models

All Qwen3.5 vision models (same vision encoder across all sizes):

| Model | Active Params | Total Params | Notes |
|-------|--------------|--------------|-------|
| Qwen3.5-0.8B | 0.8B | 0.8B | Smallest, fast iteration |
| Qwen3.5-3B | 3B | 3B | Good balance |
| Qwen3.5-35B-A3B | 3B | 35B | MoE — recommended starting point |
| Qwen3.5-122B-A10B | 10B | 122B | MoE — best quality/VRAM ratio |
| Qwen3.5-397B-A17B | 17B | 397B | Largest MoE |

You need two files per model: a **GGUF model** and a **mmproj** (vision projector).

## Requirements

| Dependency | Version | Notes |
|-----------|---------|-------|
| Python | 3.11+ | |
| ffmpeg | any recent | Frame extraction. Must be on PATH. |
| CMake | 3.21+ | Building llama.cpp |
| CUDA toolkit | 11.7+ | For GPU inference. CPU works but is very slow. |
| git | any recent | Patch application |

**Windows additional:** Visual Studio 2019+ with C++ workload (for MSVC compiler), or MinGW-w64.

## Setup

### Step 1: Install llama-video

```bash
pip install llama-video

# With Gradio WebUI:
pip install "llama-video[ui]"
```

Or from source:

```bash
git clone https://github.com/Cobdog/llama-video.git
cd llama-video
pip install ".[ui]"
```

### Step 2: Clone and patch llama.cpp

The patch targets llama.cpp commit `a8bad38`. If master has moved forward and the patch fails, pin to this commit.

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git checkout a8bad38
cd ..
llama-video-patch ./llama.cpp
```

Or manually:

```bash
cd llama.cpp
git checkout a8bad38
git apply /path/to/llama-video/patches/video-support-20260413.patch
```

### Step 3: Build llama.cpp

#### Interactive build (recommended)

If you installed from source, the included build script prompts for GPU backend, build type, and parallel job count:

```bash
./scripts/build.sh ./llama.cpp
```

If you installed via pip, grab the script directly:

```bash
curl -O https://raw.githubusercontent.com/Cobdog/llama-video/main/scripts/build.sh
chmod +x build.sh
./build.sh ./llama.cpp
```

It supports CUDA, HIP (AMD), Vulkan, Metal (macOS), and CPU-only. It will auto-detect `nvcc` in common locations (`/opt/cuda/bin`, `/usr/local/cuda/bin`) if it's not already on your PATH.

#### Manual build — Linux

```bash
cd llama.cpp

# CUDA (recommended)
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j$(nproc)

# CPU only (slow, but works)
cmake -B build
cmake --build build --config Release -j$(nproc)
```

#### Manual build — Windows (MSVC)

Open **x64 Native Tools Command Prompt** (from Visual Studio):

```cmd
cd llama.cpp

:: CUDA
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release

:: CPU only
cmake -B build
cmake --build build --config Release
```

The server binary will be at `build\bin\Release\llama-server.exe`.

#### Manual build — Windows (MinGW)

```bash
cd llama.cpp
cmake -B build -G "MinGW Makefiles" -DGGML_CUDA=ON
cmake --build build -j%NUMBER_OF_PROCESSORS%
```

> **Verify GPU support:** After building, check that `GGML_CUDA:BOOL=ON` (or your chosen backend) appears in `llama.cpp/build/CMakeCache.txt`. If it says `OFF`, `nvcc` wasn't found during configuration — see [Windows-Specific Notes](#windows-specific-notes).

### Step 4: Download a model

Download both the GGUF model and mmproj from HuggingFace. Example for Qwen3.5-35B-A3B:

```bash
# From https://huggingface.co/Qwen/Qwen3.5-35B-A3B-GGUF (or a community quant)
# You need:
#   - The model GGUF (e.g., qwen3.5-35b-a3b-q4_k_m.gguf)
#   - The mmproj GGUF (e.g., mmproj-Qwen3.5-35B-A3B-F16.gguf)
```

### Step 5: Start llama-server

#### Linux

```bash
./llama.cpp/build/bin/llama-server \
    -m /path/to/model.gguf \
    --mmproj /path/to/mmproj.gguf \
    --host 0.0.0.0 --port 8080 \
    --ctx-size 65536 --jinja
```

#### Windows

```cmd
llama.cpp\build\bin\Release\llama-server.exe ^
    -m C:\models\model.gguf ^
    --mmproj C:\models\mmproj.gguf ^
    --host 0.0.0.0 --port 8080 ^
    --ctx-size 65536 --jinja
```

**Important flags:**
- `--ctx-size 65536` — context window size. Higher = more frames but more VRAM. 65K is a good starting point.
- `--jinja` — required for Qwen3.5's chat template.
- `--port 8080` — default port that llama-video expects. Change via `LLAMA_SERVER_URL` env var.

Wait for the server to print `llama server listening` before proceeding.

### Step 6: Caption a video

**Python API:**

```python
import asyncio
from llama_video import Extractor, Preprocessor, Settings, get_preset
from llama_video.client import LlamaServerClient

async def caption(video_path: str) -> str:
    settings = Settings()
    extractor = Extractor(settings.extractor)
    preprocessor = Preprocessor(settings.model)
    client = LlamaServerClient(settings.server)

    frames = await extractor.extract_frames_async(video_path)
    video_input = preprocessor.process(frames, fps=2.0)

    result = await client.caption_video(
        video_input,
        prompt="Describe what happens in this video.",
        preset=get_preset("default"),
    )
    await client.close()
    return result

print(asyncio.run(caption("my_video.mp4")))
```

**Gradio WebUI** (requires `pip install "llama-video[ui]"`):

```bash
llama-video-ui
# Opens at http://localhost:7860
```

**FastAPI service:**

```bash
llama-video-server
# API at http://localhost:9000

curl -X POST http://localhost:9000/v1/caption \
    -H "Content-Type: application/json" \
    -d '{"video_path": "/path/to/video.mp4"}'
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_SERVER_URL` | `http://localhost:8080` | llama-server base URL |
| `LLAMA_SERVER_MODEL_NAME` | *(empty)* | Model name for router mode (optional) |
| `LLAMA_SERVER_TIMEOUT` | `600` | Request timeout in seconds |
| `LLAMA_VIDEO_DEFAULT_FPS` | `2.0` | Frame extraction rate |
| `LLAMA_VIDEO_MAX_FRAMES` | `64` | Max frames to extract per video |
| `LLAMA_VIDEO_FFMPEG_PATH` | `ffmpeg` | Path to ffmpeg binary |

### Inference Presets

Based on official Qwen team recommendations:

| Preset | Temperature | Top P | Top K | Presence Penalty | Use Case |
|--------|------------|-------|-------|-----------------|----------|
| `default` | 1.0 | 0.95 | 20 | 1.5 | General captioning (thinking mode) |
| `precise` | 0.6 | 0.95 | 20 | 0.0 | Precise/factual descriptions |

### Prompt Templates

Built-in templates with `{variable}` substitution:

| Template | Mode | Description |
|----------|------|-------------|
| `general` | both | Basic description |
| `detailed` | both | Characters, setting, actions, atmosphere |
| `motion` | video | Movement and action focus |
| `composition` | both | Framing, camera, lighting, color |
| `character` | both | Character focus (accepts `{character_name}`) |
| `narrative` | both | Screenplay-style narration |

## Caveats and Limitations

### Patch Compatibility

The patch was built against llama.cpp `master` as of 2026-04-13 (commit `a8bad38`). It modifies files in `tools/mtmd/` (clip-impl.h, clip.cpp, clip.h, models/qwen3vl.cpp, mtmd.cpp, mtmd.h, mtmd-helper.cpp) and `tools/server/` (server-common.cpp, server-common.h, server-context.cpp). If upstream has refactored these files, the patch may not apply cleanly.

**If the patch fails to apply:**

```bash
cd llama.cpp
git log --oneline -1    # Note your current commit
git apply --check /path/to/patches/video-support-20260413.patch  # Dry run to see conflicts
```

You may need to apply manually or wait for an updated patch.

### Context Window vs. VRAM

More frames and higher resolution = more vision tokens = more VRAM. A rough guide:

| Setting | Vision Tokens | Notes |
|---------|--------------|-------|
| 4 frames, 280x280 | ~200 | Minimal, fast |
| 8 frames, 560x560 | ~3,200 | Good balance |
| 16 frames, 1120x1120 | ~50,000 | High quality, needs 65K+ context |
| 64 frames, 1920x1080 | ~200,000+ | Will exceed most context windows |

The WebUI shows a live token budget bar so you can tune before running inference.

### Known Limitations

- **One llama-server at a time.** Each server instance loads the full model into VRAM. The run script guards against accidental duplicates.
- **ffmpeg must be on PATH.** The library calls ffmpeg as a subprocess. On Windows, either add it to PATH or set `LLAMA_VIDEO_FFMPEG_PATH` to the full path.
- **Thinking mode is verbose.** Qwen3.5 with `default` preset uses thinking mode, which produces internal reasoning before the final caption. This is normal — the library extracts the final answer automatically. Set timeout accordingly (120s+ for longer videos).
- **No audio processing.** Only visual frames are extracted. Audio tracks are ignored.
- **Super-frame pairing.** Frames are paired sequentially (frame 0+1, 2+3, ...). An odd number of frames duplicates the last frame to form a complete pair. Extracting an even number of frames avoids this.

### Windows-Specific Notes

- Use forward slashes or raw strings for paths in Python: `r"C:\videos\clip.mp4"` or `"C:/videos/clip.mp4"`.
- If using PowerShell, environment variables are set with `$env:LLAMA_SERVER_URL = "http://localhost:8080"`.
- The `llama-video-patch` command uses `git apply` — make sure `git` is on your PATH.
- CUDA builds require the CUDA toolkit and MSVC (not MinGW) for best compatibility.

## Project Structure

```
llama-video/
├── src/llama_video/       # Python library
│   ├── extractor.py       # ffmpeg frame extraction
│   ├── preprocessor.py    # Super-frame construction + grid THW
│   ├── client.py          # llama-server HTTP client
│   ├── server.py          # FastAPI captioning service
│   ├── webui.py           # Gradio experimentation UI
│   ├── config.py          # Settings and presets
│   ├── templates.py       # Prompt templates
│   ├── tokens.py          # Token budget estimation
│   ├── history.py         # SQLite caption history
│   ├── batch.py           # Batch captioning
│   ├── image.py           # Single-image captioning
│   ├── patch_cli.py       # llama-video-patch CLI
│   ├── debug_cli.py       # llama-video-debug CLI
│   ├── errors.py          # Exception hierarchy
│   └── types.py           # Core data types (Frame, SuperFrame, etc.)
├── patches/               # C patches for llama.cpp
├── scripts/               # Setup, build, and run scripts
└── tests/                 # Unit, integration, and smoke tests
```

## License

MIT
