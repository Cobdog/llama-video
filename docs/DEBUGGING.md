# Debugging Guide

Layer-by-layer diagnosis for the llama-video pipeline. Work from the outside in — most failures are in the outermost layers (ffmpeg, preprocessing) rather than the C patch.

## Quick checklist

```
□ Is ffmpeg installed and on PATH?          → ffmpeg -version
□ Is llama-server running?                  → curl http://localhost:8080/health
□ Is llama-server loaded with mmproj?       → check server startup logs for "mmproj loaded"
□ Is the patch applied?                     → llama-video-debug validate-patch --server-url http://localhost:8080
□ Are frames extracting correctly?          → llama-video-debug extract <video> --output-dir /tmp/f/
□ Are super-frames shaped correctly?        → llama-video-debug preprocess <video>
□ Is the model responding to images?        → test caption_image() first, then caption_video()
□ Are temporal positions non-trivial?       → debug output should show grid_thw with T > 1
```

---

## Layer 1 — Frame extraction (`extractor.py`)

### Symptoms
- `FFmpegNotFoundError`
- `NoFramesError` (0 frames extracted)
- Garbage pixel data; wrong resolution

### Diagnostics

```bash
# Sanity-check ffmpeg is reachable and the clip is decodable:
ffmpeg -i test_video.mp4 -vf "fps=2" -q:v 2 /tmp/test_frames/frame_%04d.jpg

# Extract via the debug CLI (writes frames to disk if --output-dir is given):
llama-video-debug extract test_video.mp4 --fps 2 --output-dir /tmp/debug_frames/

# From Python — note: Extractor() takes ExtractorSettings; per-call params go to ExtractorConfig:
python -c "
import asyncio
from llama_video import Extractor, ExtractorConfig
async def main():
    e = Extractor()  # reads LLAMA_VIDEO_* env vars for defaults
    frames = await e.extract_frames_async('test_video.mp4', ExtractorConfig(fps=2.0))
    print(f'Extracted {len(frames)} frames')
    for f in frames[:4]:
        print(f'  Frame {f.index}: {f.size} at t={f.timestamp:.2f}s, data={f.data.shape} {f.data.dtype}')
asyncio.run(main())
"
```

`Frame` attributes: `data` (numpy `(H, W, 3)` uint8 RGB), `index`, `timestamp`, `width`, `height`, `size`.

### Common issues

| Issue | Cause | Fix |
|-------|-------|-----|
| `FFmpegNotFoundError` | ffmpeg not on PATH | Install ffmpeg, or set `LLAMA_VIDEO_FFMPEG_PATH` |
| `NoFramesError` (0 frames) | FPS × clip-length rounded to 0 | Lower FPS, or pass `min_frames=1` in `ExtractorConfig` |
| `ExtractionTimeoutError` | Large video, slow disk, or hung ffmpeg | Raise `LLAMA_VIDEO_EXTRACTION_TIMEOUT`; reduce `max_frames` |
| `VideoDecodeError` | Corrupt / unsupported codec | Re-encode with `ffmpeg -i in.mov -c:v libx264 out.mp4` |

---

## Layer 2 — Preprocessing (`preprocessor.py`)

### Symptoms
- Super-frame has 3 channels instead of 6
- `grid_thw` looks wrong (e.g., `T=1` when you expect more)
- `temporal_positions` all zero

### Diagnostics

```bash
llama-video-debug preprocess test_video.mp4 --fps 2

# Prints the count of super-frames, grid_thw, temporal_positions,
# and the resolved pixel resolution.
```

```python
# Manual inspection
from llama_video import Extractor, ExtractorConfig, Preprocessor, ModelConfig
import asyncio

async def main():
    frames = await Extractor().extract_frames_async("test_video.mp4", ExtractorConfig(fps=2.0))
    result = Preprocessor(ModelConfig.qwen35()).process(frames, fps=2.0)

    print(f"Super-frames: {len(result.super_frames)}")
    print(f"Grid THW:     {result.grid_thw}")
    print(f"Temporal pos: {result.temporal_positions}")
    print(f"SF[0] shape:  {result.super_frames[0].shape}")  # expect (6, H, W) float32

asyncio.run(main())
```

### Expected temporal positions

`Preprocessor.compute_temporal_positions(grid_thw, fps)` returns
`[round(i * temporal_patch_size / fps) for i in range(T)]`.

With defaults (`temporal_patch_size=2`, `fps=2.0`, T=4):

```
seconds_per_temporal = 2 / 2.0 = 1.0
positions = [round(0*1), round(1*1), round(2*1), round(3*1)] = [0, 1, 2, 3]
```

If positions are all zero and T > 1 you have a bug — file an issue with the output of `llama-video-debug preprocess`.

### Common issues

| Issue | Cause | Fix |
|-------|-------|-----|
| SF has 3 channels not 6 | Pairing step bypassed | Verify `preprocessor.process()` was called; inspect `result.super_frames[0].shape` |
| `grid_thw` has `T=1` | Too few frames extracted | Upstream in Layer 1: check `Extractor` actually returned ≥ 2 frames |
| `temporal_positions` all zero | fps or `temporal_patch_size` mismatch | Check `Preprocessor(ModelConfig.qwen35()).process(frames, fps=...)` received a non-zero `fps` |
| `InvalidFrameDimensionsError` | Mixed resolutions (shouldn't happen from ffmpeg) | All frames must share dimensions; re-extract |
| `ResolutionError` | Resolved size outside `[min_pixels, max_pixels]` | Adjust `ModelConfig.max_pixels`, or downscale via `resolution_scale` arg |

### Cross-check against HuggingFace

If you suspect a preprocessing discrepancy, compare our output with the HF reference processor (`tests/integration/test_hf_reference.py` does this end-to-end):

```python
from transformers import AutoProcessor
hf_proc = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
# Run the same video through both; compare grid_thw and pixel values.
# HF's image_grid_thw uses pre-merge dimensions (patch_size=14),
# ours uses post-merge (grid_unit=28), so HF's H/W are 2× ours.
```

---

## Layer 3 — C patch, vision encoder (`clip.cpp` + `qwen3vl.cpp`)

### Symptoms
- Segfault or `GGML_ASSERT` when sending a video request
- NaN / constant embeddings
- Image requests work; video requests don't

### Diagnostics

#### Enable debug logging on `llama-server`

```bash
./llama.cpp/build/bin/llama-server \
    -m model.gguf --mmproj mmproj.gguf \
    --host 0.0.0.0 --port 8080 \
    --verbose
```

The patch emits `LOG_DBG` lines from `mtmd_tokenize_video()` of the form:

```
mtmd_tokenize_video: video nx=16 ny=16 nt=4 n_tokens=1024
```

Seeing this line confirms the video tokenize path ran. No such line = request never reached the video code path (look at the client request shape, especially `mm_processor_kwargs.is_video`).

#### Under GDB (segfaults)

```bash
# Build with debug symbols:
cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DGGML_CUDA=ON
cmake --build build -j$(nproc)

# Run under GDB:
gdb --args ./build/bin/llama-server -m model.gguf --mmproj mmproj.gguf
(gdb) run
# When it crashes:
(gdb) bt
(gdb) frame N
(gdb) print image_tokens->pos        # should be MTMD_POS_TYPE_VIDEO for video chunks
(gdb) print image_tokens->nt         # number of super-frames
(gdb) print image_tokens->nx, image_tokens->ny
```

### Expected tensor shapes

For a 4-second clip at 2 fps, 448×448 resolution:

```
Frames (Python):          8 × (448, 448, 3) uint8
Super-frames (Python):    4 × (6, 448, 448) float32
Super-frame nx × ny:      32 × 32 patches (448/14) → 16 × 16 after merge
Per-frame vision tokens:  16 × 16 = 256
Total video tokens:       nt × nx × ny = 4 × 16 × 16 = 1024
```

### Common issues

| Issue | Cause | Fix |
|-------|-------|-----|
| Segfault in `Conv2D` | Wrong tensor stride / channels | Verify `batch_f32.is_video` and channel count (6, not 3) reaching `qwen3vl.cpp` |
| NaN embeddings | Normalization divided by zero | Inspect `image_mean` / `image_std`; verify frame data isn't all-zero |
| Image path used for video chunk | `MTMD_INPUT_CHUNK_TYPE_VIDEO` not set | Trace `mm_processor_kwargs.is_video` from client → server to tokenizer |
| `GGML_ABORT("invalid position type")` | `image_tokens->pos` isn't one of the known enum values | Ensure `mtmd_tokenize_video()` ran and set `pos = MTMD_POS_TYPE_VIDEO` |

---

## Layer 4 — C patch, temporal M-RoPE (`mtmd.cpp` + `mtmd-helper.cpp`)

### Symptoms
- Captions describe a montage instead of a continuous scene ("I see four images of…" rather than "a person walks, then sits")
- Video-mode output identical to multi-image-mode output (temporal encoding ineffective)

### Diagnostics

Check `mtmd_image_tokens_get_decoder_pos()` in `tools/mtmd/mtmd.cpp` for the `MTMD_POS_TYPE_VIDEO` case. For each of the `nx * ny * nt` tokens it returns:

```cpp
case MTMD_POS_TYPE_VIDEO: {
    const uint32_t per_frame = image_tokens->nx * image_tokens->ny;
    const size_t   frame_idx = i / per_frame;
    const size_t   in_frame  = i % per_frame;
    const int32_t  t_offset  = image_tokens->temporal_positions[frame_idx];
    pos.t = pos_0 + t_offset;                                // differs per super-frame
    pos.x = pos_0 + (in_frame % image_tokens->nx);
    pos.y = pos_0 + (in_frame / image_tokens->nx);
    pos.z = 0;
} break;
```

If video captions are identical to multi-image captions:

1. Confirm `image_tokens->pos` is `MTMD_POS_TYPE_VIDEO`, not `MTMD_POS_TYPE_MROPE` (under GDB or via a `LOG_DBG` line you add).
2. Confirm `temporal_positions` is not all zeros — `llama-video-debug preprocess` prints the Python-side values.
3. Confirm the client is sending `mm_processor_kwargs.is_video: true`:

   ```bash
   curl http://localhost:9000/v1/debug/last-request | python -m json.tool
   ```

### Common issues

| Issue | Cause | Fix |
|-------|-------|-----|
| Temporal dimension collapses | All `temporal_positions[i]` equal | Fix Python side (Layer 2) |
| Spatial positions wrong | `nx` / `ny` swapped somewhere | Cross-check `clip_n_output_tokens_x` / `y` on the vision batch |
| Captions list frames instead of narrate | Patch not active | `llama-video-debug validate-patch` |

---

## Layer 5 — HTTP service + client (`server.py` + `client.py`)

### Symptoms
- `POST /v1/caption` returns 500
- Timeout
- Empty caption / "I cannot process video"

### Diagnostics

```bash
# Service health + llama-server reachability
curl http://localhost:9000/v1/health | python -m json.tool

# Caption request with a known-good video
curl -X POST http://localhost:9000/v1/caption \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "/path/to/test.mp4",
    "prompt": "Describe what happens in this video.",
    "fps": 2.0
  }' | python -m json.tool

# Inspect the most recent request (frame count, SF shapes, grid_thw, per-stage timings)
curl http://localhost:9000/v1/debug/last-request | python -m json.tool
```

### Logging

```bash
LLAMA_VIDEO_LOG_LEVEL=DEBUG llama-video-server
```

Log format: `%(asctime)s [%(name)s] %(levelname)s: %(message)s`. Expect lines from `llama_video.server`, `llama_video.extractor`, `llama_video.preprocessor`, and `llama_video.client` modules during a request.

### Request shape reaching llama-server

The client POSTs a standard OpenAI `chat/completions` payload with an additional `mm_processor_kwargs` block (this is the video-mode signal):

```json
{
  "messages": [{
    "role": "user",
    "content": [
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
      "... (one image_url per FRAME — each super-frame decomposes into 2)",
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

The patch's server-side code reads `mm_processor_kwargs` and drives the video chunk construction on the server. Without `is_video: true`, the server falls back to treating each frame as an independent image.

### Common issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| 422 from `/v1/caption` | Validation (extraction or preprocessing error); body includes the diagnostic | Check `ExtractionError` / `PreprocessingError` message |
| 502 from `/v1/caption` | `llama-server` unreachable or errored | Verify `LLAMA_SERVER_URL`; check server logs |
| 503 from llama-server | Model not loaded | Verify `--mmproj` and `-m` paths on the server command line |
| Caption says "I see 4 images" | Patch off / `is_video` not reaching server | Run `validate-patch`; verify `curl /v1/debug/last-request` shows `"is_video": true` |

---

## Debug CLI reference

```
llama-video-debug extract   <video> [--fps N] [--max-frames N] [--output-dir DIR]
llama-video-debug preprocess <video> [--fps N] [--max-frames N] [--model PROFILE]
llama-video-debug validate-patch [--server-url URL]
```

Only these three subcommands exist. `--help` on any subcommand prints its flags.

## Environment variables

See the [main README's environment-variable table](../README.md#environment-variables) for the full list.
