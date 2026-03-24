# Debugging Guide

This guide covers how to diagnose failures at every layer of the llama-video pipeline. When something goes wrong, work through these layers in order — most issues are in the outermost layers.

## Quick Diagnostic Checklist

```
□ Is ffmpeg installed and in PATH?          → ffmpeg -version
□ Is llama-server running?                  → curl http://localhost:8080/health
□ Is llama-server loaded with mmproj?       → Check server startup logs for "mmproj loaded"
□ Is the patch applied?                     → Check server startup logs for "[video] temporal support enabled"
□ Are frames extracting correctly?          → uv run llama-video-debug extract <video>
□ Are super-frames shaped correctly?        → uv run llama-video-debug preprocess <video>
□ Is the model responding to images?        → Test with a single image first
□ Is temporal encoding active?              → Check debug output for grid_thw with T>1
```

## Layer 1: Frame Extraction (Python — Extractor)

### Symptoms
- "ffmpeg not found" error
- Empty frame list returned
- Frames are black, corrupted, or wrong resolution

### Diagnostics
```bash
# Test ffmpeg directly
ffmpeg -i test_video.mp4 -vf "fps=2" -q:v 2 /tmp/test_frames/frame_%04d.jpg

# Use debug CLI
uv run llama-video-debug extract test_video.mp4 --fps 2 --output-dir /tmp/debug_frames/

# Check frame count vs expected
# For a 4-second video at 2fps: expect 8 frames
python -c "
from llama_video import Extractor, ExtractorConfig
e = Extractor(ExtractorConfig(fps=2.0))
frames = e.extract_frames('test_video.mp4')
print(f'Extracted {len(frames)} frames')
for i, f in enumerate(frames):
    print(f'  Frame {i}: {f.size}, mode={f.mode}')
"
```

### Common Issues
| Issue | Cause | Fix |
|-------|-------|-----|
| ffmpeg not found | Not in PATH | Install ffmpeg or set `FFMPEG_PATH` |
| 0 frames extracted | FPS too high for short video | Lower FPS or use `min_frames=1` |
| Wrong resolution | Video has non-standard aspect ratio | Check `max_resolution` config |
| Extraction hangs | Large video, no max_frames limit | Set `max_frames` config |

---

## Layer 2: Video Preprocessing (Python — Preprocessor)

### Symptoms
- Super-frame has wrong number of channels (expected 6, got 3)
- grid_thw values look wrong
- Temporal positions are all zeros

### Diagnostics
```bash
# Debug preprocessor output
uv run llama-video-debug preprocess test_video.mp4 --fps 2 --model qwen3.5

# This prints:
#   Frames: 8
#   Super-frames: 4 (shape: [4, 6, H, W])
#   grid_thw: [4, H/28, W/28]
#   temporal_positions: [0, 1, 2, 3]
#   Total vision tokens: 4 × (H/28) × (W/28)
```

```python
# Manual inspection
from llama_video import Preprocessor, ModelConfig
p = Preprocessor(ModelConfig.qwen35())
frames = [...]  # PIL Images
result = p.process(frames)
print(f"Super-frames: {len(result.super_frames)}")
print(f"Grid THW: {result.grid_thw}")
print(f"Temporal positions: {result.temporal_positions}")
print(f"Super-frame shape: {result.super_frames[0].shape}")  # Should be (6, H, W)
```

### Common Issues
| Issue | Cause | Fix |
|-------|-------|-----|
| 3 channels instead of 6 | Frame pairing failed | Check `temporal_patch_size` config |
| grid_thw T=1 | Not in video mode | Ensure `is_video=True` in preprocessing |
| All temporal_positions = 0 | M-RoPE computation bypassed | Check model config has correct temporal params |
| Odd frame count error | Can't pair last frame | Set `odd_frame_strategy='pad'` or `'drop'` |

### Validation Against HuggingFace Reference
```python
# Compare our preprocessor output with HuggingFace Qwen3-VL processor
from transformers import AutoProcessor
hf_proc = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

# Process same video through both pipelines
# Compare: grid_thw, pixel_values shape, temporal positions
# They should match exactly
```

---

## Layer 3: C Patch — Vision Encoder (clip.cpp)

### Symptoms
- Segfault when processing video input
- Garbage embeddings (all zeros, NaN, or constant values)
- Correct embeddings for images but wrong for video
- Dimension mismatch errors

### Diagnostics

#### Enable Debug Logging
```bash
# Start llama-server with debug logging
LLAMA_LOG_LEVEL=debug ./llama.cpp/build/bin/llama-server \
    -m model.gguf --mmproj mmproj.gguf \
    --host 0.0.0.0 --port 8080
```

#### Key Log Lines to Look For
```
# Good — video mode activated:
[mtmd] video input: T=4, H=16, W=16, channels=6
[clip] conv3d decomposition: input_shape=[6,H,W] → conv2d_0=[3,H,W] + conv2d_1=[3,H,W]
[clip] patch_embed output: [T*H*W, hidden_dim]
[mtmd] mrope temporal positions: [0, 1, 2, 3]

# Bad — fell through to image path:
[mtmd] image input: channels=3, H=..., W=...
# (This means the video flag isn't being passed through)

# Bad — tensor shape mismatch:
GGML_ASSERT: ne[0] == expected (got actual)
# (Check tensor dimensions at the assertion point)
```

#### GDB Debugging (Segfaults)
```bash
# Build with debug symbols
cd llama.cpp && cmake -B build -DCMAKE_BUILD_TYPE=Debug && cmake --build build

# Run under GDB
gdb --args ./build/bin/llama-server -m model.gguf --mmproj mmproj.gguf
(gdb) run
# When it segfaults:
(gdb) bt          # Backtrace
(gdb) frame N     # Navigate to relevant frame
(gdb) print shape # Inspect tensor shapes
```

#### Tensor Shape Verification
Expected shapes at each stage for a 4-second video at 2fps, 448×448 resolution:
```
Input:           [4, 6, 448, 448]    (4 super-frames, 6 channels)
After Conv3D:    [4, hidden, 32, 32] (patch_size=14, 448/14=32)
After merge:     [4, hidden, 16, 16] (merge_size=2, 32/2=16)
Vision tokens:   [4*16*16, hidden]   = [1024, hidden]
After DeepStack: [1024, hidden]      (multi-layer features merged)
```

### Common Issues
| Issue | Cause | Fix |
|-------|-------|-----|
| Segfault in conv2d | Wrong tensor stride/dimensions | Check input tensor layout matches ggml expectations |
| NaN embeddings | Division by zero in normalization | Check if any tensor values are zero when they shouldn't be |
| Shape mismatch assertion | grid_thw not propagated | Verify video_grid_thw reaches clip_image_encode |
| Image path used for video | is_video flag lost | Trace flag through mtmd → clip call chain |

---

## Layer 4: C Patch — Temporal M-RoPE (mtmd.cpp)

### Symptoms
- Video captions lack temporal awareness ("I see several images" instead of "A person walks, then sits down")
- Same output for video mode vs multi-image mode (M-RoPE not differentiating)
- Temporal positions not affecting output at all

### Diagnostics

#### Temporal Differentiation Test
```bash
# This is the most important test for M-RoPE:
# Send same frames as (a) multi-image and (b) video
# If outputs are identical, M-RoPE temporal encoding is NOT working

uv run llama-video-debug compare-modes test_video.mp4

# Expected output:
# Image mode response: "I see 4 photos showing a park scene..."
# Video mode response: "In this video, a dog runs across the park and catches a frisbee..."
# Cosine similarity of logits: 0.73  (should be < 0.95 if temporal encoding works)
```

#### M-RoPE Position Inspection
```python
# Verify temporal positions are computed correctly
from llama_video.preprocessor import compute_temporal_positions

grid_thw = (4, 16, 16)  # 4 temporal positions
fps = 2.0
positions = compute_temporal_positions(grid_thw, fps)
# Expected: positions increment with temporal dimension
# e.g., [0, 500, 1000, 1500] (scaled by tokens_per_second)
print(positions)
```

### Common Issues
| Issue | Cause | Fix |
|-------|-------|-----|
| No temporal differentiation | temporal_idx always 0 | Check M-RoPE code path for video vs image branching |
| Wrong temporal scale | tokens_per_second misconfigured | Compare with HF reference implementation |
| Interleaved layout wrong | Qwen3.5 vs Qwen3-VL difference | Qwen3.5 uses interleaved; check layout order |

---

## Layer 5: API Server (Python — Server/Client)

### Symptoms
- 500 error on `/v1/caption`
- Timeout waiting for response
- Caption is empty or "I cannot process video"

### Diagnostics
```bash
# Health check
curl http://localhost:9000/v1/health | python -m json.tool

# Manual caption request with verbose output
curl -X POST http://localhost:9000/v1/caption \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "/path/to/test.mp4",
    "prompt": "Describe what happens in this video.",
    "fps": 2.0
  }' | python -m json.tool

# Check last request debug info
curl http://localhost:9000/v1/debug/last-request | python -m json.tool
# Returns: frame count, super-frame shapes, grid_thw, timing, llama-server request/response
```

### Logging
```bash
# Run server with debug logging
LLAMA_VIDEO_LOG_LEVEL=DEBUG uv run llama-video-server

# Log output includes:
# [extractor] Extracting frames from /path/to/video.mp4 at 2.0 fps
# [extractor] Extracted 8 frames in 0.3s
# [preprocessor] Built 4 super-frames, grid_thw=[4, 16, 16]
# [client] Sending to llama-server: 4 images, is_video=true
# [client] Response received in 2.1s, 47 tokens
```

---

## Layer 6: Integration with Intern

### Symptoms
- Intern video editor sends clip but gets no response
- Caption appears but lacks temporal content
- Connection refused errors

### Diagnostics
```bash
# Verify llama-video service is running
curl http://localhost:9000/v1/health

# Verify Intern backend can reach it
# (from Intern's backend container/environment)
curl http://localhost:9000/v1/caption -X POST \
  -H "Content-Type: application/json" \
  -d '{"video_path": "/tmp/test.mp4", "prompt": "Describe this video."}'
```

---

## Debug CLI Reference

The `llama-video-debug` CLI provides diagnostic tools:

```bash
# Extract and inspect frames
uv run llama-video-debug extract <video> [--fps N] [--output-dir DIR]

# Preprocess and inspect super-frames
uv run llama-video-debug preprocess <video> [--fps N] [--model qwen3.5]

# Compare image-mode vs video-mode output
uv run llama-video-debug compare-modes <video> [--server-url URL]

# Full pipeline trace with timing
uv run llama-video-debug trace <video> [--prompt TEXT] [--server-url URL]

# Validate patch is working
uv run llama-video-debug validate-patch [--server-url URL]
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_VIDEO_LOG_LEVEL` | `INFO` | Python logging level |
| `LLAMA_SERVER_URL` | `http://localhost:8080` | Patched llama-server URL |
| `FFMPEG_PATH` | `ffmpeg` | Path to ffmpeg binary |
| `LLAMA_VIDEO_MAX_FRAMES` | `64` | Maximum frames to extract |
| `LLAMA_VIDEO_DEFAULT_FPS` | `2.0` | Default frame extraction FPS |
| `LLAMA_VIDEO_DEBUG` | `0` | Set to 1 for verbose debug output |
