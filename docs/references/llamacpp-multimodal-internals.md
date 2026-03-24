# llama.cpp Multimodal Internals

> **Source:** llama.cpp source code, Issue #18389, Issue #17660, PR #19468
> **Last verified:** 2026-03-23

## Architecture

Multimodal support in llama.cpp is provided by `libmtmd` (multimodal), built on top of `clip.cpp` (vision encoder).

```
User Input (text + images/audio)
    │
    ▼
libmtmd (mtmd.cpp)
├── Parses input, identifies modality markers
├── Routes images to clip.cpp
├── Routes audio to audio encoder
├── Computes position embeddings (M-RoPE for Qwen)
└── Produces interleaved token sequence for LLM
    │
    ▼
clip.cpp
├── Loads mmproj GGUF
├── Preprocesses images (resize, normalize)
├── Runs vision encoder (ViT forward pass)
├── Applies model-specific operations (Conv3D, DeepStack, etc.)
└── Returns vision embeddings
    │
    ▼
LLM (llama.cpp core)
├── Receives interleaved text + vision tokens
├── Applies positional encoding (RoPE/M-RoPE)
└── Generates text output
```

## File Map

| File | Purpose |
|------|---------|
| `tools/mtmd/mtmd.h` | Public API: `mtmd_input_text`, `mtmd_input_image`, model loading |
| `tools/mtmd/mtmd.cpp` | Implementation: input parsing, M-RoPE, token assembly |
| `tools/mtmd/clip.h` | Vision encoder API: `clip_image_encode`, model info |
| `tools/mtmd/clip.cpp` | Vision encoder: patch embed, ViT, Conv3D, normalization |
| `tools/mtmd/mtmd-cli.cpp` | CLI tool for testing multimodal input |
| `tools/mtmd/mtmd-helper.cpp` | Utility functions |
| `examples/server/server.cpp` | HTTP server with `/chat/completions` endpoint |

## clip.cpp — Vision Encoder

### Model Detection
clip.cpp detects the model architecture from the GGUF metadata and selects the appropriate processing path:
```cpp
// Simplified model detection
if (model_type == "qwen3.5" || model_type == "qwen3-vl") {
    // Qwen path: Conv3D patch embed, M-RoPE
} else if (model_type == "gemma3") {
    // Gemma path: SigLIP encoder
} // ... etc
```

### Image Processing Pipeline (Qwen3.5)
```
Input image: [3, H, W] (RGB, float32, normalized)
    │
    ▼ Resize to valid dimensions (multiple of patch_size × merge_size = 28)
    │
    ▼ Conv3D patch embedding (temporal_patch_size=1 for images)
    │   For images: this is just Conv2D [3, H, W] → [hidden, H/14, W/14]
    │
    ▼ Positional embedding (M-RoPE with temporal=0)
    │
    ▼ ViT transformer blocks (N layers)
    │
    ▼ DeepStack: extract features from configured layers
    │
    ▼ Merge: 2×2 spatial merge, reducing token count by 4×
    │
    ▼ Projection to LLM hidden dimension
    │
Output: [num_tokens, llm_hidden_dim]
```

### Conv3D Decomposition in clip.cpp
The existing code handles Conv3D for Qwen models. For images (temporal=1), it's a straight Conv2D. The Conv3D weight tensor is shaped `[out_channels, in_channels, 2, 14, 14]` and is split:

```cpp
// Pseudocode from clip.cpp
if (temporal_patch_size == 2) {
    // Split weight along temporal dimension
    weight_t0 = weight[:, :, 0, :, :]  // [out, in, 14, 14]
    weight_t1 = weight[:, :, 1, :, :]  // [out, in, 14, 14]

    if (is_video && n_channels == 6) {
        // Video: input has 6 channels (2 frames concatenated)
        frame_0 = input[:, 0:3, :, :]
        frame_1 = input[:, 3:6, :, :]
        output = conv2d(frame_0, weight_t0) + conv2d(frame_1, weight_t1)
    } else {
        // Image: input has 3 channels, temporal=1
        // Only use weight_t0 (or sum both for single frame)
        output = conv2d(input, weight_t0)  // Simplified
    }
}
```

**Key insight:** The Conv3D weight splitting is already implemented. What's missing is the video input path that feeds 6-channel super-frames and sets is_video=true.

## mtmd.cpp — Multimodal Input Assembly

### Current Input Types
```cpp
struct mtmd_input_image {
    clip_image_u8 * img;     // Raw image data
    int nx, ny;              // Dimensions
};

// No video input type exists yet — we add this
```

### M-RoPE Implementation
mtmd.cpp computes M-RoPE positions for Qwen models. The current implementation:
```cpp
// For each vision token at grid position (t, h, w):
// (Currently t is always 0 for images)
int temporal_pos = 0;  // ← THIS IS WHAT WE CHANGE FOR VIDEO
int height_pos = h;
int width_pos = w;

// Interleaved layout (Qwen3.5):
// position[i*3 + 0] = temporal_pos
// position[i*3 + 1] = height_pos
// position[i*3 + 2] = width_pos
```

## server.cpp — HTTP API

### Current Image Input Format
```json
{
    "messages": [{
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": { "url": "data:image/jpeg;base64,..." }
            },
            {
                "type": "text",
                "text": "Describe this image"
            }
        ]
    }]
}
```

### Planned Video Input Format (our patch)
```json
{
    "messages": [{
        "role": "user",
        "content": [
            {
                "type": "video_frames",
                "video_frames": {
                    "frames": [
                        "data:image/jpeg;base64,...",
                        "data:image/jpeg;base64,..."
                    ],
                    "fps": 2.0,
                    "is_video": true
                }
            },
            {
                "type": "text",
                "text": "Describe what happens in this video"
            }
        ]
    }]
}
```

Alternatively (simpler, closer to Qwen's API):
```json
{
    "messages": [{
        "role": "user",
        "content": [
            {
                "type": "video_url",
                "video_url": { "url": "file:///path/to/video.mp4" }
            },
            {
                "type": "text",
                "text": "Describe what happens"
            }
        ]
    }],
    "mm_processor_kwargs": {
        "fps": 2.0,
        "do_sample_frames": true
    }
}
```

## Build System

```bash
# Standard llama.cpp build (CMake)
cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)

# Debug build (for GDB debugging)
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --config Debug -j$(nproc)

# Key binaries
build/bin/llama-server      # HTTP server
build/bin/llama-mtmd-cli    # Multimodal CLI tool
```

## Upstream Video Support Status

- **Tracking issue:** #18389 (open since 2025-12-26)
- **Status:** Planning phase, community contributions in early stages
- **Phase 1 target:** ffmpeg frame extraction + SmolVLM2-Video (frame-by-frame)
- **Qwen3.5 (fused frames):** Explicitly noted as requiring additional work beyond Phase 1
- **No timeline** for completion

Our patch fills the Qwen3.5 gap specifically and can be contributed back or dropped when upstream catches up.
