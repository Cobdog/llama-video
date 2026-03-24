# Qwen3.5 Vision Encoder Architecture

> **Source:** HuggingFace model cards, Qwen3-VL technical report, llama.cpp Issue #17660
> **Last verified:** 2026-03-23

## Model Family

All Qwen3.5 models share the same vision encoder architecture. The LLM backbone differs in size, but the vision path is identical:

| Model | Total Params | Active Params | MoE | Vision Encoder |
|-------|-------------|---------------|-----|----------------|
| Qwen3.5-0.8B | 0.8B | 0.8B | No | Shared ViT |
| Qwen3.5-2B | 2B | 2B | No | Shared ViT |
| Qwen3.5-4B | 4B | 4B | No | Shared ViT |
| Qwen3.5-9B (Flash) | 9B | 9B | No | Shared ViT |
| Qwen3.5-27B | 27B | 27B | No | Shared ViT |
| **Qwen3.5-35B-A3B** | **35B** | **3B** | **Yes** | **Shared ViT** |
| **Qwen3.5-122B-A10B** | **122B** | **10B** | **Yes** | **Shared ViT** |
| Qwen3.5-397B-A17B | 397B | 17B | Yes | Shared ViT |

## Vision Encoder: DeepStack ViT

### Patch Embedding (Conv3D)
- **Spatial patch size:** 14×14 pixels
- **Temporal patch size:** 2 frames
- **Conv3D kernel:** (2, 14, 14) — processes 2 frames simultaneously
- **Input for images:** (1, 3, H, W) → temporal dim is 1, Conv3D degrades to Conv2D
- **Input for video:** (T, 3, H, W) → frames paired into (T/2, 6, H, W) super-frames

### Conv3D Operation
```
Input: [batch, 6, H, W]  (super-frame: 2 RGB frames concatenated on channel dim)

Conv3D(in_channels=3, out_channels=hidden, kernel_size=(2, 14, 14)):
  - Slice weight into temporal_0 and temporal_1 (each [hidden, 3, 14, 14])
  - frame_0 = input[:, 0:3, :, :]   (first 3 channels)
  - frame_1 = input[:, 3:6, :, :]   (last 3 channels)
  - output = Conv2D(frame_0, temporal_0) + Conv2D(frame_1, temporal_1)

Output: [batch, hidden, H/14, W/14]
```

### DeepStack
Unlike standard ViTs that use only the last layer's output:
- DeepStack extracts features from **multiple intermediate ViT layers**
- These are merged/weighted before projection to LLM dimension
- The mmproj GGUF file contains the DeepStack configuration
- **No patch changes needed** — DeepStack is handled inside the mmproj

### Merge Operation
After patch embedding, spatial tokens are further merged:
- **merge_size:** 2 (merges 2×2 spatial tokens into 1)
- Reduces token count by 4×
- For 448×448 input: 448/14 = 32 patches → 32/2 = 16 merged → 256 tokens per temporal position

## M-RoPE (Multimodal Rotary Position Embeddings)

### Structure
Each position has 3 components mapped to RoPE dimensions:
```
RoPE dimensions split into 3 groups:
  - Group 1: temporal position (or text position for text tokens)
  - Group 2: height position
  - Group 3: width position
```

### For Images (current llama.cpp behavior)
```
temporal_pos = 0  (constant for all patches in an image)
height_pos   = row_index  (0 to H_grid-1)
width_pos    = col_index  (0 to W_grid-1)
```

### For Video (what our patch adds)
```
temporal_pos = round(frame_idx × seconds_per_grid × tokens_per_second)
height_pos   = row_index  (0 to H_grid-1)
width_pos    = col_index  (0 to W_grid-1)
```

Where:
- `frame_idx` = temporal grid position (0 to T-1, where T = num_frames / temporal_patch_size)
- `seconds_per_grid` = temporal_patch_size / fps
- `tokens_per_second` = model config value (controls temporal resolution)

### Interleaved Layout (Qwen3.5 Upgrade)
Qwen3.5 uses **interleaved** M-RoPE layout (different from Qwen3-VL's sequential layout):
```
# Qwen3-VL (sequential):  [t,t,t,..., h,h,h,..., w,w,w,...]
# Qwen3.5 (interleaved):  [t,h,w, t,h,w, t,h,w, ...]
```
This is critical — using the wrong layout produces garbage outputs.

## Video Processing Pipeline (HuggingFace Reference)

```python
# From transformers Qwen3VLImageProcessor
def process_video(video_path, fps=2.0):
    # 1. Sample frames at specified FPS
    frames = sample_frames(video_path, fps=fps, do_sample_frames=True)

    # 2. Resize frames
    # Default: shortest_edge=4096, longest_edge=469762048
    # For practical use: bounded by max_pixels
    frames = [resize(f, min_pixels, max_pixels) for f in frames]

    # 3. Pair frames (temporal_patch_size=2)
    # [F0, F1, F2, F3] → [(F0,F1), (F2,F3)]
    # Each pair concatenated on channel dim: 3+3=6 channels

    # 4. Compute grid_thw
    T = len(frames) // temporal_patch_size  # number of temporal positions
    H = frame_height // (patch_size * merge_size)  # merged height grid
    W = frame_width // (patch_size * merge_size)    # merged width grid
    grid_thw = [T, H, W]

    # 5. Apply to model
    # pixel_values_videos: [T, 6, H, W] float tensor
    # video_grid_thw: [1, 3] long tensor ([T, H, W])
    return pixel_values_videos, video_grid_thw
```

## Configuration Files in GGUF

The mmproj GGUF contains vision encoder config. Key fields:
- `vision.patch_size` = 14
- `vision.temporal_patch_size` = 2
- `vision.merge_size` = 2
- `vision.hidden_size` = varies by model
- `vision.num_hidden_layers` = varies
- `vision.deepstack_layers` = list of layer indices for DeepStack

## Token Budget

For a video with T temporal positions, H height grid, W width grid:
```
vision_tokens = T × H × W
```

Example: 10-second video at 2fps, 448×448:
- Frames: 20
- Super-frames: 10 (T=10)
- Patches per frame: (448/14)² = 1024
- After merge: (448/28)² = 256
- Total vision tokens: 10 × 256 = 2,560 tokens

This fits easily within the 262K context window.

## Key Parameters for Our Implementation

```python
# Default model config for all Qwen3.5 models
TEMPORAL_PATCH_SIZE = 2
SPATIAL_PATCH_SIZE = 14
MERGE_SIZE = 2
DEFAULT_VIDEO_FPS = 2.0
DEFAULT_MAX_FRAMES = 64  # Conservative default
MIN_PIXELS = 4 * 28 * 28      # Minimum frame size
MAX_PIXELS = 16384 * 28 * 28  # Maximum frame size
```
