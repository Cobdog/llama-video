# Video Preprocessing Subsystem

> **Code:** `src/llama_video/preprocessor.py`
> **Tests:** `tests/unit/test_preprocessor.py`, `tests/integration/test_hf_reference.py`, `tests/integration/test_pipeline.py`
> **Reference:** `docs/references/qwen35-vision-architecture.md`
> **HF Reference Script:** `scripts/hf_reference.py`
> **Last verified:** 2026-03-23

## Purpose

Transform extracted frames into the tensor format expected by Qwen3.5's vision encoder: super-frames, grid THW, and temporal M-RoPE positions.

## Key Classes

### `Preprocessor`
- `process(frames, config?) → VideoInput`
- Handles: frame pairing, resizing, normalization, grid computation

### `ModelConfig`
- `temporal_patch_size: int = 2`
- `spatial_patch_size: int = 14`
- `merge_size: int = 2`
- `min_pixels: int = 3136` (4 × 28 × 28)
- `max_pixels: int = 12845056` (16384 × 28 × 28)
- `image_mean: tuple = (0.48145466, 0.4578275, 0.40821073)` — CLIP normalization
- `image_std: tuple = (0.26862954, 0.26130258, 0.27577711)`

### `VideoInput`
- `super_frames: list[np.ndarray]` — each (6, H, W) float32
- `grid_thw: tuple[int, int, int]` — (T, H_grid, W_grid)
- `temporal_positions: list[int]` — M-RoPE temporal indices
- `fps: float` — extraction FPS used
- `num_source_frames: int` — original frame count

## Super-Frame Construction

```python
def build_super_frames(frames: list[Frame]) -> list[np.ndarray]:
    """Pair consecutive frames into 6-channel tensors.

    temporal_patch_size = 2, so frames are paired:
    [F0, F1, F2, F3] → [concat(F0,F1), concat(F2,F3)]

    Each frame is (H, W, 3) RGB.
    Each super-frame is (6, H, W) = channel-first concat of frame pair.
    """
```

### Channel Layout

Our super-frame layout: `[R1, G1, B1, R2, G2, B2]` — frame-concatenated `(T, C, H, W)`.
HF's internal layout: `[R1, R2, G1, G2, B1, B2]` — channel-grouped `(C, T, H, W)`.

These are different orderings of identical per-frame values. Our layout is natural for
clip.cpp's Conv3D decomposition: `channels[0:3]` = frame_a, `channels[3:6]` = frame_b,
then `Conv2D(frame_a, weight_0) + Conv2D(frame_b, weight_1)`.

### Odd Frame Count Handling
- `pad`: Duplicate last frame to make even count
- `drop`: Drop last frame

## Grid THW Computation

```python
T = num_super_frames  # = len(frames) // temporal_patch_size
H = frame_height // (spatial_patch_size * merge_size)  # e.g., 448 // 28 = 16
W = frame_width // (spatial_patch_size * merge_size)    # e.g., 448 // 28 = 16
grid_thw = (T, H, W)
```

**Convention note:** Our grid_thw uses **post-merge** spatial dimensions (`grid_unit=28`).
HF's `image_grid_thw` uses **pre-merge** dimensions (`patch_size=14`), so HF's H and W
values are exactly `merge_size` (2×) ours. This is intentional — our clip.cpp patch
works with post-merge token counts.

## Resolution Handling

Frames are resized to be divisible by `spatial_patch_size × merge_size = 28`:
1. Compute target resolution within `[min_pixels, max_pixels]` (matches HF `smart_resize`)
2. Round to nearest multiple of 28
3. Resize all frames to this resolution using **BICUBIC** interpolation (matches HF default)
4. All frames in a video must be the same resolution

## Temporal Position Computation

```python
def compute_temporal_positions(
    grid_thw: tuple[int, int, int],
    fps: float,
    temporal_patch_size: int = 2,
) -> list[int]:
    """Compute M-RoPE temporal indices.

    For each temporal grid position t (0 to T-1):
      seconds = t * temporal_patch_size / fps
      position = round(seconds * tokens_per_second)

    tokens_per_second is a model config value. For Qwen3.5 it's
    implicitly 1.0 in the default config.
    """
```

## Validation

The preprocessor validates:
- Frame dimensions are consistent
- Frame count ≥ 1
- Resolution within model bounds
- Grid THW values are positive integers
- Super-frame channel count = 2 × 3 = 6
