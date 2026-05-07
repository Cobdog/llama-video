# Preprocessing Subsystem

> **Code:** `src/llama_video/preprocessor.py`
> **Tests:** `tests/unit/test_preprocessor.py`, `tests/integration/test_hf_reference.py`, `tests/integration/test_pipeline.py`
> **Reference:** [`docs/references/qwen35-vision-architecture.md`](../references/qwen35-vision-architecture.md)
> **Last verified:** 2026-04-24 (against `src/` at HEAD)

## Purpose

Turn extracted `Frame`s into the tensor shape the Qwen3.5 vision encoder expects: 6-channel super-frames, a `grid_thw` triple, and M-RoPE temporal indices. All of that is packed into `VideoInput` and handed to `LlamaServerClient.caption_video()`.

> **Note:** This document describes the Qwen3.5-specific `Preprocessor` pipeline. Other model families (e.g., Gemma4) use the adapter system (`src/llama_video/adapters/`) which handles preprocessing, payload construction, and response parsing per family. The adapter layer delegates to this `Preprocessor` for Qwen3.5 models but uses its own logic for others.

## Key classes

### `Preprocessor`

```python
Preprocessor(model_config: ModelConfig | None = None)
```

If `model_config` is `None`, uses `ModelConfig()` which reads `LLAMA_*` env vars and defaults to Qwen3.5 encoder params.

Primary entry point:

```python
Preprocessor.process(
    frames: list[Frame],
    fps: float = 2.0,
    odd_strategy: OddFrameStrategy = OddFrameStrategy.PAD,
    resolution_scale: float = 1.0,
) -> VideoInput
```

That's the whole signature — there is no `config` parameter. `fps` is the extraction FPS (used for temporal-position math), `odd_strategy` controls what happens when the frame count is odd (`PAD` duplicates the last frame, `DROP` drops it), and `resolution_scale` lets you shrink the resolved resolution before grid snapping (1.0 = native, 0.5 = half).

Helper methods (called internally; useful in tests):

| Method | Signature | Purpose |
|--------|-----------|---------|
| `compute_grid_thw` | `(num_super_frames: int, target_w: int, target_h: int) -> tuple[int, int, int]` | `(T, H/grid_unit, W/grid_unit)` |
| `compute_temporal_positions` | `(grid_thw: tuple[int, int, int], fps: float) -> list[int]` | See below |

### `ModelConfig`

Vision encoder profile. Defaults are for Qwen3.5:

| Field | Default | Meaning |
|-------|---------|---------|
| `temporal_patch_size` | `2` | Frames per super-frame |
| `spatial_patch_size` | `14` | Pixels per patch |
| `merge_size` | `2` | Spatial merge factor |
| `min_pixels` | `3136` | `4 × 28²` |
| `max_pixels` | `12_845_056` | `16384 × 28²` |
| `image_mean` | `(0.48145466, 0.4578275, 0.40821073)` | CLIP RGB mean |
| `image_std` | `(0.26862954, 0.26130258, 0.27577711)` | CLIP RGB std |
| `grid_unit` | (property) `spatial_patch_size × merge_size` = 28 | Pixel size frames must be divisible by |

### `VideoInput`

Output of `process()`.

| Attribute | Type | Meaning |
|-----------|------|---------|
| `super_frames` | `list[SuperFrame]` | Each `SuperFrame.data` is `(6, H, W)` float32, channels `[R_a, G_a, B_a, R_b, G_b, B_b]` |
| `grid_thw` | `tuple[int, int, int]` | `(T, H_grid, W_grid)` — post-merge spatial dims |
| `temporal_positions` | `list[int]` | M-RoPE temporal indices, length T |
| `fps` | `float` | Extraction FPS used |
| `num_source_frames` | `int` | Original frame count before pairing |
| `resolution` | `tuple[int, int]` | `(width, height)` after resize |

---

## Super-frame construction

Pair consecutive frames into one 6-channel tensor:

```
frame_a: (H, W, 3) uint8, channels [R, G, B]
frame_b: (H, W, 3) uint8

after resize + normalize:
frame_a' (3, H, W) float32
frame_b' (3, H, W) float32

super_frame = concat([frame_a', frame_b'], axis=0)
            # (6, H, W) float32, channels [R_a, G_a, B_a, R_b, G_b, B_b]
```

This is the layout `qwen3vl.cpp`'s Conv3D decomposition expects: `channels[0:3]` = frame_a, `channels[3:6]` = frame_b, then `Conv2D(frame_a, weight_t0) + Conv2D(frame_b, weight_t1)`.

HuggingFace's internal layout is different (channel-grouped `[R_a, R_b, G_a, G_b, B_a, B_b]` in a `(C, T, H, W)` tensor) but carries the same per-frame values. We chose channel-concatenated `(T, C, H, W)` because it matches the patch's Conv3D split directly.

### Odd frame counts

| Strategy | Behavior |
|----------|----------|
| `OddFrameStrategy.PAD` (default) | Duplicate the last frame to make the count even |
| `OddFrameStrategy.DROP` | Drop the last frame |

Extracting an even number of frames avoids the question entirely.

---

## Grid THW

```
T = num_super_frames          # len(frames) // temporal_patch_size (with PAD)
H = H_resized // grid_unit    # e.g. 448 // 28 = 16
W = W_resized // grid_unit    # e.g. 448 // 28 = 16
grid_thw = (T, H, W)
```

**Convention:** our `grid_thw` uses **post-merge** dimensions (`grid_unit = spatial_patch_size × merge_size = 28`). HuggingFace's `image_grid_thw` uses **pre-merge** dimensions (`spatial_patch_size = 14`), so HF's H and W are exactly `merge_size` (2×) ours. Both describe the same tensor; it's just a choice about which point in the pipeline the labels refer to.

---

## Resolution handling

Frames are resized so the pixel count stays in `[min_pixels, max_pixels]` and both dims are divisible by `grid_unit`:

1. Compute the aspect-ratio-preserving resize target within bounds (matches HF `smart_resize`).
2. Round to nearest multiple of `grid_unit`.
3. Resize every frame to that size with **BICUBIC** interpolation (matches HF default).
4. Enforce that all frames share the resolved `(H, W)`.

`resolution_scale < 1.0` shrinks the target before grid snapping — useful for reducing vision-token count on a memory-constrained run.

---

## Temporal M-RoPE positions

```python
def compute_temporal_positions(
    self,
    grid_thw: tuple[int, int, int],
    fps: float,
) -> list[int]:
    t, _, _ = grid_thw
    seconds_per_temporal = self._config.temporal_patch_size / fps
    return [round(i * seconds_per_temporal) for i in range(t)]
```

Worked example (`temporal_patch_size=2`, `fps=2.0`, T=4):

```
seconds_per_temporal = 2 / 2.0 = 1.0
positions = [round(0), round(1), round(2), round(3)] = [0, 1, 2, 3]
```

These integers land in `mm_processor_kwargs.temporal_positions` in the HTTP payload and, after the C patch reads them, become the `t` coordinate in each super-frame's M-RoPE positions.

There is no `tokens_per_second` multiplier in the current code — every token within a super-frame shares the super-frame's temporal position, and spatial `x`/`y` advance within the frame.

---

## Validation

`process()` raises before it returns bad data:

| Error | Raised when |
|-------|-------------|
| `PreprocessingError` | `frames` is empty |
| `InvalidFrameDimensionsError` | Frames have inconsistent `(W, H)` |
| `ResolutionError` | Resolved pixel count is outside `[min_pixels, max_pixels]` |

Every error carries a `context: dict` with diagnostic fields (frame count, resolved size, pixel count, etc.).
