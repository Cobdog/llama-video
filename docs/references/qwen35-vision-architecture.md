# Qwen3.5 Vision Encoder Architecture

> **Source:** HuggingFace model cards, Qwen3-VL technical notes, llama.cpp `tools/mtmd/models/qwen3vl.cpp`
> **Last verified:** 2026-04-24 (against llama.cpp master `0adede8`)

Reference notes for what the vision encoder does and what our patch has to match. Claims here are kept to what we can verify from code we can read — the Qwen3.5 HuggingFace reference processor and llama.cpp's `tools/mtmd/models/qwen3vl.cpp`. Size-specific numbers (parameter counts, MoE configs) are in the [main README](../../README.md#supported-models) and not duplicated here.

## Vision encoder: shared across sizes

All Qwen3.5 VL models share the same vision encoder. The LLM backbone differs per size (dense vs. MoE, active params, hidden size), but the vision path — Conv3D patch embed, ViT, DeepStack, merge, projection — is identical. One mmproj per size, but the encoder logic our patch hooks into is the same.

## Patch embedding (Conv3D)

| Parameter | Value |
|-----------|-------|
| Spatial patch size | 14 × 14 pixels |
| Temporal patch size | 2 frames |
| Conv3D kernel | `(2, 14, 14)` |
| Input for **images** | `(1, 3, H, W)` → temporal dim is 1, Conv3D degrades to Conv2D |
| Input for **video** | `(T/2, 6, H, W)` super-frames, 2 source frames concatenated on the channel axis |

### Conv3D decomposition (what our patch rides)

```
Input: (batch, 6, H, W)   # super-frame: 2 RGB frames concatenated on channel dim

Conv3D(in=3, out=hidden, kernel=(2, 14, 14)):
  - Split weight along temporal dim → weight_t0, weight_t1 (each (hidden, 3, 14, 14))
  - frame_a = input[:, 0:3, :, :]
  - frame_b = input[:, 3:6, :, :]
  - output  = Conv2D(frame_a, weight_t0) + Conv2D(frame_b, weight_t1)

Output: (batch, hidden, H/14, W/14)
```

llama.cpp already implements this decomposition for images (where `frame_b` is either absent or zeroed). Our patch feeds it 6-channel input and sets `clip_image_f32_batch.is_video = true` so the qwen3vl model path takes the two-Conv2D sum branch.

## DeepStack

Unlike a vanilla ViT that uses only the last layer's output, DeepStack extracts features from several intermediate layers and merges them before projection to the LLM hidden dimension. The configuration (which layer indices to pull from, merge weights) lives in the **mmproj GGUF** — not in code we ship. No patch changes needed for DeepStack support.

## Spatial merge

After patch embedding, spatial tokens are merged 2×2:

- `merge_size = 2` → reduces token count by 4×
- Example: 448×448 input → 32×32 patches (pre-merge) → 16×16 tokens (post-merge) → 256 vision tokens per super-frame

Our `grid_thw` uses the **post-merge** dimensions. HF's `image_grid_thw` uses **pre-merge**. See [`preprocessing.md`](../subsystems/preprocessing.md#grid-thw) for why.

## M-RoPE positions

M-RoPE splits the RoPE dimensions into groups: temporal, height, width (plus a reserved fourth group used by HunyuanVL). Each vision token gets a `(t, x, y, z)` tuple that the attention layers use in place of a scalar position.

### Image path (unchanged by our patch)

```
t = pos_0           # same for every token in the image
x = pos_0 + col_in_grid
y = pos_0 + row_in_grid
z = 0
```

### Video path (added by our patch)

```
per_frame = nx * ny
frame_idx = token_i // per_frame           # which super-frame this token belongs to
in_frame  = token_i %  per_frame           # token's offset within its frame

t = pos_0 + temporal_positions[frame_idx]  # each super-frame has its own temporal index
x = pos_0 + (in_frame %  nx)
y = pos_0 + (in_frame // nx)
z = 0
```

Where `temporal_positions[i]` is computed in Python as `round(i * temporal_patch_size / fps)` — see [`preprocessing.md`](../subsystems/preprocessing.md#temporal-m-rope-positions).

### Position-array layout

llama.cpp writes the `(t, y, x, z)` tuples into the batch as **slabs** of length `n_tokens`, not interleaved per token. The helper `set_position_mrope_2d(rel_pos, seq_id)` does:

```cpp
pos[i                  ] = rel_pos[i].t;
pos[i + n_tokens       ] = rel_pos[i].y;
pos[i + n_tokens * 2   ] = rel_pos[i].x;
pos[i + n_tokens * 3   ] = rel_pos[i].z;
```

So the four dimensions live in four contiguous blocks. This is an implementation detail of llama.cpp, not a property of M-RoPE; the mathematical attention computation is the same either way.

## Token budget

```
vision_tokens = T × H_post × W_post
              = T × (H_pixels / 28) × (W_pixels / 28)
```

Example — 10-second clip at 2 fps, 448×448:

| Quantity | Value |
|----------|-------|
| Frames extracted | 20 |
| Super-frames (T) | 10 |
| Pre-merge patches/frame | 32² = 1024 |
| Post-merge tokens/frame | 16² = 256 |
| Total vision tokens | 10 × 256 = 2,560 |

This fits comfortably within a 65K context.

## mmproj GGUF metadata

The mmproj GGUF carries vision encoder config in its metadata:

- `vision.patch_size` = 14
- `vision.temporal_patch_size` = 2
- `vision.merge_size` = 2
- `vision.hidden_size` — varies
- `vision.num_hidden_layers` — varies
- `vision.deepstack_layers` — list of layer indices merged by DeepStack

Our patch doesn't read these — it relies on llama.cpp's existing mmproj loader.

## Defaults our implementation uses

From `src/llama_video/config.py::ModelConfig.qwen35()`:

```python
temporal_patch_size = 2
spatial_patch_size  = 14
merge_size          = 2
min_pixels          = 3_136          # 4   * 28^2
max_pixels          = 12_845_056     # 16384 * 28^2
image_mean          = (0.48145466, 0.4578275, 0.40821073)
image_std           = (0.26862954, 0.26130258, 0.27577711)
```

Defaults for extraction (`ExtractorSettings` / `ExtractorConfig`):

```python
default_fps = 2.0
max_frames  = 64
```
