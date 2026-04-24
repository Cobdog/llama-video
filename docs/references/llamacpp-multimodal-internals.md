# llama.cpp Multimodal Internals

> **Source:** llama.cpp source code at `0adede8` — primary files: `tools/mtmd/mtmd.{h,cpp}`, `tools/mtmd/mtmd-helper.cpp`, `tools/mtmd/clip.{h,cpp}`, `tools/mtmd/models/qwen3vl.cpp`
> **Last verified:** 2026-04-24

Reference notes for how llama.cpp's multimodal layer is organized, oriented around the pieces our patch touches. For what the patch actually adds, see [`subsystems/c-patch.md`](../subsystems/c-patch.md).

## Architecture

```
User Input (text + images/audio/video)
    │
    ▼
libmtmd (tools/mtmd/mtmd.cpp)
├── Parses input, routes by modality marker (default "<__media__>")
├── Calls the appropriate vision / audio encoder
├── Builds mtmd_input_chunk list (TEXT / IMAGE / AUDIO / VIDEO)
└── Computes per-token decoder positions via mtmd_image_tokens_get_decoder_pos()
    │
    ▼
clip (tools/mtmd/clip.cpp + tools/mtmd/models/*.cpp)
├── Loads mmproj GGUF
├── Preprocesses images (resize, normalize) via mtmd-image.cpp
├── Runs vision encoder (ViT forward pass — model-specific)
└── Returns vision embeddings
    │
    ▼
LLM (llama.cpp core)
├── Consumes interleaved text + vision tokens
├── Applies positional encoding (RoPE / M-RoPE) using the decoder positions
└── Generates text output
```

## File map

| File | Purpose |
|------|---------|
| `tools/mtmd/mtmd.h` | Public API: chunk types, tokenize entry points, image-tokens accessors |
| `tools/mtmd/mtmd.cpp` | Input parsing, pos-type switch, token assembly |
| `tools/mtmd/mtmd-helper.cpp` | Decode loop: positions → batch → `llama_decode` |
| `tools/mtmd/mtmd-image.cpp` | Image-specific preprocessing |
| `tools/mtmd/mtmd-audio.cpp` | Audio-specific preprocessing |
| `tools/mtmd/clip.h` / `clip.cpp` | Shared vision encoder code |
| `tools/mtmd/models/qwen3vl.cpp` | Qwen3-VL / Qwen3.5 encoder (where our Conv3D path is) |
| `tools/mtmd/mtmd-cli.cpp` | Standalone CLI |
| `tools/server/server-common.{h,cpp}` | HTTP server shared utilities; request parsing |
| `tools/server/server-context.{h,cpp}` | `/v1/chat/completions` handler |

## Relevant data types

### Chunk types (`mtmd_input_chunk_type`)

After our patch applies:

```cpp
enum mtmd_input_chunk_type {
    MTMD_INPUT_CHUNK_TYPE_TEXT,
    MTMD_INPUT_CHUNK_TYPE_IMAGE,
    MTMD_INPUT_CHUNK_TYPE_AUDIO,
    MTMD_INPUT_CHUNK_TYPE_VIDEO,   // added by our patch
};
```

### Position types (`mtmd_pos_type`, in `mtmd.cpp`)

```cpp
enum mtmd_pos_type {
    MTMD_POS_TYPE_NORMAL,     // sequential positions
    MTMD_POS_TYPE_MROPE,      // Qwen M-RoPE (each image = 1 temporal slot)
    MTMD_POS_TYPE_HUNYUANVL,  // HunyuanVL BOI/EOI + newline layout
    MTMD_POS_TYPE_VIDEO,      // Qwen M-RoPE with per-super-frame temporal indices — added by our patch
};
```

### Token container (`mtmd_image_tokens`, in `mtmd.cpp`)

After our patch:

```cpp
struct mtmd_image_tokens {
    uint32_t nx, ny;
    uint32_t nt = 1;                           // temporal positions (1 for images, >1 for video)
    mtmd_pos_type pos = MTMD_POS_TYPE_NORMAL;
    uint32_t image_idx = 0;                    // used by HunyuanVL
    std::vector<int32_t> temporal_positions;   // used by VIDEO
    uint32_t n_tokens() const {
        if (pos == MTMD_POS_TYPE_HUNYUANVL) return (nx + 1) * ny + 2;
        return nx * ny * nt;
    }
    clip_image_f32_batch batch_f32;
    std::string id;
};
```

### Per-token decoder positions (`mtmd_decoder_pos`, in `mtmd.h`)

```cpp
struct mtmd_decoder_pos {
    uint32_t t, x, y, z;  // z reserved
};

MTMD_API mtmd_decoder_pos mtmd_image_tokens_get_decoder_pos(
    const mtmd_image_tokens * image_tokens,
    llama_pos pos_0,
    size_t i);
```

One call per token `i ∈ [0, n_tokens)`. The switch on `image_tokens->pos` returns the right tuple for each position type. Our `MTMD_POS_TYPE_VIDEO` case reads `temporal_positions[i / (nx * ny)]` for the `t` coordinate.

Upstream deprecated `mtmd_image_tokens_get_nx()` / `get_ny()` in favor of the decoder-pos API; the deprecated functions still exist for now.

## Qwen3.5 vision path in `models/qwen3vl.cpp`

Qwen3-VL and Qwen3.5 share the `clip_graph_qwen3vl::build()` path. Our patch teaches it:

- Detect `batch_f32.is_video`.
- When true and the input has 6 channels, take the Conv3D decomposition branch: compute `Conv2D(frame_a, weight_t0) + Conv2D(frame_b, weight_t1)`.
- When false (images), continue down the existing `Conv2D(frame, weight_t0)` path unchanged.

`clip_n_output_tokens_x` / `y` report the per-frame token grid; `mtmd_tokenize_video()` uses them to populate `image_tokens->nx` and `ny`.

## HTTP request shape that reaches the server

The patched `tools/server/server-common.cpp` / `server-context.cpp` accept a standard OpenAI `/v1/chat/completions` request plus an optional `mm_processor_kwargs` object:

```json
{
  "messages": [{
    "role": "user",
    "content": [
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
      "... (one image_url per source frame)",
      {"type": "text", "text": "Describe what happens in this video."}
    ]
  }],
  "mm_processor_kwargs": {
    "fps": 2.0,
    "is_video": true,
    "grid_thw": [T, H_grid, W_grid],
    "temporal_positions": [...]
  },
  "max_tokens": 2048,
  "temperature": 1.0
}
```

Key point: **the patch does not introduce a new content type**. There is no `video_url` or `video_frames`. Video is signaled by `mm_processor_kwargs.is_video`; frames travel as standard `image_url` entries (which means any OpenAI-compatible client already knows how to send them). The server's job is to pair the frames, set `image_tokens->pos = MTMD_POS_TYPE_VIDEO`, populate `nt` and `temporal_positions`, and hand off to the mtmd decode path.

## Build

```bash
cd llama.cpp
# Release build (what you want for inference)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build build --config Release -j$(nproc)

# Debug build (for GDB)
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DGGML_CUDA=ON
cmake --build build --config Debug -j$(nproc)
```

Binaries of interest:

- `build/bin/llama-server` — HTTP server
- `build/bin/llama-mtmd-cli` — multimodal CLI (useful for out-of-server testing)

## Upstream video support status

Upstream tracks video support in issue `#18389`. The issue's Phase 1 target is frame-by-frame input (SmolVLM2-Video etc.); Qwen3.5's fused-frame Conv3D path is explicitly called out as requiring additional work. Our patch fills that specific gap and can be dropped when upstream catches up — the public `llama_video` Python surface is designed to survive that transition.
