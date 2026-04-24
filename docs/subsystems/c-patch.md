# C Patch Subsystem

> **Code:** Changes in `llama.cpp/tools/mtmd/` and `llama.cpp/tools/server/`
> **Patches:** `patches/` directory
> **Reference:** `docs/references/llamacpp-multimodal-internals.md`
> **Last verified:** 2026-04-24 (against llama.cpp master commit `0adede8`)

## Purpose

Minimal modifications to llama.cpp's multimodal pipeline to accept video frame sequences with proper temporal encoding for Qwen3.5 models.

## Patch File

A single unified patch lives in `patches/`:

| File | Target files | Summary |
|------|--------------|---------|
| `video-support-YYYYMMDD.patch` | `tools/mtmd/{clip-impl.h, clip.cpp, clip.h, models/qwen3vl.cpp, mtmd-helper.cpp, mtmd.cpp, mtmd.h}`, `tools/server/{server-common.cpp, server-common.h, server-context.cpp}` | 6-channel super-frame input (Conv3D), `MTMD_INPUT_CHUNK_TYPE_VIDEO`, temporal M-RoPE via `MTMD_POS_TYPE_VIDEO`, `mtmd_tokenize_video()` API, server `mm_processor_kwargs` passthrough |

The date suffix tracks the upstream rebase target; extract overwrites it with today's date.

## Applying the Patch

```bash
# Preferred (shim on PATH after `uv sync`):
llama-video-patch /path/to/llama.cpp

# Or the script directly:
./scripts/apply-patches.sh /path/to/llama.cpp

# Or manually:
cd /path/to/llama.cpp
git apply /path/to/llama-video/patches/video-support-YYYYMMDD.patch
```

## Extracting the Patch

After modifying llama.cpp source:
```bash
./scripts/extract-patches.sh
# Writes patches/video-support-$(date +%Y%m%d).patch from `git diff`
```

## Patch Design Principles

1. **Minimal diff** — only touch lines that must change
2. **No behavior change for images** — existing image path is selected by `pos == MTMD_POS_TYPE_MROPE`; video uses a sibling `MTMD_POS_TYPE_VIDEO` branch
3. **Reuse upstream abstractions** — video tokens flow through the same `mtmd_image_tokens_get_decoder_pos()` switch as images and HunyuanVL, not a parallel code path
4. **Debug logging** — `mtmd_tokenize_video` emits `nx/ny/nt/n_tokens` at DBG level
5. **GGML_ASSERT** — frame-pair dimensions and temporal-position counts checked at tokenize time

## How the Video Path Hooks In

Reading the rebased patch as an architectural overview:

1. `clip-impl.h` / `clip.cpp`: `clip_image_f32_batch` gains an `is_video` flag; the Qwen3VL model path in `qwen3vl.cpp` routes 6-channel super-frames through the Conv3D decomposition.
2. `mtmd.h`: adds `MTMD_INPUT_CHUNK_TYPE_VIDEO` enum value, `mtmd_image_tokens_get_nt()` + `mtmd_image_tokens_get_temporal_positions()` accessors, and the `mtmd_tokenize_video()` entry point.
3. `mtmd.cpp`:
   - Adds `MTMD_POS_TYPE_VIDEO` to the `mtmd_pos_type` enum alongside `MROPE` / `NORMAL` / `HUNYUANVL`.
   - Adds `nt` and `temporal_positions` fields to `struct mtmd_image_tokens`.
   - Extends `mtmd_image_tokens_get_decoder_pos()` with a VIDEO case that returns per-super-frame temporal indices plus spatial (x, y) inside each frame.
   - Extends `mtmd_image_tokens_get_n_pos()` with a VIDEO case returning `max({nt, nx, ny})`.
4. `mtmd-helper.cpp`: widens the M-RoPE decode branch to accept both IMAGE and VIDEO chunks — both flow through `mtmd_helper_image_get_decoder_pos` → `set_position_mrope_2d(rel_pos, seq_id)` unchanged.
5. `server-*.{cpp,h}`: passes `mm_processor_kwargs` (fps, temporal params) through the `/chat/completions` endpoint to the tokenizer.

## Testing the Patch

```bash
# 1. Build (full)
./scripts/build.sh

# 2. Image regression (must still work)
./llama.cpp/build/bin/llama-mtmd-cli \
    -m model.gguf --mmproj mmproj.gguf \
    --image test.jpg -p "Describe this image"

# 3. Video smoke test (Python harness against llama-server):
uv run pytest tests/smoke/test_video_caption.py -v

# 4. Confirm the patch is active on the running server
uv run llama-video-debug validate-patch --server-url http://localhost:8080
```

## Rebasing on Upstream

```bash
cd llama.cpp
git fetch origin
git stash push -m "video-support (pre-rebase)"
git checkout origin/master
git apply --3way /path/to/llama-video/patches/video-support-YYYYMMDD.patch
# Resolve any conflicts, rebuild, retest
cd ..
./scripts/extract-patches.sh   # overwrites the dated patch
```

`git apply --3way` is the useful probe — it cleanly applies hunks whose surrounding context merely shifted and only surfaces the hunks that genuinely collide with upstream refactors. See the commit history for worked rebase examples.
