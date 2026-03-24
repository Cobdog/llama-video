# C Patch Subsystem

> **Code:** Changes in `llama.cpp/tools/mtmd/` (clip.cpp, mtmd.cpp, mtmd.h)
> **Patches:** `patches/` directory
> **Reference:** `docs/references/llamacpp-multimodal-internals.md`
> **Last verified:** 2026-03-23

## Purpose

Minimal modifications to llama.cpp's multimodal pipeline to accept video frame sequences with proper temporal encoding for Qwen3.5 models.

## Patch Files

| Patch | Target | Description |
|-------|--------|-------------|
| `01-mtmd-video-input.patch` | mtmd.h, mtmd.cpp | Video input struct + API |
| `02-clip-video-conv3d.patch` | clip.cpp | Super-frame → Conv3D path |
| `03-mtmd-temporal-mrope.patch` | mtmd.cpp | Temporal M-RoPE positions |
| `04-server-video-url.patch` | server.cpp | Video URL in chat completions |

Patches are numbered for application order. Each is independently testable.

## Applying Patches

```bash
cd llama.cpp
git apply ../patches/01-mtmd-video-input.patch
git apply ../patches/02-clip-video-conv3d.patch
# ... etc

# Or all at once:
../scripts/apply-patches.sh
```

## Extracting Patches

After modifying llama.cpp source:
```bash
../scripts/extract-patches.sh
# Creates/updates patches/ from diff against upstream HEAD
```

## Patch Design Principles

1. **Minimal diff** — only touch lines that must change
2. **No behavior change for images** — existing image path must be completely unaffected
3. **Feature-flagged** — video path activated only when `is_video=true`
4. **Debug logging** — every new code path logs tensor shapes at DBG level
5. **GGML_ASSERT** — invariants checked at every stage

## Testing the Patch

```bash
# 1. Build
cd llama.cpp && cmake -B build -DCMAKE_BUILD_TYPE=Debug && cmake --build build -j$(nproc)

# 2. Test image regression (must still work)
./build/bin/llama-mtmd-cli -m model.gguf --mmproj mmproj.gguf --image test.jpg -p "Describe this image"

# 3. Test video (our new capability)
# Currently via the Python test suite:
cd .. && uv run pytest tests/smoke/test_video_caption.py -v

# 4. Validate temporal encoding
uv run llama-video-debug compare-modes test_video.mp4
```

## Rebasing on Upstream

When llama.cpp updates:
```bash
cd llama.cpp
git fetch origin
git stash  # Stash any uncommitted changes
git rebase origin/master
# Resolve conflicts in our patched files
# Re-test everything
cd .. && ./scripts/extract-patches.sh  # Re-extract clean patches
```
