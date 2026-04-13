# Video Support Patches for llama.cpp

Adds temporal video support to llama.cpp's multimodal pipeline (super-frames, temporal M-RoPE, server video mode).

## Current Patch

| File | Description |
|------|-------------|
| `video-support-20260413.patch` | Unified patch: 6-channel super-frame input, temporal M-RoPE positions, video chunk type, server `mm_processor_kwargs` passthrough |

**Target files:** `tools/mtmd/clip-impl.h`, `tools/mtmd/clip.cpp`, `tools/mtmd/clip.h`, `tools/mtmd/models/qwen3vl.cpp`, `tools/mtmd/mtmd-helper.cpp`, `tools/mtmd/mtmd.cpp`, `tools/mtmd/mtmd.h`, `tools/server/server-common.cpp`, `tools/server/server-common.h`, `tools/server/server-context.cpp`

## How to Apply

Preferred (from anywhere):

```bash
llama-video-patch /path/to/llama.cpp
```

Or with the script (from project root):

```bash
./scripts/apply-patches.sh /path/to/llama.cpp
```

Or manually:

```bash
cd /path/to/llama.cpp
git apply /path/to/llama-video/patches/video-support-20260413.patch
```

## How to Extract

After modifying llama.cpp source:

```bash
./scripts/extract-patches.sh
```

## Rebasing on Upstream

```bash
cd llama.cpp
git stash
git pull --rebase
git stash pop
# Resolve any conflicts, then:
cd ..
./scripts/extract-patches.sh
```
