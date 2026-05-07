# Implementation Plan: Model Adapter Architecture

**Spec:** `docs/SPEC_MODEL_ADAPTER.md`
**Test server:** llama-server router on port 7800 (stock build, no superframe patch)

## Context

The codebase hardcodes Qwen3.5 assumptions into every module: `preprocessor.py` builds 6-channel superframes, `client.py` sends `mm_processor_kwargs` with `grid_thw`/`temporal_positions`, `config.py` has Qwen-specific defaults, `tokens.py` uses Qwen's grid formula. Issue #1 requests Gemma4 support which uses an entirely different pipeline (1 FPS individual frames, timestamp-based, native llama.cpp support). The adapter pattern isolates these differences behind a common interface.

## Dependency Graph

```
types.py (Frame, VideoInput, SuperFrame, CaptionRequest, etc.)
  ↑
config.py (ModelConfig, InferencePreset, Settings)
  ↑
extractor.py (Extractor → Frame[])          ← model-agnostic, STAYS
  ↑
adapters/base.py (ModelAdapter ABC)
  ↑              ↑
adapters/qwen.py   adapters/gemma.py
  ↑                 ↑
preprocessor.py    (delegates to adapter)
  ↑
client.py          (delegates payload to adapter)
  ↑
tokens.py          (delegates to adapter)
  ↑
server.py          (selects adapter, wires pipeline)
  ↑
webui.py           (profile selector + auto-detect)
```

## Phases

### Phase 1: Foundation (adapters/base.py + registry)

Create the adapter package with the abstract interface and registry. No behavior change yet — existing code untouched. Tests validate the interface contract.

### Phase 2: Qwen Adapter Extraction

Extract current Qwen3.5 behavior into `QwenAdapter`. Wire it behind the registry so `get_adapter("qwen3.5")` returns it. Run full existing test suite to verify zero regressions. This is the riskiest phase — any behavioral drift here breaks everything.

### Phase 3: Core Pipeline Delegation

Wire `preprocessor.py`, `client.py`, and `tokens.py` to delegate to the adapter. The server pipeline (`server.py`, `batch.py`) calls through the adapter instead of directly calling Preprocessor/Client internals. Existing tests still pass because QwenAdapter reproduces exact same behavior.

### Phase 4: Gemma4 Adapter

Implement `GemmaAdapter` with 1 FPS frames, timestamp payloads, Gemma-specific sampler, and `<|channel>thought` parsing. Unit tests + integration test against router on port 7800.

### Phase 5: Video Segmentation

Add chunking logic for videos exceeding adapter max_duration. Segmented processing with per-chunk metadata.

### Phase 6: Auto-Detect + WebUI

Profile auto-detection from llama-server's `/v1/models` + WebUI dropdown.

### Phase 7: Final Verification

Full regression suite, CI green, spec review.

## Checkpoint Rules

After each phase:
1. Run `uv run pytest tests/unit/ tests/integration/ --ignore=tests/smoke -q` — must see 147+ passed, 0 new failures
2. Run `uv run ruff check . && uv run ruff format --check .` — clean
3. Run `uv run mypy src/` — clean
4. Commit with descriptive message referencing spec section
