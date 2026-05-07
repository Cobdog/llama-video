# Task List: Model Adapter Architecture

Reference: `docs/SPEC_MODEL_ADAPTER.md`

---

## Phase 1: Foundation

### Task 1.1 — Create adapter package with abstract base
- **Acceptance:** `adapters/base.py` defines `AdapterPreset`, `ModelAdapter` ABC with all methods from spec. `adapters/__init__.py` exists and exports the base class.
- **Verify:** `python -c "from llama_video.adapters import ModelAdapter"` succeeds. `mypy src/` clean.
- **Files:** `src/llama_video/adapters/__init__.py`, `src/llama_video/adapters/base.py`

### Task 1.2 — Create adapter registry
- **Acceptance:** `adapters/registry.py` implements `register_adapter(name, cls)`, `get_adapter(name)`, `list_adapters()`. Unknown name raises `ValueError` listing available adapters. Default returns `"qwen3.5"`.
- **Verify:** Unit tests in `tests/unit/test_adapter_registry.py` — registration, lookup, error cases, default selection.
- **Files:** `src/llama_video/adapters/registry.py`, `tests/unit/test_adapter_registry.py`

---

## Phase 2: Qwen Adapter Extraction

### Task 2.1 — Implement QwenAdapter.preprocess
- **Acceptance:** `QwenAdapter.preprocess(frames, fps)` produces identical `VideoInput` (same superframes, grid_thw, temporal_positions) as `Preprocessor.process()` for the same inputs. Uses extracted code from `preprocessor.py`.
- **Verify:** Unit tests with known frame inputs comparing old Preprocessor output vs new QwenAdapter output — bit-exact match.
- **Files:** `src/llama_video/adapters/qwen.py`, `tests/unit/test_qwen_adapter.py`
- **Reuse:** `preprocessor.py:Preprocessor` (lines 59-317), `config.py:ModelConfig` (lines 51-79), `types.py:SuperFrame`, `types.py:OddFrameStrategy`

### Task 2.2 — Implement QwenAdapter.build_payload
- **Acceptance:** `QwenAdapter.build_payload(video_input, prompt, preset)` produces identical payload dict (same `messages`, `mm_processor_kwargs`, sampler params) as `LlamaServerClient._build_video_message()` + `caption_video()` for the same inputs.
- **Verify:** Unit tests comparing old client payload construction vs new adapter output — exact dict match.
- **Files:** `src/llama_video/adapters/qwen.py` (add method)
- **Reuse:** `client.py:_super_frame_to_base64_pair()` (lines 116-150), `client.py:_build_video_message()` (lines 152-188)

### Task 2.3 — Implement QwenAdapter.parse_response and estimate_tokens
- **Acceptance:** `parse_response` handles `<think/>...</think/>` tags identically to `client.py:parse_model_response()`. `estimate_tokens` uses T*H*W grid formula identical to `tokens.py:TokenEstimator.estimate()`.
- **Verify:** Unit tests with thinking/no-thinking/truncated response strings. Token estimation matches old estimator for known grid_thw values.
- **Files:** `src/llama_video/adapters/qwen.py` (add methods)
- **Reuse:** `client.py:parse_model_response()` (lines 44-65), `tokens.py:TokenEstimator` (lines 30-131)

### Task 2.4 — Register Qwen adapter and wire config
- **Acceptance:** `get_adapter("qwen3.5")` returns a `QwenAdapter` instance. `AdapterPreset` for Qwen has correct defaults (temp=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5). `CaptionRequest.model_profile` default still `"qwen3.5"`.
- **Verify:** Registry unit test. All 147 existing tests still pass.
- **Files:** `src/llama_video/adapters/qwen.py` (registration), `src/llama_video/config.py` (minimal changes if needed)

---

## Phase 3: Core Pipeline Delegation

### Task 3.1 — Generalize VideoInput type
- **Acceptance:** `VideoInput` in `types.py` (or adapters/base.py) is family-agnostic. Qwen's `super_frames`, `grid_thw`, `temporal_positions` become optional or move into a Qwen-specific subclass. New `VideoInput` carries the adapter name and generic frame count/metadata.
- **Verify:** Existing tests pass unchanged (Qwen code path creates Qwen-shaped VideoInput via adapter).
- **Files:** `src/llama_video/types.py`, `src/llama_video/adapters/base.py`

### Task 3.2 — Wire server.py pipeline through adapter
- **Acceptance:** `server.py:caption_video()` selects adapter via `get_adapter(request.model_profile)`, calls `adapter.preprocess()` instead of `_preprocessor.process()`, calls `adapter.build_payload()` instead of client internals. All 147 existing tests pass.
- **Verify:** Run full test suite. Manual test: `CaptionRequest(model_profile="qwen3.5")` produces same result as before.
- **Files:** `src/llama_video/server.py`, `src/llama_video/client.py` (simplified)

### Task 3.3 — Wire client.py to use adapter payload
- **Acceptance:** `LlamaServerClient.caption_video()` accepts adapter-built payload (or delegates to `adapter.build_payload()`). Superframe encoding moves into QwenAdapter. Client becomes model-agnostic HTTP transport.
- **Verify:** Existing tests pass. New unit test: mock adapter, verify client sends correct HTTP.
- **Files:** `src/llama_video/client.py`

### Task 3.4 — Wire tokens.py through adapter
- **Acceptance:** `TokenEstimator` delegates to `adapter.estimate_tokens()`. Qwen adapter uses T*H*W formula.
- **Verify:** Existing token estimation tests pass.
- **Files:** `src/llama_video/tokens.py`

### Checkpoint 3 — Full regression gate
- **Verify:** `uv run pytest tests/unit/ tests/integration/ --ignore=tests/smoke` — 147+ passed, 0 new failures. `ruff check`, `ruff format --check`, `mypy src/` all clean.

---

## Phase 4: Gemma4 Adapter

### Task 4.1 — Implement GemmaAdapter.preprocess
- **Acceptance:** `GemmaAdapter.preprocess(frames, fps=1.0)` returns VideoInput with individual frames (no pairing), timestamps in MM:SS format, fps=1.0, max 60 frames. Validates frame count ≤ 60.
- **Verify:** Unit tests: known frame inputs → correct VideoInput. Edge cases: single frame, 60 frames, >60 frames raises error.
- **Files:** `src/llama_video/adapters/gemma.py`, `tests/unit/test_gemma_adapter.py`

### Task 4.2 — Implement GemmaAdapter.build_payload
- **Acceptance:** Payload has frames as individual `image_url` entries with `MM:SS` timestamp text between them. Images appear BEFORE text prompt (modality order requirement). Uses Gemma sampler defaults (temp=1.0, top_p=0.95, top_k=64). No `mm_processor_kwargs` — native llama.cpp handles Gemma.
- **Verify:** Unit tests validate payload structure, image-before-text ordering, correct sampler values.
- **Files:** `src/llama_video/adapters/gemma.py` (add method)

### Task 4.3 — Implement GemmaAdapter.parse_response and estimate_tokens
- **Acceptance:** `parse_response` handles `<|channel>thought\n...<channel|>` tags. `estimate_tokens` uses `frames * image_max_tokens` formula (default 280).
- **Verify:** Unit tests with Gemma thinking/no-thinking/truncated response strings.
- **Files:** `src/llama_video/adapters/gemma.py` (add methods)

### Task 4.4 — Register Gemma adapter and add integration test
- **Acceptance:** `get_adapter("gemma4")` returns GemmaAdapter. Integration test extracts frames from a test video, preprocesses with GemmaAdapter, sends to `MM-Sprinkle-Gemma4-31B-Q4` on port 7800, receives valid caption.
- **Verify:** Integration test passes against router. Unload model after test via `/v1/unload`.
- **Files:** `src/llama_video/adapters/gemma.py` (registration), `tests/integration/test_gemma_adapter.py`

---

## Phase 5: Video Segmentation

### Task 5.1 — Implement video segmenter
- **Acceptance:** `segment_video(duration, chunk_seconds, max_chunk_seconds)` returns list of `(start, end)` tuples. Last chunk may be shorter. Chunk size capped at adapter max_duration.
- **Verify:** Unit tests: under max (no segmentation), over max (correct chunks), exact boundary, small last chunk.
- **Files:** `src/llama_video/segmenter.py` (new), `tests/unit/test_video_segmenter.py`

### Task 5.2 — Wire segmentation into caption pipeline
- **Acceptance:** `server.py` detects when video duration exceeds adapter max, segments into chunks, processes each chunk independently, combines captions. `CaptionRequest` gains optional `chunk_duration_seconds` field. Response metadata includes chunk count.
- **Verify:** Integration test with a video >60s processed through Gemma4 adapter (multiple chunks).
- **Files:** `src/llama_video/server.py`, `src/llama_video/types.py` (CaptionRequest, CaptionMetadata)

---

## Phase 6: Auto-Detect + WebUI

### Task 6.1 — Implement auto-detect from llama-server
- **Acceptance:** `detect_adapter(server_url)` queries `/v1/models`, matches loaded model name against patterns, returns adapter name. Falls back to default. Caches result.
- **Verify:** Unit tests with mocked `/v1/models` responses: `"gemma-4-31b-it-Q4"` → `"gemma4"`, `"qwen3.5-35b"` → `"qwen3.5"`, no match → default.
- **Files:** `src/llama_video/adapters/detect.py` (new), `tests/unit/test_adapter_detect.py`

### Task 6.2 — Wire auto-detect into server and WebUI
- **Acceptance:** `CaptionRequest.model_profile="auto"` triggers auto-detect. WebUI shows dropdown of registered adapters plus "Auto-detect" option.
- **Verify:** Manual test in WebUI. Unit test for `"auto"` profile routing.
- **Files:** `src/llama_video/server.py`, `src/llama_video/webui.py`, `src/llama_video/types.py`

---

## Phase 7: Final Verification

### Task 7.1 — Full regression + CI
- **Acceptance:** All existing tests pass. New adapter tests ≥90% coverage. `ruff check`, `ruff format --check`, `mypy src/` all clean. CI green on GitHub Actions.
- **Verify:** `uv run pytest tests/unit/ tests/integration/ --ignore=tests/smoke -v`. Push to branch, confirm CI passes.
- **Files:** All

### Task 7.2 — Update docs and API description
- **Acceptance:** `README.md` mentions multi-model support. `ROADMAP.md` marks adapter item complete. Server description updated. `SPEC_MODEL_ADAPTER.md` finalized.
- **Verify:** Review docs for accuracy.
- **Files:** `README.md`, `docs/ROADMAP.md`, `src/llama_video/server.py` (description)
