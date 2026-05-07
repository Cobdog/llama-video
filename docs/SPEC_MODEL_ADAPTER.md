# Spec: Model Adapter Architecture

## Objective

Decompose llama-video's hardcoded Qwen3.5 video pipeline into a `ModelAdapter` interface so that different model families (Qwen3.5, Gemma4, future models) can each define their own frame extraction, preprocessing, payload construction, and sampler settings.

**Why now:** Issue #1 requests Gemma4 support. Gemma4 processes video fundamentally differently from Qwen3.5 (single frames at 1 FPS with timestamps, no superframes, no C patch). The current codebase bakes Qwen3.5 assumptions into every module. Adding Gemma4 without an adapter boundary would create a tangled mess of if/else branches.

**Who is the user:** Developers integrating llama-video with different model families served by llama.cpp (or compatible servers). End users selecting model profiles through the API or WebUI.

**Success looks like:**
- All existing Qwen3.5 tests pass unchanged (zero regressions)
- A new `GemmaAdapter` can be registered and selected via `model_profile`
- Adding a future model family requires implementing one class, no core changes
- The public API surface (`/v1/caption`, CLI, Python API) is backward-compatible
- The WebUI allows manual profile selection OR auto-detection from llama-server
- Each adapter has fully isolated sampler and token budget settings

## Tech Stack

- Python 3.12+ (already established)
- Pydantic for config/schema validation
- NumPy for frame manipulation
- httpx for server communication
- pytest + pytest-asyncio for testing

## Commands

```bash
# Install
uv sync --dev

# Lint
uv run ruff check .
uv run ruff format --check .

# Typecheck
uv run mypy src/

# Unit tests
uv run pytest tests/unit/ -v --tb=short --cov=llama_video --cov-report=term-missing

# Integration tests (requires ffmpeg + test videos)
uv run pytest tests/integration/ -v --tb=short

# All tests
uv run pytest tests/unit/ tests/integration/ -v --tb=short
```

## Project Structure

```
src/llama_video/
  adapters/               → NEW: adapter package
    __init__.py           → Registry + get_adapter() factory + auto-detect
    base.py               → Abstract ModelAdapter class + AdapterConfig
    qwen.py               → Qwen3.5 adapter (extracted from current code)
    gemma.py              → Gemma4 adapter (new)
  config.py               → Modified: ModelConfig becomes family-aware
  preprocessor.py         → Simplified: delegates to adapter
  client.py               → Simplified: delegates payload to adapter
  extractor.py            → Unchanged (model-agnostic)
  image.py                → Unchanged (model-agnostic)
  tokens.py               → Modified: per-adapter token estimation
  types.py                → Modified: family-agnostic types
  server.py               → Minimal change: adapter selection in pipeline
  webui.py                → Modified: profile selector + auto-detect toggle
  ...

tests/
  unit/
    test_adapters.py      → NEW: adapter unit tests
    test_adapter_registry.py → NEW: registry and factory tests
    test_qwen_adapter.py  → NEW: extracted Qwen adapter tests
    test_gemma_adapter.py → NEW: Gemma4 adapter tests
    test_video_segmenter.py → NEW: chunking/segmentation tests
    test_extractor.py     → Existing, should pass unchanged
    test_preprocessor.py  → Existing, should pass unchanged
    ...
  integration/
    test_extractor_real_video.py → Existing, should pass unchanged
    test_hf_reference.py         → Existing, should pass unchanged
    ...
```

## Code Style

Follow existing project conventions (ruff-formatted, type-annotated, minimal comments).

### Adapter interface

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from llama_video.types import Frame, VideoInput


@dataclass(frozen=True)
class AdapterConfig:
    """Base config for adapter-specific settings."""
    name: str


@dataclass(frozen=True)
class AdapterPreset:
    """Model-family-specific inference settings."""

    temperature: float
    top_p: float
    top_k: int
    min_p: float = 0.0
    presence_penalty: float = 0.0


class ModelAdapter(ABC):
    """Interface for model-family-specific video processing."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Adapter identifier (e.g. 'qwen3.5', 'gemma4')."""

    @property
    @abstractmethod
    def default_fps(self) -> float:
        """Recommended frame extraction FPS for this model family."""

    @property
    @abstractmethod
    def max_duration_seconds(self) -> float:
        """Max video duration this adapter supports in one chunk."""

    @property
    @abstractmethod
    def max_frames(self) -> int:
        """Maximum frames this adapter supports in one chunk."""

    @property
    @abstractmethod
    def default_preset(self) -> AdapterPreset:
        """Default inference preset for this model family."""

    @abstractmethod
    def preprocess(self, frames: list[Frame], fps: float) -> VideoInput:
        """Transform raw frames into model-specific VideoInput."""

    @abstractmethod
    def build_payload(
        self,
        video_input: VideoInput,
        prompt: str,
        preset: AdapterPreset | None = None,
        **kwargs,
    ) -> dict:
        """Build the OpenAI-compatible request payload for this model."""

    @abstractmethod
    def parse_response(self, raw: str) -> tuple[str, str, bool]:
        """Parse model response into (caption, thinking, truncated).

        Each model family uses different thinking tag formats.
        """

    @abstractmethod
    def estimate_tokens(self, video_input: VideoInput) -> int:
        """Estimate vision token consumption."""
```

### Adapter isolation principle

Each adapter encapsulates **all** model-family-specific behavior:

| Concern | Qwen3.5 | Gemma4 |
|---------|---------|--------|
| FPS | 2.0 (superframe pairing) | 1.0 (individual frames) |
| Max frames per chunk | 64 | 60 |
| Max duration | No hard limit | 60 seconds |
| Frame transform | 6-channel superframes (2-frame pairs) | Individual frames with MM:SS timestamps |
| Payload | `mm_processor_kwargs` with grid_thw, temporal_positions, is_video | Individual `image_url` entries with timestamp text between frames |
| Sampler defaults | temp=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5 | temp=1.0, top_p=0.95, top_k=64 |
| Thinking tags | `<think/>...</think/>` | `<|channel>thought\n...<channel|>` |
| Modality order | Text after frames (flexible) | **Images before text** (required for optimal performance) |
| Token estimation | T * H * W (grid formula) | frames * image_max_tokens (budget formula) |
| Token budget control | grid resolution (min/max pixels) | `image_max_tokens` (70/140/280/560/1120) |

Key conventions:
- Adapter methods are stateless pure functions where possible
- Each adapter owns its own config dataclass
- The registry maps profile names to adapter classes
- Unknown profile names raise a clear error listing available adapters
- Auto-detect probes `/v1/models` and matches name patterns

## Model Profile Selection

### Manual selection (API)

The existing `model_profile` field on `CaptionRequest` selects the adapter:

```json
{"model_profile": "gemma4", "video_path": "...", "prompt": "..."}
```

Default: `"qwen3.5"` (backward compatible).

### Manual selection (WebUI)

The WebUI presents a dropdown of registered adapters. Selection persists for the session.

### Auto-detection

When `model_profile` is set to `"auto"` (or left empty with auto-detect enabled), the system:

1. Queries llama-server's `/v1/models` endpoint
2. Matches the returned model name against adapter name patterns:
   - `"qwen"`, `"qwen2"`, `"qwen3"` → `qwen3.5`
   - `"gemma"`, `"gemma-4"`, `"gemma4"` → `gemma4`
3. Falls back to the default adapter if no match
4. Caches the result for the session (avoid probing on every request)

## Video Segmentation

Videos exceeding an adapter's `max_duration_seconds` are automatically segmented into chunks:

- **Chunk size:** User-configurable via `chunk_duration_seconds` parameter (max: adapter's max_duration). Default: adapter's max_duration.
- **Processing:** Each chunk is processed independently — extract frames, preprocess, caption.
- **Cohesion pass (open question):** After all chunks are captioned, an optional second pass could stitch the per-chunk captions into a single coherent narrative. This is a research item — may not be feasible with context limits or may require a separate summarization call.

```json
{
  "model_profile": "gemma4",
  "video_path": "/path/to/3min_video.mp4",
  "chunk_duration_seconds": 30,
  "prompt": "Describe what happens in this video."
}
```

The response includes per-chunk metadata plus the combined caption:

```json
{
  "caption": "...",
  "metadata": {
    "chunks": 6,
    "chunk_duration_seconds": 30,
    "frames_extracted": 180,
    "processing_time_ms": 45000
  }
}
```

## Testing Strategy

### Existing tests (must pass unchanged)

All tests in `tests/unit/` and `tests/integration/` that currently pass must continue to pass. This is the non-regression gate.

### New test layers

**Unit — adapter internals** (`tests/unit/test_qwen_adapter.py`, `test_gemma_adapter.py`):
- Frame-to-VideoInput transformation with known inputs
- Payload structure validation (correct mm_processor_kwargs shape for Qwen, correct image+timestamp layout for Gemma)
- Token estimation accuracy
- Preset defaults are correct per model family
- Thinking tag parsing (Qwen's `<think/>` vs Gemma's `<|channel>thought`)
- Edge cases: empty frames, single frame, max frames, odd frame counts

**Unit — registry** (`tests/unit/test_adapter_registry.py`):
- `get_adapter("qwen3.5")` returns QwenAdapter
- `get_adapter("gemma4")` returns GemmaAdapter
- `get_adapter("unknown")` raises informative error
- Default adapter selection when no profile specified
- Auto-detect: mock `/v1/models` responses, verify correct adapter selected

**Unit — segmentation** (`tests/unit/test_video_segmenter.py`):
- Video under max duration: no segmentation
- Video over max duration: correct chunk boundaries
- User-specified smaller chunk size
- Edge case: video duration exactly equals max
- Edge case: last chunk shorter than chunk size

**Integration — end-to-end with real ffmpeg** (`tests/integration/`):
- Existing integration tests implicitly test Qwen through the server
- New Gemma integration test: extract frames from test video, preprocess with GemmaAdapter, validate payload structure

### Coverage expectation

New adapter code should have ≥90% coverage. Existing code coverage must not decrease.

## Boundaries

### Always do
- Run the full test suite before committing
- Maintain backward compatibility with existing `model_profile="qwen3.5"` default
- Validate adapter selection at config load time (fail fast, not at inference time)
- Each adapter in its own file, no cross-imports between adapter implementations
- Keep the Extractor model-agnostic (it extracts raw frames, period)
- Keep sampler settings fully isolated per adapter
- Place images before text in Gemma payloads (model requirement)

### Ask first
- Changing the public API surface (CaptionRequest fields, endpoint signatures)
- Adding new dependencies
- Changing the server's lifespan or startup sequence
- Modifying the WebUI layout

### Never do
- Break existing integration test assertions
- Mix model-specific logic into extractor.py, image.py, or the error hierarchy
- Add adapter-specific fields to Frame or other model-agnostic types
- Commit without running lint + typecheck + unit tests
- Remove or weaken existing validation
- Use Qwen sampler defaults for Gemma or vice versa

## Success Criteria

1. **Zero regressions:** All existing unit and integration tests pass without modification
2. **Qwen parity:** The Qwen adapter produces identical VideoInput and payload output as the current hardcoded path for the same inputs
3. **Gemma4 functional:** A Gemma4 adapter correctly processes frames at 1 FPS, builds timestamped image payloads with images-before-text ordering, uses Gemma-specific sampler defaults, and parses `<|channel>thought` thinking tags
4. **Full isolation:** Each adapter has its own sampler preset, token estimation formula, and response parser. No cross-contamination.
5. **Clean boundary:** No model-specific imports in extractor.py, image.py, server.py (only in adapters/ and their config)
6. **Auto-detect:** WebUI can auto-select the correct adapter by probing llama-server's loaded model
7. **Segmentation:** Videos exceeding max_duration are automatically chunked and processed
8. **Extensible:** Adding a third adapter (e.g., InternVL) requires only a new file in `adapters/` and a registry entry — no changes to core modules
9. **Coverage:** New adapter code ≥90% line coverage
10. **CI green:** All lint, typecheck, and test jobs pass on GitHub Actions

## Open Questions

1. **Cohesion pass for multi-chunk videos:** After segmenting and captioning chunks independently, is there a viable way to stitch them into a single coherent narrative? Options:
   - A second summarization call that takes all chunk captions as input
   - Sliding-window overlap between chunks (last N seconds of chunk N prepended to chunk N+1)
   - Accept per-chunk captions as-is and let the user stitch manually
   *Research needed — this may be deferred to a follow-up.*

2. **Gemma4 thinking mode control:** Should the adapter always enable thinking (include `<|think|>` in system prompt), make it configurable, or disable by default? The E2B/E4B models always emit thought tags even when thinking is "disabled" (just empty). *Initial assumption: enable by default, make configurable via API/WebUI.*

3. **Auto-detect name patterns:** The current pattern matching (`"gemma" in name → gemma4`) is simplistic. Some GGUF files have names like `gemma-4-26b-it-Q8_0`. Should we maintain a configurable name→adapter mapping, or keep it pattern-based? *Initial assumption: pattern-based with clear documentation on expected naming conventions.*
