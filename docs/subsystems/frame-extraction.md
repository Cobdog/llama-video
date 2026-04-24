# Frame Extraction Subsystem

> **Code:** `src/llama_video/extractor.py`
> **Tests:** `tests/unit/test_extractor.py`, `tests/integration/test_extractor_real_video.py`
> **Last verified:** 2026-04-24 (against `src/` at HEAD)

## Purpose

Extract video frames by spawning ffmpeg as a subprocess and reading raw `rgb24` pixels over a pipe. Output is a `list[Frame]` ready for the preprocessor.

## Key classes

### `Extractor`

```python
Extractor(settings: ExtractorSettings | None = None)
```

- Constructor takes `ExtractorSettings` (env-configured defaults: `ffmpeg_path`, `default_fps`, `max_frames`, `extraction_timeout`). If `None`, constructs from env vars.
- **Per-call parameters** (fps, max_frames, strategy) go in `ExtractorConfig`, passed to `extract_frames` / `extract_frames_async`.

Methods:

| Method | Signature | Notes |
|--------|-----------|-------|
| `extract_frames` | `(video_path: str \| Path, config: ExtractorConfig \| None = None) -> list[Frame]` | Synchronous. |
| `extract_frames_async` | `(video_path: str \| Path, config: ExtractorConfig \| None = None) -> list[Frame]` | Preferred from async code. |
| `ffmpeg_path` | property | Resolves and caches the ffmpeg binary path. |

### `ExtractorConfig`

Per-call extraction knobs:

```python
ExtractorConfig(
    fps: float = 2.0,
    max_frames: int = 64,
    min_frames: int = 1,
    strategy: SamplingStrategy = SamplingStrategy.UNIFORM,
)
```

That's the complete field list — `__slots__ = ("fps", "max_frames", "min_frames", "strategy")`. There is no `output_format` or `ffmpeg_path` here; ffmpeg path resolution is on `Extractor` via `ExtractorSettings`.

### `ExtractorSettings`

Env-configured defaults (prefix `LLAMA_VIDEO_`):

```python
ExtractorSettings(
    ffmpeg_path: str = "ffmpeg",
    default_fps: float = 2.0,
    max_frames: int = 64,
    extraction_timeout: float = 60.0,
)
```

### `Frame`

Single extracted frame. `__slots__ = ("data", "height", "index", "timestamp", "width")`.

| Attribute | Type | Meaning |
|-----------|------|---------|
| `data` | `np.ndarray` | `(H, W, 3)` uint8 RGB |
| `index` | `int` | Frame number in source |
| `timestamp` | `float` | Seconds into the source |
| `width`, `height` | `int` | Pixel dims |
| `size` | `tuple[int, int]` | `(width, height)` (property) |

---

## Sampling strategies

`SamplingStrategy` (a `StrEnum` in `types.py`) declares three values:

| Strategy | Status |
|----------|--------|
| `UNIFORM` | **Implemented** — uses ffmpeg `-vf fps=N`. |
| `KEYFRAME` | Declared; not yet wired through ffmpeg. Passing it has no effect beyond the default uniform sampling. |
| `SCENE_CHANGE` | Declared; not yet wired through ffmpeg. |

If you pick a non-UNIFORM strategy today, the extractor still runs uniform sampling. See [`docs/ROADMAP.md`](../ROADMAP.md) for planned work.

---

## ffmpeg command (uniform sampling)

The actual invocation (see `extractor.py::extract_frames_async`):

```
ffmpeg -i <video> -vf fps=<fps> -frames:v <max_frames> \
       -f rawvideo -pix_fmt rgb24 -v error \
       pipe:1
```

Plus a preceding `ffprobe` call for width/height/duration metadata.

Frames stream out `pipe:1`; the extractor reads `height × width × 3` bytes per frame until EOF or `max_frames` is hit. Expanding the strategy field means teaching the extractor to pick a different `-vf` filter graph (e.g., `select='eq(pict_type,I)'` for `KEYFRAME`), not a config flip.

---

## Error handling

All exceptions inherit from `llama_video.errors.ExtractionError` (which is an `LlamaVideoError`).

| Error | Raised when | Recovery |
|-------|-------------|----------|
| `FFmpegNotFoundError` | ffmpeg binary not found on PATH or at `LLAMA_VIDEO_FFMPEG_PATH` | Install ffmpeg, or fix the env var |
| `VideoNotFoundError` | Video path doesn't exist | Check the path |
| `VideoDecodeError` | ffmpeg failed to open/decode the input | Inspect ffmpeg stderr (captured into `context`); re-encode if codec is exotic |
| `NoFramesError` | ffmpeg produced zero complete frames | Raise FPS or use a longer clip |
| `ExtractionTimeoutError` | `extraction_timeout` seconds elapsed before ffmpeg finished | Raise `LLAMA_VIDEO_EXTRACTION_TIMEOUT`, reduce `max_frames` |
| `InvalidFrameDimensionsError` | Extracted frame bytes don't match `(H × W × 3)` | Usually a partial read at EOF — file an issue with reproducer |

Every exception carries a `context: dict` with diagnostic fields (e.g., `ffmpeg_stderr`, `video_path`, `expected_bytes`).
