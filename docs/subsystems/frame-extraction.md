# Frame Extraction Subsystem

> **Code:** `src/llama_video/extractor.py`
> **Tests:** `tests/unit/test_extractor.py`, `tests/integration/test_extractor_real_video.py`
> **Last verified:** 2026-03-23

## Purpose

Extract video frames using ffmpeg subprocess, with configurable FPS, frame limits, and sampling strategies.

## Key Classes

### `Extractor`
- `extract_frames(video_path, config?) → list[Frame]`
- `extract_frames_async(video_path, config?) → list[Frame]`

### `ExtractorConfig`
- `fps: float = 2.0` — frames per second to extract
- `max_frames: int = 64` — maximum frames (prevents OOM)
- `min_frames: int = 1` — minimum frames (short clips)
- `strategy: SamplingStrategy = UNIFORM` — frame selection strategy
- `output_format: str = "rgb"` — raw RGB or JPEG
- `ffmpeg_path: str = "ffmpeg"` — path to ffmpeg binary

### `Frame`
- `data: np.ndarray` — RGB pixel data (H, W, 3)
- `index: int` — frame number in source video
- `timestamp: float` — seconds into video
- `size: tuple[int, int]` — (width, height)

## Sampling Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| `UNIFORM` | Even spacing at specified FPS | General use |
| `KEYFRAME` | Prefer I-frames, fill with P-frames | Action scenes |
| `SCENE_CHANGE` | Detect scene changes, sample around them | Montages, cuts |

## ffmpeg Commands

```bash
# Uniform sampling at 2fps
ffmpeg -i input.mp4 -vf "fps=2" -f rawvideo -pix_fmt rgb24 pipe:1

# With frame limit
ffmpeg -i input.mp4 -vf "fps=2" -frames:v 64 -f rawvideo -pix_fmt rgb24 pipe:1

# Keyframe extraction
ffmpeg -i input.mp4 -vf "select='eq(pict_type,I)'" -vsync vfr -f rawvideo -pix_fmt rgb24 pipe:1
```

## Error Handling

| Error | Exception | Recovery |
|-------|-----------|----------|
| ffmpeg not found | `FFmpegNotFoundError` | Check FFMPEG_PATH env var |
| Video file not found | `VideoNotFoundError` | — |
| Corrupt/unreadable video | `VideoDecodeError` | — |
| 0 frames extracted | `NoFramesError` | Lower FPS or check video |
| ffmpeg timeout | `ExtractionTimeoutError` | Increase timeout or reduce max_frames |
