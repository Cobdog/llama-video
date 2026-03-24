"""Batch captioning API.

Captions multiple files sequentially using the same settings.
Supports video and image modes with auto-detection from file extensions.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from llama_video.types import CaptionMetadata, CaptionResult

logger = logging.getLogger(__name__)

_VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".avi", ".mkv"}
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def detect_mode(path: str) -> str:
    """Detect captioning mode from file extension.

    Returns:
        "video" or "image".

    Raises:
        ValueError: If extension is not recognized.
    """
    ext = Path(path).suffix.lower()
    if ext in _VIDEO_EXTENSIONS:
        return "video"
    if ext in _IMAGE_EXTENSIONS:
        return "image"
    raise ValueError(f"Unsupported file extension: {ext!r}")


def validate_batch_mode(paths: list[str], mode: str) -> str:
    """Validate and resolve batch mode.

    Args:
        paths: List of file paths.
        mode: "video", "image", or "auto".

    Returns:
        Resolved mode ("video" or "image").

    Raises:
        ValueError: If paths is empty or contains mixed types in auto mode.
    """
    if not paths:
        raise ValueError("Batch is empty — no files to caption")

    if mode in ("video", "image"):
        return mode

    # Auto-detect from extensions
    modes = {detect_mode(p) for p in paths}
    if len(modes) > 1:
        raise ValueError(
            "Batch contains mixed file types (video and image). "
            "Use mode='video' or mode='image' to process one type at a time."
        )
    return modes.pop()


async def batch_caption(
    paths: list[str],
    mode: str = "auto",
    prompt: str = "Describe what happens in this {media_type}.",
    variables: dict[str, str] | None = None,
    preset: str = "default",
    fps: float = 2.0,
    max_frames: int = 64,
    max_tokens: int = 2048,
    on_progress: Callable[[int, int, str, CaptionResult | None], Any] | None = None,
) -> list[CaptionResult]:
    """Caption multiple files sequentially.

    Args:
        paths: List of file paths to caption.
        mode: "video", "image", or "auto" (detect from extensions).
        prompt: Prompt string (may contain {variables}).
        variables: Variable substitutions for the prompt.
        preset: Inference preset name.
        fps: Frame extraction FPS (video only).
        max_frames: Maximum frames to extract (video only).
        max_tokens: Maximum generation tokens.
        on_progress: Callback(index, total, path, result) after each file.

    Returns:
        List of CaptionResult objects.

    Raises:
        ValueError: If paths is empty or contains mixed types in auto mode.
    """
    from llama_video.client import LlamaServerClient
    from llama_video.config import Settings, get_preset
    from llama_video.extractor import Extractor, ExtractorConfig
    from llama_video.preprocessor import Preprocessor

    resolved_mode = validate_batch_mode(paths, mode)
    resolved_vars = variables or {}
    media_type = resolved_mode

    # Render prompt
    rendered_prompt = prompt.format(media_type=media_type, **resolved_vars)

    settings = Settings()
    preset_obj = get_preset(preset)
    client = LlamaServerClient(settings.server)

    results: list[CaptionResult] = []

    try:
        for i, path in enumerate(paths):
            start = time.monotonic()

            if resolved_mode == "video":
                extractor = Extractor(settings.extractor)
                preprocessor = Preprocessor(settings.model)
                frames = await extractor.extract_frames_async(
                    path, ExtractorConfig(fps=fps, max_frames=max_frames)
                )
                video_input = preprocessor.process(frames, fps=fps)
                caption = await client.caption_video(
                    video_input,
                    prompt=rendered_prompt,
                    max_tokens=max_tokens,
                    preset=preset_obj,
                )
                metadata = CaptionMetadata(
                    frames_extracted=len(frames),
                    super_frames=len(video_input.super_frames),
                    grid_thw=video_input.grid_thw,
                    processing_time_ms=(time.monotonic() - start) * 1000,
                )
            else:
                caption = await client.caption_image(
                    path,
                    prompt=rendered_prompt,
                    max_tokens=max_tokens,
                    preset=preset_obj,
                )
                metadata = CaptionMetadata(
                    frames_extracted=1,
                    super_frames=0,
                    grid_thw=(0, 0, 0),
                    processing_time_ms=(time.monotonic() - start) * 1000,
                )

            duration_ms = (time.monotonic() - start) * 1000

            result = CaptionResult(
                caption=caption,
                source_path=path,
                mode=resolved_mode,
                prompt_rendered=rendered_prompt,
                template_name=None,
                variables=resolved_vars,
                preset_name=preset,
                settings={
                    "fps": fps,
                    "max_frames": max_frames,
                    "max_tokens": max_tokens,
                },
                metadata=metadata,
                token_usage=None,
                duration_ms=duration_ms,
            )
            results.append(result)

            logger.info(
                "Batch [%d/%d] %s: %d chars in %.1fs",
                i + 1,
                len(paths),
                path,
                len(caption),
                duration_ms / 1000,
            )

            if on_progress is not None:
                on_progress(i, len(paths), path, result)

    finally:
        await client.close()

    return results
