"""Exception hierarchy for llama-video.

Every exception includes diagnostic context for debugging.
"""

from __future__ import annotations

from typing import Any


class LlamaVideoError(Exception):
    """Base exception for all llama-video errors."""

    def __init__(self, message: str, context: dict[str, Any] | None = None) -> None:
        self.context = context or {}
        super().__init__(message)


# --- Frame Extraction Errors ---


class ExtractionError(LlamaVideoError):
    """Base for frame extraction errors."""


class FFmpegNotFoundError(ExtractionError):
    """ffmpeg binary not found in PATH or configured location."""


class VideoNotFoundError(ExtractionError):
    """Video file does not exist or is not readable."""


class VideoDecodeError(ExtractionError):
    """ffmpeg failed to decode the video file."""


class NoFramesError(ExtractionError):
    """Zero frames were extracted from the video."""


class ExtractionTimeoutError(ExtractionError):
    """ffmpeg process timed out."""


# --- Preprocessing Errors ---


class PreprocessingError(LlamaVideoError):
    """Base for video preprocessing errors."""


class InvalidFrameDimensionsError(PreprocessingError):
    """Frames have inconsistent or invalid dimensions."""


class ResolutionError(PreprocessingError):
    """Frame resolution outside model's supported range."""


# --- Server/Client Errors ---


class ClientError(LlamaVideoError):
    """Base for llama-server client errors."""


class ServerUnreachableError(ClientError):
    """Cannot connect to llama-server."""


class ServerResponseError(ClientError):
    """llama-server returned an error response."""


class ModelNotLoadedError(ClientError):
    """llama-server is running but no model is loaded."""


# --- Patch Validation Errors ---


class PatchError(LlamaVideoError):
    """Base for C patch related errors."""


class PatchNotAppliedError(PatchError):
    """llama-server is running but the video patch is not active."""


class TemporalEncodingError(PatchError):
    """Temporal M-RoPE encoding is not working correctly."""
