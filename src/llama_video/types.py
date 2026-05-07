"""Core data types for llama-video."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class SamplingStrategy(StrEnum):
    """Frame sampling strategy for video extraction."""

    UNIFORM = "uniform"
    KEYFRAME = "keyframe"
    SCENE_CHANGE = "scene_change"


class OddFrameStrategy(StrEnum):
    """How to handle odd frame counts when pairing."""

    PAD = "pad"
    DROP = "drop"


class Frame:
    """A single extracted video frame.

    Not a Pydantic model because it holds numpy data.
    """

    __slots__ = ("data", "height", "index", "timestamp", "width")

    def __init__(
        self,
        data: np.ndarray,
        index: int,
        timestamp: float,
        width: int,
        height: int,
    ) -> None:
        self.data = data  # (H, W, 3) uint8 RGB
        self.index = index
        self.timestamp = timestamp
        self.width = width
        self.height = height

    @property
    def size(self) -> tuple[int, int]:
        return (self.width, self.height)

    def __repr__(self) -> str:
        return f"Frame(index={self.index}, timestamp={self.timestamp:.2f}s, size={self.size})"


class SuperFrame:
    """A paired video super-frame (2 consecutive frames merged).

    Channel layout: [R1, G1, B1, R2, G2, B2] — 6 channels total.
    """

    __slots__ = ("data", "source_frames", "temporal_index")

    def __init__(
        self,
        data: np.ndarray,
        temporal_index: int,
        source_frames: tuple[int, int],
    ) -> None:
        self.data = data  # (6, H, W) float32, normalized
        self.temporal_index = temporal_index
        self.source_frames = source_frames  # (frame_idx_0, frame_idx_1)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.data.shape)

    def __repr__(self) -> str:
        return (
            f"SuperFrame(temporal_index={self.temporal_index}, "
            f"shape={self.shape}, source_frames={self.source_frames})"
        )


class CaptionRequest(BaseModel):
    """Request body for /v1/caption endpoint."""

    model_config = ConfigDict(strict=True)

    video_path: str = Field(description="Path to the video file")
    prompt: str = Field(
        default="Describe what happens in this video.",
        description="Caption prompt",
    )
    fps: float = Field(default=2.0, gt=0, le=30, description="Frame extraction FPS")
    max_frames: int = Field(default=64, gt=0, le=512, description="Maximum frames to extract")
    model_profile: str = Field(default="qwen3.5", description="Model configuration profile")
    max_tokens: int = Field(default=2048, gt=0, description="Max tokens for caption generation")
    preset: str = Field(default="default", description="Inference preset name")
    temperature: float | None = Field(
        default=None, ge=0, le=2, description="Override preset temperature"
    )
    chunk_duration_seconds: float | None = Field(
        default=None, gt=0, description="Chunk duration for long videos (auto-detected if None)"
    )


class CaptionMetadata(BaseModel):
    """Metadata about a caption generation request."""

    frames_extracted: int
    super_frames: int
    grid_thw: tuple[int, int, int]
    processing_time_ms: float
    model_tokens_used: int | None = None


class CaptionResponse(BaseModel):
    """Response body for /v1/caption endpoint."""

    caption: str
    metadata: CaptionMetadata


class HealthResponse(BaseModel):
    """Response body for /v1/health endpoint."""

    status: str
    llama_server_reachable: bool
    llama_server_url: str
    version: str


class DebugInfo(BaseModel):
    """Debug information about the last request."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    request: dict[str, Any] | None = None
    frames_extracted: int | None = None
    super_frame_shapes: list[tuple[int, ...]] | None = None
    grid_thw: tuple[int, int, int] | None = None
    temporal_positions: list[int] | None = None
    llama_server_request: dict[str, Any] | None = None
    llama_server_response: dict[str, Any] | None = None
    timing: dict[str, float] | None = None
    error: str | None = None


@dataclass
class CaptionResult:
    """Complete result of a captioning operation, for history and export."""

    caption: str
    source_path: str
    mode: str  # "video" | "image"
    prompt_rendered: str
    template_name: str | None
    variables: dict[str, str]
    preset_name: str
    settings: dict[str, Any]
    metadata: CaptionMetadata
    token_usage: Any | None  # TokenBudget or None
    duration_ms: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dict for serialization."""
        return {
            "caption": self.caption,
            "source_path": self.source_path,
            "mode": self.mode,
            "prompt_rendered": self.prompt_rendered,
            "template_name": self.template_name,
            "variables": self.variables,
            "preset_name": self.preset_name,
            "settings": self.settings,
            "metadata": self.metadata.model_dump(),
            "token_usage": None,  # Will be populated when TokenBudget is available
            "duration_ms": self.duration_ms,
        }
