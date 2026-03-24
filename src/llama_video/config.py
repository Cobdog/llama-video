"""Configuration management for llama-video."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import Field
from pydantic_settings import BaseSettings


@dataclass(frozen=True)
class InferencePreset:
    """Named set of inference parameters for caption generation.

    Based on official Qwen team recommendations for thinking mode.
    """

    name: str
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    presence_penalty: float


PRESETS: dict[str, InferencePreset] = {
    "default": InferencePreset(
        name="default",
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=1.5,
    ),
    "precise": InferencePreset(
        name="precise",
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=0.0,
    ),
}


def get_preset(name: str) -> InferencePreset:
    """Get an inference preset by name. Raises KeyError if not found."""
    return PRESETS[name]


class ModelConfig(BaseSettings):
    """Vision encoder configuration for a model family.

    Defaults are for Qwen3.5 — all sizes share the same vision encoder.
    """

    temporal_patch_size: int = Field(default=2, description="Frames per temporal patch")
    spatial_patch_size: int = Field(default=14, description="Pixels per spatial patch")
    merge_size: int = Field(default=2, description="Spatial merge factor")
    min_pixels: int = Field(default=3136, description="Minimum frame pixels (4 x 28^2)")
    max_pixels: int = Field(default=12845056, description="Maximum frame pixels (16384 x 28^2)")
    image_mean: tuple[float, float, float] = Field(
        default=(0.48145466, 0.4578275, 0.40821073),
        description="CLIP normalization mean (RGB)",
    )
    image_std: tuple[float, float, float] = Field(
        default=(0.26862954, 0.26130258, 0.27577711),
        description="CLIP normalization std (RGB)",
    )

    @property
    def grid_unit(self) -> int:
        """Pixel size that frames must be divisible by."""
        return self.spatial_patch_size * self.merge_size

    @classmethod
    def qwen35(cls) -> ModelConfig:
        """Default config for all Qwen3.5 models."""
        return cls()


class ServerConfig(BaseSettings):
    """Configuration for the llama-server connection."""

    model_config = {"env_prefix": "LLAMA_SERVER_"}

    url: str = Field(default="http://localhost:8080", description="llama-server URL")
    timeout: float = Field(default=120.0, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Max retry attempts")
    retry_delay: float = Field(default=1.0, description="Base retry delay in seconds")


class ExtractorSettings(BaseSettings):
    """Configuration for frame extraction."""

    model_config = {"env_prefix": "LLAMA_VIDEO_"}

    ffmpeg_path: str = Field(default="ffmpeg", description="Path to ffmpeg binary")
    default_fps: float = Field(default=2.0, description="Default extraction FPS")
    max_frames: int = Field(default=64, description="Default max frames")
    extraction_timeout: float = Field(default=60.0, description="ffmpeg timeout in seconds")


class ServiceConfig(BaseSettings):
    """Configuration for the Python API service."""

    model_config = {"env_prefix": "LLAMA_VIDEO_"}

    host: str = Field(default="0.0.0.0", description="Bind host")
    port: int = Field(default=9000, description="Bind port")
    workers: int = Field(default=1, description="Uvicorn worker count")
    log_level: str = Field(default="INFO", description="Logging level")
    debug: bool = Field(default=False, description="Enable debug mode")


class Settings(BaseSettings):
    """Root settings combining all config sections."""

    model: ModelConfig = Field(default_factory=ModelConfig.qwen35)
    server: ServerConfig = Field(default_factory=ServerConfig)
    extractor: ExtractorSettings = Field(default_factory=ExtractorSettings)
    service: ServiceConfig = Field(default_factory=ServiceConfig)
