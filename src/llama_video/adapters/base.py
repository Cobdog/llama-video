"""Abstract base for model-family-specific video processing adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

__all__ = ["AdapterPreset", "ModelAdapter"]


@dataclass(frozen=True)
class AdapterPreset:
    """Model-family-specific inference settings.

    Each adapter defines its own defaults — no cross-contamination.
    """

    temperature: float
    top_p: float
    top_k: int
    min_p: float = 0.0
    presence_penalty: float = 0.0


class ModelAdapter(ABC):
    """Interface for model-family-specific video processing.

    Each adapter encapsulates ALL model-specific behavior:
    frame extraction parameters, preprocessing, payload construction,
    response parsing, token estimation, and sampler defaults.

    Adding a new model family = implement this class + register it.
    """

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
        """Max video duration (seconds) this adapter supports in one chunk."""

    @property
    @abstractmethod
    def max_frames(self) -> int:
        """Maximum frames this adapter supports in one chunk."""

    @property
    @abstractmethod
    def default_preset(self) -> AdapterPreset:
        """Default inference preset for this model family."""

    @abstractmethod
    def preprocess(self, frames: list[Any], fps: float) -> Any:
        """Transform raw Frame objects into a model-specific preprocessed input.

        Args:
            frames: Extracted Frame objects from the Extractor.
            fps: Frame extraction FPS used.

        Returns:
            Adapter-specific preprocessed video input (e.g. VideoInput).
        """

    @abstractmethod
    def build_payload(
        self,
        video_input: Any,
        prompt: str,
        max_tokens: int = 2048,
        preset: AdapterPreset | None = None,
        cache_prompt: bool = True,
        model_name: str = "",
    ) -> dict[str, Any]:
        """Build the OpenAI-compatible request payload for this model.

        Args:
            video_input: Preprocessed input from preprocess().
            prompt: User caption prompt.
            max_tokens: Max generation tokens.
            preset: Inference preset (uses default if None).
            cache_prompt: Whether to enable prompt caching.
            model_name: Model name for router mode (empty = single-model).

        Returns:
            Complete payload dict for /v1/chat/completions.
        """

    @abstractmethod
    def parse_response(self, raw: str) -> tuple[str, str, bool]:
        """Parse model response into (caption, thinking, truncated).

        Each model family uses different thinking tag formats.

        Args:
            raw: Raw text from model response.

        Returns:
            (caption, thinking, truncated) tuple.
        """

    @abstractmethod
    def estimate_tokens(self, video_input: Any) -> int:
        """Estimate vision token consumption for a preprocessed input."""
