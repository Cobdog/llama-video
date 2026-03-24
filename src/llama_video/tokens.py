"""Token budget estimation for video and image captioning.

Estimates vision tokens, prompt tokens, and generation budget to help
users understand context window consumption before running inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llama_video.preprocessor import VideoInput


@dataclass(frozen=True)
class TokenBudget:
    """Estimated token consumption breakdown."""

    vision_tokens: int
    prompt_tokens: int
    generation_budget: int
    total_estimated: int
    context_limit: int
    headroom: int
    warning: str | None


class TokenEstimator:
    """Estimates token consumption for captioning operations."""

    def __init__(
        self,
        spatial_patch_size: int = 14,
        merge_size: int = 2,
        temporal_patch_size: int = 2,
    ) -> None:
        self._spatial_patch = spatial_patch_size
        self._merge = merge_size
        self._temporal_patch = temporal_patch_size

    def estimate(
        self,
        video_input: VideoInput | None = None,
        prompt: str = "",
        max_tokens: int = 2048,
        context_limit: int = 65536,
    ) -> TokenBudget:
        """Estimate token budget from a preprocessed VideoInput (exact)."""
        if video_input is not None:
            t, h, w = video_input.grid_thw
            vision_tokens = t * h * w
        else:
            vision_tokens = 0

        return self._build_budget(vision_tokens, prompt, max_tokens, context_limit)

    def estimate_from_settings(
        self,
        frame_count: int,
        resolution: tuple[int, int],
        fps: float,
        prompt: str = "",
        max_tokens: int = 2048,
        context_limit: int = 65536,
    ) -> TokenBudget:
        """Estimate token budget from raw parameters (approximate).

        Used for pre-inference UI display before frames are extracted.
        """
        width, height = resolution
        grid_unit = self._spatial_patch * self._merge

        # Approximate grid dimensions (matches preprocessor logic)
        h_grid = height // grid_unit
        w_grid = width // grid_unit
        t_grid = max(1, frame_count // self._temporal_patch)

        vision_tokens = t_grid * h_grid * w_grid
        return self._build_budget(vision_tokens, prompt, max_tokens, context_limit)

    def _build_budget(
        self,
        vision_tokens: int,
        prompt: str,
        max_tokens: int,
        context_limit: int,
    ) -> TokenBudget:
        prompt_tokens = max(1, len(prompt) // 4)  # Rough estimate
        total = vision_tokens + prompt_tokens + max_tokens
        headroom = context_limit - total
        warning = (
            f"Estimated tokens ({total}) exceed context limit ({context_limit})"
            if headroom < 0
            else None
        )
        return TokenBudget(
            vision_tokens=vision_tokens,
            prompt_tokens=prompt_tokens,
            generation_budget=max_tokens,
            total_estimated=total,
            context_limit=context_limit,
            headroom=headroom,
            warning=warning,
        )
