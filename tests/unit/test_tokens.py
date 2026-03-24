"""Tests for token budget estimation."""

from __future__ import annotations

import numpy as np

from llama_video.preprocessor import VideoInput
from llama_video.tokens import TokenBudget, TokenEstimator
from llama_video.types import SuperFrame


class TestTokenBudget:
    """Test TokenBudget dataclass."""

    def test_headroom_calculation(self):
        b = TokenBudget(
            vision_tokens=2048,
            prompt_tokens=50,
            generation_budget=1024,
            total_estimated=3122,
            context_limit=65536,
            headroom=62414,
            warning=None,
        )
        assert b.headroom == b.context_limit - b.total_estimated

    def test_warning_when_over_budget(self):
        b = TokenBudget(
            vision_tokens=60000,
            prompt_tokens=50,
            generation_budget=8192,
            total_estimated=68242,
            context_limit=65536,
            headroom=-2706,
            warning="Estimated tokens (68242) exceed context limit (65536)",
        )
        assert b.warning is not None
        assert b.headroom < 0


class TestTokenEstimator:
    """Test token estimation from VideoInput and raw settings."""

    def setup_method(self):
        self.estimator = TokenEstimator()

    def test_estimate_from_video_input(self):
        sf = SuperFrame(
            data=np.zeros((6, 28, 28), dtype=np.float32),
            temporal_index=0,
            source_frames=(0, 1),
        )
        video_input = VideoInput(
            super_frames=[sf],
            grid_thw=(1, 1, 1),
            temporal_positions=[0],
            fps=2.0,
            num_source_frames=2,
            resolution=(28, 28),
        )
        budget = self.estimator.estimate(
            video_input=video_input,
            prompt="Describe this video.",
            max_tokens=1024,
        )
        assert budget.vision_tokens == 1  # 1*1*1 from grid_thw
        assert budget.generation_budget == 1024
        assert (
            budget.total_estimated
            == budget.vision_tokens + budget.prompt_tokens + budget.generation_budget
        )
        assert budget.context_limit == 65536
        assert budget.headroom > 0
        assert budget.warning is None

    def test_estimate_from_video_input_large_grid(self):
        sf = SuperFrame(
            data=np.zeros((6, 448, 448), dtype=np.float32),
            temporal_index=0,
            source_frames=(0, 1),
        )
        video_input = VideoInput(
            super_frames=[sf] * 4,
            grid_thw=(4, 16, 16),
            temporal_positions=[0, 1, 2, 3],
            fps=2.0,
            num_source_frames=8,
            resolution=(448, 448),
        )
        budget = self.estimator.estimate(video_input=video_input, prompt="test")
        assert budget.vision_tokens == 4 * 16 * 16  # 1024

    def test_estimate_from_settings(self):
        budget = self.estimator.estimate_from_settings(
            frame_count=8,
            resolution=(448, 448),
            fps=2.0,
            prompt="Describe this video.",
            max_tokens=2048,
            context_limit=65536,
        )
        assert budget.vision_tokens > 0
        assert budget.prompt_tokens > 0
        assert budget.generation_budget == 2048
        assert budget.context_limit == 65536
        assert isinstance(budget.headroom, int)

    def test_estimate_warns_when_over_budget(self):
        budget = self.estimator.estimate_from_settings(
            frame_count=64,
            resolution=(1920, 1080),
            fps=4.0,
            prompt="A very long prompt " * 100,
            max_tokens=8192,
            context_limit=4096,  # Tiny context
        )
        assert budget.warning is not None
        assert budget.headroom < 0

    def test_custom_context_limit(self):
        budget = self.estimator.estimate_from_settings(
            frame_count=4,
            resolution=(448, 448),
            fps=2.0,
            prompt="test",
            max_tokens=512,
            context_limit=131072,
        )
        assert budget.context_limit == 131072
