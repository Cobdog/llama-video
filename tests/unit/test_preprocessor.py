"""Tests for video preprocessing: super-frames, grid computation, temporal positions."""

from __future__ import annotations

import numpy as np
import pytest

from llama_video.config import ModelConfig
from llama_video.errors import InvalidFrameDimensionsError, PreprocessingError
from llama_video.preprocessor import Preprocessor, VideoInput
from llama_video.types import Frame, OddFrameStrategy


class TestSuperFrameConstruction:
    """Test super-frame pairing logic."""

    def test_even_frame_count_produces_correct_super_frames(
        self, sample_frames: list[Frame], model_config: ModelConfig
    ):
        """8 frames → 4 super-frames, each with 6 channels."""
        preprocessor = Preprocessor(model_config)
        super_frames = preprocessor.build_super_frames(sample_frames, 448, 448)

        assert len(super_frames) == 4
        for sf in super_frames:
            assert sf.shape == (6, 448, 448)

    def test_super_frame_source_indices_are_correct(
        self, sample_frames: list[Frame], model_config: ModelConfig
    ):
        """Super-frame pairs track which source frames they came from."""
        preprocessor = Preprocessor(model_config)
        super_frames = preprocessor.build_super_frames(sample_frames, 448, 448)

        assert super_frames[0].source_frames == (0, 1)
        assert super_frames[1].source_frames == (2, 3)
        assert super_frames[2].source_frames == (4, 5)
        assert super_frames[3].source_frames == (6, 7)

    def test_super_frame_temporal_indices_are_sequential(
        self, sample_frames: list[Frame], model_config: ModelConfig
    ):
        preprocessor = Preprocessor(model_config)
        super_frames = preprocessor.build_super_frames(sample_frames, 448, 448)

        for i, sf in enumerate(super_frames):
            assert sf.temporal_index == i

    def test_odd_frame_count_pad_duplicates_last(
        self, odd_frames: list[Frame], model_config: ModelConfig
    ):
        """7 frames with PAD → 4 super-frames (last frame duplicated)."""
        preprocessor = Preprocessor(model_config)
        super_frames = preprocessor.build_super_frames(odd_frames, 448, 448, OddFrameStrategy.PAD)
        assert len(super_frames) == 4

    def test_odd_frame_count_drop_removes_last(
        self, odd_frames: list[Frame], model_config: ModelConfig
    ):
        """7 frames with DROP → 3 super-frames (last frame dropped)."""
        preprocessor = Preprocessor(model_config)
        super_frames = preprocessor.build_super_frames(odd_frames, 448, 448, OddFrameStrategy.DROP)
        assert len(super_frames) == 3

    def test_super_frame_channels_are_frame_pair(self, model_config: ModelConfig):
        """First 3 channels should come from frame A, last 3 from frame B."""
        # Create two distinctive frames
        frame_a_data = np.full((28, 28, 3), 100, dtype=np.uint8)
        frame_b_data = np.full((28, 28, 3), 200, dtype=np.uint8)

        frames = [
            Frame(data=frame_a_data, index=0, timestamp=0.0, width=28, height=28),
            Frame(data=frame_b_data, index=1, timestamp=0.5, width=28, height=28),
        ]

        preprocessor = Preprocessor(model_config)
        super_frames = preprocessor.build_super_frames(frames, 28, 28)

        assert len(super_frames) == 1
        sf = super_frames[0]

        # After normalization, channel values should differ between first 3 and last 3
        first_half_mean = sf.data[:3].mean()
        second_half_mean = sf.data[3:].mean()
        assert first_half_mean != pytest.approx(second_half_mean, abs=0.1)


class TestGridTHW:
    """Test grid_thw computation."""

    def test_grid_thw_standard_case(self, model_config: ModelConfig):
        """4 super-frames at 448x448 → grid (4, 16, 16)."""
        preprocessor = Preprocessor(model_config)
        grid = preprocessor.compute_grid_thw(4, 448, 448)
        # 448 / (14 * 2) = 16
        assert grid == (4, 16, 16)

    def test_grid_thw_rectangular(self, model_config: ModelConfig):
        """Non-square resolution."""
        preprocessor = Preprocessor(model_config)
        grid = preprocessor.compute_grid_thw(2, 672, 448)
        # 672 / 28 = 24, 448 / 28 = 16
        assert grid == (2, 16, 24)

    def test_grid_thw_minimum_resolution(self, model_config: ModelConfig):
        """Minimum viable resolution (28x28)."""
        preprocessor = Preprocessor(model_config)
        grid = preprocessor.compute_grid_thw(1, 28, 28)
        assert grid == (1, 1, 1)


class TestTemporalPositions:
    """Test M-RoPE temporal position computation."""

    def test_positions_at_2fps(self, model_config: ModelConfig):
        """At 2fps with temporal_patch_size=2: each position = 1 second."""
        preprocessor = Preprocessor(model_config)
        positions = preprocessor.compute_temporal_positions((4, 16, 16), fps=2.0)
        assert len(positions) == 4
        assert positions == [0, 1, 2, 3]

    def test_positions_at_1fps(self, model_config: ModelConfig):
        """At 1fps with temporal_patch_size=2: each position = 2 seconds."""
        preprocessor = Preprocessor(model_config)
        positions = preprocessor.compute_temporal_positions((4, 16, 16), fps=1.0)
        assert positions == [0, 2, 4, 6]

    def test_positions_at_4fps(self, model_config: ModelConfig):
        """At 4fps with temporal_patch_size=2: each position = 0.5 seconds."""
        preprocessor = Preprocessor(model_config)
        positions = preprocessor.compute_temporal_positions((4, 16, 16), fps=4.0)
        # round(0) = 0, round(0.5) = 0, round(1.0) = 1, round(1.5) = 2
        assert positions == [0, 0, 1, 2]

    def test_single_temporal_position(self, model_config: ModelConfig):
        """Single super-frame → single temporal position at 0."""
        preprocessor = Preprocessor(model_config)
        positions = preprocessor.compute_temporal_positions((1, 16, 16), fps=2.0)
        assert positions == [0]


class TestFullPipeline:
    """Test the complete process() pipeline."""

    def test_process_8_frames(self, sample_frames: list[Frame], model_config: ModelConfig):
        """Full pipeline with 8 standard frames."""
        preprocessor = Preprocessor(model_config)
        result = preprocessor.process(sample_frames, fps=2.0)

        assert isinstance(result, VideoInput)
        assert len(result.super_frames) == 4
        assert result.grid_thw[0] == 4  # T
        assert result.num_source_frames == 8
        assert result.fps == 2.0
        assert len(result.temporal_positions) == 4

    def test_process_empty_frames_raises(self, model_config: ModelConfig):
        """Empty frame list should raise PreprocessingError."""
        preprocessor = Preprocessor(model_config)
        with pytest.raises(PreprocessingError, match="No frames"):
            preprocessor.process([], fps=2.0)

    def test_process_inconsistent_dimensions_raises(self, model_config: ModelConfig):
        """Frames with different sizes should raise InvalidFrameDimensionsError."""
        rng = np.random.default_rng(42)
        frames = [
            Frame(
                data=rng.integers(0, 256, (448, 448, 3), dtype=np.uint8),
                index=0,
                timestamp=0.0,
                width=448,
                height=448,
            ),
            Frame(
                data=rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
                index=1,
                timestamp=0.5,
                width=224,
                height=224,
            ),
        ]
        preprocessor = Preprocessor(model_config)
        with pytest.raises(InvalidFrameDimensionsError):
            preprocessor.process(frames, fps=2.0)


class TestResolutionComputation:
    """Test target resolution calculation."""

    def test_standard_resolution_unchanged(self, model_config: ModelConfig):
        """448x448 is already grid-aligned → no change."""
        preprocessor = Preprocessor(model_config)
        w, h = preprocessor._compute_target_resolution(448, 448)
        assert w == 448
        assert h == 448

    def test_non_aligned_resolution_rounds(self, model_config: ModelConfig):
        """Non-28-aligned dimensions get rounded."""
        preprocessor = Preprocessor(model_config)
        w, h = preprocessor._compute_target_resolution(450, 300)
        assert w % 28 == 0
        assert h % 28 == 0

    def test_very_large_resolution_scales_down(self, model_config: ModelConfig):
        """4K resolution gets scaled to fit max_pixels."""
        preprocessor = Preprocessor(model_config)
        w, h = preprocessor._compute_target_resolution(3840, 2160)
        assert w * h <= model_config.max_pixels
        assert w % 28 == 0
        assert h % 28 == 0

    def test_very_small_resolution_scales_up(self, model_config: ModelConfig):
        """Tiny resolution gets scaled to meet min_pixels."""
        preprocessor = Preprocessor(model_config)
        w, h = preprocessor._compute_target_resolution(10, 10)
        assert w * h >= model_config.min_pixels
        assert w % 28 == 0
        assert h % 28 == 0


class TestEdgeCases:
    """Test edge cases for preprocessing."""

    def test_single_frame_pads_to_one_super_frame(
        self, single_frame: list[Frame], model_config: ModelConfig
    ):
        """Single frame with PAD → duplicates to 2 frames → 1 super-frame."""
        preprocessor = Preprocessor(model_config)
        result = preprocessor.process(single_frame, fps=2.0, odd_strategy=OddFrameStrategy.PAD)

        assert len(result.super_frames) == 1
        assert result.num_source_frames == 1  # Original count before padding
        assert result.super_frames[0].source_frames == (0, 0)  # Same frame duplicated

    def test_two_frames_exactly_one_super_frame(self, model_config: ModelConfig):
        """Two frames (minimum viable) → exactly one super-frame."""
        rng = np.random.default_rng(42)
        frames = [
            Frame(
                data=rng.integers(0, 256, (448, 448, 3), dtype=np.uint8),
                index=0,
                timestamp=0.0,
                width=448,
                height=448,
            ),
            Frame(
                data=rng.integers(0, 256, (448, 448, 3), dtype=np.uint8),
                index=1,
                timestamp=0.5,
                width=448,
                height=448,
            ),
        ]

        preprocessor = Preprocessor(model_config)
        result = preprocessor.process(frames, fps=2.0)

        assert len(result.super_frames) == 1
        assert result.num_source_frames == 2
        assert result.grid_thw == (1, 16, 16)

    def test_single_frame_drop_strategy_raises(
        self, single_frame: list[Frame], model_config: ModelConfig
    ):
        """Single frame with DROP would produce 0 frames → raises PreprocessingError."""
        preprocessor = Preprocessor(model_config)
        # With DROP, 1 frame becomes 0 frames (dropped to make even), which fails
        # Actually: DROP on odd removes the last frame, so 1 becomes 0
        # This should raise because no frames remain for pairing
        with pytest.raises(PreprocessingError, match="No super-frames"):
            preprocessor.process(single_frame, fps=2.0, odd_strategy=OddFrameStrategy.DROP)

    def test_very_wide_aspect_ratio(self, model_config: ModelConfig):
        """Very wide aspect ratio (e.g., 1920x100) is handled correctly."""
        rng = np.random.default_rng(42)
        frames = [
            Frame(
                data=rng.integers(0, 256, (100, 1920, 3), dtype=np.uint8),
                index=i,
                timestamp=i / 2.0,
                width=1920,
                height=100,
            )
            for i in range(2)
        ]

        preprocessor = Preprocessor(model_config)
        result = preprocessor.process(frames, fps=2.0)

        # Target resolution should be grid-aligned
        w, h = result.resolution
        assert w % 28 == 0
        assert h % 28 == 0
        # Should preserve aspect ratio (roughly)
        assert w > h  # Still wide
        assert len(result.super_frames) == 1

    def test_very_tall_aspect_ratio(self, model_config: ModelConfig):
        """Very tall aspect ratio (e.g., 100x1920) is handled correctly."""
        rng = np.random.default_rng(42)
        frames = [
            Frame(
                data=rng.integers(0, 256, (1920, 100, 3), dtype=np.uint8),
                index=i,
                timestamp=i / 2.0,
                width=100,
                height=1920,
            )
            for i in range(2)
        ]

        preprocessor = Preprocessor(model_config)
        result = preprocessor.process(frames, fps=2.0)

        # Target resolution should be grid-aligned
        w, h = result.resolution
        assert w % 28 == 0
        assert h % 28 == 0
        # Should preserve aspect ratio (roughly)
        assert h > w  # Still tall
        assert len(result.super_frames) == 1

    def test_process_with_single_frame_fixture(
        self, single_frame: list[Frame], model_config: ModelConfig
    ):
        """Verify process() works with existing single_frame fixture."""
        preprocessor = Preprocessor(model_config)
        result = preprocessor.process(single_frame, fps=2.0)

        # Should succeed with padding
        assert isinstance(result, VideoInput)
        assert len(result.super_frames) >= 1
        assert result.fps == 2.0
