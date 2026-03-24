"""Integration tests for frame extraction with real video files.

These tests use the Studio Ghibli test videos in testvid/ and require ffmpeg.
Marked with @pytest.mark.integration so they can be skipped with -m "not integration".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from llama_video.extractor import Extractor, ExtractorConfig

# Path to test videos relative to project root
TESTVID_DIR = Path(__file__).parent.parent.parent / "testvid"


@pytest.fixture
def kiki_video() -> Path:
    """Path to kiki.mp4 test video (4.6s, 1920x1040)."""
    video_path = TESTVID_DIR / "kiki.mp4"
    if not video_path.exists():
        pytest.skip(f"Test video not found: {video_path}")
    return video_path


@pytest.fixture
def howl_video() -> Path:
    """Path to howl.mp4 test video."""
    video_path = TESTVID_DIR / "howl.mp4"
    if not video_path.exists():
        pytest.skip(f"Test video not found: {video_path}")
    return video_path


@pytest.mark.integration
class TestExtractorWithRealVideo:
    """Integration tests using real video files."""

    def test_extract_frames_from_kiki_at_2fps(self, kiki_video: Path) -> None:
        """Extract frames from kiki.mp4 at 2fps should yield approximately 9 frames.

        kiki.mp4 is 4.625 seconds at 2fps = 9.25 frames, so we expect 9 frames.
        """
        extractor = Extractor()
        config = ExtractorConfig(fps=2.0, max_frames=64)

        frames = extractor.extract_frames(kiki_video, config)

        # Should get approximately 9 frames (4.6s * 2fps)
        assert len(frames) == 9, f"Expected 9 frames, got {len(frames)}"

    def test_frame_dimensions_match_video(self, kiki_video: Path) -> None:
        """Frame dimensions should match the source video (1920x1040)."""
        extractor = Extractor()
        config = ExtractorConfig(fps=2.0)

        frames = extractor.extract_frames(kiki_video, config)

        for frame in frames:
            assert frame.width == 1920, f"Expected width 1920, got {frame.width}"
            assert frame.height == 1040, f"Expected height 1040, got {frame.height}"
            assert frame.data.shape == (1040, 1920, 3), (
                f"Expected data shape (1040, 1920, 3), got {frame.data.shape}"
            )

    def test_frame_data_is_valid_rgb(self, kiki_video: Path) -> None:
        """Frame data should be valid RGB (not all zeros, not corrupted)."""
        extractor = Extractor()
        config = ExtractorConfig(fps=2.0)

        frames = extractor.extract_frames(kiki_video, config)

        assert len(frames) > 0, "No frames extracted"

        for i, frame in enumerate(frames):
            # Check dtype is uint8
            assert frame.data.dtype == np.uint8, (
                f"Frame {i}: expected dtype uint8, got {frame.data.dtype}"
            )

            # Check not all zeros (would indicate corruption)
            assert not np.all(frame.data == 0), f"Frame {i} is all zeros"

            # Check values are in valid range [0, 255]
            assert frame.data.min() >= 0, f"Frame {i} has negative values"
            assert frame.data.max() <= 255, f"Frame {i} has values > 255"

            # Check there's some variation (not a solid color)
            assert frame.data.std() > 0, f"Frame {i} has no variation (solid color)"

    def test_timestamps_are_monotonically_increasing(self, kiki_video: Path) -> None:
        """Frame timestamps should be monotonically increasing."""
        extractor = Extractor()
        config = ExtractorConfig(fps=2.0)

        frames = extractor.extract_frames(kiki_video, config)

        timestamps = [f.timestamp for f in frames]

        # Check monotonic increase
        for i in range(1, len(timestamps)):
            assert timestamps[i] > timestamps[i - 1], (
                f"Timestamps not monotonically increasing: {timestamps}"
            )

        # Check timestamps match expected pattern (0.0, 0.5, 1.0, ... for 2fps)
        for i, frame in enumerate(frames):
            expected_timestamp = i / config.fps
            assert abs(frame.timestamp - expected_timestamp) < 0.01, (
                f"Frame {i}: expected timestamp {expected_timestamp}, got {frame.timestamp}"
            )

    def test_frame_indices_are_sequential(self, kiki_video: Path) -> None:
        """Frame indices should be 0, 1, 2, ... in order."""
        extractor = Extractor()
        config = ExtractorConfig(fps=2.0)

        frames = extractor.extract_frames(kiki_video, config)

        for i, frame in enumerate(frames):
            assert frame.index == i, f"Frame index mismatch: expected {i}, got {frame.index}"

    def test_extract_at_1fps(self, kiki_video: Path) -> None:
        """Extract at 1fps should yield approximately 5 frames (4.6s * 1fps)."""
        extractor = Extractor()
        config = ExtractorConfig(fps=1.0, max_frames=64)

        frames = extractor.extract_frames(kiki_video, config)

        # 4.6s at 1fps = 4-5 frames depending on rounding
        assert 4 <= len(frames) <= 5, f"Expected 4-5 frames at 1fps, got {len(frames)}"

        # Check timestamps are 1 second apart
        for i, frame in enumerate(frames):
            expected_timestamp = float(i)
            assert abs(frame.timestamp - expected_timestamp) < 0.01

    def test_extract_at_4fps(self, kiki_video: Path) -> None:
        """Extract at 4fps should yield approximately 18 frames (4.6s * 4fps)."""
        extractor = Extractor()
        config = ExtractorConfig(fps=4.0, max_frames=64)

        frames = extractor.extract_frames(kiki_video, config)

        # 4.6s at 4fps = ~18 frames
        assert 17 <= len(frames) <= 19, f"Expected ~18 frames at 4fps, got {len(frames)}"

        # Check timestamps are 0.25 seconds apart (1/4fps)
        for i, frame in enumerate(frames):
            expected_timestamp = i / 4.0
            assert abs(frame.timestamp - expected_timestamp) < 0.01

    def test_max_frames_limit(self, kiki_video: Path) -> None:
        """max_frames limit should cap the number of extracted frames."""
        extractor = Extractor()
        config = ExtractorConfig(fps=2.0, max_frames=4)

        frames = extractor.extract_frames(kiki_video, config)

        assert len(frames) == 4, f"Expected exactly 4 frames with max_frames=4, got {len(frames)}"

    def test_max_frames_of_1(self, kiki_video: Path) -> None:
        """max_frames=1 should extract exactly one frame."""
        extractor = Extractor()
        config = ExtractorConfig(fps=2.0, max_frames=1)

        frames = extractor.extract_frames(kiki_video, config)

        assert len(frames) == 1, f"Expected exactly 1 frame with max_frames=1, got {len(frames)}"
        assert frames[0].index == 0
        assert frames[0].timestamp == 0.0

    def test_extract_from_different_video(self, howl_video: Path) -> None:
        """Extraction should work on different video files."""
        extractor = Extractor()
        config = ExtractorConfig(fps=2.0, max_frames=4)

        frames = extractor.extract_frames(howl_video, config)

        assert len(frames) == 4
        # All frames should be valid
        for frame in frames:
            assert frame.width > 0
            assert frame.height > 0
            assert frame.data.shape[2] == 3  # RGB

    def test_frames_have_different_content(self, kiki_video: Path) -> None:
        """Frames from different timestamps should have different pixel content."""
        extractor = Extractor()
        config = ExtractorConfig(fps=1.0, max_frames=5)

        frames = extractor.extract_frames(kiki_video, config)

        # Compare first and last frame - they should be different
        if len(frames) >= 2:
            first_frame = frames[0].data
            last_frame = frames[-1].data

            # Calculate difference
            diff = np.abs(first_frame.astype(np.int16) - last_frame.astype(np.int16))
            mean_diff = np.mean(diff)

            # There should be some difference between frames at different timestamps
            assert mean_diff > 1.0, f"First and last frames too similar (mean diff: {mean_diff})"


@pytest.mark.integration
@pytest.mark.asyncio
class TestExtractorAsyncWithRealVideo:
    """Async integration tests using real video files."""

    async def test_extract_frames_async(self, kiki_video: Path) -> None:
        """Async extraction should work the same as sync."""
        extractor = Extractor()
        config = ExtractorConfig(fps=2.0, max_frames=4)

        frames = await extractor.extract_frames_async(kiki_video, config)

        assert len(frames) == 4
        for frame in frames:
            assert frame.width == 1920
            assert frame.height == 1040
