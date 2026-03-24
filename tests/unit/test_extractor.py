"""Tests for frame extraction."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from llama_video.config import ExtractorSettings
from llama_video.errors import FFmpegNotFoundError, VideoDecodeError, VideoNotFoundError
from llama_video.extractor import Extractor, ExtractorConfig


class TestExtractorConfig:
    """Test ExtractorConfig defaults and validation."""

    def test_default_config(self):
        config = ExtractorConfig()
        assert config.fps == 2.0
        assert config.max_frames == 64
        assert config.min_frames == 1

    def test_custom_config(self):
        config = ExtractorConfig(fps=4.0, max_frames=128)
        assert config.fps == 4.0
        assert config.max_frames == 128


class TestExtractorInit:
    """Test Extractor initialization and ffmpeg detection."""

    def test_ffmpeg_not_found_raises(self):
        settings = ExtractorSettings(ffmpeg_path="nonexistent_ffmpeg_binary_xyz")
        extractor = Extractor(settings)
        with pytest.raises(FFmpegNotFoundError):
            _ = extractor.ffmpeg_path

    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    def test_ffmpeg_found(self, mock_which):
        extractor = Extractor()
        assert extractor.ffmpeg_path == "/usr/bin/ffmpeg"


class TestExtractorValidation:
    """Test input validation."""

    def test_nonexistent_video_raises(self):
        extractor = Extractor()
        with pytest.raises(VideoNotFoundError):
            extractor.extract_frames("/nonexistent/video.mp4")

    def test_directory_path_raises(self, tmp_path):
        extractor = Extractor()
        with pytest.raises(VideoNotFoundError, match="not a file"):
            extractor.extract_frames(str(tmp_path))


class TestFrameParsing:
    """Test raw frame data parsing (mock ffmpeg output)."""

    @pytest.mark.asyncio
    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    async def test_parse_raw_rgb_frames(self, mock_which, tmp_path):
        """Verify correct parsing of raw RGB24 data from ffmpeg stdout."""
        # Create a fake video file (just needs to exist for validation)
        video_file = tmp_path / "test.mp4"
        video_file.touch()

        # Create synthetic raw RGB data: 2 frames of 4x4 pixels
        width, height = 4, 4
        frame_size = width * height * 3
        rng = np.random.default_rng(42)
        frame_data = rng.integers(0, 256, size=frame_size * 2, dtype=np.uint8)

        # Mock the ffprobe call (video info)
        ffprobe_result = AsyncMock()
        ffprobe_result.communicate = AsyncMock(
            return_value=(f"{width}x{height}x1.0\n".encode(), b"")
        )
        ffprobe_result.returncode = 0

        # Mock the ffmpeg call (frame extraction)
        ffmpeg_result = AsyncMock()
        ffmpeg_result.communicate = AsyncMock(return_value=(frame_data.tobytes(), b""))
        ffmpeg_result.returncode = 0

        extractor = Extractor()

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            # First call is ffprobe, second is ffmpeg
            mock_exec.side_effect = [ffprobe_result, ffmpeg_result]

            frames = await extractor.extract_frames_async(str(video_file))

        assert len(frames) == 2
        assert frames[0].width == width
        assert frames[0].height == height
        assert frames[0].data.shape == (height, width, 3)
        assert frames[0].index == 0
        assert frames[1].index == 1


class TestExtractorEdgeCases:
    """Test edge cases for frame extraction."""

    @pytest.mark.asyncio
    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    async def test_single_frame_video(self, mock_which, tmp_path):
        """Very short video with only one frame (e.g., 0.1s at 2fps)."""
        video_file = tmp_path / "short.mp4"
        video_file.touch()

        width, height = 4, 4
        frame_size = width * height * 3
        # Only 1 frame of data
        rng = np.random.default_rng(42)
        frame_data = rng.integers(0, 256, size=frame_size, dtype=np.uint8)

        ffprobe_result = AsyncMock()
        ffprobe_result.communicate = AsyncMock(
            return_value=(f"{width}x{height}x0.1\n".encode(), b"")
        )
        ffprobe_result.returncode = 0

        ffmpeg_result = AsyncMock()
        ffmpeg_result.communicate = AsyncMock(return_value=(frame_data.tobytes(), b""))
        ffmpeg_result.returncode = 0

        extractor = Extractor()

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_exec.side_effect = [ffprobe_result, ffmpeg_result]
            frames = await extractor.extract_frames_async(str(video_file))

        assert len(frames) == 1
        assert frames[0].index == 0
        assert frames[0].timestamp == 0.0

    @pytest.mark.asyncio
    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    async def test_audio_only_video_raises_video_decode_error(self, mock_which, tmp_path):
        """Video with no video stream (audio-only) should raise VideoDecodeError."""
        video_file = tmp_path / "audio_only.mp3"
        video_file.touch()

        # ffprobe returns no video stream info (width=0, height=0)
        ffprobe_result = AsyncMock()
        ffprobe_result.communicate = AsyncMock(return_value=(b"0x0x3.0\n", b""))
        ffprobe_result.returncode = 0

        extractor = Extractor()

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_exec.return_value = ffprobe_result
            with pytest.raises(VideoDecodeError, match="Could not determine video dimensions"):
                await extractor.extract_frames_async(str(video_file))

    @pytest.mark.asyncio
    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    async def test_very_high_fps_produces_many_frames(self, mock_which, tmp_path):
        """High fps request (30fps on 3s clip) respects max_frames limit."""
        video_file = tmp_path / "high_fps.mp4"
        video_file.touch()

        width, height = 4, 4
        frame_size = width * height * 3
        # 30fps * 3s = 90 frames, but max_frames=64 by default
        num_frames = 64
        rng = np.random.default_rng(42)
        frame_data = rng.integers(0, 256, size=frame_size * num_frames, dtype=np.uint8)

        ffprobe_result = AsyncMock()
        ffprobe_result.communicate = AsyncMock(
            return_value=(f"{width}x{height}x3.0\n".encode(), b"")
        )
        ffprobe_result.returncode = 0

        ffmpeg_result = AsyncMock()
        ffmpeg_result.communicate = AsyncMock(return_value=(frame_data.tobytes(), b""))
        ffmpeg_result.returncode = 0

        extractor = Extractor()

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_exec.side_effect = [ffprobe_result, ffmpeg_result]
            # Request very high fps but should be limited by max_frames
            config = ExtractorConfig(fps=30.0, max_frames=64)
            frames = await extractor.extract_frames_async(str(video_file), config)

        assert len(frames) == 64
        # Verify frames are indexed correctly
        assert frames[0].index == 0
        assert frames[-1].index == 63

    @pytest.mark.asyncio
    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    async def test_max_frames_one_extracts_single_frame(self, mock_which, tmp_path):
        """max_frames=1 should limit extraction to exactly one frame."""
        video_file = tmp_path / "long_video.mp4"
        video_file.touch()

        width, height = 4, 4
        frame_size = width * height * 3
        rng = np.random.default_rng(42)
        frame_data = rng.integers(0, 256, size=frame_size, dtype=np.uint8)

        ffprobe_result = AsyncMock()
        ffprobe_result.communicate = AsyncMock(
            return_value=(f"{width}x{height}x10.0\n".encode(), b"")
        )
        ffprobe_result.returncode = 0

        ffmpeg_result = AsyncMock()
        ffmpeg_result.communicate = AsyncMock(return_value=(frame_data.tobytes(), b""))
        ffmpeg_result.returncode = 0

        extractor = Extractor()

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_exec.side_effect = [ffprobe_result, ffmpeg_result]
            config = ExtractorConfig(fps=2.0, max_frames=1)
            frames = await extractor.extract_frames_async(str(video_file), config)

        assert len(frames) == 1
        assert frames[0].index == 0

    @pytest.mark.asyncio
    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    async def test_ffmpeg_failure_raises_video_decode_error(self, mock_which, tmp_path):
        """ffmpeg returning non-zero exit code should raise VideoDecodeError."""
        video_file = tmp_path / "corrupt.mp4"
        video_file.touch()

        width, height = 4, 4

        ffprobe_result = AsyncMock()
        ffprobe_result.communicate = AsyncMock(
            return_value=(f"{width}x{height}x1.0\n".encode(), b"")
        )
        ffprobe_result.returncode = 0

        ffmpeg_result = AsyncMock()
        ffmpeg_result.communicate = AsyncMock(
            return_value=(b"", b"Invalid data found when processing input\n")
        )
        ffmpeg_result.returncode = 1

        extractor = Extractor()

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_exec.side_effect = [ffprobe_result, ffmpeg_result]
            with pytest.raises(VideoDecodeError, match="ffmpeg failed"):
                await extractor.extract_frames_async(str(video_file))
