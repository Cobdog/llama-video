"""Frame extraction from video files using ffmpeg."""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from llama_video.errors import (
    ExtractionTimeoutError,
    FFmpegNotFoundError,
    NoFramesError,
    VideoDecodeError,
    VideoNotFoundError,
)
from llama_video.types import Frame, SamplingStrategy

if TYPE_CHECKING:
    from llama_video.config import ExtractorSettings

logger = logging.getLogger(__name__)


class ExtractorConfig:
    """Configuration for a single extraction run."""

    __slots__ = ("duration", "fps", "max_frames", "min_frames", "start_time", "strategy")

    def __init__(
        self,
        fps: float = 2.0,
        max_frames: int = 64,
        min_frames: int = 1,
        strategy: SamplingStrategy = SamplingStrategy.UNIFORM,
        start_time: float | None = None,
        duration: float | None = None,
    ) -> None:
        self.fps = fps
        self.max_frames = max_frames
        self.min_frames = min_frames
        self.strategy = strategy
        self.start_time = start_time
        self.duration = duration


class Extractor:
    """Extract frames from video files using ffmpeg subprocess."""

    def __init__(self, settings: ExtractorSettings | None = None) -> None:
        if settings is None:
            from llama_video.config import ExtractorSettings

            settings = ExtractorSettings()
        self._settings = settings
        self._ffmpeg_path: str | None = None

    @property
    def ffmpeg_path(self) -> str:
        """Resolve and cache ffmpeg binary path."""
        if self._ffmpeg_path is None:
            path = shutil.which(self._settings.ffmpeg_path)
            if path is None:
                raise FFmpegNotFoundError(
                    f"ffmpeg not found at '{self._settings.ffmpeg_path}'. "
                    "Install ffmpeg or set LLAMA_VIDEO_FFMPEG_PATH.",
                    context={"configured_path": self._settings.ffmpeg_path},
                )
            self._ffmpeg_path = path
        return self._ffmpeg_path

    def _validate_video_path(self, video_path: Path) -> None:
        if not video_path.exists():
            raise VideoNotFoundError(
                f"Video file not found: {video_path}",
                context={"video_path": str(video_path)},
            )
        if not video_path.is_file():
            raise VideoNotFoundError(
                f"Path is not a file: {video_path}",
                context={"video_path": str(video_path)},
            )

    async def _get_video_info(self, video_path: Path) -> tuple[int, int, float]:
        """Get video dimensions and duration using ffprobe."""
        cmd = [
            str(
                Path(self.ffmpeg_path).with_name(
                    Path(self.ffmpeg_path).name.replace("ffmpeg", "ffprobe")
                )
            ),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,duration",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0:s=x",
            str(video_path),
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=self._settings.extraction_timeout,
        )
        if proc.returncode != 0:
            raise VideoDecodeError(
                f"ffprobe failed: {stderr.decode()}",
                context={"video_path": str(video_path), "returncode": proc.returncode},
            )
        # Parse "width x height x duration" or similar
        parts = stdout.decode().strip().split("\n")
        # ffprobe output varies; parse robustly
        width, height, duration = 0, 0, 0.0
        for line in parts:
            fields = line.strip().split("x")
            if len(fields) >= 2:
                try:
                    width = int(fields[0])
                    height = int(fields[1])
                    if len(fields) >= 3:
                        duration = float(fields[2])
                except ValueError:
                    continue
            elif len(fields) == 1 and duration == 0.0:
                # Format duration on its own line (fallback when
                # stream duration is N/A or missing)
                try:
                    duration = float(fields[0])
                except ValueError:
                    continue
        return width, height, duration

    async def extract_frames_async(
        self,
        video_path: str | Path,
        config: ExtractorConfig | None = None,
    ) -> list[Frame]:
        """Extract frames from a video file asynchronously.

        Args:
            video_path: Path to the video file.
            config: Extraction configuration. Uses defaults if not provided.

        Returns:
            List of extracted Frame objects.

        Raises:
            FFmpegNotFoundError: ffmpeg binary not found.
            VideoNotFoundError: Video file does not exist.
            VideoDecodeError: ffmpeg failed to decode the video.
            NoFramesError: No frames were extracted.
            ExtractionTimeoutError: ffmpeg timed out.
        """
        video_path = Path(video_path)
        self._validate_video_path(video_path)

        if config is None:
            config = ExtractorConfig(
                fps=self._settings.default_fps,
                max_frames=self._settings.max_frames,
            )

        logger.info(
            "Extracting frames from %s at %.1f fps (max %d)",
            video_path.name,
            config.fps,
            config.max_frames,
        )

        # Build ffmpeg command for raw RGB output to stdout
        cmd = [self.ffmpeg_path]

        # Seek to start position before input for fast seeking
        if config.start_time is not None:
            cmd.extend(["-ss", f"{config.start_time:.3f}"])

        cmd.extend(["-i", str(video_path)])

        # Limit duration after input for accurate clipping
        if config.duration is not None:
            cmd.extend(["-t", f"{config.duration:.3f}"])

        cmd.extend(
            [
                "-vf",
                f"fps={config.fps}",
                "-frames:v",
                str(config.max_frames),
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-v",
                "error",
                "pipe:1",
            ]
        )

        # Also need dimensions — get them first
        width, height, duration = await self._get_video_info(video_path)
        if width == 0 or height == 0:
            raise VideoDecodeError(
                "Could not determine video dimensions",
                context={"video_path": str(video_path)},
            )

        logger.debug("Video info: %dx%d, %.1fs", width, height, duration)

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self._settings.extraction_timeout,
            )
        except TimeoutError as e:
            raise ExtractionTimeoutError(
                f"ffmpeg timed out after {self._settings.extraction_timeout}s",
                context={
                    "video_path": str(video_path),
                    "timeout": self._settings.extraction_timeout,
                },
            ) from e

        if proc.returncode != 0:
            raise VideoDecodeError(
                f"ffmpeg failed (exit {proc.returncode}): {stderr.decode()[:500]}",
                context={
                    "video_path": str(video_path),
                    "returncode": proc.returncode,
                },
            )

        # Parse raw RGB frames from stdout
        frame_size = width * height * 3  # RGB24
        raw_data = stdout
        num_frames = len(raw_data) // frame_size

        if num_frames == 0:
            raise NoFramesError(
                "No frames extracted from video",
                context={
                    "video_path": str(video_path),
                    "fps": config.fps,
                    "raw_bytes": len(raw_data),
                    "expected_frame_size": frame_size,
                },
            )

        frames: list[Frame] = []
        start_offset = config.start_time or 0.0
        for i in range(num_frames):
            offset = i * frame_size
            frame_bytes = raw_data[offset : offset + frame_size]
            data = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(height, width, 3)
            timestamp = start_offset + i / config.fps

            frames.append(
                Frame(
                    data=data.copy(),  # Copy so we don't hold the full buffer
                    index=i,
                    timestamp=timestamp,
                    width=width,
                    height=height,
                )
            )

        logger.info(
            "Extracted %d frames (%dx%d) from %s", len(frames), width, height, video_path.name
        )
        return frames

    def extract_frames(
        self,
        video_path: str | Path,
        config: ExtractorConfig | None = None,
    ) -> list[Frame]:
        """Synchronous wrapper for extract_frames_async."""
        return asyncio.run(self.extract_frames_async(video_path, config))
