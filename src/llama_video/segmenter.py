"""Video segmentation for adapters with duration limits.

Splits long videos into chunks that fit within an adapter's
max_duration_seconds constraint.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VideoChunk:
    """A time-bounded segment of a video."""

    start_seconds: float
    end_seconds: float
    index: int

    @property
    def duration(self) -> float:
        return self.end_seconds - self.start_seconds


def segment_video(
    total_duration: float,
    chunk_seconds: float,
    max_chunk_seconds: float | None = None,
) -> list[VideoChunk]:
    """Segment a video into time-bounded chunks.

    Args:
        total_duration: Total video duration in seconds.
        chunk_seconds: Desired chunk length. Capped at max_chunk_seconds.
        max_chunk_seconds: Maximum allowed chunk length (adapter constraint).
            None means no cap.

    Returns:
        List of VideoChunk instances. If total_duration <= effective chunk size,
        returns a single chunk covering the full duration.
    """
    effective_chunk = chunk_seconds
    if max_chunk_seconds is not None:
        effective_chunk = min(effective_chunk, max_chunk_seconds)

    if total_duration <= effective_chunk:
        return [VideoChunk(start_seconds=0.0, end_seconds=total_duration, index=0)]

    chunks: list[VideoChunk] = []
    start = 0.0
    idx = 0

    while start < total_duration:
        end = min(start + effective_chunk, total_duration)
        chunks.append(VideoChunk(start_seconds=start, end_seconds=end, index=idx))
        start = end
        idx += 1

    return chunks
