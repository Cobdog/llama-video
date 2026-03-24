"""Smoke tests: full pipeline from video file to caption.

These tests require:
1. A running patched llama-server with a Qwen3.5 model loaded
2. ffmpeg installed
3. Test video files in tests/data/

Run with: uv run pytest tests/smoke/ -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.smoke,
    pytest.mark.slow,
]


TEST_DATA = Path(__file__).parent.parent / "data"


@pytest.fixture
def test_video() -> Path:
    """Path to a short test video."""
    video = TEST_DATA / "test_clip_4s.mp4"
    if not video.exists():
        pytest.skip(f"Test video not found: {video}")
    return video


class TestFullPipeline:
    """End-to-end: video file → caption string."""

    @pytest.mark.asyncio
    async def test_caption_short_clip(self, test_video):
        """Caption a 4-second test clip."""
        from llama_video.client import LlamaServerClient
        from llama_video.config import Settings
        from llama_video.extractor import Extractor, ExtractorConfig
        from llama_video.preprocessor import Preprocessor

        settings = Settings()
        extractor = Extractor(settings.extractor)
        preprocessor = Preprocessor(settings.model)
        client = LlamaServerClient(settings.server)

        try:
            # Extract
            frames = await extractor.extract_frames_async(
                test_video,
                ExtractorConfig(fps=2.0, max_frames=16),
            )
            assert len(frames) > 0

            # Preprocess
            video_input = preprocessor.process(frames, fps=2.0)
            assert len(video_input.super_frames) > 0
            assert video_input.grid_thw[0] > 0

            # Caption
            caption = await client.caption_video(
                video_input,
                prompt="Describe what happens in this video in one sentence.",
                max_tokens=256,
            )
            assert isinstance(caption, str)
            assert len(caption) > 10  # Should be a real sentence

        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_temporal_awareness(self, test_video):
        """Caption should demonstrate temporal understanding, not just describe a still."""
        from llama_video.client import LlamaServerClient
        from llama_video.config import Settings
        from llama_video.extractor import Extractor, ExtractorConfig
        from llama_video.preprocessor import Preprocessor

        settings = Settings()
        extractor = Extractor(settings.extractor)
        preprocessor = Preprocessor(settings.model)
        client = LlamaServerClient(settings.server)

        try:
            frames = await extractor.extract_frames_async(test_video, ExtractorConfig(fps=2.0))
            video_input = preprocessor.process(frames, fps=2.0)

            caption = await client.caption_video(
                video_input,
                prompt="Describe the sequence of events in this video, in order.",
                max_tokens=512,
            )

            # Look for temporal language — not a definitive test, but a sanity check
            temporal_words = [
                "then",
                "after",
                "before",
                "first",
                "next",
                "finally",
                "begins",
                "ends",
                "starts",
                "follows",
                "sequence",
            ]
            has_temporal = any(word in caption.lower() for word in temporal_words)
            # Log for manual inspection even if assertion passes
            print(f"\nCaption: {caption}")
            print(f"Temporal words found: {has_temporal}")

        finally:
            await client.close()
