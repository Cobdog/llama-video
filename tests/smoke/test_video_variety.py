"""Smoke tests: video variety validation across different clips.

Tests all 6 testvid clips at 2fps with keyword spot-checks,
and a subset at 1fps to validate different frame rates.

Requirements:
    - Patched llama-server running on LLAMA_SERVER_URL (default: http://localhost:7801)
    - Qwen3.5 model loaded with mmproj
    - Test videos in testvid/ directory

Run: uv run pytest tests/smoke/test_video_variety.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.smoke,
    pytest.mark.slow,
    pytest.mark.timeout(300),
]

# Expected keywords for each video based on reference captions.
# These are distinctive terms that should appear in any reasonable caption.
VIDEO_KEYWORDS: dict[str, list[str]] = {
    "kiki": [
        "girl",
        "pink",
        "field",
        "flower",
        "radio",
        "grass",
    ],  # girl in pink dress in flower field
    "howl": ["girl", "hat", "mirror", "braid", "room"],  # girl with hat in front of mirror
    "arrietty_1": [
        "girl",
        "yellow",
        "dress",
        "mushroom",
        "chair",
        "plant",
    ],  # girl on mushroom chair
    "boy_and_heron": ["village", "people", "flag", "street", "japanese"],  # village with flags
    "coquelicots": ["school", "student", "building", "uniform", "walk"],  # students at school
    "earthsea": ["stair", "fall", "boy", "girl", "step"],  # boy falls off stairs
}

# Videos to test at 1fps (subset for efficiency)
FPS_1_TEST_VIDEOS = ["kiki", "howl", "arrietty_1"]


def get_video_path(testvid_dir: str, video_name: str) -> Path:
    """Get path to a test video file."""
    return Path(testvid_dir) / f"{video_name}.mp4"


async def caption_video_file(
    video_path: Path,
    fps: float = 2.0,
    prompt: str = "Describe what happens in this video in 2-3 sentences.",
    max_tokens: int = 512,
) -> str:
    """Run the full pipeline on a video file and return the caption.

    This helper uses the complete llama_video pipeline:
    extractor -> preprocessor -> client
    """
    from llama_video.client import LlamaServerClient
    from llama_video.config import Settings
    from llama_video.extractor import Extractor, ExtractorConfig
    from llama_video.preprocessor import Preprocessor

    settings = Settings()
    extractor = Extractor(settings.extractor)
    preprocessor = Preprocessor(settings.model)
    client = LlamaServerClient(settings.server)

    try:
        # Extract frames
        frames = await extractor.extract_frames_async(
            video_path,
            ExtractorConfig(fps=fps, max_frames=64),
        )
        assert len(frames) > 0, f"No frames extracted from {video_path}"

        # Preprocess into super-frames
        video_input = preprocessor.process(frames, fps=fps)
        assert len(video_input.super_frames) > 0, "No super-frames generated"

        # Get caption from llama-server
        caption = await client.caption_video(
            video_input,
            prompt=prompt,
            max_tokens=max_tokens,
        )
        return caption

    finally:
        await client.close()


class TestVideoVariety2fps:
    """Test all 6 testvid clips at 2fps."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("video_name", list(VIDEO_KEYWORDS.keys()))
    async def test_caption_is_non_empty(
        self,
        video_name: str,
        testvid_dir: str,
        server_ready: None,  # Ensures server is up
    ):
        """Each video should produce a non-empty caption."""
        video_path = get_video_path(testvid_dir, video_name)
        if not video_path.exists():
            pytest.skip(f"Test video not found: {video_path}")

        caption = await caption_video_file(video_path, fps=2.0)

        assert isinstance(caption, str), f"Caption should be string, got {type(caption)}"
        assert len(caption) > 20, f"Caption too short ({len(caption)} chars): {caption!r}"
        print(f"\n[{video_name}] Caption ({len(caption)} chars): {caption[:200]}...")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("video_name", list(VIDEO_KEYWORDS.keys()))
    async def test_caption_contains_relevant_keywords(
        self,
        video_name: str,
        testvid_dir: str,
        server_ready: None,
    ):
        """Caption should contain at least one keyword relevant to the video content."""
        video_path = get_video_path(testvid_dir, video_name)
        if not video_path.exists():
            pytest.skip(f"Test video not found: {video_path}")

        caption = await caption_video_file(video_path, fps=2.0)
        caption_lower = caption.lower()

        keywords = VIDEO_KEYWORDS[video_name]
        found_keywords = [kw for kw in keywords if kw.lower() in caption_lower]

        # Require at least one keyword match
        assert len(found_keywords) >= 1, (
            f"Caption for {video_name} missing expected keywords.\n"
            f"Expected one of: {keywords}\n"
            f"Caption: {caption!r}"
        )
        print(f"\n[{video_name}] Found keywords: {found_keywords}")


class TestVideoVariety1fps:
    """Test a subset of videos at 1fps to validate different frame rates."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("video_name", FPS_1_TEST_VIDEOS)
    async def test_caption_at_1fps(
        self,
        video_name: str,
        testvid_dir: str,
        server_ready: None,
    ):
        """Videos should produce reasonable captions at 1fps (half the default rate)."""
        video_path = get_video_path(testvid_dir, video_name)
        if not video_path.exists():
            pytest.skip(f"Test video not found: {video_path}")

        caption = await caption_video_file(video_path, fps=1.0)

        assert isinstance(caption, str), f"Caption should be string, got {type(caption)}"
        assert len(caption) > 20, f"Caption too short ({len(caption)} chars): {caption!r}"

        # Also check keywords at 1fps
        caption_lower = caption.lower()
        keywords = VIDEO_KEYWORDS[video_name]
        found_keywords = [kw for kw in keywords if kw.lower() in caption_lower]

        print(f"\n[{video_name} @ 1fps] Caption ({len(caption)} chars): {caption[:200]}...")
        print(f"[{video_name} @ 1fps] Found keywords: {found_keywords}")
