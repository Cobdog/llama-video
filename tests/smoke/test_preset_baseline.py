"""Smoke tests: baseline preset validation.

Validates that the default inference preset (Qwen thinking-mode recommended)
produces reasonable captions. Outputs are logged for human review.

Requirements:
    - Patched llama-server running (LLAMA_SERVER_URL)
    - ffmpeg installed
    - testvid/ directory with test clips

Run: uv run pytest tests/smoke/test_preset_baseline.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.smoke,
    pytest.mark.slow,
    pytest.mark.timeout(300),
]


def get_video_path(testvid_dir: str, video_name: str) -> Path:
    """Get path to a test video file."""
    return Path(testvid_dir) / f"{video_name}.mp4"


async def caption_with_preset(
    video_path: Path,
    server_url: str,
    preset_name: str,
) -> str:
    """Caption a video using a specific preset."""
    from llama_video.client import LlamaServerClient
    from llama_video.config import ServerConfig, Settings, get_preset
    from llama_video.extractor import Extractor, ExtractorConfig
    from llama_video.preprocessor import Preprocessor

    settings = Settings()
    extractor = Extractor(settings.extractor)
    preprocessor = Preprocessor(settings.model)
    server_config = ServerConfig(url=server_url)
    client = LlamaServerClient(server_config)
    preset = get_preset(preset_name)

    try:
        frames = await extractor.extract_frames_async(
            str(video_path),
            ExtractorConfig(fps=2.0, max_frames=16),
        )
        video_input = preprocessor.process(frames, fps=2.0)
        return await client.caption_video(
            video_input,
            prompt="Describe what happens in this video in detail.",
            max_tokens=1024,
            preset=preset,
        )
    finally:
        await client.close()


class TestPresetBaseline:
    """Validate the default preset produces reasonable captions."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("video_name", ["kiki", "howl", "boy_and_heron"])
    async def test_default_preset_caption_length(
        self,
        video_name: str,
        testvid_dir: str,
        server_url: str,
        server_ready: None,
    ):
        """Default preset should produce captions of reasonable length."""
        video_path = get_video_path(testvid_dir, video_name)
        if not video_path.exists():
            pytest.skip(f"Test video not found: {video_path}")

        caption = await caption_with_preset(video_path, server_url, "default")

        print(f"\n=== {video_name} - default preset ===")
        print(f"Length: {len(caption)} chars")
        print(f"Caption:\n{caption}")
        print("=" * 60)

        assert len(caption) > 50, (
            f"Caption too short ({len(caption)} chars) - preset may be misconfigured.\n"
            f"Caption: {caption!r}"
        )
        assert len(caption) < 5000, (
            f"Caption too long ({len(caption)} chars) - may indicate runaway generation."
        )

    @pytest.mark.asyncio
    async def test_default_preset_is_descriptive(
        self,
        testvid_dir: str,
        server_url: str,
        server_ready: None,
    ):
        """Default preset caption should contain descriptive language, not just noise."""
        video_path = get_video_path(testvid_dir, "kiki")
        if not video_path.exists():
            pytest.skip("Test video not found: kiki.mp4")

        caption = await caption_with_preset(video_path, server_url, "default")
        caption_lower = caption.lower()

        # A descriptive caption should have visual language
        visual_words = [
            "girl",
            "boy",
            "woman",
            "man",
            "person",
            "scene",
            "video",
            "color",
            "green",
            "red",
            "blue",
            "pink",
            "white",
            "black",
            "field",
            "room",
            "building",
            "tree",
            "flower",
            "sky",
            "walk",
            "sit",
            "stand",
            "move",
            "look",
            "wear",
        ]
        matched = [w for w in visual_words if w in caption_lower]

        print(f"\nVisual words matched: {matched}")

        assert len(matched) >= 3, (
            f"Caption matched only {len(matched)} visual words ({matched}).\n"
            f"Caption may not be descriptive enough.\n"
            f"Caption: {caption!r}"
        )
