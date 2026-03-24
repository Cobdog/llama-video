"""Smoke tests: temporal reasoning reliability.

Validates that video mode (with M-RoPE temporal positions) produces
better temporal understanding than image mode (frames as separate images).

Requirements:
    - Patched llama-server running on LLAMA_SERVER_URL (default: http://localhost:7801)
    - Qwen3.5 model loaded with mmproj
    - Test videos in testvid/ directory

Run: uv run pytest tests/smoke/test_temporal_reasoning.py -v
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import httpx
import numpy as np
import pytest
from PIL import Image

pytestmark = [
    pytest.mark.smoke,
    pytest.mark.slow,
    pytest.mark.timeout(300),
]

# Words that indicate temporal/sequential understanding
TEMPORAL_WORDS = [
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
    "changes",
    "moves",
    "transitions",
    "continues",
    "later",
    "earlier",
    "during",
    "while",
    "gradually",
    "suddenly",
    "eventually",
    "meanwhile",
    "subsequently",
]


def get_video_path(testvid_dir: str, video_name: str) -> Path:
    """Get path to a test video file."""
    return Path(testvid_dir) / f"{video_name}.mp4"


async def caption_video_mode(
    video_path: Path,
    server_url: str,
    prompt: str,
) -> str:
    """Caption a video using video mode (temporal positions enabled)."""
    from llama_video.client import LlamaServerClient
    from llama_video.config import Settings
    from llama_video.extractor import Extractor, ExtractorConfig
    from llama_video.preprocessor import Preprocessor

    settings = Settings()
    extractor = Extractor(settings.extractor)
    preprocessor = Preprocessor(settings.model)
    client = LlamaServerClient(settings.server)

    try:
        frames = await extractor.extract_frames_async(
            video_path,
            ExtractorConfig(fps=2.0, max_frames=16),
        )
        video_input = preprocessor.process(frames, fps=2.0)
        return await client.caption_video(
            video_input,
            prompt=prompt,
            max_tokens=1024,
        )
    finally:
        await client.close()


async def caption_image_mode(
    video_path: Path,
    server_url: str,
    prompt: str,
) -> str:
    """Caption a video using image mode (no temporal positions, frames as separate images).

    Sends the same frames but without is_video flag or temporal positions,
    so the model treats them as independent images.
    """
    from llama_video.config import ModelConfig, Settings, get_preset
    from llama_video.extractor import Extractor, ExtractorConfig
    from llama_video.preprocessor import Preprocessor

    settings = Settings()
    extractor = Extractor(settings.extractor)
    preprocessor = Preprocessor(settings.model)
    model = ModelConfig.qwen35()
    preset = get_preset("default")

    frames = await extractor.extract_frames_async(
        video_path,
        ExtractorConfig(fps=2.0, max_frames=16),
    )
    video_input = preprocessor.process(frames, fps=2.0)

    # Build image-only message (no mm_processor_kwargs)
    content: list[dict[str, object]] = []
    mean = np.array(model.image_mean, dtype=np.float32).reshape(3, 1, 1)
    std = np.array(model.image_std, dtype=np.float32).reshape(3, 1, 1)

    for sf in video_input.super_frames:
        for offset in (0, 3):
            channels = sf.data[offset : offset + 3]
            channels = channels * std + mean
            channels = np.clip(channels * 255, 0, 255).astype(np.uint8)
            channels = np.transpose(channels, (1, 2, 0))
            img = Image.fromarray(channels, "RGB")
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=95)
            b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                }
            )

    content.append({"type": "text", "text": prompt})

    payload = {
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 1024,
        "temperature": preset.temperature,
        "top_p": preset.top_p,
        "top_k": preset.top_k,
        "min_p": preset.min_p,
        "presence_penalty": preset.presence_penalty,
        # No mm_processor_kwargs — image mode
    }

    with httpx.Client(base_url=server_url, timeout=300.0) as client:
        resp = client.post("/v1/chat/completions", json=payload)
        resp.raise_for_status()
        msg = resp.json()["choices"][0]["message"]
        caption = msg.get("content", "") or ""
        if not caption:
            caption = msg.get("reasoning_content", "") or ""
        return caption


def count_temporal_words(text: str) -> int:
    """Count how many temporal/sequence words appear in text."""
    text_lower = text.lower()
    return sum(1 for word in TEMPORAL_WORDS if word in text_lower)


class TestTemporalReasoningReliability:
    """Video mode should produce more temporal language than image mode."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("video_name", ["kiki", "howl", "arrietty_1"])
    async def test_sequence_prompt_video_vs_image(
        self,
        video_name: str,
        testvid_dir: str,
        server_ready: None,  # Ensures server is up
        server_url: str,
    ):
        """Video mode should contain temporal language for sequence prompts."""
        video_path = get_video_path(testvid_dir, video_name)
        if not video_path.exists():
            pytest.skip(f"Test video not found: {video_path}")

        prompt = "Describe the sequence of events in this video, in chronological order."

        video_caption = await caption_video_mode(video_path, server_url, prompt)
        image_caption = await caption_image_mode(video_path, server_url, prompt)

        video_temporal = count_temporal_words(video_caption)
        image_temporal = count_temporal_words(image_caption)

        print(f"\n--- {video_name}: temporal reasoning ---")
        print(f"Video mode ({video_temporal} temporal words): {video_caption[:300]}...")
        print(f"Image mode ({image_temporal} temporal words): {image_caption[:300]}...")

        # Video mode should have at least some temporal language
        assert video_temporal > 0, (
            f"Video mode caption for {video_name} had zero temporal words.\n"
            f"Caption: {video_caption!r}"
        )

    @pytest.mark.asyncio
    async def test_change_detection_prompt(
        self,
        testvid_dir: str,
        server_ready: None,
        server_url: str,
    ):
        """Prompt asking about changes should elicit temporal reasoning in video mode."""
        video_path = get_video_path(testvid_dir, "arrietty_1")
        if not video_path.exists():
            pytest.skip("arrietty_1.mp4 not found")

        prompt = "Does anything change during this video? Describe what changes and in what order."

        caption = await caption_video_mode(video_path, server_url, prompt)

        print("\n--- arrietty_1: change detection ---")
        print(f"Caption: {caption[:400]}...")

        temporal_count = count_temporal_words(caption)
        assert temporal_count > 0, (
            f"Change detection prompt got zero temporal words.\nCaption: {caption!r}"
        )

    @pytest.mark.asyncio
    async def test_first_and_last_prompt(
        self,
        testvid_dir: str,
        server_ready: None,
        server_url: str,
    ):
        """Asking about first/last events should produce ordered descriptions."""
        video_path = get_video_path(testvid_dir, "howl")
        if not video_path.exists():
            pytest.skip("howl.mp4 not found")

        prompt = "What happens at the beginning of this video and what happens at the end?"

        caption = await caption_video_mode(video_path, server_url, prompt)

        print("\n--- howl: first and last ---")
        print(f"Caption: {caption[:400]}...")

        # Should mention beginning/end/first/last concepts
        ordering_words = [
            "begin",
            "end",
            "first",
            "last",
            "start",
            "finish",
            "open",
            "close",
        ]
        caption_lower = caption.lower()
        matched = [w for w in ordering_words if w in caption_lower]
        assert len(matched) > 0, f"First/last prompt got no ordering words.\nCaption: {caption!r}"
