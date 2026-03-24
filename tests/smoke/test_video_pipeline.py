"""Smoke tests for the video pipeline.

Requirements:
    - Patched llama-server running on LLAMA_SERVER_URL (default: http://localhost:7801)
    - Qwen3.5 model loaded with mmproj

Run: uv run pytest tests/smoke/test_video_pipeline.py -v
"""

import base64
import os
import struct

import httpx
import pytest

SERVER_URL = os.environ.get("LLAMA_SERVER_URL", "http://localhost:7801")
TIMEOUT = 120.0  # video processing can be slow


def make_solid_color_png(width: int, height: int, r: int, g: int, b: int) -> bytes:
    """Create a minimal valid PNG with a solid color (no compression library needed)."""
    import zlib

    def chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        crc = struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        return struct.pack(">I", len(data)) + c + crc

    # IHDR
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    # IDAT: uncompressed rows, each row = filter_byte + RGB * width
    raw_data = b""
    for _ in range(height):
        raw_data += b"\x00"  # filter: none
        raw_data += bytes([r, g, b]) * width
    idat_data = zlib.compress(raw_data)

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", ihdr_data)
    png += chunk(b"IDAT", idat_data)
    png += chunk(b"IEND", b"")
    return png


def image_url_from_png(png_bytes: bytes) -> dict:
    """Create an image_url content part from PNG bytes."""
    b64 = base64.b64encode(png_bytes).decode()
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{b64}"},
    }


def send_chat(
    messages: list[dict],
    mm_processor_kwargs: dict | None = None,
    max_tokens: int = 512,
) -> str:
    """Send a chat completion request and return the response text.

    Returns content if non-empty, otherwise falls back to reasoning_content
    (Qwen3.5 with thinking mode spends tokens on reasoning before content).
    """
    body: dict = {
        "model": "qwen3.5",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }
    if mm_processor_kwargs:
        body["mm_processor_kwargs"] = mm_processor_kwargs

    with httpx.Client(timeout=TIMEOUT) as client:
        resp = client.post(f"{SERVER_URL}/v1/chat/completions", json=body)
        resp.raise_for_status()
        data = resp.json()
        msg = data["choices"][0]["message"]
        content = msg.get("content", "")
        if content:
            return content
        # Fall back to reasoning_content for thinking models
        return msg.get("reasoning_content", "")


@pytest.fixture
def red_frame() -> bytes:
    return make_solid_color_png(224, 224, 255, 0, 0)


@pytest.fixture
def blue_frame() -> bytes:
    return make_solid_color_png(224, 224, 0, 0, 255)


@pytest.fixture
def green_frame() -> bytes:
    return make_solid_color_png(224, 224, 0, 255, 0)


class TestVideoSmoke:
    """Level 1: Does the server accept video input and return a response?"""

    def test_video_two_frames_returns_caption(self, red_frame, blue_frame):
        """Send a 2-frame video and verify we get a non-empty response."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this video."},
                    image_url_from_png(red_frame),
                    image_url_from_png(blue_frame),
                ],
            }
        ]
        result = send_chat(
            messages,
            mm_processor_kwargs={
                "is_video": True,
                "temporal_positions": [0],
            },
        )
        assert len(result) > 0, "Expected non-empty caption"

    def test_video_four_frames_returns_caption(self, red_frame, blue_frame, green_frame):
        """Send a 4-frame video and verify we get a non-empty response."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this video."},
                    image_url_from_png(red_frame),
                    image_url_from_png(blue_frame),
                    image_url_from_png(green_frame),
                    image_url_from_png(red_frame),
                ],
            }
        ]
        result = send_chat(
            messages,
            mm_processor_kwargs={
                "is_video": True,
                "temporal_positions": [0, 1],
            },
        )
        assert len(result) > 0, "Expected non-empty caption"


class TestTemporalDifferentiation:
    """Level 2: Does video mode produce different output than image mode?"""

    def test_video_vs_images_differ(self, red_frame, blue_frame):
        """Same frames sent as video vs two separate images must produce different outputs."""
        prompt_text = "Describe what you see."

        # Video mode: 2 frames as video
        video_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    image_url_from_png(red_frame),
                    image_url_from_png(blue_frame),
                ],
            }
        ]
        video_result = send_chat(
            video_messages,
            mm_processor_kwargs={
                "is_video": True,
                "temporal_positions": [0],
            },
        )

        # Image mode: same 2 frames as separate images (no video kwargs)
        image_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    image_url_from_png(red_frame),
                    image_url_from_png(blue_frame),
                ],
            }
        ]
        image_result = send_chat(image_messages)

        assert video_result != image_result, (
            "Video and image modes produced identical output — "
            "M-RoPE temporal positions may not be reaching the model.\n"
            f"Video: {video_result!r}\n"
            f"Image: {image_result!r}"
        )


class TestTemporalReasoning:
    """Level 3: Can the model reason about temporal sequence?"""

    def test_sequence_description(self, red_frame, blue_frame):
        """Send video with visible color change and ask about sequence."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "This video shows a color changing over time. "
                        "What color appears first, and what color appears next?",
                    },
                    image_url_from_png(red_frame),
                    image_url_from_png(blue_frame),
                ],
            }
        ]
        result = send_chat(
            messages,
            mm_processor_kwargs={
                "is_video": True,
                "temporal_positions": [0],
            },
            max_tokens=256,
        )
        # Qualitative check: response should mention color(s)
        result_lower = result.lower()
        has_color = any(c in result_lower for c in ["red", "blue", "color", "chang"])
        assert has_color, (
            f"Expected model to mention colors in temporal reasoning.\nGot: {result!r}"
        )
