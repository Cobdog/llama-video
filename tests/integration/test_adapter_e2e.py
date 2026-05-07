"""End-to-end integration tests for the adapter pipeline against a real llama-server.

These tests exercise the full path: frame extraction -> adapter preprocessing ->
payload construction -> HTTP inference -> response parsing. They require:
  - A running llama-server (router mode) with auto-load capability
  - Test video files in testvid/
  - ffmpeg on PATH

Run with:  pytest tests/integration/test_adapter_e2e.py -v -s --timeout=300
Skip with: pytest -k "not e2e_adapter"
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import httpx
import pytest

from llama_video.adapters import AdapterPreset, get_adapter
from llama_video.adapters.gemma import GEMMA_PRESET, GemmaAdapter
from llama_video.client import LlamaServerClient
from llama_video.config import ServerConfig
from llama_video.extractor import Extractor, ExtractorConfig

logger = logging.getLogger(__name__)

TESTVID_DIR = Path(__file__).parent.parent.parent / "testvid"
SERVER_URL = "http://localhost:7800"
MODEL_NAME = "MM-Sprinkle-Gemma4-31B-Q4"
CAPTION_PROMPT = "Describe what is happening in this video clip in detail."


# ── Helpers ────────────────────────────────────────────────────────────


def _skip_if_no_server() -> None:
    """Skip test if llama-server is not reachable."""
    try:
        r = httpx.get(f"{SERVER_URL}/health", timeout=5.0)
        if r.status_code != 200:
            pytest.skip(f"Server returned {r.status_code}")
    except (httpx.ConnectError, httpx.TimeoutException):
        pytest.skip("llama-server not running on port 7800")


def _skip_if_no_video(name: str) -> Path:
    """Return video path or skip test if file missing."""
    p = TESTVID_DIR / name
    if not p.exists():
        pytest.skip(f"Test video not found: {p}")
    return p


@pytest.fixture
def server_config() -> ServerConfig:
    return ServerConfig(url=SERVER_URL, model_name=MODEL_NAME, timeout=300.0)


@pytest.fixture
def client(server_config: ServerConfig) -> LlamaServerClient:
    return LlamaServerClient(server_config)


@pytest.fixture
def gemma() -> GemmaAdapter:
    return GemmaAdapter()


@pytest.fixture
def coquelicots() -> Path:
    return _skip_if_no_video("coquelicots.mp4")


# ── Tests: Gemma4 adapter pipeline ───────────────────────────────────


class TestGemmaAdapterE2E:
    """End-to-end test of the Gemma4 adapter pipeline."""

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_full_gemma_pipeline(self, client, gemma, coquelicots):
        """Extract frames -> preprocess -> build payload -> infer -> parse.

        This tests the entire adapter pipeline against a real llama-server
        using the Gemma4 model. The router will auto-load the model on the
        first request, which can take 30-60 seconds.
        """
        _skip_if_no_server()

        # 1. Extract frames at Gemma4's native 1 FPS
        extractor = Extractor()
        config = ExtractorConfig(fps=gemma.default_fps, max_frames=gemma.max_frames)
        frames = await extractor.extract_frames_async(str(coquelicots), config)

        assert len(frames) > 0, "No frames extracted"
        assert len(frames) <= gemma.max_frames
        logger.info("Extracted %d frames from %s", len(frames), coquelicots.name)

        # 2. Preprocess through adapter
        video_input = gemma.preprocess(frames, fps=gemma.default_fps)
        assert len(video_input.super_frames) == len(frames)
        assert video_input.grid_thw[0] == len(frames)

        # 3. Build payload
        payload = gemma.build_payload(
            video_input,
            prompt=CAPTION_PROMPT,
            max_tokens=1024,
            model_name=MODEL_NAME,
        )
        assert payload["model"] == MODEL_NAME
        assert payload["temperature"] == GEMMA_PRESET.temperature
        assert payload["top_k"] == GEMMA_PRESET.top_k
        # Gemma4 should NOT have mm_processor_kwargs
        assert "mm_processor_kwargs" not in payload
        # Messages: system + user
        assert len(payload["messages"]) == 2
        # User message: images interleaved with timestamps, prompt last
        user_content = payload["messages"][1]["content"]
        assert user_content[-1]["type"] == "text"
        # Timestamp text precedes each image; verify images are present
        image_items = [c for c in user_content if c["type"] == "image_url"]
        assert len(image_items) == len(frames)
        logger.info("Payload built: %d content items", len(user_content))

        # 4. Send to llama-server
        t0 = time.monotonic()
        raw = await client.send_completion(payload)
        elapsed = time.monotonic() - t0
        logger.info("Inference completed in %.1fs, response: %d chars", elapsed, len(raw))

        assert len(raw) > 0, "Empty response from server"

        # 5. Parse response
        caption, thinking, truncated = gemma.parse_response(raw)
        logger.info(
            "Parsed: caption=%d chars, thinking=%d chars, truncated=%s",
            len(caption),
            len(thinking),
            truncated,
        )

        # The model should produce SOMETHING — either a caption or thinking
        assert len(caption) > 0 or len(thinking) > 0, (
            f"Model produced no output. Raw response (first 500 chars): {raw[:500]}"
        )

        # If we got thinking but no caption, that's truncation — acceptable
        # for first load but worth noting
        if not caption and thinking:
            logger.warning(
                "Model only produced thinking (%d chars), no caption. "
                "This may indicate max_tokens was too low or model is still warming up.",
                len(thinking),
            )

        await client.close()

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_gemma_adapter_properties(self, gemma):
        """Verify GemmaAdapter properties match expectations."""
        assert gemma.name == "gemma4"
        assert gemma.default_fps == 1.0
        assert gemma.max_duration_seconds == 60.0
        assert gemma.max_frames == 60
        assert gemma.default_preset == GEMMA_PRESET
        assert gemma.default_preset.top_k == 64
        assert gemma.default_preset.presence_penalty == 0.0

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_adapter_registry_gemma(self):
        """Verify gemma4 is registered and retrievable."""
        adapter = get_adapter("gemma4")
        assert isinstance(adapter, GemmaAdapter)
        assert adapter.name == "gemma4"

        adapter_lower = get_adapter("GEMMA4")
        assert isinstance(adapter_lower, GemmaAdapter)

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_auto_detect_gemma_model(self):
        """Auto-detect should identify Gemma4 model from llama-server."""
        _skip_if_no_server()

        from llama_video.adapters.detect import detect_adapter

        adapter_name = await detect_adapter(SERVER_URL, timeout=10.0)
        # Server has many models; the first loaded/available one determines the result.
        # We just verify detection works without error.
        assert adapter_name in ("gemma4", "qwen3.5"), f"Unexpected adapter: {adapter_name}"
        logger.info("Auto-detected adapter: %s", adapter_name)

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_gemma_with_custom_preset(self, client, gemma, coquelicots):
        """Verify custom preset overrides work end-to-end."""
        _skip_if_no_server()

        extractor = Extractor()
        config = ExtractorConfig(fps=gemma.default_fps, max_frames=4)
        frames = await extractor.extract_frames_async(str(coquelicots), config)
        assert len(frames) > 0

        video_input = gemma.preprocess(frames, fps=gemma.default_fps)

        custom_preset = AdapterPreset(
            temperature=0.7,
            top_p=0.9,
            top_k=32,
            min_p=0.05,
            presence_penalty=0.1,
        )

        payload = gemma.build_payload(
            video_input,
            prompt="Briefly describe this scene.",
            max_tokens=512,
            preset=custom_preset,
            model_name=MODEL_NAME,
        )

        assert payload["temperature"] == 0.7
        assert payload["top_k"] == 32
        assert payload["min_p"] == 0.05

        raw = await client.send_completion(payload)
        assert len(raw) > 0

        caption, _, _ = gemma.parse_response(raw)
        logger.info("Custom preset caption: %d chars", len(caption))

        await client.close()


class TestGemmaPreprocessingParity:
    """Verify GemmaAdapter preprocessing produces valid, consistent output."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_frame_timestamps_are_integers(self, gemma, coquelicots):
        """Gemma4 uses integer-second timestamps (MM:SS)."""
        extractor = Extractor()
        config = ExtractorConfig(fps=1.0, max_frames=4)
        frames = await extractor.extract_frames_async(str(coquelicots), config)

        video_input = gemma.preprocess(frames, fps=1.0)

        # All temporal positions should be integers (rounded seconds)
        for pos in video_input.temporal_positions:
            assert isinstance(pos, int), f"Expected int, got {type(pos)}: {pos}"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_no_mm_processor_kwargs_in_payload(self, gemma, coquelicots):
        """Gemma4 payload must NOT contain mm_processor_kwargs."""
        extractor = Extractor()
        config = ExtractorConfig(fps=1.0, max_frames=4)
        frames = await extractor.extract_frames_async(str(coquelicots), config)

        video_input = gemma.preprocess(frames, fps=1.0)
        payload = gemma.build_payload(video_input, prompt="test", model_name=MODEL_NAME)

        assert "mm_processor_kwargs" not in payload
        # Verify images are present as base64 data URIs
        user_content = payload["messages"][1]["content"]
        image_items = [c for c in user_content if c["type"] == "image_url"]
        assert len(image_items) == len(frames)
        for item in image_items:
            assert item["image_url"]["url"].startswith("data:image/jpeg;base64,")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_images_before_text_ordering(self, gemma, coquelicots):
        """Gemma4 requires all images before text in user message."""
        extractor = Extractor()
        config = ExtractorConfig(fps=1.0, max_frames=4)
        frames = await extractor.extract_frames_async(str(coquelicots), config)

        video_input = gemma.preprocess(frames, fps=1.0)
        payload = gemma.build_payload(video_input, prompt="Describe this.", model_name=MODEL_NAME)

        user_content = payload["messages"][1]["content"]
        # Find the text prompt position
        text_indices = [i for i, c in enumerate(user_content) if c["type"] == "text" and len(c["text"]) > 10]
        image_indices = [i for i, c in enumerate(user_content) if c["type"] == "image_url"]

        # All images must come before the main text prompt
        if text_indices and image_indices:
            assert max(image_indices) < text_indices[0], (
                f"Images at {image_indices} but text at {text_indices} — "
                "Gemma4 requires images before text"
            )
