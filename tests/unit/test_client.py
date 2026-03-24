"""Tests for llama-server client."""

from __future__ import annotations

import numpy as np
import pytest

from llama_video.client import LlamaServerClient, _parse_stream_state, parse_model_response
from llama_video.config import ServerConfig
from llama_video.preprocessor import VideoInput
from llama_video.types import SuperFrame


@pytest.fixture
def mock_video_input() -> VideoInput:
    """Minimal VideoInput for client tests."""
    rng = np.random.default_rng(42)
    super_frames = [
        SuperFrame(
            data=rng.standard_normal((6, 28, 28)).astype(np.float32),
            temporal_index=i,
            source_frames=(i * 2, i * 2 + 1),
        )
        for i in range(2)
    ]
    return VideoInput(
        super_frames=super_frames,
        grid_thw=(2, 1, 1),
        temporal_positions=[0, 1],
        fps=2.0,
        num_source_frames=4,
        resolution=(28, 28),
    )


class TestLlamaServerClient:
    """Test client message construction and error handling."""

    def test_build_video_message_has_correct_image_count(self, mock_video_input):
        """2 super-frames → 4 images (each super-frame splits to 2 images)."""
        client = LlamaServerClient(ServerConfig(url="http://localhost:8080"))
        message = client._build_video_message(mock_video_input, "Describe this video.")

        content = message["content"]
        image_entries = [c for c in content if c["type"] == "image_url"]
        text_entries = [c for c in content if c["type"] == "text"]

        assert len(image_entries) == 4  # 2 super-frames x 2 images each
        assert len(text_entries) == 1
        assert text_entries[0]["text"] == "Describe this video."

    def test_build_video_message_images_are_base64(self, mock_video_input):
        """Image URLs should be base64-encoded data URIs."""
        client = LlamaServerClient(ServerConfig(url="http://localhost:8080"))
        message = client._build_video_message(mock_video_input, "test")

        for entry in message["content"]:
            if entry["type"] == "image_url":
                url = entry["image_url"]["url"]
                assert url.startswith("data:image/jpeg;base64,")

    @pytest.mark.asyncio
    async def test_health_check_unreachable(self):
        """Health check returns False when server is unreachable."""
        client = LlamaServerClient(ServerConfig(url="http://localhost:1"))
        try:
            result = await client.health_check()
            assert result is False
        finally:
            await client.close()


class TestResponseParsing:
    """Test response parsing handles thinking models."""

    @pytest.mark.asyncio
    async def test_caption_falls_back_to_reasoning_content(self, respx_mock):
        """When content is empty, fall back to reasoning_content."""
        config = ServerConfig(url="http://test-server:8080")
        client = LlamaServerClient(config)

        # Mock a thinking-model response where content is empty
        respx_mock.post("http://test-server:8080/v1/chat/completions").respond(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "reasoning_content": "The video shows a red screen changing to blue.",
                        }
                    }
                ]
            },
        )

        # Create a minimal VideoInput
        sf = SuperFrame(
            data=np.zeros((6, 28, 28), dtype=np.float32),
            temporal_index=0,
            source_frames=(0, 1),
        )
        video_input = VideoInput(
            super_frames=[sf],
            grid_thw=(1, 1, 1),
            temporal_positions=[0],
            fps=2.0,
            num_source_frames=2,
            resolution=(28, 28),
        )

        result = await client.caption_video(video_input, "Describe this video.")
        await client.close()

        assert result == "The video shows a red screen changing to blue."

    @pytest.mark.asyncio
    async def test_caption_prefers_content_when_non_empty(self, respx_mock):
        """When content is non-empty, use it (don't fall back)."""
        config = ServerConfig(url="http://test-server:8080")
        client = LlamaServerClient(config)

        respx_mock.post("http://test-server:8080/v1/chat/completions").respond(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "A video of a sunset.",
                            "reasoning_content": "Let me analyze this...",
                        }
                    }
                ]
            },
        )

        sf = SuperFrame(
            data=np.zeros((6, 28, 28), dtype=np.float32),
            temporal_index=0,
            source_frames=(0, 1),
        )
        video_input = VideoInput(
            super_frames=[sf],
            grid_thw=(1, 1, 1),
            temporal_positions=[0],
            fps=2.0,
            num_source_frames=2,
            resolution=(28, 28),
        )

        result = await client.caption_video(video_input, "Describe this video.")
        await client.close()

        assert result == "A video of a sunset."


class TestPresetPassthrough:
    """Test that preset parameters are sent to the server."""

    @pytest.mark.asyncio
    async def test_caption_sends_preset_params(self, respx_mock, mock_video_input):
        """Verify all preset params are included in the request payload."""
        config = ServerConfig(url="http://test-server:8080")
        client = LlamaServerClient(config)

        route = respx_mock.post("http://test-server:8080/v1/chat/completions").respond(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "A caption."}}]},
        )

        from llama_video.config import get_preset

        preset = get_preset("default")
        result = await client.caption_video(
            mock_video_input,
            "Describe this video.",
            max_tokens=512,
            preset=preset,
        )
        await client.close()

        # Check the sent payload
        sent = route.calls[0].request
        import json

        body = json.loads(sent.content)
        assert body["temperature"] == 1.0
        assert body["top_p"] == 0.95
        assert body["top_k"] == 20
        assert body["min_p"] == 0.0
        assert body["presence_penalty"] == 1.5
        assert body["max_tokens"] == 512
        assert "mm_processor_kwargs" in body
        assert result == "A caption."

    @pytest.mark.asyncio
    async def test_caption_uses_default_preset_when_none(self, respx_mock, mock_video_input):
        """When no preset is passed, use the default preset values."""
        config = ServerConfig(url="http://test-server:8080")
        client = LlamaServerClient(config)

        route = respx_mock.post("http://test-server:8080/v1/chat/completions").respond(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "A caption."}}]},
        )

        _result = await client.caption_video(
            mock_video_input,
            "Describe this video.",
        )
        await client.close()

        import json

        body = json.loads(route.calls[0].request.content)
        # Should use default preset values
        assert body["temperature"] == 1.0
        assert body["presence_penalty"] == 1.5


class TestCachePrompt:
    """Test cache_prompt passthrough to server payload."""

    @pytest.mark.asyncio
    async def test_cache_prompt_defaults_true(self, respx_mock, mock_video_input):
        """Default cache_prompt is True."""
        config = ServerConfig(url="http://test-server:8080")
        client = LlamaServerClient(config)
        route = respx_mock.post("http://test-server:8080/v1/chat/completions").respond(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
        )
        await client.caption_video(mock_video_input, "test")
        await client.close()
        import json

        body = json.loads(route.calls[0].request.content)
        assert body["cache_prompt"] is True

    @pytest.mark.asyncio
    async def test_cache_prompt_false_passthrough(self, respx_mock, mock_video_input):
        """cache_prompt=False is sent to server."""
        config = ServerConfig(url="http://test-server:8080")
        client = LlamaServerClient(config)
        route = respx_mock.post("http://test-server:8080/v1/chat/completions").respond(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
        )
        await client.caption_video(mock_video_input, "test", cache_prompt=False)
        await client.close()
        import json

        body = json.loads(route.calls[0].request.content)
        assert body["cache_prompt"] is False

    @pytest.mark.asyncio
    async def test_image_cache_prompt_passthrough(self, respx_mock, tmp_path):
        """cache_prompt is threaded through image captioning."""
        from PIL import Image as PILImage

        img = PILImage.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        img_path = tmp_path / "test.jpg"
        img.save(str(img_path))

        config = ServerConfig(url="http://test-server:8080")
        client = LlamaServerClient(config)
        route = respx_mock.post("http://test-server:8080/v1/chat/completions").respond(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
        )
        await client.caption_image(str(img_path), "test", cache_prompt=False)
        await client.close()
        import json

        body = json.loads(route.calls[0].request.content)
        assert body["cache_prompt"] is False


class TestImageCaptioning:
    """Test image captioning via client."""

    @pytest.mark.asyncio
    async def test_caption_image_sends_no_mm_kwargs(self, respx_mock, tmp_path):
        """Image captioning should not send mm_processor_kwargs."""
        from PIL import Image as PILImage

        # Create a test image
        img = PILImage.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        img_path = tmp_path / "test.jpg"
        img.save(str(img_path))

        config = ServerConfig(url="http://test-server:8080")
        client = LlamaServerClient(config)

        route = respx_mock.post("http://test-server:8080/v1/chat/completions").respond(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "A sunset."}}]},
        )

        result = await client.caption_image(str(img_path), "Describe this image.")
        await client.close()

        import json

        body = json.loads(route.calls[0].request.content)
        assert "mm_processor_kwargs" not in body
        assert body["messages"][0]["content"][0]["type"] == "image_url"
        assert result == "A sunset."


class TestParseModelResponse:
    """Test parse_model_response for thinking tag extraction."""

    def test_empty_input(self):
        assert parse_model_response("") == ("", "", False)

    def test_no_thinking(self):
        assert parse_model_response("A sunset.") == ("A sunset.", "", False)

    def test_complete_thinking(self):
        text = "<think>Let me analyze this.</think>A beautiful sunset."
        cap, thinking, truncated = parse_model_response(text)
        assert cap == "A beautiful sunset."
        assert thinking == "Let me analyze this."
        assert truncated is False

    def test_truncated_thinking(self):
        text = "<think>Let me analyze this but I ran out of"
        cap, thinking, truncated = parse_model_response(text)
        assert cap == ""
        assert "Let me analyze this" in thinking
        assert truncated is True


class TestParseStreamState:
    """Test _parse_stream_state for incremental streaming."""

    def test_no_think_tag_yet(self):
        thinking, caption, still = _parse_stream_state("Hello world")
        assert thinking == ""
        assert caption == "Hello world"
        assert still is False

    def test_think_tag_open_still_thinking(self):
        thinking, caption, still = _parse_stream_state("<think>analyzing the")
        assert thinking == "analyzing the"
        assert caption == ""
        assert still is True

    def test_think_tag_closed(self):
        thinking, caption, still = _parse_stream_state(
            "<think>done thinking</think>The caption"
        )
        assert thinking == "done thinking"
        assert caption == "The caption"
        assert still is False

    def test_empty_input(self):
        thinking, caption, still = _parse_stream_state("")
        assert thinking == ""
        assert caption == ""
        assert still is False

    def test_partial_close_tag(self):
        # </thin is not </think> — still inside thinking
        thinking, caption, still = _parse_stream_state("<think>reasoning</thin")
        assert "reasoning" in thinking
        assert still is True
