"""Tests for GemmaAdapter — verifies Gemma4-specific video pipeline."""

import numpy as np
import pytest

from llama_video.adapters.base import AdapterPreset
from llama_video.adapters.gemma import GEMMA_PRESET, GemmaAdapter
from llama_video.errors import PreprocessingError
from llama_video.types import Frame


def _make_frame(w: int = 224, h: int = 224, index: int = 0) -> Frame:
    data = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    return Frame(data=data, index=index, timestamp=index * 1.0, width=w, height=h)


class TestGemmaAdapterPreprocess:
    """Verify GemmaAdapter.preprocess produces correct individual-frame output."""

    def test_basic_preprocess(self):
        adapter = GemmaAdapter()
        frames = [_make_frame(index=i) for i in range(8)]
        vi = adapter.preprocess(frames, fps=1.0)

        assert len(vi.super_frames) == 8
        assert vi.grid_thw == (8, 1, 1)
        assert vi.fps == 1.0
        assert vi.num_source_frames == 8

    def test_single_frame(self):
        adapter = GemmaAdapter()
        frames = [_make_frame(index=0)]
        vi = adapter.preprocess(frames, fps=1.0)

        assert len(vi.super_frames) == 1
        assert vi.grid_thw == (1, 1, 1)

    def test_max_frames_allowed(self):
        adapter = GemmaAdapter()
        frames = [_make_frame(index=i) for i in range(60)]
        vi = adapter.preprocess(frames, fps=1.0)
        assert len(vi.super_frames) == 60

    def test_exceeds_max_frames_raises(self):
        adapter = GemmaAdapter()
        frames = [_make_frame(index=i) for i in range(61)]
        with pytest.raises(PreprocessingError, match="max 60 frames"):
            adapter.preprocess(frames, fps=1.0)

    def test_empty_frames_raises(self):
        adapter = GemmaAdapter()
        with pytest.raises(PreprocessingError, match="No frames"):
            adapter.preprocess([], fps=1.0)

    def test_temporal_positions_are_timestamps(self):
        adapter = GemmaAdapter()
        frames = [
            Frame(
                data=np.zeros((28, 28, 3), dtype=np.uint8),
                index=i,
                timestamp=i * 1.0,
                width=28,
                height=28,
            )
            for i in range(5)
        ]
        vi = adapter.preprocess(frames, fps=1.0)
        assert vi.temporal_positions == [0, 1, 2, 3, 4]

    def test_super_frame_data_is_3channel(self):
        adapter = GemmaAdapter()
        frames = [_make_frame(index=0)]
        vi = adapter.preprocess(frames, fps=1.0)
        # Gemma stores 3-channel (C, H, W), not 6-channel like Qwen
        assert vi.super_frames[0].data.shape[0] == 3

    def test_resolution_preserved(self):
        adapter = GemmaAdapter()
        frames = [_make_frame(w=640, h=480, index=0)]
        vi = adapter.preprocess(frames, fps=1.0)
        assert vi.resolution == (640, 480)


class TestGemmaAdapterPayload:
    """Verify GemmaAdapter.build_payload produces correct structure."""

    def test_payload_no_mm_processor_kwargs(self):
        adapter = GemmaAdapter()
        frames = [_make_frame(index=i) for i in range(4)]
        vi = adapter.preprocess(frames, fps=1.0)
        payload = adapter.build_payload(vi, "Describe this video.")

        assert "mm_processor_kwargs" not in payload

    def test_payload_has_messages(self):
        adapter = GemmaAdapter()
        frames = [_make_frame(index=i) for i in range(4)]
        vi = adapter.preprocess(frames, fps=1.0)
        payload = adapter.build_payload(vi, "Test prompt")

        assert len(payload["messages"]) == 2
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][1]["role"] == "user"

    def test_images_before_text(self):
        adapter = GemmaAdapter()
        frames = [_make_frame(index=i) for i in range(4)]
        vi = adapter.preprocess(frames, fps=1.0)
        payload = adapter.build_payload(vi, "Test prompt")
        content = payload["messages"][1]["content"]

        # Last entry should be the text prompt
        assert content[-1]["type"] == "text"
        assert content[-1]["text"] == "Test prompt"

        # All image entries should come before the final text
        image_entries = [c for c in content[:-1] if c["type"] == "image_url"]
        assert len(image_entries) == 4

    def test_timestamps_in_payload(self):
        adapter = GemmaAdapter()
        frames = [
            Frame(
                data=np.zeros((28, 28, 3), dtype=np.uint8),
                index=i,
                timestamp=float(i),
                width=28,
                height=28,
            )
            for i in range(3)
        ]
        vi = adapter.preprocess(frames, fps=1.0)
        payload = adapter.build_payload(vi, "Test")
        content = payload["messages"][1]["content"]

        # Each frame has timestamp text + image
        timestamp_texts = [
            c["text"] for c in content if c["type"] == "text" and c["text"] != "Test"
        ]
        assert "00:00 " in timestamp_texts[0]
        assert "00:01 " in timestamp_texts[1]
        assert "00:02 " in timestamp_texts[2]

    def test_payload_sampler_defaults(self):
        adapter = GemmaAdapter()
        frames = [_make_frame(index=i) for i in range(4)]
        vi = adapter.preprocess(frames, fps=1.0)
        payload = adapter.build_payload(vi, "Test")

        assert payload["temperature"] == 1.0
        assert payload["top_p"] == 0.95
        assert payload["top_k"] == 64
        assert payload["min_p"] == 0.0
        assert payload["presence_penalty"] == 0.0

    def test_payload_custom_preset(self):
        adapter = GemmaAdapter()
        frames = [_make_frame(index=i) for i in range(4)]
        vi = adapter.preprocess(frames, fps=1.0)
        custom = AdapterPreset(temperature=0.5, top_p=0.9, top_k=10)
        payload = adapter.build_payload(vi, "Test", preset=custom)

        assert payload["temperature"] == 0.5
        assert payload["top_k"] == 10

    def test_payload_model_name(self):
        adapter = GemmaAdapter()
        frames = [_make_frame(index=i) for i in range(4)]
        vi = adapter.preprocess(frames, fps=1.0)
        payload = adapter.build_payload(vi, "Test", model_name="gemma-4-31b")
        assert payload["model"] == "gemma-4-31b"

    def test_payload_no_model_name_omitted(self):
        adapter = GemmaAdapter()
        frames = [_make_frame(index=i) for i in range(4)]
        vi = adapter.preprocess(frames, fps=1.0)
        payload = adapter.build_payload(vi, "Test")
        assert "model" not in payload


class TestGemmaAdapterResponseParsing:
    """Verify GemmaAdapter.parse_response handles channel-thought tags."""

    def test_no_thinking(self):
        adapter = GemmaAdapter()
        caption, thinking, truncated = adapter.parse_response("Just a caption.")
        assert caption == "Just a caption."
        assert thinking == ""
        assert truncated is False

    def test_with_thinking(self):
        adapter = GemmaAdapter()
        raw = "<|channel|>thought\nLet me analyze this...\n<|channel|>A sunset over the ocean."
        caption, thinking, truncated = adapter.parse_response(raw)
        assert "analyze" in thinking
        assert "sunset" in caption
        assert truncated is False

    def test_truncated_thinking(self):
        adapter = GemmaAdapter()
        raw = "<|channel|>thought\nStill thinking..."
        caption, thinking, truncated = adapter.parse_response(raw)
        assert caption == ""
        assert "thinking" in thinking
        assert truncated is True

    def test_empty_response(self):
        adapter = GemmaAdapter()
        caption, thinking, truncated = adapter.parse_response("")
        assert caption == ""
        assert thinking == ""
        assert truncated is False

    def test_empty_thinking_tags(self):
        adapter = GemmaAdapter()
        raw = "<|channel|>thought\n<|channel|>A plain caption."
        caption, thinking, truncated = adapter.parse_response(raw)
        assert thinking == ""
        assert caption == "A plain caption."
        assert truncated is False

    def test_turn_tokens_stripped(self):
        adapter = GemmaAdapter()
        raw = "<|turn|>model A plain caption."
        caption, _thinking, truncated = adapter.parse_response(raw)
        assert caption == "A plain caption."
        assert truncated is False


class TestGemmaAdapterTokenEstimation:
    """Verify GemmaAdapter.estimate_tokens uses frames * budget * 2 formula."""

    def test_token_count_default_budget(self):
        adapter = GemmaAdapter()
        frames = [_make_frame(index=i) for i in range(8)]
        vi = adapter.preprocess(frames, fps=1.0)
        tokens = adapter.estimate_tokens(vi)
        # 8 frames * 280 default * 2 (two image tokens per frame) = 4480
        assert tokens == 8 * 280 * 2

    def test_custom_image_max_tokens(self):
        adapter = GemmaAdapter(image_max_tokens=560)
        frames = [_make_frame(index=i) for i in range(4)]
        vi = adapter.preprocess(frames, fps=1.0)
        tokens = adapter.estimate_tokens(vi)
        assert tokens == 4 * 560 * 2

    def test_single_frame(self):
        adapter = GemmaAdapter()
        frames = [_make_frame(index=0)]
        vi = adapter.preprocess(frames, fps=1.0)
        tokens = adapter.estimate_tokens(vi)
        assert tokens > 0


class TestGemmaAdapterProperties:
    def test_name(self):
        assert GemmaAdapter().name == "gemma4"

    def test_default_fps(self):
        assert GemmaAdapter().default_fps == 1.0

    def test_max_frames(self):
        assert GemmaAdapter().max_frames == 60

    def test_max_duration(self):
        assert GemmaAdapter().max_duration_seconds == 60.0

    def test_preset(self):
        assert GEMMA_PRESET.temperature == 1.0
        assert GEMMA_PRESET.top_k == 64
        assert GEMMA_PRESET.presence_penalty == 0.0


class TestGemmaAdapterTimestampFormat:
    def test_zero(self):
        assert GemmaAdapter._format_timestamp(0.0) == "00:00"

    def test_thirty_seconds(self):
        assert GemmaAdapter._format_timestamp(30.0) == "00:30"

    def test_one_minute(self):
        assert GemmaAdapter._format_timestamp(60.0) == "01:00"

    def test_nine_minutes(self):
        assert GemmaAdapter._format_timestamp(540.0) == "09:00"

    def test_fractional_seconds_truncated(self):
        assert GemmaAdapter._format_timestamp(5.7) == "00:05"
