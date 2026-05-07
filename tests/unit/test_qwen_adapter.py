"""Tests for QwenAdapter — verifies parity with original Preprocessor/Client."""

import numpy as np
import pytest

from llama_video.adapters.base import AdapterPreset
from llama_video.adapters.qwen import QWEN_PRESET, QwenAdapter
from llama_video.config import ModelConfig
from llama_video.preprocessor import Preprocessor
from llama_video.types import Frame, OddFrameStrategy


def _make_frame(w: int = 224, h: int = 224, index: int = 0) -> Frame:
    data = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    return Frame(data=data, index=index, timestamp=index * 0.5, width=w, height=h)


class TestQwenAdapterPreprocess:
    """Verify QwenAdapter.preprocess matches Preprocessor.process."""

    def test_parity_with_original_preprocessor(self):
        config = ModelConfig.qwen35()
        adapter = QwenAdapter(model_config=config)
        preprocessor = Preprocessor(model_config=config)

        frames = [_make_frame(index=i) for i in range(8)]
        fps = 2.0

        original = preprocessor.process(frames, fps=fps)
        adapted = adapter.preprocess(frames, fps=fps)

        assert adapted.grid_thw == original.grid_thw
        assert adapted.temporal_positions == original.temporal_positions
        assert adapted.fps == original.fps
        assert adapted.num_source_frames == original.num_source_frames
        assert adapted.resolution == original.resolution
        assert len(adapted.super_frames) == len(original.super_frames)

        for a_sf, o_sf in zip(adapted.super_frames, original.super_frames, strict=True):
            assert a_sf.temporal_index == o_sf.temporal_index
            assert a_sf.source_frames == o_sf.source_frames
            np.testing.assert_array_almost_equal(a_sf.data, o_sf.data, decimal=6)

    def test_parity_single_frame_pads(self):
        config = ModelConfig.qwen35()
        adapter = QwenAdapter(model_config=config)
        preprocessor = Preprocessor(model_config=config)

        frames = [_make_frame(index=0)]
        original = preprocessor.process(frames, fps=2.0)
        adapted = adapter.preprocess(frames, fps=2.0)

        assert adapted.grid_thw == original.grid_thw
        assert len(adapted.super_frames) == 1

    def test_odd_frame_count_drop(self):
        adapter = QwenAdapter()
        frames = [_make_frame(index=i) for i in range(3)]
        result = adapter.preprocess(frames, fps=2.0, odd_strategy=OddFrameStrategy.DROP)
        assert len(result.super_frames) == 1

    def test_empty_frames_raises(self):
        adapter = QwenAdapter()
        with pytest.raises(Exception, match="No frames"):
            adapter.preprocess([], fps=2.0)


class TestQwenAdapterPayload:
    """Verify QwenAdapter.build_payload produces correct structure."""

    def test_payload_has_mm_processor_kwargs(self):
        adapter = QwenAdapter()
        frames = [_make_frame(index=i) for i in range(4)]
        vi = adapter.preprocess(frames, fps=2.0)
        payload = adapter.build_payload(vi, "Describe this video.")

        assert "mm_processor_kwargs" in payload
        mm = payload["mm_processor_kwargs"]
        assert mm["is_video"] is True
        assert mm["fps"] == 2.0
        assert mm["grid_thw"] == list(vi.grid_thw)
        assert mm["temporal_positions"] == vi.temporal_positions

    def test_payload_has_messages(self):
        adapter = QwenAdapter()
        frames = [_make_frame(index=i) for i in range(4)]
        vi = adapter.preprocess(frames, fps=2.0)
        payload = adapter.build_payload(vi, "Test prompt")

        assert len(payload["messages"]) == 2
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][1]["role"] == "user"
        content = payload["messages"][1]["content"]
        last = content[-1]
        assert last["type"] == "text"
        assert last["text"] == "Test prompt"

    def test_payload_sampler_defaults(self):
        adapter = QwenAdapter()
        frames = [_make_frame(index=i) for i in range(4)]
        vi = adapter.preprocess(frames, fps=2.0)
        payload = adapter.build_payload(vi, "Test")

        assert payload["temperature"] == 1.0
        assert payload["top_p"] == 0.95
        assert payload["top_k"] == 20
        assert payload["min_p"] == 0.0
        assert payload["presence_penalty"] == 1.5

    def test_payload_custom_preset(self):
        adapter = QwenAdapter()
        frames = [_make_frame(index=i) for i in range(4)]
        vi = adapter.preprocess(frames, fps=2.0)
        custom = AdapterPreset(temperature=0.5, top_p=0.9, top_k=10)
        payload = adapter.build_payload(vi, "Test", preset=custom)

        assert payload["temperature"] == 0.5
        assert payload["top_k"] == 10

    def test_payload_model_name(self):
        adapter = QwenAdapter()
        frames = [_make_frame(index=i) for i in range(4)]
        vi = adapter.preprocess(frames, fps=2.0)
        payload = adapter.build_payload(vi, "Test", model_name="my-model")
        assert payload["model"] == "my-model"

    def test_payload_no_model_name_omitted(self):
        adapter = QwenAdapter()
        frames = [_make_frame(index=i) for i in range(4)]
        vi = adapter.preprocess(frames, fps=2.0)
        payload = adapter.build_payload(vi, "Test")
        assert "model" not in payload


class TestQwenAdapterResponseParsing:
    """Verify QwenAdapter.parse_response handles thinking tags."""

    def test_no_thinking(self):
        adapter = QwenAdapter()
        caption, thinking, truncated = adapter.parse_response("Just a caption.")
        assert caption == "Just a caption."
        assert thinking == ""
        assert truncated is False

    def test_with_thinking(self):
        adapter = QwenAdapter()
        raw = "<think >Let me analyze this...</think >A sunset over the ocean."
        caption, thinking, truncated = adapter.parse_response(raw)
        assert "analyze" in thinking
        assert "sunset" in caption
        assert truncated is False

    def test_truncated_thinking(self):
        adapter = QwenAdapter()
        raw = "<think >Still thinking..."
        caption, thinking, truncated = adapter.parse_response(raw)
        assert caption == ""
        assert "thinking" in thinking
        assert truncated is True

    def test_empty_response(self):
        adapter = QwenAdapter()
        caption, thinking, truncated = adapter.parse_response("")
        assert caption == ""
        assert thinking == ""
        assert truncated is False


class TestQwenAdapterTokenEstimation:
    """Verify QwenAdapter.estimate_tokens uses T*H*W formula."""

    def test_token_count_matches_grid(self):
        adapter = QwenAdapter()
        frames = [_make_frame(index=i) for i in range(8)]
        vi = adapter.preprocess(frames, fps=2.0)
        tokens = adapter.estimate_tokens(vi)
        t, h, w = vi.grid_thw
        assert tokens == t * h * w

    def test_single_frame(self):
        adapter = QwenAdapter()
        frames = [_make_frame(index=0)]
        vi = adapter.preprocess(frames, fps=2.0)
        tokens = adapter.estimate_tokens(vi)
        assert tokens > 0


class TestQwenAdapterProperties:
    def test_name(self):
        assert QwenAdapter().name == "qwen3.5"

    def test_default_fps(self):
        assert QwenAdapter().default_fps == 2.0

    def test_max_frames(self):
        assert QwenAdapter().max_frames == 64

    def test_max_duration(self):
        assert QwenAdapter().max_duration_seconds == float("inf")

    def test_preset(self):
        assert QWEN_PRESET.temperature == 1.0
        assert QWEN_PRESET.top_k == 20
        assert QWEN_PRESET.presence_penalty == 1.5
