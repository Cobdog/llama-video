"""Tests for CaptionResult dataclass."""

from __future__ import annotations

from llama_video.types import CaptionMetadata, CaptionResult


class TestCaptionResult:
    """Test CaptionResult creation and fields."""

    def test_create_minimal(self):
        metadata = CaptionMetadata(
            frames_extracted=4,
            super_frames=2,
            grid_thw=(2, 1, 1),
            processing_time_ms=1234.5,
        )
        result = CaptionResult(
            caption="A girl in a field.",
            source_path="/path/to/video.mp4",
            mode="video",
            prompt_rendered="Describe this video.",
            template_name="general",
            variables={},
            preset_name="default",
            settings={"fps": 2.0, "max_frames": 64},
            metadata=metadata,
            token_usage=None,
            duration_ms=5000.0,
        )
        assert result.caption == "A girl in a field."
        assert result.mode == "video"
        assert result.template_name == "general"
        assert result.duration_ms == 5000.0

    def test_create_with_custom_template(self):
        metadata = CaptionMetadata(
            frames_extracted=1,
            super_frames=0,
            grid_thw=(0, 0, 0),
            processing_time_ms=500.0,
        )
        result = CaptionResult(
            caption="A sunset.",
            source_path="/path/to/image.jpg",
            mode="image",
            prompt_rendered="What is this?",
            template_name=None,
            variables={"custom_var": "value"},
            preset_name="precise",
            settings={"max_tokens": 512},
            metadata=metadata,
            token_usage=None,
            duration_ms=1000.0,
        )
        assert result.mode == "image"
        assert result.template_name is None
        assert result.variables == {"custom_var": "value"}

    def test_to_dict(self):
        metadata = CaptionMetadata(
            frames_extracted=4,
            super_frames=2,
            grid_thw=(2, 1, 1),
            processing_time_ms=1234.5,
        )
        result = CaptionResult(
            caption="A girl.",
            source_path="/path/to/video.mp4",
            mode="video",
            prompt_rendered="Describe.",
            template_name="general",
            variables={},
            preset_name="default",
            settings={"fps": 2.0},
            metadata=metadata,
            token_usage=None,
            duration_ms=5000.0,
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["caption"] == "A girl."
        assert d["mode"] == "video"
        assert d["settings"] == {"fps": 2.0}
