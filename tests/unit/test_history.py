"""Tests for caption history (SQLite persistence)."""

from __future__ import annotations

import json

import pytest

from llama_video.history import CaptionHistory
from llama_video.types import CaptionMetadata, CaptionResult


@pytest.fixture
def tmp_history(tmp_path):
    """Create a CaptionHistory with a temporary database."""
    db_path = str(tmp_path / "test_captions.db")
    history = CaptionHistory(db_path=db_path)
    yield history
    history.close()


@pytest.fixture
def sample_result() -> CaptionResult:
    return CaptionResult(
        caption="A girl lying in a field of flowers.",
        source_path="/testvid/kiki.mp4",
        mode="video",
        prompt_rendered="Describe this video.",
        template_name="general",
        variables={},
        preset_name="default",
        settings={"fps": 2.0, "max_frames": 64, "max_tokens": 2048},
        metadata=CaptionMetadata(
            frames_extracted=8,
            super_frames=4,
            grid_thw=(4, 16, 16),
            processing_time_ms=5000.0,
        ),
        token_usage=None,
        duration_ms=5500.0,
    )


class TestCaptionHistory:
    """Test SQLite caption history."""

    def test_save_and_get(self, tmp_history, sample_result):
        row_id = tmp_history.save(sample_result)
        assert row_id == 1

        retrieved = tmp_history.get(row_id)
        assert retrieved["caption"] == "A girl lying in a field of flowers."
        assert retrieved["mode"] == "video"
        assert retrieved["template"] == "general"
        assert retrieved["preset"] == "default"

    def test_save_multiple(self, tmp_history, sample_result):
        id1 = tmp_history.save(sample_result)
        id2 = tmp_history.save(sample_result)
        assert id2 == id1 + 1

    def test_list_returns_all(self, tmp_history, sample_result):
        tmp_history.save(sample_result)
        tmp_history.save(sample_result)
        results = tmp_history.list_captions()
        assert len(results) == 2

    def test_list_filter_by_mode(self, tmp_history, sample_result):
        tmp_history.save(sample_result)

        image_result = CaptionResult(
            caption="A sunset.",
            source_path="/path/image.jpg",
            mode="image",
            prompt_rendered="Describe.",
            template_name=None,
            variables={},
            preset_name="default",
            settings={},
            metadata=CaptionMetadata(
                frames_extracted=1,
                super_frames=0,
                grid_thw=(0, 0, 0),
                processing_time_ms=100.0,
            ),
            token_usage=None,
            duration_ms=200.0,
        )
        tmp_history.save(image_result)

        video_only = tmp_history.list_captions(mode="video")
        assert len(video_only) == 1
        assert video_only[0]["mode"] == "video"

    def test_list_with_limit(self, tmp_history, sample_result):
        for _ in range(5):
            tmp_history.save(sample_result)
        results = tmp_history.list_captions(limit=3)
        assert len(results) == 3

    def test_get_nonexistent_raises(self, tmp_history):
        with pytest.raises(KeyError):
            tmp_history.get(999)

    def test_export_json(self, tmp_history, sample_result):
        tmp_history.save(sample_result)
        exported = tmp_history.export(format="json")
        data = json.loads(exported)
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["caption"] == "A girl lying in a field of flowers."

    def test_export_csv(self, tmp_history, sample_result):
        tmp_history.save(sample_result)
        exported = tmp_history.export(format="csv")
        assert "caption" in exported  # Header row
        assert "A girl lying in a field of flowers." in exported

    def test_export_specific_ids(self, tmp_history, sample_result):
        id1 = tmp_history.save(sample_result)
        tmp_history.save(sample_result)
        exported = tmp_history.export(ids=[id1], format="json")
        data = json.loads(exported)
        assert len(data) == 1

    def test_scrub(self, tmp_history, sample_result):
        tmp_history.save(sample_result)
        tmp_history.save(sample_result)
        count = tmp_history.scrub()
        assert count == 2
        assert len(tmp_history.list_captions()) == 0

    def test_scrub_empty_db(self, tmp_history):
        count = tmp_history.scrub()
        assert count == 0
