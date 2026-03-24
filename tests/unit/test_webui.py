"""Tests for WebUI helper functions."""

import pytest

pytest.importorskip("gradio")

from llama_video.webui import (
    build_budget_html,
    build_metadata_html,
    compute_budget,
)


class TestBuildBudgetHtml:
    def test_basic_budget(self):
        html = build_budget_html(1000, 25, 2048, 65536)
        assert "Vision: 1,000" in html
        assert "Gen: 2,048" in html
        assert "65,536" in html

    def test_zero_vision(self):
        html = build_budget_html(0, 0, 2048, 65536)
        assert "Vision" not in html
        assert "Gen: 2,048" in html

    def test_over_budget(self):
        html = build_budget_html(60000, 100, 8000, 65536)
        # total=68100, over=2564
        assert "Over by 2,564" in html

    def test_near_limit_warning_color(self):
        # 90%+ usage → warning color
        html = build_budget_html(56000, 100, 2000, 65536)
        # 58100/65536 = 88.6%
        assert "ffaa00" in html

    def test_low_usage_normal_color(self):
        html = build_budget_html(1000, 25, 2048, 65536)
        # 3073/65536 = 4.7%
        assert "ff4444" not in html

    def test_zero_context_limit(self):
        # Should not crash on divide-by-zero
        html = build_budget_html(100, 10, 2048, 0)
        assert html  # Just check it doesn't crash

    def test_all_segments_present(self):
        html = build_budget_html(5000, 50, 2048, 65536)
        assert "Vision: 5,000" in html
        assert "Prompt: 50" in html
        assert "Gen: 2,048" in html
        assert "Free:" in html


class TestBuildMetadataHtml:
    def test_video_metadata(self):
        html = build_metadata_html(
            "video",
            10,
            5,
            (5, 37, 69),
            (1932, 1036),
            15000,
        )
        assert "video" in html
        assert "15.0s" in html
        # Grid values present (using HTML entities)
        assert "12,765 tokens" in html  # 5*37*69

    def test_image_metadata(self):
        html = build_metadata_html(
            "image",
            1,
            0,
            (1, 20, 30),
            (840, 560),
            5000,
        )
        assert "image" in html
        assert "5.0s" in html
        assert "600 tokens" in html  # 1*20*30

    def test_zero_duration(self):
        html = build_metadata_html(
            "video",
            4,
            2,
            (2, 10, 10),
            (280, 280),
            0,
        )
        assert "0.0s" in html


class TestComputeBudget:
    def test_empty_info(self):
        html = compute_budget({}, 2.0, 64, "test", 2048, 65536)
        assert "Gen: 2,048" in html
        assert "Vision" not in html

    def test_zero_width(self):
        info = {"width": 0, "height": 0, "mode": "video"}
        html = compute_budget(info, 2.0, 64, "test", 2048, 65536)
        assert "Vision" not in html

    def test_video_budget(self):
        info = {
            "width": 1920,
            "height": 1080,
            "duration": 5.0,
            "mode": "video",
        }
        html = compute_budget(
            info,
            2.0,
            64,
            "test prompt",
            2048,
            65536,
        )
        assert "Vision" in html
        assert "65,536" in html

    def test_image_budget(self):
        info = {
            "width": 800,
            "height": 600,
            "duration": 0,
            "mode": "image",
        }
        html = compute_budget(
            info,
            1.0,
            1,
            "test",
            2048,
            65536,
        )
        assert "Vision" in html

    def test_video_duration_limits_frames(self):
        # 2s video at 2fps = 4 frames max
        info = {
            "width": 280,
            "height": 280,
            "duration": 2.0,
            "mode": "video",
        }
        html_short = compute_budget(
            info,
            2.0,
            64,
            "",
            2048,
            65536,
        )
        # 10s video at 2fps = 20 frames
        info2 = {
            "width": 280,
            "height": 280,
            "duration": 10.0,
            "mode": "video",
        }
        html_long = compute_budget(
            info2,
            2.0,
            64,
            "",
            2048,
            65536,
        )
        # Longer video should use more vision tokens
        # (both have same resolution, so more frames = more tokens)
        assert html_short != html_long

    def test_resolution_scale_reduces_tokens(self):
        info = {
            "width": 1920,
            "height": 1080,
            "duration": 5.0,
            "mode": "video",
        }
        html_full = compute_budget(info, 2.0, 64, "", 2048, 65536, 1.0)
        html_half = compute_budget(info, 2.0, 64, "", 2048, 65536, 0.5)
        # Half resolution should use fewer vision tokens
        assert html_full != html_half
        # Full should have Vision tokens, half should too (but fewer)
        assert "Vision" in html_full
        assert "Vision" in html_half

    def test_custom_context_limit(self):
        info = {
            "width": 1920,
            "height": 1080,
            "duration": 5.0,
            "mode": "video",
        }
        html = compute_budget(
            info,
            2.0,
            64,
            "test",
            2048,
            131072,
        )
        assert "131,072" in html
