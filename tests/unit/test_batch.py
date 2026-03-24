"""Tests for batch captioning API."""

from __future__ import annotations

import pytest

from llama_video.batch import detect_mode, validate_batch_mode

VIDEO_EXTENSIONS = [".mp4", ".webm", ".mov", ".avi", ".mkv"]
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]


class TestDetectMode:
    """Test file extension → mode detection."""

    @pytest.mark.parametrize("ext", VIDEO_EXTENSIONS)
    def test_video_extensions(self, ext):
        assert detect_mode(f"/path/to/file{ext}") == "video"

    @pytest.mark.parametrize("ext", IMAGE_EXTENSIONS)
    def test_image_extensions(self, ext):
        assert detect_mode(f"/path/to/file{ext}") == "image"

    def test_unknown_extension_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            detect_mode("/path/to/file.txt")

    def test_case_insensitive(self):
        assert detect_mode("/path/to/FILE.MP4") == "video"
        assert detect_mode("/path/to/IMAGE.JPG") == "image"


class TestValidateBatchMode:
    """Test batch mode validation."""

    def test_auto_mode_all_video(self):
        paths = ["/a/v1.mp4", "/a/v2.mov"]
        mode = validate_batch_mode(paths, "auto")
        assert mode == "video"

    def test_auto_mode_all_image(self):
        paths = ["/a/i1.jpg", "/a/i2.png"]
        mode = validate_batch_mode(paths, "auto")
        assert mode == "image"

    def test_auto_mode_mixed_raises(self):
        paths = ["/a/v1.mp4", "/a/i1.jpg"]
        with pytest.raises(ValueError, match="mixed"):
            validate_batch_mode(paths, "auto")

    def test_explicit_video_mode(self):
        paths = ["/a/v1.mp4"]
        mode = validate_batch_mode(paths, "video")
        assert mode == "video"

    def test_explicit_image_mode(self):
        paths = ["/a/i1.jpg"]
        mode = validate_batch_mode(paths, "image")
        assert mode == "image"

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="empty"):
            validate_batch_mode([], "auto")
