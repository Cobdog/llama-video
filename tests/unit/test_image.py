"""Tests for image captioning support."""

from __future__ import annotations

import numpy as np
import pytest

from llama_video.image import build_image_message, load_image


class TestLoadImage:
    """Test image loading and validation."""

    def test_load_jpg(self, tmp_path):
        from PIL import Image

        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        path = tmp_path / "test.jpg"
        img.save(str(path))

        result = load_image(str(path))
        assert result.shape == (100, 100, 3)
        assert result.dtype == np.uint8

    def test_load_png(self, tmp_path):
        from PIL import Image

        img = Image.fromarray(np.random.randint(0, 255, (50, 80, 3), dtype=np.uint8))
        path = tmp_path / "test.png"
        img.save(str(path))

        result = load_image(str(path))
        assert result.shape == (50, 80, 3)

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_image("/nonexistent/image.jpg")

    def test_load_rgba_converts_to_rgb(self, tmp_path):
        from PIL import Image

        img = Image.fromarray(np.random.randint(0, 255, (50, 50, 4), dtype=np.uint8), "RGBA")
        path = tmp_path / "test.png"
        img.save(str(path))

        result = load_image(str(path))
        assert result.shape == (50, 50, 3)


class TestBuildImageMessage:
    """Test image message construction for llama-server."""

    def test_message_structure(self, tmp_path):
        from PIL import Image

        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        path = tmp_path / "test.jpg"
        img.save(str(path))

        message = build_image_message(str(path), "Describe this image.")

        assert message["role"] == "user"
        content = message["content"]
        assert len(content) == 2  # 1 image + 1 text

        image_entry = content[0]
        assert image_entry["type"] == "image_url"
        assert image_entry["image_url"]["url"].startswith("data:image/jpeg;base64,")

        text_entry = content[1]
        assert text_entry["type"] == "text"
        assert text_entry["text"] == "Describe this image."
