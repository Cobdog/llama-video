"""Tests for configuration management."""

from __future__ import annotations

from llama_video.config import (
    PRESETS,
    ModelConfig,
    ServerConfig,
    Settings,
    get_preset,
)


class TestModelConfig:
    """Test ModelConfig defaults and properties."""

    def test_qwen35_defaults(self):
        config = ModelConfig.qwen35()
        assert config.temporal_patch_size == 2
        assert config.spatial_patch_size == 14
        assert config.merge_size == 2

    def test_grid_unit(self):
        config = ModelConfig.qwen35()
        assert config.grid_unit == 28  # 14 * 2

    def test_normalization_values_are_clip(self):
        config = ModelConfig.qwen35()
        # Standard CLIP normalization
        assert len(config.image_mean) == 3
        assert len(config.image_std) == 3
        assert all(0 < m < 1 for m in config.image_mean)
        assert all(0 < s < 1 for s in config.image_std)


class TestServerConfig:
    """Test ServerConfig defaults and env loading."""

    def test_defaults(self):
        config = ServerConfig()
        assert config.url == "http://localhost:8080"
        assert config.timeout == 120.0
        assert config.max_retries == 3

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("LLAMA_SERVER_URL", "http://remote:9090")
        monkeypatch.setenv("LLAMA_SERVER_TIMEOUT", "60")
        config = ServerConfig()
        assert config.url == "http://remote:9090"
        assert config.timeout == 60.0


class TestSettings:
    """Test root Settings composition."""

    def test_default_settings(self):
        settings = Settings()
        assert settings.model.temporal_patch_size == 2
        assert settings.server.url == "http://localhost:8080"
        assert settings.extractor.default_fps == 2.0
        assert settings.service.port == 9000


class TestInferencePreset:
    """Test inference preset system."""

    def test_default_preset_exists(self):
        assert "default" in PRESETS

    def test_default_preset_values(self):
        p = PRESETS["default"]
        assert p.name == "default"
        assert p.temperature == 1.0
        assert p.top_p == 0.95
        assert p.top_k == 20
        assert p.min_p == 0.0
        assert p.presence_penalty == 1.5

    def test_precise_preset_exists(self):
        assert "precise" in PRESETS

    def test_precise_preset_values(self):
        p = PRESETS["precise"]
        assert p.temperature == 0.6
        assert p.presence_penalty == 0.0

    def test_get_preset_returns_default(self):
        p = get_preset("default")
        assert p.name == "default"
        assert p.temperature == 1.0

    def test_get_preset_unknown_raises(self):
        import pytest

        with pytest.raises(KeyError):
            get_preset("nonexistent")

    def test_preset_is_dataclass(self):
        from dataclasses import fields

        p = PRESETS["default"]
        field_names = {f.name for f in fields(p)}
        assert "temperature" in field_names
        assert "top_p" in field_names
        assert "top_k" in field_names
        assert "min_p" in field_names
        assert "presence_penalty" in field_names
