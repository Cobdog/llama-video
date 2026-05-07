"""Tests for the adapter registry."""

import pytest

from llama_video.adapters.base import AdapterPreset, ModelAdapter
from llama_video.adapters.registry import (
    AdapterNotFoundError,
    default_adapter_name,
    get_adapter,
    list_adapters,
    register_adapter,
)


class _StubAdapter(ModelAdapter):
    """Minimal adapter for registry testing."""

    @property
    def name(self) -> str:
        return "stub"

    @property
    def default_fps(self) -> float:
        return 1.0

    @property
    def max_duration_seconds(self) -> float:
        return 60.0

    @property
    def max_frames(self) -> int:
        return 60

    @property
    def default_preset(self) -> AdapterPreset:
        return AdapterPreset(temperature=1.0, top_p=0.95, top_k=64)

    def preprocess(self, frames, fps):
        return None

    def build_payload(self, video_input, prompt, **kwargs):
        return {}

    def parse_response(self, raw):
        return raw, "", False

    def estimate_tokens(self, video_input):
        return 0


class TestRegisterAdapter:
    def test_register_and_retrieve(self):
        register_adapter("test-stub", _StubAdapter)
        adapter = get_adapter("test-stub")
        assert adapter.name == "stub"
        assert isinstance(adapter, _StubAdapter)

    def test_list_includes_registered(self):
        register_adapter("test-list", _StubAdapter)
        names = list_adapters()
        assert "test-list" in names

    def test_overwrite_warns(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            register_adapter("test-overwrite", _StubAdapter)
            register_adapter("test-overwrite", _StubAdapter)
        assert "Overwriting adapter" in caplog.text


class TestGetAdapter:
    def test_unknown_raises(self):
        with pytest.raises(AdapterNotFoundError, match="no-such-adapter"):
            get_adapter("no-such-adapter")

    def test_error_lists_available(self):
        register_adapter("test-available", _StubAdapter)
        with pytest.raises(AdapterNotFoundError, match="test-available"):
            get_adapter("zzz-nonexistent")

    def test_none_returns_default(self):
        register_adapter(default_adapter_name(), _StubAdapter)
        adapter = get_adapter(None)
        assert isinstance(adapter, _StubAdapter)

    def test_empty_string_returns_default(self):
        register_adapter(default_adapter_name(), _StubAdapter)
        adapter = get_adapter("")
        assert isinstance(adapter, _StubAdapter)

    def test_case_insensitive(self):
        register_adapter("test-ci", _StubAdapter)
        adapter = get_adapter("TEST-CI")
        assert isinstance(adapter, _StubAdapter)


class TestDefaultAdapter:
    def test_default_name(self):
        assert default_adapter_name() == "qwen3.5"

    def test_list_returns_sorted(self):
        register_adapter("zebra", _StubAdapter)
        register_adapter("alpha", _StubAdapter)
        names = list_adapters()
        assert names == sorted(names)
