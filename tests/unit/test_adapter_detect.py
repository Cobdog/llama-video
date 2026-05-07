"""Tests for adapter auto-detection from llama-server."""

import pytest

from llama_video.adapters.detect import detect_adapter, match_model_to_adapter


class TestMatchModelToAdapter:
    """Test model name pattern matching."""

    def test_gemma4_exact(self):
        assert match_model_to_adapter("gemma-4-31b-it-Q4") == "gemma4"

    def test_gemma4_sprinkle(self):
        assert match_model_to_adapter("MM-Sprinkle-Gemma4-31B-Q4") == "gemma4"

    def test_gemma4_space(self):
        assert match_model_to_adapter("gemma 4 31b") == "gemma4"

    def test_qwen35_full(self):
        assert match_model_to_adapter("MM-Qwen3.5-35-A3B") == "qwen3.5"

    def test_qwen35_dash(self):
        assert match_model_to_adapter("qwen-3.5-35b") == "qwen3.5"

    def test_qwen_bare(self):
        assert match_model_to_adapter("Qwen-7B-Chat") == "qwen3.5"

    def test_unknown_model(self):
        assert match_model_to_adapter("llama-3.1-8b") is None

    def test_empty_string(self):
        assert match_model_to_adapter("") is None

    def test_case_insensitive(self):
        assert match_model_to_adapter("GEMMA-4-31B") == "gemma4"
        assert match_model_to_adapter("QWEN3.5-35B") == "qwen3.5"


class TestDetectAdapter:
    """Test full auto-detect flow with mocked HTTP."""

    @pytest.mark.asyncio
    async def test_detect_gemma4(self, respx_mock):
        respx_mock.get("http://test:8080/v1/models").respond(
            200,
            json={"data": [{"id": "MM-Sprinkle-Gemma4-31B-Q4", "object": "model"}]},
        )
        result = await detect_adapter("http://test:8080")
        assert result == "gemma4"

    @pytest.mark.asyncio
    async def test_detect_qwen(self, respx_mock):
        respx_mock.get("http://test:8080/v1/models").respond(
            200,
            json={"data": [{"id": "MM-Qwen3.5-35-A3B", "object": "model"}]},
        )
        result = await detect_adapter("http://test:8080")
        assert result == "qwen3.5"

    @pytest.mark.asyncio
    async def test_detect_unknown_falls_back(self, respx_mock):
        respx_mock.get("http://test:8080/v1/models").respond(
            200,
            json={"data": [{"id": "llama-3.1-8b", "object": "model"}]},
        )
        result = await detect_adapter("http://test:8080")
        assert result == "qwen3.5"  # default

    @pytest.mark.asyncio
    async def test_detect_empty_models_falls_back(self, respx_mock):
        respx_mock.get("http://test:8080/v1/models").respond(
            200,
            json={"data": []},
        )
        result = await detect_adapter("http://test:8080")
        assert result == "qwen3.5"

    @pytest.mark.asyncio
    async def test_detect_unreachable_falls_back(self):
        result = await detect_adapter("http://localhost:1", timeout=0.5)
        assert result == "qwen3.5"

    @pytest.mark.asyncio
    async def test_detect_error_status_falls_back(self, respx_mock):
        respx_mock.get("http://test:8080/v1/models").respond(500)
        result = await detect_adapter("http://test:8080")
        assert result == "qwen3.5"
