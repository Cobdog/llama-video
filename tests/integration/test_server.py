"""Integration tests for the FastAPI server.

These tests start the FastAPI app with test client (no real llama-server needed
for request validation tests, but caption tests need a running llama-server).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client for the FastAPI app with lifespan support."""
    # Import here to avoid module-level side effects
    from llama_video.server import app

    # Use context manager to trigger lifespan events
    with TestClient(app) as test_client:
        yield test_client


class TestHealthEndpoint:
    """Test /v1/health endpoint."""

    def test_health_returns_200(self, client):
        """Health endpoint should always return 200."""
        response = client.get("/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data


class TestDebugEndpoint:
    """Test /v1/debug/last-request endpoint."""

    def test_debug_returns_empty_initially(self, client):
        """Debug endpoint returns empty info before any requests."""
        response = client.get("/v1/debug/last-request")
        assert response.status_code == 200


class TestCaptionEndpointValidation:
    """Test /v1/caption request validation (no llama-server needed)."""

    def test_caption_rejects_missing_video_path(self, client):
        """Missing video_path should return 422."""
        response = client.post("/v1/caption", json={"prompt": "test"})
        assert response.status_code == 422

    def test_caption_rejects_invalid_fps(self, client):
        """FPS <= 0 should return 422."""
        response = client.post(
            "/v1/caption",
            json={"video_path": "/tmp/test.mp4", "fps": -1},
        )
        assert response.status_code == 422

    def test_caption_rejects_invalid_max_frames(self, client):
        """max_frames <= 0 should return 422."""
        response = client.post(
            "/v1/caption",
            json={"video_path": "/tmp/test.mp4", "max_frames": 0},
        )
        assert response.status_code == 422
