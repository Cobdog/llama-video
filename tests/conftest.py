"""Shared test fixtures for llama-video."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from llama_video.config import ModelConfig, ServerConfig
from llama_video.types import Frame


@pytest.fixture
def model_config() -> ModelConfig:
    """Default Qwen3.5 model config."""
    return ModelConfig.qwen35()


@pytest.fixture
def server_config() -> ServerConfig:
    """Test server config pointing to localhost."""
    return ServerConfig(url="http://localhost:8080")


@pytest.fixture
def sample_frame() -> Frame:
    """A single 448x448 synthetic test frame."""
    rng = np.random.default_rng(42)
    data = rng.integers(0, 256, size=(448, 448, 3), dtype=np.uint8)
    return Frame(data=data, index=0, timestamp=0.0, width=448, height=448)


@pytest.fixture
def sample_frames() -> list[Frame]:
    """8 synthetic 448x448 frames (4 seconds at 2fps)."""
    rng = np.random.default_rng(42)
    frames: list[Frame] = []
    for i in range(8):
        data = rng.integers(0, 256, size=(448, 448, 3), dtype=np.uint8)
        frames.append(
            Frame(
                data=data,
                index=i,
                timestamp=i / 2.0,
                width=448,
                height=448,
            )
        )
    return frames


@pytest.fixture
def odd_frames() -> list[Frame]:
    """7 synthetic frames (odd count for testing padding/dropping)."""
    rng = np.random.default_rng(42)
    frames: list[Frame] = []
    for i in range(7):
        data = rng.integers(0, 256, size=(448, 448, 3), dtype=np.uint8)
        frames.append(
            Frame(
                data=data,
                index=i,
                timestamp=i / 2.0,
                width=448,
                height=448,
            )
        )
    return frames


@pytest.fixture
def single_frame() -> list[Frame]:
    """Single frame for edge case testing."""
    rng = np.random.default_rng(42)
    data = rng.integers(0, 256, size=(448, 448, 3), dtype=np.uint8)
    return [Frame(data=data, index=0, timestamp=0.0, width=448, height=448)]


@pytest.fixture
def small_frames() -> list[Frame]:
    """4 small 56x56 frames (minimum viable resolution)."""
    rng = np.random.default_rng(42)
    frames: list[Frame] = []
    for i in range(4):
        data = rng.integers(0, 256, size=(56, 56, 3), dtype=np.uint8)
        frames.append(
            Frame(
                data=data,
                index=i,
                timestamp=i / 2.0,
                width=56,
                height=56,
            )
        )
    return frames


@pytest.fixture
def test_data_dir() -> Path:
    """Path to test data directory."""
    return Path(__file__).parent / "data"
