"""Shared fixtures for smoke tests.

All smoke tests require a running patched llama-server.
The server_ready fixture waits for it to be available before any tests run.
"""

from __future__ import annotations

import os
import time
from collections.abc import Generator

import httpx
import pytest

SERVER_URL = os.environ.get("LLAMA_SERVER_URL", "http://localhost:7801")
SERVER_READY_TIMEOUT = 120  # seconds to wait for model loading
TEST_TIMEOUT = 300  # 5 minutes per test — thinking mode is slow


@pytest.fixture(scope="session")
def server_url() -> str:
    """The llama-server URL from env or default."""
    return SERVER_URL


@pytest.fixture(scope="session", autouse=True)
def server_ready(server_url: str) -> None:
    """Wait for llama-server to be ready before running smoke tests.

    Polls /health every 5 seconds for up to SERVER_READY_TIMEOUT seconds.
    Skips the entire test session if the server never becomes ready.
    """
    deadline = time.monotonic() + SERVER_READY_TIMEOUT
    last_error = ""

    while time.monotonic() < deadline:
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(f"{server_url}/health")
                if resp.status_code == 200:
                    print(f"\nllama-server ready at {server_url}")
                    return
                last_error = f"status {resp.status_code}"
        except httpx.ConnectError:
            last_error = "connection refused"
        except httpx.TimeoutException:
            last_error = "timeout"

        time.sleep(5)

    pytest.skip(
        f"llama-server not ready at {server_url} after {SERVER_READY_TIMEOUT}s "
        f"(last error: {last_error}). Start it first with ./scripts/run-server.sh"
    )


@pytest.fixture(scope="session")
def caption_client(server_url: str) -> Generator[httpx.Client, None, None]:
    """A configured httpx client pointing at the llama-server."""
    with httpx.Client(base_url=server_url, timeout=TEST_TIMEOUT) as client:
        yield client


@pytest.fixture
def testvid_dir() -> str:
    """Path to the test video directory."""
    from pathlib import Path

    path = Path(__file__).parent.parent.parent / "testvid"
    if not path.is_dir():
        pytest.skip(f"testvid directory not found at {path}")
    return str(path)
