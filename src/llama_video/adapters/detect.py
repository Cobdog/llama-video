"""Auto-detect model adapter from llama-server's loaded model.

Queries /v1/models on the llama-server, matches model names against
known patterns, and returns the appropriate adapter name.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import httpx

from llama_video.adapters.registry import default_adapter_name

logger = logging.getLogger(__name__)

# Model name patterns mapped to adapter names.
# Patterns are matched case-insensitively against the loaded model name.
_MODEL_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"gemma[\s_-]?4", re.IGNORECASE), "gemma4"),
    (re.compile(r"qwen[\s_-]?3\.?5", re.IGNORECASE), "qwen3.5"),
    (re.compile(r"qwen", re.IGNORECASE), "qwen3.5"),
]


def match_model_to_adapter(model_name: str) -> str | None:
    """Match a model name to an adapter using known patterns.

    Args:
        model_name: Model name from llama-server (e.g. "gemma-4-31b-it-q4_k_m").

    Returns:
        Adapter name or None if no pattern matches.
    """
    for pattern, adapter_name in _MODEL_PATTERNS:
        if pattern.search(model_name):
            return adapter_name
    return None


async def detect_adapter(
    server_url: str,
    timeout: float = 5.0,
) -> str:
    """Detect the appropriate adapter from llama-server's loaded model.

    Queries /v1/models on the server, extracts the first model's name,
    and matches it against known patterns. Falls back to default adapter
    if detection fails.

    Args:
        server_url: llama-server base URL (e.g. "http://localhost:8080").
        timeout: Request timeout in seconds.

    Returns:
        Adapter name string (e.g. "qwen3.5", "gemma4").
    """
    try:
        async with httpx.AsyncClient(
            base_url=server_url,
            timeout=httpx.Timeout(timeout),
        ) as client:
            response = await client.get("/v1/models")
            if response.status_code != 200:
                logger.warning(
                    "Failed to query /v1/models (status %d), using default adapter",
                    response.status_code,
                )
                return default_adapter_name()

            data: dict[str, Any] = response.json()
            models = data.get("data", [])
            if not models:
                logger.warning("No models returned from /v1/models, using default adapter")
                return default_adapter_name()

            model_id = models[0].get("id", "")
            if not model_id:
                logger.warning("Empty model ID, using default adapter")
                return default_adapter_name()

            adapter_name = match_model_to_adapter(model_id)
            if adapter_name:
                logger.info(
                    "Auto-detected adapter '%s' from model '%s'",
                    adapter_name,
                    model_id,
                )
                return adapter_name

            logger.warning(
                "No adapter pattern matched model '%s', using default '%s'",
                model_id,
                default_adapter_name(),
            )
            return default_adapter_name()

    except (httpx.ConnectError, httpx.TimeoutException) as e:
        logger.warning(
            "Cannot reach llama-server for auto-detect (%s), using default adapter",
            type(e).__name__,
        )
        return default_adapter_name()
