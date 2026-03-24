"""HTTP client for communicating with patched llama-server."""

from __future__ import annotations

import base64
import io
import json
import logging
import re
from typing import TYPE_CHECKING, Any

import httpx
import numpy as np
from PIL import Image

from llama_video.errors import (
    ModelNotLoadedError,
    ServerResponseError,
    ServerUnreachableError,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from llama_video.config import InferencePreset, ServerConfig
    from llama_video.preprocessor import VideoInput

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_THINK_OPEN = "<think>"


def parse_model_response(text: str) -> tuple[str, str, bool]:
    """Parse thinking tags from a model response.

    Returns:
        (caption, thinking, truncated) where truncated is True if the
        model was cut off mid-thought (unclosed <think> tag).
    """
    if not text:
        return "", "", False

    m = _THINK_RE.search(text)
    if m:
        thinking = m.group(1).strip()
        caption = text[m.end() :].strip()
        return caption, thinking, False

    # Unclosed <think> — model was truncated during reasoning
    if _THINK_OPEN in text:
        thinking = text.split(_THINK_OPEN, 1)[1].strip()
        return "", thinking, True

    return text.strip(), "", False


def _parse_stream_state(raw: str) -> tuple[str, str, bool]:
    """Parse accumulated streaming text into (thinking, caption, still_thinking).

    Unlike parse_model_response (used for final results), this handles
    the intermediate state where tokens are still arriving.
    """
    if _THINK_OPEN not in raw:
        return "", raw, False

    _, after_open = raw.split(_THINK_OPEN, 1)

    if "</think>" in after_open:
        thinking, caption = after_open.split("</think>", 1)
        return thinking, caption, False

    return after_open, "", True


class LlamaServerClient:
    """Client for patched llama-server with video support.

    Communicates via the OpenAI-compatible /chat/completions endpoint,
    sending preprocessed video frames as base64-encoded images with
    video metadata.
    """

    def __init__(self, config: ServerConfig | None = None) -> None:
        if config is None:
            from llama_video.config import ServerConfig

            config = ServerConfig()
        self._config = config
        self._client: httpx.AsyncClient | None = None
        self.last_thinking: str = ""
        self.last_truncated: bool = False

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._config.url,
                timeout=httpx.Timeout(self._config.timeout),
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    def _super_frame_to_base64_pair(self, super_frame_data: np.ndarray) -> list[str]:
        """Convert a 6-channel super-frame back to two base64 JPEG images.

        For now, we send individual frames as separate images.
        The C patch is responsible for recognizing these as video frames
        and applying temporal encoding.

        Args:
            super_frame_data: (6, H, W) float32 normalized array.

        Returns:
            Two base64-encoded JPEG strings.
        """
        # Denormalize for transmission (the server re-normalizes)
        from llama_video.config import ModelConfig

        model = ModelConfig.qwen35()
        mean = np.array(model.image_mean, dtype=np.float32).reshape(3, 1, 1)
        std = np.array(model.image_std, dtype=np.float32).reshape(3, 1, 1)

        images: list[str] = []
        for offset in (0, 3):
            # Extract 3-channel slice and denormalize
            channels = super_frame_data[offset : offset + 3]  # (3, H, W)
            channels = channels * std + mean
            channels = np.clip(channels * 255, 0, 255).astype(np.uint8)
            channels = np.transpose(channels, (1, 2, 0))  # (H, W, 3)

            img = Image.fromarray(channels, "RGB")
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=95)
            b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
            images.append(f"data:image/jpeg;base64,{b64}")

        return images

    def _build_video_message(
        self,
        video_input: VideoInput,
        prompt: str,
    ) -> dict[str, Any]:
        """Build an OpenAI-compatible message with video frames.

        The message includes:
        - All frames as image_url entries
        - Video metadata for the C patch to detect video mode
        - The text prompt
        """
        content: list[dict[str, Any]] = []

        # Add all frames as images
        for sf in video_input.super_frames:
            frame_images = self._super_frame_to_base64_pair(sf.data)
            for img_url in frame_images:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": img_url},
                    }
                )

        # Add the text prompt
        content.append(
            {
                "type": "text",
                "text": prompt,
            }
        )

        return {
            "role": "user",
            "content": content,
        }

    @staticmethod
    def _extract_content(data: dict[str, Any]) -> str:
        """Extract text content from a chat completions response."""
        msg = data["choices"][0]["message"]
        content = msg.get("content", "") or ""
        if not content:
            content = msg.get("reasoning_content", "") or ""
        return str(content)

    async def caption_video(
        self,
        video_input: VideoInput,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float | None = None,
        preset: InferencePreset | None = None,
    ) -> str:
        """Send preprocessed video to llama-server and get a caption.

        Args:
            video_input: Preprocessed VideoInput from Preprocessor.
            prompt: Caption prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (overrides preset if given).
            preset: Inference preset to use. Defaults to the "default" preset.

        Returns:
            Generated caption string.

        Raises:
            ServerUnreachableError: Cannot connect to llama-server.
            ServerResponseError: Server returned an error.
            ModelNotLoadedError: No model loaded on server.
        """
        if preset is None:
            from llama_video.config import get_preset

            preset = get_preset("default")

        # Explicit temperature overrides preset
        effective_temp = temperature if temperature is not None else preset.temperature

        client = await self._get_client()
        message = self._build_video_message(video_input, prompt)

        payload: dict[str, Any] = {
            "messages": [message],
            "max_tokens": max_tokens,
            "temperature": effective_temp,
            "top_p": preset.top_p,
            "top_k": preset.top_k,
            "min_p": preset.min_p,
            "presence_penalty": preset.presence_penalty,
            # Video metadata for the C patch
            "mm_processor_kwargs": {
                "fps": video_input.fps,
                "is_video": True,
                "grid_thw": list(video_input.grid_thw),
                "temporal_positions": video_input.temporal_positions,
            },
        }

        if self._config.model_name:
            payload["model"] = self._config.model_name

        logger.info(
            "Sending %d frames to llama-server (grid_thw=%s)",
            video_input.num_source_frames,
            video_input.grid_thw,
        )

        attempt = 0
        last_error: Exception | None = None

        while attempt <= self._config.max_retries:
            try:
                response = await client.post("/v1/chat/completions", json=payload)

                if response.status_code == 503:
                    raise ModelNotLoadedError(
                        "llama-server has no model loaded",
                        context={"status": 503, "body": response.text[:500]},
                    )

                if response.status_code != 200:
                    raise ServerResponseError(
                        f"llama-server error {response.status_code}: {response.text[:500]}",
                        context={"status": response.status_code},
                    )

                data = response.json()
                raw = self._extract_content(data)
                caption, thinking, truncated = parse_model_response(raw)
                self.last_thinking = thinking
                self.last_truncated = truncated
                if thinking:
                    logger.info(
                        "Thinking: %d chars%s",
                        len(thinking),
                        " (truncated)" if truncated else "",
                    )
                logger.info("Caption received: %d chars", len(caption))
                return caption

            except httpx.ConnectError as e:
                last_error = ServerUnreachableError(
                    f"Cannot connect to llama-server at {self._config.url}",
                    context={"url": self._config.url, "error": str(e)},
                )
            except httpx.TimeoutException as e:
                last_error = ServerResponseError(
                    f"llama-server request timed out after {self._config.timeout}s",
                    context={"timeout": self._config.timeout, "error": str(e)},
                )

            attempt += 1
            if attempt <= self._config.max_retries:
                import asyncio

                delay = self._config.retry_delay * (2 ** (attempt - 1))
                logger.warning("Retry %d/%d after %.1fs", attempt, self._config.max_retries, delay)
                await asyncio.sleep(delay)

        if last_error is not None:
            raise last_error
        raise ServerUnreachableError("Failed to connect to llama-server after retries")

    async def caption_image(
        self,
        image_path: str,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float | None = None,
        preset: InferencePreset | None = None,
    ) -> str:
        """Send an image to llama-server and get a caption.

        Unlike caption_video, this sends a single image without
        mm_processor_kwargs (no temporal positions or video mode).

        Args:
            image_path: Path to image file.
            prompt: Caption prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (overrides preset if given).
            preset: Inference preset to use. Defaults to the "default" preset.

        Returns:
            Generated caption string.
        """
        if preset is None:
            from llama_video.config import get_preset

            preset = get_preset("default")

        effective_temp = temperature if temperature is not None else preset.temperature

        from llama_video.image import build_image_message

        message = build_image_message(image_path, prompt)
        client = await self._get_client()

        payload: dict[str, Any] = {
            "messages": [message],
            "max_tokens": max_tokens,
            "temperature": effective_temp,
            "top_p": preset.top_p,
            "top_k": preset.top_k,
            "min_p": preset.min_p,
            "presence_penalty": preset.presence_penalty,
        }

        if self._config.model_name:
            payload["model"] = self._config.model_name

        logger.info("Sending image to llama-server: %s", image_path)

        attempt = 0
        last_error: Exception | None = None

        while attempt <= self._config.max_retries:
            try:
                response = await client.post("/v1/chat/completions", json=payload)

                if response.status_code == 503:
                    raise ModelNotLoadedError(
                        "llama-server has no model loaded",
                        context={"status": 503, "body": response.text[:500]},
                    )

                if response.status_code != 200:
                    raise ServerResponseError(
                        f"llama-server error {response.status_code}: {response.text[:500]}",
                        context={"status": response.status_code},
                    )

                data = response.json()
                raw = self._extract_content(data)
                caption, thinking, truncated = parse_model_response(raw)
                self.last_thinking = thinking
                self.last_truncated = truncated
                if thinking:
                    logger.info(
                        "Image thinking: %d chars%s",
                        len(thinking),
                        " (truncated)" if truncated else "",
                    )
                logger.info("Image caption received: %d chars", len(caption))
                return caption

            except httpx.ConnectError as e:
                last_error = ServerUnreachableError(
                    f"Cannot connect to llama-server at {self._config.url}",
                    context={"url": self._config.url, "error": str(e)},
                )
            except httpx.TimeoutException as e:
                last_error = ServerResponseError(
                    f"llama-server request timed out after {self._config.timeout}s",
                    context={"timeout": self._config.timeout, "error": str(e)},
                )

            attempt += 1
            if attempt <= self._config.max_retries:
                import asyncio

                delay = self._config.retry_delay * (2 ** (attempt - 1))
                logger.warning("Retry %d/%d after %.1fs", attempt, self._config.max_retries, delay)
                await asyncio.sleep(delay)

        if last_error is not None:
            raise last_error
        raise ServerUnreachableError("Failed to connect to llama-server after retries")

    async def _iter_sse_tokens(
        self,
        payload: dict[str, Any],
    ) -> AsyncGenerator[str, None]:
        """Stream SSE tokens from /v1/chat/completions.

        Yields individual token strings as they arrive.
        """
        client = await self._get_client()
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            json=payload,
        ) as response:
            if response.status_code == 503:
                await response.aread()
                raise ModelNotLoadedError(
                    "llama-server has no model loaded",
                    context={"status": 503},
                )
            if response.status_code != 200:
                await response.aread()
                raise ServerResponseError(
                    f"llama-server error {response.status_code}",
                    context={"status": response.status_code},
                )

            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                token = delta.get("content", "") or ""
                if token:
                    yield token

    async def stream_caption_video(
        self,
        video_input: VideoInput,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float | None = None,
        preset: InferencePreset | None = None,
    ) -> AsyncGenerator[tuple[str, str, bool], None]:
        """Stream video caption, yielding (thinking, caption, still_thinking).

        After the generator is exhausted, last_thinking and last_truncated
        are set on the client instance.
        """
        if preset is None:
            from llama_video.config import get_preset

            preset = get_preset("default")

        effective_temp = temperature if temperature is not None else preset.temperature
        message = self._build_video_message(video_input, prompt)

        payload: dict[str, Any] = {
            "messages": [message],
            "max_tokens": max_tokens,
            "temperature": effective_temp,
            "top_p": preset.top_p,
            "top_k": preset.top_k,
            "min_p": preset.min_p,
            "presence_penalty": preset.presence_penalty,
            "stream": True,
            "mm_processor_kwargs": {
                "fps": video_input.fps,
                "is_video": True,
                "grid_thw": list(video_input.grid_thw),
                "temporal_positions": video_input.temporal_positions,
            },
        }

        if self._config.model_name:
            payload["model"] = self._config.model_name

        logger.info(
            "Streaming %d frames to llama-server (grid_thw=%s)",
            video_input.num_source_frames,
            video_input.grid_thw,
        )

        raw = ""
        async for token in self._iter_sse_tokens(payload):
            raw += token
            thinking, caption, still_thinking = _parse_stream_state(raw)
            yield thinking, caption, still_thinking

        caption, thinking, truncated = parse_model_response(raw)
        self.last_thinking = thinking
        self.last_truncated = truncated

    async def stream_caption_image(
        self,
        image_path: str,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float | None = None,
        preset: InferencePreset | None = None,
    ) -> AsyncGenerator[tuple[str, str, bool], None]:
        """Stream image caption, yielding (thinking, caption, still_thinking)."""
        if preset is None:
            from llama_video.config import get_preset

            preset = get_preset("default")

        effective_temp = temperature if temperature is not None else preset.temperature

        from llama_video.image import build_image_message

        message = build_image_message(image_path, prompt)

        payload: dict[str, Any] = {
            "messages": [message],
            "max_tokens": max_tokens,
            "temperature": effective_temp,
            "top_p": preset.top_p,
            "top_k": preset.top_k,
            "min_p": preset.min_p,
            "presence_penalty": preset.presence_penalty,
            "stream": True,
        }

        if self._config.model_name:
            payload["model"] = self._config.model_name

        logger.info("Streaming image to llama-server: %s", image_path)

        raw = ""
        async for token in self._iter_sse_tokens(payload):
            raw += token
            thinking, caption, still_thinking = _parse_stream_state(raw)
            yield thinking, caption, still_thinking

        caption, thinking, truncated = parse_model_response(raw)
        self.last_thinking = thinking
        self.last_truncated = truncated

    async def health_check(self) -> bool:
        """Check if llama-server is reachable and healthy."""
        try:
            client = await self._get_client()
            response = await client.get("/health")
            return response.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False
