"""Gemma4 model adapter: individual frames, timestamps, native llama.cpp.

Gemma4 processes video as individual frames at 1 FPS with MM:SS timestamps.
Unlike Qwen3.5, it uses native llama.cpp (no C patch) and has no superframe
pairing or grid THW computation.
"""

from __future__ import annotations

import base64
import io
import logging
import re
from typing import Any

import numpy as np
from PIL import Image

from llama_video.adapters.base import AdapterPreset, ModelAdapter
from llama_video.adapters.registry import register_adapter
from llama_video.errors import PreprocessingError
from llama_video.preprocessor import VideoInput
from llama_video.types import Frame, SuperFrame

logger = logging.getLogger(__name__)

_THOUGHT_RE = re.compile(r"<\|channel\|>thought\n(.*?)<\|channel\|>", re.DOTALL)
_THOUGHT_OPEN = "<|channel|>thought\n"

GEMMA_PRESET = AdapterPreset(
    temperature=1.0,
    top_p=0.95,
    top_k=64,
    min_p=0.0,
    presence_penalty=0.0,
)

_DEFAULT_IMAGE_MAX_TOKENS = 280

_SYSTEM_MESSAGE: dict[str, str] = {
    "role": "system",
    "content": (
        "You are a media captioning assistant. Your reasoning is private and "
        "will not be shown to the user. Your response must contain the complete, "
        "detailed caption — do not summarize or abbreviate what you described "
        "in your reasoning. Write the full description in your response."
    ),
}


class GemmaAdapter(ModelAdapter):
    """Gemma4 video adapter: 1 FPS individual frames with timestamps."""

    def __init__(self, image_max_tokens: int = _DEFAULT_IMAGE_MAX_TOKENS) -> None:
        self._image_max_tokens = image_max_tokens

    @property
    def name(self) -> str:
        return "gemma4"

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
        return GEMMA_PRESET

    # ── Preprocessing ──────────────────────────────────────────────

    def _frame_to_base64(self, frame: Frame, quality: int = 95) -> str:
        img = Image.fromarray(frame.data)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        m = int(seconds) // 60
        s = int(seconds) % 60
        return f"{m:02d}:{s:02d}"

    def preprocess(
        self,
        frames: list[Frame],
        fps: float = 1.0,
        odd_strategy: Any = None,
        resolution_scale: float = 1.0,
    ) -> VideoInput:
        if not frames:
            raise PreprocessingError("No frames to process", context={"frame_count": 0})

        if len(frames) > self.max_frames:
            raise PreprocessingError(
                f"Gemma4 supports max {self.max_frames} frames, got {len(frames)}",
                context={"frame_count": len(frames), "max_frames": self.max_frames},
            )

        # Gemma4 processes frames individually — no pairing.
        # Store each frame as a single-channel SuperFrame for VideoInput compat.
        super_frames: list[SuperFrame] = []
        for i, frame in enumerate(frames):
            arr = np.transpose(frame.data, (2, 0, 1)).astype(np.float32) / 255.0
            super_frames.append(
                SuperFrame(
                    data=arr,
                    temporal_index=i,
                    source_frames=(frame.index, frame.index),
                )
            )

        # No grid THW — use (num_frames, 1, 1) as placeholder
        grid_thw = (len(super_frames), 1, 1)

        # Timestamps for each frame
        temporal_positions = [round(frame.timestamp) for frame in frames]

        w, h = frames[0].width, frames[0].height

        return VideoInput(
            super_frames=super_frames,
            grid_thw=grid_thw,
            temporal_positions=temporal_positions,
            fps=fps,
            num_source_frames=len(frames),
            resolution=(w, h),
        )

    # ── Payload construction ───────────────────────────────────────

    def _build_video_message(self, video_input: VideoInput, prompt: str) -> dict[str, Any]:
        content: list[dict[str, Any]] = []

        for sf, ts in zip(
            video_input.super_frames,
            video_input.temporal_positions,
            strict=True,
        ):
            timestamp_str = self._format_timestamp(ts)
            # Reconstruct image from stored frame data
            ch = np.clip(sf.data * 255, 0, 255).astype(np.uint8)
            ch = np.transpose(ch, (1, 2, 0))
            img = Image.fromarray(ch, "RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=95)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            img_url = f"data:image/jpeg;base64,{b64}"

            # Timestamp text before each image
            content.append({"type": "text", "text": f"{timestamp_str} "})
            content.append({"type": "image_url", "image_url": {"url": img_url}})

        # Text prompt comes AFTER all images (Gemma4 modality order)
        content.append({"type": "text", "text": prompt})

        return {"role": "user", "content": content}

    def build_payload(
        self,
        video_input: VideoInput,
        prompt: str,
        max_tokens: int = 2048,
        preset: AdapterPreset | None = None,
        cache_prompt: bool = True,
        model_name: str = "",
    ) -> dict[str, Any]:
        p = preset or self.default_preset
        message = self._build_video_message(video_input, prompt)

        payload: dict[str, Any] = {
            "messages": [_SYSTEM_MESSAGE, message],
            "max_tokens": max_tokens,
            "temperature": p.temperature,
            "top_p": p.top_p,
            "top_k": p.top_k,
            "min_p": p.min_p,
            "presence_penalty": p.presence_penalty,
            "cache_prompt": cache_prompt,
        }

        if model_name:
            payload["model"] = model_name

        return payload

    # ── Response parsing ───────────────────────────────────────────

    def parse_response(self, raw: str) -> tuple[str, str, bool]:
        if not raw:
            return "", "", False

        m = _THOUGHT_RE.search(raw)
        if m:
            thinking = m.group(1).strip()
            caption = raw[m.end() :].strip()
            # Strip any leading turn tokens
            caption = re.sub(r"^<\|turn\|>model\s*", "", caption)
            return caption, thinking, False

        if _THOUGHT_OPEN in raw:
            thinking = raw.split(_THOUGHT_OPEN, 1)[1].strip()
            return "", thinking, True

        # Strip turn tokens from plain response
        cleaned = re.sub(r"<\|turn\|>\w+\s*", "", raw).strip()
        return cleaned, "", False

    # ── Token estimation ───────────────────────────────────────────

    def estimate_tokens(self, video_input: VideoInput) -> int:
        # Each frame uses image_max_tokens * 2 (two image tokens per frame)
        t, _, _ = video_input.grid_thw
        return t * self._image_max_tokens * 2


register_adapter("gemma4", GemmaAdapter)
