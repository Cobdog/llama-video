"""Qwen3.5 model adapter: superframe pairing, grid THW, M-RoPE positions.

Extracts the Qwen3.5-specific video pipeline from the original
preprocessor.py, client.py, and tokens.py into a single adapter.
"""

from __future__ import annotations

import base64
import io
import logging
import math
import re
from typing import Any

import numpy as np
from PIL import Image

from llama_video.adapters.base import AdapterPreset, ModelAdapter
from llama_video.adapters.registry import register_adapter
from llama_video.config import ModelConfig
from llama_video.errors import InvalidFrameDimensionsError, PreprocessingError
from llama_video.preprocessor import VideoInput
from llama_video.types import Frame, OddFrameStrategy, SuperFrame

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think >(.*?)</think >", re.DOTALL)
_THINK_OPEN = "<think >"

QWEN_PRESET = AdapterPreset(
    temperature=1.0,
    top_p=0.95,
    top_k=20,
    min_p=0.0,
    presence_penalty=1.5,
)

_SYSTEM_MESSAGE: dict[str, str] = {
    "role": "system",
    "content": (
        "You are a media captioning assistant. Your reasoning is private and "
        "will not be shown to the user. Your response must contain the complete, "
        "detailed caption — do not summarize or abbreviate what you described "
        "in your reasoning. Write the full description in your response."
    ),
}


class QwenAdapter(ModelAdapter):
    """Qwen3.5 video adapter: 2-frame superframes with grid THW."""

    def __init__(self, model_config: ModelConfig | None = None) -> None:
        self._config = model_config or ModelConfig.qwen35()

    @property
    def name(self) -> str:
        return "qwen3.5"

    @property
    def default_fps(self) -> float:
        return 2.0

    @property
    def max_duration_seconds(self) -> float:
        return float("inf")

    @property
    def max_frames(self) -> int:
        return 64

    @property
    def default_preset(self) -> AdapterPreset:
        return QWEN_PRESET

    # ── Preprocessing ──────────────────────────────────────────────

    def _compute_target_resolution(self, width: int, height: int) -> tuple[int, int]:
        grid_unit = self._config.grid_unit
        total_pixels = width * height
        if total_pixels > self._config.max_pixels:
            scale = math.sqrt(self._config.max_pixels / total_pixels)
            width = int(width * scale)
            height = int(height * scale)
        total_pixels = width * height
        if total_pixels < self._config.min_pixels:
            scale = math.sqrt(self._config.min_pixels / total_pixels)
            width = int(width * scale)
            height = int(height * scale)
        target_w = max(grid_unit, round(width / grid_unit) * grid_unit)
        target_h = max(grid_unit, round(height / grid_unit) * grid_unit)
        return target_w, target_h

    def _resize_frame(self, frame: Frame, target_w: int, target_h: int) -> np.ndarray:
        img = Image.fromarray(frame.data)
        img = img.resize((target_w, target_h), Image.Resampling.BICUBIC)
        return np.array(img, dtype=np.float32) / 255.0

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        mean = np.array(self._config.image_mean, dtype=np.float32)
        std = np.array(self._config.image_std, dtype=np.float32)
        return (data - mean) / std

    def _build_super_frames(
        self,
        frames: list[Frame],
        target_w: int,
        target_h: int,
        odd_strategy: OddFrameStrategy = OddFrameStrategy.PAD,
    ) -> list[SuperFrame]:
        working = list(frames)
        if len(working) % self._config.temporal_patch_size != 0:
            if odd_strategy == OddFrameStrategy.PAD:
                working.append(working[-1])
            elif odd_strategy == OddFrameStrategy.DROP:
                working = working[:-1]

        result: list[SuperFrame] = []
        for i in range(0, len(working), self._config.temporal_patch_size):
            fa = working[i]
            fb = working[i + 1]
            arr_a = self._normalize(self._resize_frame(fa, target_w, target_h))
            arr_b = self._normalize(self._resize_frame(fb, target_w, target_h))
            arr_a = np.transpose(arr_a, (2, 0, 1))
            arr_b = np.transpose(arr_b, (2, 0, 1))
            sf_data = np.concatenate([arr_a, arr_b], axis=0)
            tidx = i // self._config.temporal_patch_size
            result.append(
                SuperFrame(data=sf_data, temporal_index=tidx, source_frames=(fa.index, fb.index))
            )
        return result

    def _compute_grid_thw(self, num_sf: int, tw: int, th: int) -> tuple[int, int, int]:
        gu = self._config.grid_unit
        return (num_sf, th // gu, tw // gu)

    def _compute_temporal_positions(self, grid_thw: tuple[int, int, int], fps: float) -> list[int]:
        t, _, _ = grid_thw
        spt = self._config.temporal_patch_size / fps
        return [round(i * spt) for i in range(t)]

    def preprocess(
        self,
        frames: list[Frame],
        fps: float = 2.0,
        odd_strategy: OddFrameStrategy = OddFrameStrategy.PAD,
        resolution_scale: float = 1.0,
    ) -> VideoInput:
        if not frames:
            raise PreprocessingError("No frames to process", context={"frame_count": 0})

        widths = {f.width for f in frames}
        heights = {f.height for f in frames}
        if len(widths) > 1 or len(heights) > 1:
            raise InvalidFrameDimensionsError(
                "Frames have inconsistent dimensions",
                context={"widths": sorted(widths), "heights": sorted(heights)},
            )

        w, h = frames[0].width, frames[0].height
        if resolution_scale != 1.0:
            w = max(1, int(w * resolution_scale))
            h = max(1, int(h * resolution_scale))

        tw, th = self._compute_target_resolution(w, h)
        sfs = self._build_super_frames(frames, tw, th, odd_strategy)
        if not sfs:
            raise PreprocessingError(
                "No super-frames produced",
                context={"frame_count": len(frames), "odd_strategy": odd_strategy.value},
            )

        grid_thw = self._compute_grid_thw(len(sfs), tw, th)
        temporal = self._compute_temporal_positions(grid_thw, fps)

        return VideoInput(
            super_frames=sfs,
            grid_thw=grid_thw,
            temporal_positions=temporal,
            fps=fps,
            num_source_frames=len(frames),
            resolution=(tw, th),
        )

    # ── Payload construction ───────────────────────────────────────

    def _super_frame_to_base64_pair(self, sf_data: np.ndarray) -> list[str]:
        mean = np.array(self._config.image_mean, dtype=np.float32).reshape(3, 1, 1)
        std = np.array(self._config.image_std, dtype=np.float32).reshape(3, 1, 1)
        images: list[str] = []
        for offset in (0, 3):
            ch = sf_data[offset : offset + 3]
            ch = ch * std + mean
            ch = np.clip(ch * 255, 0, 255).astype(np.uint8)
            ch = np.transpose(ch, (1, 2, 0))
            img = Image.fromarray(ch, "RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=95)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            images.append(f"data:image/jpeg;base64,{b64}")
        return images

    def _build_video_message(self, video_input: VideoInput, prompt: str) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        for sf in video_input.super_frames:
            for img_url in self._super_frame_to_base64_pair(sf.data):
                content.append({"type": "image_url", "image_url": {"url": img_url}})
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
            "mm_processor_kwargs": {
                "fps": video_input.fps,
                "is_video": True,
                "grid_thw": list(video_input.grid_thw),
                "temporal_positions": video_input.temporal_positions,
            },
        }
        if model_name:
            payload["model"] = model_name
        return payload

    # ── Response parsing ───────────────────────────────────────────

    def parse_response(self, raw: str) -> tuple[str, str, bool]:
        if not raw:
            return "", "", False
        m = _THINK_RE.search(raw)
        if m:
            thinking = m.group(1).strip()
            caption = raw[m.end() :].strip()
            return caption, thinking, False
        if _THINK_OPEN in raw:
            thinking = raw.split(_THINK_OPEN, 1)[1].strip()
            return "", thinking, True
        return raw.strip(), "", False

    # ── Token estimation ───────────────────────────────────────────

    def estimate_tokens(self, video_input: VideoInput) -> int:
        t, h, w = video_input.grid_thw
        return t * h * w


register_adapter("qwen3.5", QwenAdapter)
