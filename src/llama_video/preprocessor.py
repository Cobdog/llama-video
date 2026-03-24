"""Video preprocessing: super-frame construction, grid computation, temporal positions."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from llama_video.errors import InvalidFrameDimensionsError, PreprocessingError
from llama_video.types import Frame, OddFrameStrategy, SuperFrame

if TYPE_CHECKING:
    from llama_video.config import ModelConfig

logger = logging.getLogger(__name__)


class VideoInput:
    """Preprocessed video ready for the vision encoder.

    Contains super-frames, grid dimensions, and temporal positions.
    """

    __slots__ = (
        "fps",
        "grid_thw",
        "num_source_frames",
        "resolution",
        "super_frames",
        "temporal_positions",
    )

    def __init__(
        self,
        super_frames: list[SuperFrame],
        grid_thw: tuple[int, int, int],
        temporal_positions: list[int],
        fps: float,
        num_source_frames: int,
        resolution: tuple[int, int],
    ) -> None:
        self.super_frames = super_frames
        self.grid_thw = grid_thw
        self.temporal_positions = temporal_positions
        self.fps = fps
        self.num_source_frames = num_source_frames
        self.resolution = resolution  # (width, height) after resize

    def __repr__(self) -> str:
        return (
            f"VideoInput(super_frames={len(self.super_frames)}, "
            f"grid_thw={self.grid_thw}, resolution={self.resolution})"
        )


class Preprocessor:
    """Transform extracted frames into vision encoder input.

    Handles:
    - Frame resizing to model-compatible resolution
    - CLIP normalization
    - Super-frame construction (temporal pairing)
    - Grid THW computation
    - Temporal M-RoPE position computation
    """

    def __init__(self, model_config: ModelConfig | None = None) -> None:
        if model_config is None:
            from llama_video.config import ModelConfig

            model_config = ModelConfig.qwen35()
        self._config = model_config

    def _compute_target_resolution(
        self,
        width: int,
        height: int,
    ) -> tuple[int, int]:
        """Compute target resolution within model bounds, divisible by grid_unit.

        Follows Qwen3.5's smart resize: find the closest resolution
        where total pixels is within [min_pixels, max_pixels] and
        both dimensions are divisible by grid_unit (patch_size x merge_size = 28).
        """
        grid_unit = self._config.grid_unit  # 28

        # Scale to fit within max_pixels while preserving aspect ratio
        total_pixels = width * height
        if total_pixels > self._config.max_pixels:
            scale = math.sqrt(self._config.max_pixels / total_pixels)
            width = int(width * scale)
            height = int(height * scale)

        # Scale up to meet min_pixels
        total_pixels = width * height
        if total_pixels < self._config.min_pixels:
            scale = math.sqrt(self._config.min_pixels / total_pixels)
            width = int(width * scale)
            height = int(height * scale)

        # Round to nearest multiple of grid_unit
        target_w = max(grid_unit, round(width / grid_unit) * grid_unit)
        target_h = max(grid_unit, round(height / grid_unit) * grid_unit)

        return target_w, target_h

    def _resize_frame(self, frame: Frame, target_w: int, target_h: int) -> np.ndarray:
        """Resize a frame and return as float32 normalized array."""
        img = Image.fromarray(frame.data)
        img = img.resize((target_w, target_h), Image.Resampling.BICUBIC)
        arr = np.array(img, dtype=np.float32) / 255.0
        return arr  # (H, W, 3) float32 [0, 1]

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """Apply CLIP normalization: (pixel - mean) / std."""
        mean = np.array(self._config.image_mean, dtype=np.float32)
        std = np.array(self._config.image_std, dtype=np.float32)
        return (data - mean) / std

    def build_super_frames(
        self,
        frames: list[Frame],
        target_w: int,
        target_h: int,
        odd_strategy: OddFrameStrategy = OddFrameStrategy.PAD,
    ) -> list[SuperFrame]:
        """Pair consecutive frames into 6-channel super-frames.

        Args:
            frames: Extracted video frames.
            target_w: Target width (must be divisible by grid_unit).
            target_h: Target height (must be divisible by grid_unit).
            odd_strategy: How to handle odd frame counts.

        Returns:
            List of SuperFrame objects.
        """
        working_frames = list(frames)

        # Handle odd frame count
        if len(working_frames) % self._config.temporal_patch_size != 0:
            if odd_strategy == OddFrameStrategy.PAD:
                # Duplicate last frame
                working_frames.append(working_frames[-1])
                logger.debug("Padded odd frame count %d → %d", len(frames), len(working_frames))
            elif odd_strategy == OddFrameStrategy.DROP:
                working_frames = working_frames[:-1]
                logger.debug("Dropped last frame: %d → %d", len(frames), len(working_frames))

        super_frames: list[SuperFrame] = []
        for i in range(0, len(working_frames), self._config.temporal_patch_size):
            frame_a = working_frames[i]
            frame_b = working_frames[i + 1]

            # Resize and normalize each frame
            arr_a = self._normalize(self._resize_frame(frame_a, target_w, target_h))
            arr_b = self._normalize(self._resize_frame(frame_b, target_w, target_h))

            # Convert to channel-first: (H, W, 3) → (3, H, W)
            arr_a = np.transpose(arr_a, (2, 0, 1))
            arr_b = np.transpose(arr_b, (2, 0, 1))

            # Concatenate on channel dim: (3, H, W) + (3, H, W) → (6, H, W)
            super_frame_data = np.concatenate([arr_a, arr_b], axis=0)

            temporal_idx = i // self._config.temporal_patch_size
            super_frames.append(
                SuperFrame(
                    data=super_frame_data,
                    temporal_index=temporal_idx,
                    source_frames=(frame_a.index, frame_b.index),
                )
            )

        logger.info(
            "Built %d super-frames from %d frames (target %dx%d)",
            len(super_frames),
            len(frames),
            target_w,
            target_h,
        )
        return super_frames

    def compute_grid_thw(
        self,
        num_super_frames: int,
        target_w: int,
        target_h: int,
    ) -> tuple[int, int, int]:
        """Compute the temporal-height-width grid dimensions.

        Args:
            num_super_frames: Number of super-frames (= temporal positions).
            target_w: Frame width after resize.
            target_h: Frame height after resize.

        Returns:
            (T, H, W) grid dimensions.
        """
        grid_unit = self._config.grid_unit
        t = num_super_frames
        h = target_h // grid_unit
        w = target_w // grid_unit
        return (t, h, w)

    def compute_temporal_positions(
        self,
        grid_thw: tuple[int, int, int],
        fps: float,
    ) -> list[int]:
        """Compute M-RoPE temporal indices for each temporal grid position.

        For each temporal position t (0 to T-1):
          seconds = t * temporal_patch_size / fps
          position = round(seconds)  # In Qwen3.5, positions are in frame-pair units

        Args:
            grid_thw: (T, H, W) grid dimensions.
            fps: Frame extraction FPS.

        Returns:
            List of temporal position indices, length T.
        """
        t, _, _ = grid_thw
        seconds_per_temporal = self._config.temporal_patch_size / fps

        positions: list[int] = []
        for i in range(t):
            seconds = i * seconds_per_temporal
            positions.append(round(seconds))

        return positions

    def process(
        self,
        frames: list[Frame],
        fps: float = 2.0,
        odd_strategy: OddFrameStrategy = OddFrameStrategy.PAD,
        resolution_scale: float = 1.0,
    ) -> VideoInput:
        """Full preprocessing pipeline: resize, pair, compute grid and positions.

        Args:
            frames: Extracted video frames.
            fps: Frame extraction FPS used.
            odd_strategy: How to handle odd frame counts.
            resolution_scale: Scale factor applied to native resolution before
                grid snapping. 1.0 = full, 0.5 = half, etc.

        Returns:
            VideoInput ready for the vision encoder.

        Raises:
            PreprocessingError: If frames are empty.
            InvalidFrameDimensionsError: If frames have inconsistent sizes.
            ResolutionError: If resolution is outside model bounds.
        """
        if not frames:
            raise PreprocessingError(
                "No frames to process",
                context={"frame_count": 0},
            )

        # Validate consistent dimensions
        widths = {f.width for f in frames}
        heights = {f.height for f in frames}
        if len(widths) > 1 or len(heights) > 1:
            raise InvalidFrameDimensionsError(
                "Frames have inconsistent dimensions",
                context={"widths": sorted(widths), "heights": sorted(heights)},
            )

        width = frames[0].width
        height = frames[0].height

        # Apply resolution scale before smart resize
        if resolution_scale != 1.0:
            width = max(1, int(width * resolution_scale))
            height = max(1, int(height * resolution_scale))

        # Compute target resolution
        target_w, target_h = self._compute_target_resolution(width, height)
        logger.debug("Target resolution: %dx%d → %dx%d", width, height, target_w, target_h)

        # Build super-frames
        super_frames = self.build_super_frames(frames, target_w, target_h, odd_strategy)

        if not super_frames:
            raise PreprocessingError(
                "No super-frames produced",
                context={"frame_count": len(frames), "odd_strategy": odd_strategy.value},
            )

        # Compute grid and positions
        grid_thw = self.compute_grid_thw(len(super_frames), target_w, target_h)
        temporal_positions = self.compute_temporal_positions(grid_thw, fps)

        result = VideoInput(
            super_frames=super_frames,
            grid_thw=grid_thw,
            temporal_positions=temporal_positions,
            fps=fps,
            num_source_frames=len(frames),
            resolution=(target_w, target_h),
        )

        logger.info(
            "Preprocessed: %d frames → %d super-frames, grid_thw=%s, resolution=%s",
            len(frames),
            len(super_frames),
            grid_thw,
            (target_w, target_h),
        )
        return result
