"""End-to-end integration tests for the full Extractor -> Preprocessor pipeline.

These tests use real video files from testvid/ and verify:
- Frames are correctly extracted by Extractor
- Preprocessor.process() produces valid VideoInput
- Super-frames have correct 6-channel structure
- grid_thw dimensions are consistent with frame count and resolution
- temporal_positions length matches grid_thw[0]
- Resolution is divisible by 28 (grid_unit)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from llama_video.config import ModelConfig
from llama_video.extractor import Extractor, ExtractorConfig
from llama_video.preprocessor import Preprocessor, VideoInput
from llama_video.types import OddFrameStrategy

# Path to test videos relative to project root
TESTVID_DIR = Path(__file__).parent.parent.parent / "testvid"


@pytest.fixture
def kiki_video() -> Path:
    """Path to kiki.mp4 test video (4.6s, 1920x1040)."""
    video_path = TESTVID_DIR / "kiki.mp4"
    if not video_path.exists():
        pytest.skip(f"Test video not found: {video_path}")
    return video_path


@pytest.fixture
def howl_video() -> Path:
    """Path to howl.mp4 test video."""
    video_path = TESTVID_DIR / "howl.mp4"
    if not video_path.exists():
        pytest.skip(f"Test video not found: {video_path}")
    return video_path


@pytest.fixture
def boy_and_heron_video() -> Path:
    """Path to boy_and_heron.mp4 test video."""
    video_path = TESTVID_DIR / "boy_and_heron.mp4"
    if not video_path.exists():
        pytest.skip(f"Test video not found: {video_path}")
    return video_path


@pytest.fixture
def arrietty_video() -> Path:
    """Path to arrietty_1.mp4 test video."""
    video_path = TESTVID_DIR / "arrietty_1.mp4"
    if not video_path.exists():
        pytest.skip(f"Test video not found: {video_path}")
    return video_path


@pytest.fixture
def coquelicots_video() -> Path:
    """Path to coquelicots.mp4 test video."""
    video_path = TESTVID_DIR / "coquelicots.mp4"
    if not video_path.exists():
        pytest.skip(f"Test video not found: {video_path}")
    return video_path


@pytest.fixture
def earthsea_video() -> Path:
    """Path to earthsea.mp4 test video."""
    video_path = TESTVID_DIR / "earthsea.mp4"
    if not video_path.exists():
        pytest.skip(f"Test video not found: {video_path}")
    return video_path


@pytest.fixture
def model_config() -> ModelConfig:
    """Default Qwen3.5 model config."""
    return ModelConfig.qwen35()


@pytest.mark.integration
class TestFullPipeline:
    """End-to-end tests for Extractor -> Preprocessor pipeline."""

    def test_kiki_video_full_pipeline(self, kiki_video: Path, model_config: ModelConfig) -> None:
        """Full pipeline: extract frames from kiki.mp4 and preprocess."""
        # Extract frames at 2fps (default for Qwen3.5)
        extractor = Extractor()
        extract_config = ExtractorConfig(fps=2.0, max_frames=64)
        frames = extractor.extract_frames(kiki_video, extract_config)

        # Verify extraction
        assert len(frames) > 0, "No frames extracted"

        # Run through preprocessor
        preprocessor = Preprocessor(model_config)
        video_input = preprocessor.process(frames, fps=extract_config.fps)

        # Verify VideoInput structure
        self._assert_valid_video_input(video_input, len(frames), extract_config.fps, model_config)

    def test_howl_video_full_pipeline(self, howl_video: Path, model_config: ModelConfig) -> None:
        """Full pipeline with a different video file."""
        extractor = Extractor()
        extract_config = ExtractorConfig(fps=2.0, max_frames=32)
        frames = extractor.extract_frames(howl_video, extract_config)

        assert len(frames) > 0

        preprocessor = Preprocessor(model_config)
        video_input = preprocessor.process(frames, fps=extract_config.fps)

        self._assert_valid_video_input(video_input, len(frames), extract_config.fps, model_config)

    def test_boy_and_heron_full_pipeline(
        self, boy_and_heron_video: Path, model_config: ModelConfig
    ) -> None:
        """Full pipeline with boy_and_heron.mp4."""
        extractor = Extractor()
        extract_config = ExtractorConfig(fps=2.0, max_frames=32)
        frames = extractor.extract_frames(boy_and_heron_video, extract_config)

        preprocessor = Preprocessor(model_config)
        video_input = preprocessor.process(frames, fps=extract_config.fps)

        self._assert_valid_video_input(video_input, len(frames), extract_config.fps, model_config)

    def test_arrietty_full_pipeline(self, arrietty_video: Path, model_config: ModelConfig) -> None:
        """Full pipeline with arrietty_1.mp4."""
        extractor = Extractor()
        extract_config = ExtractorConfig(fps=2.0, max_frames=32)
        frames = extractor.extract_frames(arrietty_video, extract_config)

        preprocessor = Preprocessor(model_config)
        video_input = preprocessor.process(frames, fps=extract_config.fps)

        self._assert_valid_video_input(video_input, len(frames), extract_config.fps, model_config)

    def test_coquelicots_full_pipeline(
        self, coquelicots_video: Path, model_config: ModelConfig
    ) -> None:
        """Full pipeline with coquelicots.mp4."""
        extractor = Extractor()
        extract_config = ExtractorConfig(fps=2.0, max_frames=32)
        frames = extractor.extract_frames(coquelicots_video, extract_config)

        preprocessor = Preprocessor(model_config)
        video_input = preprocessor.process(frames, fps=extract_config.fps)

        self._assert_valid_video_input(video_input, len(frames), extract_config.fps, model_config)

    def test_earthsea_full_pipeline(self, earthsea_video: Path, model_config: ModelConfig) -> None:
        """Full pipeline with earthsea.mp4."""
        extractor = Extractor()
        extract_config = ExtractorConfig(fps=2.0, max_frames=32)
        frames = extractor.extract_frames(earthsea_video, extract_config)

        preprocessor = Preprocessor(model_config)
        video_input = preprocessor.process(frames, fps=extract_config.fps)

        self._assert_valid_video_input(video_input, len(frames), extract_config.fps, model_config)

    def test_pipeline_at_different_fps(self, kiki_video: Path, model_config: ModelConfig) -> None:
        """Pipeline should work correctly at different FPS settings."""
        for fps in [1.0, 2.0, 4.0]:
            extractor = Extractor()
            extract_config = ExtractorConfig(fps=fps, max_frames=64)
            frames = extractor.extract_frames(kiki_video, extract_config)

            preprocessor = Preprocessor(model_config)
            video_input = preprocessor.process(frames, fps=fps)

            self._assert_valid_video_input(video_input, len(frames), fps, model_config)

    def test_pipeline_with_odd_frame_count_drop(
        self, kiki_video: Path, model_config: ModelConfig
    ) -> None:
        """Pipeline with DROP strategy for odd frame counts."""
        # Use 5 max_frames to guarantee odd count
        extractor = Extractor()
        extract_config = ExtractorConfig(fps=2.0, max_frames=5)
        frames = extractor.extract_frames(kiki_video, extract_config)

        preprocessor = Preprocessor(model_config)
        video_input = preprocessor.process(
            frames, fps=extract_config.fps, odd_strategy=OddFrameStrategy.DROP
        )

        # With DROP, odd frames -> one dropped
        expected_super_frames = len(frames) // 2
        assert len(video_input.super_frames) == expected_super_frames

    def test_pipeline_with_odd_frame_count_pad(
        self, kiki_video: Path, model_config: ModelConfig
    ) -> None:
        """Pipeline with PAD strategy for odd frame counts."""
        extractor = Extractor()
        extract_config = ExtractorConfig(fps=2.0, max_frames=5)
        frames = extractor.extract_frames(kiki_video, extract_config)

        preprocessor = Preprocessor(model_config)
        video_input = preprocessor.process(
            frames, fps=extract_config.fps, odd_strategy=OddFrameStrategy.PAD
        )

        # With PAD, odd frames -> one frame padded
        expected_super_frames = (len(frames) + 1) // 2
        assert len(video_input.super_frames) == expected_super_frames

    def test_pipeline_with_single_frame(self, kiki_video: Path, model_config: ModelConfig) -> None:
        """Pipeline with only 1 frame (edge case)."""
        extractor = Extractor()
        extract_config = ExtractorConfig(fps=2.0, max_frames=1)
        frames = extractor.extract_frames(kiki_video, extract_config)

        assert len(frames) == 1

        preprocessor = Preprocessor(model_config)
        video_input = preprocessor.process(frames, fps=extract_config.fps)

        # Single frame with PAD -> 1 super-frame
        assert len(video_input.super_frames) == 1
        assert video_input.grid_thw[0] == 1
        assert len(video_input.temporal_positions) == 1

    def test_pipeline_with_two_frames(self, kiki_video: Path, model_config: ModelConfig) -> None:
        """Pipeline with exactly 2 frames (one super-frame pair)."""
        extractor = Extractor()
        extract_config = ExtractorConfig(fps=2.0, max_frames=2)
        frames = extractor.extract_frames(kiki_video, extract_config)

        assert len(frames) == 2

        preprocessor = Preprocessor(model_config)
        video_input = preprocessor.process(frames, fps=extract_config.fps)

        assert len(video_input.super_frames) == 1
        assert video_input.grid_thw[0] == 1
        assert len(video_input.temporal_positions) == 1

    def _assert_valid_video_input(
        self,
        video_input: VideoInput,
        num_frames: int,
        fps: float,
        model_config: ModelConfig,
    ) -> None:
        """Assert that VideoInput has correct structure."""
        # 1. VideoInput should be returned
        assert isinstance(video_input, VideoInput)

        # 2. super_frames should have at least one element
        assert len(video_input.super_frames) >= 1, "No super-frames produced"

        # 3. Each super_frame should be 6-channel tensor with correct shape
        target_w, target_h = video_input.resolution
        for i, sf in enumerate(video_input.super_frames):
            assert sf.shape[0] == 6, f"Super-frame {i} has {sf.shape[0]} channels, expected 6"
            assert sf.shape[1] == target_h, (
                f"Super-frame {i} height {sf.shape[1]} != resolution height {target_h}"
            )
            assert sf.shape[2] == target_w, (
                f"Super-frame {i} width {sf.shape[2]} != resolution width {target_w}"
            )
            # Data type should be float32
            assert sf.data.dtype == np.float32, (
                f"Super-frame {i} has dtype {sf.data.dtype}, expected float32"
            )

        # 4. grid_thw dimensions should be consistent
        t, h, w = video_input.grid_thw
        assert t == len(video_input.super_frames), (
            f"grid_thw T={t} != super_frames count {len(video_input.super_frames)}"
        )
        assert h == target_h // model_config.grid_unit, (
            f"grid_thw H={h} != height/grid_unit = {target_h // model_config.grid_unit}"
        )
        assert w == target_w // model_config.grid_unit, (
            f"grid_thw W={w} != width/grid_unit = {target_w // model_config.grid_unit}"
        )

        # 5. temporal_positions length should equal grid_thw[0]
        assert len(video_input.temporal_positions) == t, (
            f"temporal_positions length {len(video_input.temporal_positions)} != grid_thw T={t}"
        )

        # 6. Resolution should be divisible by grid_unit (28)
        assert target_w % model_config.grid_unit == 0, (
            f"Width {target_w} not divisible by grid_unit {model_config.grid_unit}"
        )
        assert target_h % model_config.grid_unit == 0, (
            f"Height {target_h} not divisible by grid_unit {model_config.grid_unit}"
        )

        # 7. FPS should match what was passed
        assert video_input.fps == fps

        # 8. num_source_frames should match input frame count
        assert video_input.num_source_frames == num_frames

        # 9. Super-frames should have valid (not NaN/Inf) data
        for i, sf in enumerate(video_input.super_frames):
            assert not np.any(np.isnan(sf.data)), f"Super-frame {i} contains NaN values"
            assert not np.any(np.isinf(sf.data)), f"Super-frame {i} contains Inf values"

        # 10. Super-frames should have sequential temporal indices
        for i, sf in enumerate(video_input.super_frames):
            assert sf.temporal_index == i, f"Super-frame {i} has temporal_index {sf.temporal_index}"


@pytest.mark.integration
@pytest.mark.asyncio
class TestFullPipelineAsync:
    """Async end-to-end tests for Extractor -> Preprocessor pipeline."""

    async def test_async_pipeline_kiki(self, kiki_video: Path, model_config: ModelConfig) -> None:
        """Full async pipeline: extract and preprocess."""
        extractor = Extractor()
        extract_config = ExtractorConfig(fps=2.0, max_frames=16)
        frames = await extractor.extract_frames_async(kiki_video, extract_config)

        preprocessor = Preprocessor(model_config)
        video_input = preprocessor.process(frames, fps=extract_config.fps)

        # Basic validation
        assert isinstance(video_input, VideoInput)
        assert len(video_input.super_frames) >= 1
        assert video_input.resolution[0] % 28 == 0
        assert video_input.resolution[1] % 28 == 0
