"""Compare our preprocessor output against HuggingFace's Qwen2VLProcessor reference.

Qwen3.5 uses the same preprocessing as Qwen2-VL/Qwen3-VL (same vision encoder).
This test validates that our implementation matches the reference behavior.

The reference data is pre-generated via scripts/hf_reference.py and stored in
tests/data/hf_reference_*.npz files.

If transformers is installed, also run live comparisons.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from llama_video.config import ModelConfig
from llama_video.preprocessor import Preprocessor
from llama_video.types import Frame

# Path to test data directory
TEST_DATA_DIR = Path(__file__).parent.parent / "data"


def create_synthetic_frames(
    num_frames: int,
    width: int,
    height: int,
    seed: int = 42,
) -> list[Frame]:
    """Create synthetic video frames matching HF reference script.

    Args:
        num_frames: Number of frames to create.
        width: Frame width.
        height: Frame height.
        seed: Random seed for reproducibility.

    Returns:
        List of Frame objects.
    """
    rng = np.random.default_rng(seed)
    frames: list[Frame] = []
    for i in range(num_frames):
        data = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        frames.append(
            Frame(
                data=data,
                index=i,
                timestamp=i / 2.0,  # Default 2 fps
                width=width,
                height=height,
            )
        )
    return frames


def load_reference_data(name: str) -> dict[str, Any]:
    """Load pre-generated HF reference data.

    Args:
        name: Reference file name without extension (e.g., "448x448_8frames_2fps").

    Returns:
        Dict with reference data arrays.
    """
    ref_path = TEST_DATA_DIR / f"hf_reference_{name}.npz"
    if not ref_path.exists():
        pytest.skip(f"Reference data not found: {ref_path}")
    data = np.load(ref_path)
    result = {}
    for key in data.files:
        result[key] = data[key]
    return result


def get_reference_test_cases() -> list[str]:
    """Get list of available reference test case names."""
    if not TEST_DATA_DIR.exists():
        return []
    return [p.stem.replace("hf_reference_", "") for p in TEST_DATA_DIR.glob("hf_reference_*.npz")]


@pytest.fixture
def model_config() -> ModelConfig:
    """Default Qwen3.5 model config."""
    return ModelConfig.qwen35()


@pytest.fixture
def preprocessor(model_config: ModelConfig) -> Preprocessor:
    """Preprocessor with default config."""
    return Preprocessor(model_config)


@pytest.mark.integration
class TestHFReferenceComparison:
    """Compare our preprocessor against HuggingFace reference implementation."""

    @pytest.mark.parametrize("test_case", get_reference_test_cases())
    def test_target_resolution_matches_hf(
        self,
        preprocessor: Preprocessor,
        test_case: str,
    ) -> None:
        """Our _compute_target_resolution should match HF's smart_resize."""
        ref = load_reference_data(test_case)

        input_w = int(ref["input_width"])
        input_h = int(ref["input_height"])
        expected_w = int(ref["target_width"])
        expected_h = int(ref["target_height"])

        actual_w, actual_h = preprocessor._compute_target_resolution(input_w, input_h)

        assert actual_w == expected_w, (
            f"Width mismatch for {input_w}x{input_h}: got {actual_w}, expected {expected_w}"
        )
        assert actual_h == expected_h, (
            f"Height mismatch for {input_w}x{input_h}: got {actual_h}, expected {expected_h}"
        )

    @pytest.mark.parametrize("test_case", get_reference_test_cases())
    def test_grid_thw_matches_hf(
        self,
        preprocessor: Preprocessor,
        test_case: str,
    ) -> None:
        """Our grid_thw should exactly match HF's computation."""
        ref = load_reference_data(test_case)

        num_frames = int(ref["num_frames"])
        fps = float(ref["fps"])
        expected_grid = tuple(ref["grid_thw"])

        # Create frames and process
        frames = create_synthetic_frames(
            num_frames,
            int(ref["input_width"]),
            int(ref["input_height"]),
        )
        video_input = preprocessor.process(frames, fps=fps)

        assert video_input.grid_thw == expected_grid, (
            f"grid_thw mismatch: got {video_input.grid_thw}, expected {expected_grid}"
        )

    @pytest.mark.parametrize("test_case", get_reference_test_cases())
    def test_super_frame_count_matches_hf(
        self,
        preprocessor: Preprocessor,
        test_case: str,
    ) -> None:
        """Number of super-frames should match HF."""
        ref = load_reference_data(test_case)

        num_frames = int(ref["num_frames"])
        fps = float(ref["fps"])
        expected_shape = tuple(ref["pixel_values_shape"])
        expected_t = expected_shape[0]  # First dim is number of super-frames

        frames = create_synthetic_frames(
            num_frames,
            int(ref["input_width"]),
            int(ref["input_height"]),
        )
        video_input = preprocessor.process(frames, fps=fps)

        assert len(video_input.super_frames) == expected_t, (
            f"Super-frame count mismatch: got {len(video_input.super_frames)}, "
            f"expected {expected_t}"
        )

    @pytest.mark.parametrize("test_case", get_reference_test_cases())
    def test_temporal_positions_match_hf(
        self,
        preprocessor: Preprocessor,
        test_case: str,
    ) -> None:
        """Our temporal positions should match HF's computation."""
        ref = load_reference_data(test_case)

        num_frames = int(ref["num_frames"])
        fps = float(ref["fps"])
        expected_positions = list(ref["temporal_positions"])

        # Create frames and process
        frames = create_synthetic_frames(
            num_frames,
            int(ref["input_width"]),
            int(ref["input_height"]),
        )
        video_input = preprocessor.process(frames, fps=fps)

        assert video_input.temporal_positions == expected_positions, (
            f"Temporal positions mismatch: got {video_input.temporal_positions}, "
            f"expected {expected_positions}"
        )

    @pytest.mark.parametrize("test_case", get_reference_test_cases())
    def test_super_frame_shape_matches_hf(
        self,
        preprocessor: Preprocessor,
        test_case: str,
    ) -> None:
        """Super-frame tensor shape should match HF."""
        ref = load_reference_data(test_case)

        num_frames = int(ref["num_frames"])
        fps = float(ref["fps"])
        expected_shape = tuple(ref["pixel_values_shape"])  # (T, 6, H, W)

        frames = create_synthetic_frames(
            num_frames,
            int(ref["input_width"]),
            int(ref["input_height"]),
        )
        video_input = preprocessor.process(frames, fps=fps)

        # Verify each super-frame has 6 channels and correct spatial dims
        _, c, h, w = expected_shape
        for i, sf in enumerate(video_input.super_frames):
            assert sf.shape[0] == c, f"Super-frame {i} channels: got {sf.shape[0]}, expected {c}"
            assert sf.shape[1] == h, f"Super-frame {i} height: got {sf.shape[1]}, expected {h}"
            assert sf.shape[2] == w, f"Super-frame {i} width: got {sf.shape[2]}, expected {w}"

    @pytest.mark.parametrize("test_case", get_reference_test_cases())
    def test_pixel_values_match_hf(
        self,
        preprocessor: Preprocessor,
        test_case: str,
    ) -> None:
        """Super-frame pixel values should match HF (within tolerance).

        This is the critical test: our normalized super-frame data should
        be numerically identical to HF's output.
        """
        ref = load_reference_data(test_case)

        num_frames = int(ref["num_frames"])
        fps = float(ref["fps"])
        expected_sample = ref["pixel_values_sample"]  # (6, 14, 14) center crop

        frames = create_synthetic_frames(
            num_frames,
            int(ref["input_width"]),
            int(ref["input_height"]),
        )
        video_input = preprocessor.process(frames, fps=fps)

        # Extract same center crop from our first super-frame
        target_h = int(ref["target_height"])
        target_w = int(ref["target_width"])
        center_h = target_h // 2
        center_w = target_w // 2
        sample_size = 14

        our_sample = video_input.super_frames[0].data[
            :,
            center_h : center_h + sample_size,
            center_w : center_w + sample_size,
        ]

        # Compare with tolerance for floating-point differences
        np.testing.assert_allclose(
            our_sample,
            expected_sample,
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"Pixel values differ from HF reference for {test_case}",
        )

    def test_normalization_values_match_hf(
        self,
        model_config: ModelConfig,
    ) -> None:
        """Our CLIP normalization values should match HF defaults."""
        # Load HF processor config
        config_path = TEST_DATA_DIR / "hf_processor_config.npz"
        if not config_path.exists():
            pytest.skip(f"HF config not found: {config_path}")

        hf_config = np.load(config_path)

        # Check mean values
        np.testing.assert_allclose(
            model_config.image_mean,
            hf_config["image_mean"],
            rtol=1e-6,
            err_msg="image_mean differs from HF",
        )

        # Check std values
        np.testing.assert_allclose(
            model_config.image_std,
            hf_config["image_std"],
            rtol=1e-6,
            err_msg="image_std differs from HF",
        )


@pytest.mark.integration
class TestHFReferenceLive:
    """Live comparison tests using transformers (if available).

    These tests run the actual HF processor and compare with our output.
    Skipped if transformers is not installed.
    """

    @pytest.fixture
    def hf_processor(self):
        """Get HuggingFace Qwen2VL image processor if available."""
        try:
            from transformers import Qwen2VLImageProcessor
        except ImportError:
            pytest.skip("transformers not installed")

        model_config = ModelConfig.qwen35()
        return Qwen2VLImageProcessor(
            min_pixels=model_config.min_pixels,
            max_pixels=model_config.max_pixels,
            patch_size=model_config.spatial_patch_size,
            merge_size=model_config.merge_size,
            temporal_patch_size=model_config.temporal_patch_size,
        )

    def test_live_resolution_matches(
        self,
        preprocessor: Preprocessor,
        hf_processor,
    ) -> None:
        """Live test: our resolution matches HF's smart_resize."""
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

        test_dims = [
            (448, 448),
            (672, 448),
            (1920, 1080),
            (56, 56),
            (100, 200),
        ]

        for width, height in test_dims:
            # Our resolution
            our_w, our_h = preprocessor._compute_target_resolution(width, height)

            # HF resolution
            hf_h, hf_w = smart_resize(
                height=height,
                width=width,
                factor=hf_processor.patch_size * hf_processor.merge_size,
                min_pixels=hf_processor.min_pixels,
                max_pixels=hf_processor.max_pixels,
            )

            assert our_w == hf_w, f"Width mismatch for {width}x{height}"
            assert our_h == hf_h, f"Height mismatch for {width}x{height}"

    def test_live_grid_thw_matches(
        self,
        preprocessor: Preprocessor,
        hf_processor,
    ) -> None:
        """Live test: our grid_thw matches HF's computation."""
        test_cases = [
            (8, 448, 448, 2.0),
            (4, 672, 448, 2.0),
            (4, 1920, 1080, 2.0),
        ]

        for num_frames, width, height, fps in test_cases:
            frames = create_synthetic_frames(num_frames, width, height)
            video_input = preprocessor.process(frames, fps=fps)

            # Compute expected grid from HF logic
            target_w, target_h = video_input.resolution
            grid_unit = hf_processor.patch_size * hf_processor.merge_size
            expected_t = len(video_input.super_frames)
            expected_h = target_h // grid_unit
            expected_w = target_w // grid_unit

            assert video_input.grid_thw == (expected_t, expected_h, expected_w), (
                f"grid_thw mismatch for {width}x{height}"
            )

    def test_live_grid_thw_convention_documented(
        self,
        preprocessor: Preprocessor,
        hf_processor,
    ) -> None:
        """Verify our grid_thw uses post-merge dims (grid_unit=28) vs HF's pre-merge (patch_size=14).

        Our grid_thw = (T, H/28, W/28) — post-merge spatial dimensions.
        HF's grid_thw = (T, H/14, W/14) — pre-merge spatial dimensions.
        HF values are exactly 2× ours in H and W (merge_size=2).

        This is intentional: our clip.cpp patch expects post-merge dims.
        """
        from PIL import Image

        test_dims = [(56, 56), (448, 448), (672, 448)]
        merge = hf_processor.merge_size

        for width, height in test_dims:
            # HF grid for single image
            frame = Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))
            hf_result = hf_processor.preprocess(images=[frame], return_tensors="np")
            hf_grid = hf_result["image_grid_thw"][0]
            hf_h, hf_w = int(hf_grid[1]), int(hf_grid[2])

            # Our grid
            our_w, our_h = preprocessor._compute_target_resolution(width, height)
            our_grid_h = our_h // preprocessor._config.grid_unit
            our_grid_w = our_w // preprocessor._config.grid_unit

            # HF pre-merge = 2× our post-merge
            assert hf_h == our_grid_h * merge, (
                f"Expected HF H ({hf_h}) = our H ({our_grid_h}) × merge ({merge})"
            )
            assert hf_w == our_grid_w * merge, (
                f"Expected HF W ({hf_w}) = our W ({our_grid_w}) × merge ({merge})"
            )

    def test_live_per_frame_pixel_values_match_hf(
        self,
        preprocessor: Preprocessor,
        hf_processor,
    ) -> None:
        """Definitive test: our per-frame pixel values exactly match HF's.

        This is the ground truth comparison. We:
        1. Process each frame through HF's actual Qwen2VLImageProcessor
        2. Reconstruct per-frame (3, H, W) data from HF's patchified output
        3. Extract per-frame (3, H, W) data from our super-frames
        4. Assert numerical equality

        Channel ordering differs (HF: C,T,H,W → ours: T,C,H,W) but
        per-frame RGB values must be identical.

        HF's pixel_values use merge-block patch ordering (not simple row-major):
        patches are grouped in merge_size×merge_size blocks, each block containing
        the patches that will be pixel-shuffled during the merge operation.
        """
        from PIL import Image

        test_cases = [
            (4, 56, 56),
            (4, 448, 448),
            (4, 672, 448),
            (8, 56, 56),
        ]

        for num_frames, width, height in test_cases:
            frames = create_synthetic_frames(num_frames, width, height)

            # === Our pipeline ===
            video_input = preprocessor.process(frames, fps=2.0)

            # Extract per-frame data from super-frames: each super-frame is (6, H, W)
            # channels [0:3] = frame_a, channels [3:6] = frame_b
            our_frames_3ch: list[np.ndarray] = []
            for sf in video_input.super_frames:
                our_frames_3ch.append(sf.data[:3])  # frame_a (3, H, W)
                our_frames_3ch.append(sf.data[3:])  # frame_b (3, H, W)

            # === HF pipeline ===
            # Process each frame individually through HF's actual processor
            pil_frames = [Image.fromarray(f.data) for f in frames]
            hf_result = hf_processor.preprocess(
                images=pil_frames,
                return_tensors="np",
            )

            hf_pv = hf_result["pixel_values"]  # (total_patches, patch_dim)
            hf_grid = hf_result["image_grid_thw"]  # (num_images, 3)

            # Reconstruct per-frame (3, H, W) from HF's patchified output.
            # HF patches use merge-block ordering: patches within each
            # merge_size×merge_size block are contiguous, blocks are row-major.
            # Patch dim layout: (C, T, H_patch, W_patch) = (3, 2, 14, 14)
            patch_size = hf_processor.patch_size
            merge = hf_processor.merge_size
            tp = hf_processor.temporal_patch_size
            hf_frames_3ch: list[np.ndarray] = []

            patch_offset = 0
            for img_idx in range(len(hf_grid)):
                grid_t = int(hf_grid[img_idx][0])
                grid_h = int(hf_grid[img_idx][1])
                grid_w = int(hf_grid[img_idx][2])
                num_patches = grid_t * grid_h * grid_w
                img_patches = hf_pv[patch_offset : patch_offset + num_patches]
                patch_offset += num_patches

                # Merged grid dimensions
                m_h = grid_h // merge
                m_w = grid_w // merge

                # Reshape accounting for merge-block ordering:
                # (T, m_h, m_w, merge_h, merge_w, C, Tp, Ph, Pw)
                patches = img_patches.reshape(
                    grid_t, m_h, m_w, merge, merge, 3, tp, patch_size, patch_size
                )
                # Rearrange to (T*Tp, C, full_H, full_W)
                full = patches.transpose(0, 6, 5, 1, 3, 7, 2, 4, 8).reshape(
                    grid_t * tp,
                    3,
                    m_h * merge * patch_size,
                    m_w * merge * patch_size,
                )
                # For single image: T=1, full is (2, 3, H, W) with duplicated
                # temporal dim. Take frame 0 as the processed frame.
                hf_frames_3ch.append(full[0])

            # === Compare per-frame values ===
            assert len(our_frames_3ch) == len(hf_frames_3ch), (
                f"Frame count mismatch for {width}x{height}: "
                f"ours={len(our_frames_3ch)}, HF={len(hf_frames_3ch)}"
            )

            for i in range(len(our_frames_3ch)):
                np.testing.assert_allclose(
                    our_frames_3ch[i],
                    hf_frames_3ch[i],
                    rtol=1e-5,
                    atol=1e-5,
                    err_msg=(
                        f"Frame {i} pixel values differ from HF "
                        f"for {width}x{height} ({num_frames} frames)"
                    ),
                )
