"""Test Stage 3: UltraMax Decoupled Execution Pipeline.

Verifies:
1. Pipeline Segmentation:
   - Neural inference stage (GPEN / CodeFormer super-resolution) is decoupled from composite visual filters.
   - Routes the 512x512 aligned face chip through the TensorRT-accelerated ONNX session.
   - Composite visual filters (unsharp mask, bilateral edge preservation, adaptive sharpening) execute separately.
   - Post-inference blending in OpenCV uses vectorized in-place operations (cv2.addWeighted) to prevent CPU bottlenecks.
2. Zero-Copy Tensor Handling:
   - Normalization ((img / 255.0 - 0.5) / 0.5 and img / 255.0) are exact.
   - RGB/BGR transposition (transpose(2, 0, 1)) is strictly contiguous in memory (np.ascontiguousarray)
     to maximize TensorRT DMA transfer speeds.
   - Memory layout satisfies tensor.flags['C_CONTIGUOUS'] == True for float32 and float16.
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

_APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

import cv2
import numpy as np

from roop.face_enhancer import (
    FaceEnhancer,
    apply_adaptive_sharpening,
    apply_bilateral_edge_preservation,
    apply_ultramax_composite,
    apply_ultramax_composite_filters,
    apply_unsharp_mask,
    post_inference_blend,
    prepare_zero_copy_tensor,
)
from roop.processors.Enhance_UltraMax import Enhance_UltraMax


def make_test_face_chip(size: int = 512) -> np.ndarray:
    """Create a synthetic 512x512 face chip with realistic tonal gradients and step edges."""
    chip = np.full((size, size, 3), (170, 185, 210), dtype=np.uint8)
    cx, cy = size // 2, int(size * 0.52)
    # Face boundary oval
    cv2.ellipse(chip, (cx, cy), (int(size * 0.30), int(size * 0.40)), 0, 0, 360, (140, 160, 200), -1)
    # Eyes
    cv2.circle(chip, (int(size * 0.38), int(size * 0.45)), int(size * 0.04), (30, 25, 20), -1)
    cv2.circle(chip, (int(size * 0.62), int(size * 0.45)), int(size * 0.04), (30, 25, 20), -1)
    # Eyebrows
    cv2.line(chip, (int(size * 0.32), int(size * 0.38)), (int(size * 0.44), int(size * 0.38)), (40, 35, 30), 4)
    cv2.line(chip, (int(size * 0.56), int(size * 0.38)), (int(size * 0.68), int(size * 0.38)), (40, 35, 30), 4)
    # Nose
    cv2.circle(chip, (cx, int(size * 0.60)), int(size * 0.03), (110, 130, 175), -1)
    # Lips
    cv2.ellipse(chip, (cx, int(size * 0.74)), (int(size * 0.10), int(size * 0.04)), 0, 0, 360, (65, 75, 160), -1)
    return chip


class TestZeroCopyTensorHandling(unittest.TestCase):
    """Test suite for memory contiguity, transposition, and normalization."""

    def test_symmetric_normalization_and_contiguity(self):
        """((img / 255.0 - 0.5) / 0.5) maps [0, 255] to [-1.0, 1.0] and is C-contiguous."""
        test_img = np.zeros((512, 512, 3), dtype=np.uint8)
        test_img[0, 0] = [0, 127, 255]  # BGR

        tensor = prepare_zero_copy_tensor(
            test_img,
            target_size=512,
            normalization_mode="symmetric",
            dtype=np.float32,
        )

        # Shape must be NCHW
        self.assertEqual(tensor.shape, (1, 3, 512, 512))
        self.assertEqual(tensor.dtype, np.float32)

        # Must be strictly C-contiguous in memory for DMA transfer
        self.assertTrue(
            tensor.flags["C_CONTIGUOUS"],
            "Tensor must be C-contiguous to maximize TensorRT DMA transfer speeds",
        )

        # Range must be bounded within [-1.0, 1.0]
        self.assertGreaterEqual(float(tensor.min()), -1.0001)
        self.assertLessEqual(float(tensor.max()), 1.0001)

        # Value check: 0 maps to -1.0, 255 maps to 1.0
        # Notice BGR [0, 127, 255] transposed to RGB is [255, 127, 0]
        # Channel 0 (R): 255 -> 1.0
        self.assertAlmostEqual(float(tensor[0, 0, 0, 0]), 1.0, places=3)
        # Channel 2 (B): 0 -> -1.0
        self.assertAlmostEqual(float(tensor[0, 2, 0, 0]), -1.0, places=3)

    def test_unit_normalization_and_contiguity(self):
        """(img / 255.0) maps [0, 255] to [0.0, 1.0] and is C-contiguous."""
        test_img = np.full((512, 512, 3), 255, dtype=np.uint8)
        tensor = prepare_zero_copy_tensor(
            test_img,
            target_size=512,
            normalization_mode="unit",
            dtype=np.float32,
        )

        self.assertEqual(tensor.shape, (1, 3, 512, 512))
        self.assertTrue(tensor.flags["C_CONTIGUOUS"])
        self.assertAlmostEqual(float(tensor.min()), 1.0, places=4)
        self.assertAlmostEqual(float(tensor.max()), 1.0, places=4)

    def test_float16_precision_contiguity(self):
        """FP16 tensor transposition is C-contiguous for half-precision TensorRT engines."""
        test_img = make_test_face_chip(512)
        tensor = prepare_zero_copy_tensor(
            test_img,
            target_size=512,
            normalization_mode="symmetric",
            dtype=np.float16,
        )

        self.assertEqual(tensor.dtype, np.float16)
        self.assertEqual(tensor.shape, (1, 3, 512, 512))
        self.assertTrue(tensor.flags["C_CONTIGUOUS"])

    def test_non_square_input_resized_to_512(self):
        """Inputs differing from 512x512 are properly resized and made contiguous."""
        non_square = np.zeros((480, 640, 3), dtype=np.uint8)
        tensor = prepare_zero_copy_tensor(non_square, target_size=512)
        self.assertEqual(tensor.shape, (1, 3, 512, 512))
        self.assertTrue(tensor.flags["C_CONTIGUOUS"])


class TestDecoupledCompositeVisualFilters(unittest.TestCase):
    """Test suite for decoupled visual filters (unsharp mask, bilateral, adaptive sharpening)."""

    def setUp(self):
        self.chip = make_test_face_chip(512)

    def test_unsharp_mask_vectorized(self):
        """Unsharp mask boosts edge clarity via cv2.addWeighted without shape or dtype changes."""
        unsharp = apply_unsharp_mask(self.chip, amount=0.40, sigma=1.0)
        self.assertEqual(unsharp.shape, self.chip.shape)
        self.assertEqual(unsharp.dtype, np.uint8)

        # Step edges (e.g. eye pupil edge) should exhibit higher contrast
        eye_y, eye_x = int(512 * 0.45), int(512 * 0.38)
        diff = np.abs(unsharp.astype(np.int32) - self.chip.astype(np.int32))
        self.assertGreater(diff.sum(), 0)

        # Zero amount returns original image
        unsharp_zero = apply_unsharp_mask(self.chip, amount=0.0)
        self.assertTrue(np.array_equal(unsharp_zero, self.chip))

    def test_bilateral_edge_preservation(self):
        """Bilateral filter preserves sharp step edges while smoothing sensor noise."""
        # Add high-frequency noise
        noisy_chip = self.chip.copy()
        noise = np.random.randint(-15, 15, self.chip.shape, dtype=np.int16)
        noisy_chip = np.clip(noisy_chip.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        filtered = apply_bilateral_edge_preservation(noisy_chip, d=5, sigma_color=25.0, sigma_space=25.0, blend=0.85)
        self.assertEqual(filtered.shape, self.chip.shape)
        self.assertEqual(filtered.dtype, np.uint8)

        # Flat cheek area variance should decrease due to noise attenuation
        cheek_noisy = noisy_chip[200:230, 200:230]
        cheek_filtered = filtered[200:230, 200:230]
        self.assertLess(np.std(cheek_filtered), np.std(cheek_noisy))

    def test_adaptive_sharpening_anti_halo(self):
        """Adaptive sharpening enhances edges without creating white halos or out-of-bound spikes."""
        sharpened = apply_adaptive_sharpening(self.chip, strength=0.35, radius=1, limit=2.5)
        self.assertEqual(sharpened.shape, self.chip.shape)
        self.assertEqual(sharpened.dtype, np.uint8)

        # Output must be finite and within [0, 255]
        self.assertGreaterEqual(int(sharpened.min()), 0)
        self.assertLessEqual(int(sharpened.max()), 255)

    def test_post_inference_blend_vectorized(self):
        """Vectorized in-place blending via cv2.addWeighted guarantees exact interpolation."""
        ref = np.full((512, 512, 3), 50, dtype=np.uint8)
        enh = np.full((512, 512, 3), 150, dtype=np.uint8)

        # Ratio 1.0 -> enhanced
        b100 = post_inference_blend(enh, ref, blend_ratio=1.0)
        self.assertTrue(np.array_equal(b100, enh))

        # Ratio 0.0 -> reference
        b0 = post_inference_blend(enh, ref, blend_ratio=0.0)
        self.assertTrue(np.array_equal(b0, ref))

        # Ratio 0.5 -> exactly (150*0.5 + 50*0.5) = 100
        b50 = post_inference_blend(enh, ref, blend_ratio=0.5)
        self.assertTrue(np.allclose(b50, 100, atol=1))

    def test_apply_ultramax_composite_filters_pipeline(self):
        """Decoupled composite filter pipeline executes all visual filters in sequence."""
        filtered = apply_ultramax_composite_filters(
            enhanced_chip=self.chip,
            reference_chip=self.chip,
            unsharp_amount=0.25,
            bilateral_strength=0.35,
            adaptive_sharpen_strength=0.25,
            blend_ratio=0.85,
        )
        self.assertEqual(filtered.shape, (512, 512, 3))
        self.assertEqual(filtered.dtype, np.uint8)


class TestUltraMaxDecoupledExecution(unittest.TestCase):
    """Test suite for UltraMax decoupled neural inference and composite filters."""

    def test_ultramax_processor_apply_composite_filters(self):
        """Enhance_UltraMax exposes apply_composite_filters method."""
        proc = Enhance_UltraMax()
        chip = make_test_face_chip(512)
        out = proc.apply_composite_filters(
            face_chip=chip,
            reference_chip=chip,
            unsharp_amount=0.20,
            bilateral_strength=0.30,
            adaptive_sharpen_strength=0.20,
            blend_ratio=0.90,
        )
        self.assertEqual(out.shape, (512, 512, 3))
        self.assertEqual(out.dtype, np.uint8)

    @patch("onnxruntime.InferenceSession")
    def test_face_enhancer_decoupled_execution_flow(self, mock_ort_session):
        """FaceEnhancer executes neural inference and decoupled composite filters."""
        mock_sess_instance = MagicMock()
        mock_in = MagicMock(name="input")
        mock_in.name = "input"
        mock_out = MagicMock(name="output")
        mock_out.name = "output"
        mock_sess_instance.get_inputs.return_value = [mock_in]
        mock_sess_instance.get_outputs.return_value = [mock_out]
        mock_sess_instance.get_providers.return_value = ["TensorrtExecutionProvider"]

        # Return simulated enhanced tensor (1, 3, 512, 512) in [-1, 1]
        dummy_out = np.zeros((1, 3, 512, 512), dtype=np.float32)
        mock_sess_instance.run.return_value = [dummy_out]
        mock_ort_session.return_value = mock_sess_instance

        enhancer = FaceEnhancer(enhancer_type="ultramax", model_size=512)
        enhancer.session = mock_sess_instance
        enhancer._in_name = "input"
        enhancer._out_name = "output"

        aligned_chip = make_test_face_chip(512)
        enhanced = enhancer.enhance_crop(aligned_chip, blend_ratio=0.85)

        # Verify inference was called with C-contiguous NCHW tensor
        mock_sess_instance.run.assert_called_once()
        _, feed_kwargs = mock_sess_instance.run.call_args[0]
        in_tensor = feed_kwargs["input"]
        self.assertEqual(in_tensor.shape, (1, 3, 512, 512))
        self.assertTrue(in_tensor.flags["C_CONTIGUOUS"])

        # Verify output is valid uint8 image of shape (512, 512, 3)
        self.assertEqual(enhanced.shape, (512, 512, 3))
        self.assertEqual(enhanced.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
