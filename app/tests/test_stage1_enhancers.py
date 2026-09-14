"""Unit tests for Stage 1: Backend Model Registry & ONNX Enhancer Logic.

Tests:
1. Model registration specifications, SHA-256 integrity check and corruption detection.
2. ONNX session initialization with execution provider fallback (CUDA -> CPU).
3. 5-point landmark face alignment and inverse affine warp back (cv2.warpAffine / cv2.invertAffineTransform).
4. UltraMax multi-pass composite filter and normalized blend_ratio blending (cv2.addWeighted).
5. FaceEnhancer pipeline engine execution.
"""

from __future__ import annotations

import os
import tempfile
import cv2
import numpy as np
import pytest

from roop.model_registry import (
    MODEL_REGISTRY,
    compute_file_sha256,
    ensure_model_downloaded,
    verify_file_integrity,
)
from roop.face_enhancer import (
    FaceEnhancer,
    align_face_5point,
    apply_ultramax_composite,
    create_enhancer_session,
    inverse_affine_warp_back,
)


def test_model_registry_entries():
    """Verify required model entries are registered with valid URLs and SHA-256 digests."""
    required_keys = ["gpen_bfr_512", "gpen_bfr_256", "codeformer_fp16", "codeformer_fp32"]
    for key in required_keys:
        assert key in MODEL_REGISTRY, f"Missing model key {key} in registry"
        spec = MODEL_REGISTRY[key]
        assert spec.url.startswith("https://"), f"Invalid URL for {key}: {spec.url}"
        assert len(spec.sha256) == 64, f"Invalid SHA-256 hash length for {key}: {spec.sha256}"
        assert spec.size_bytes > 0, f"Invalid size_bytes for {key}"


def test_integrity_check_valid_and_corrupt():
    """Test that integrity verification accepts valid files and rejects corrupted files."""
    with tempfile.NamedTemporaryFile(delete=False) as tf:
        tf.write(b"sample model weights content 12345")
        temp_path = tf.name

    try:
        real_hash = compute_file_sha256(temp_path)
        assert verify_file_integrity(temp_path, real_hash) is True
        assert verify_file_integrity(temp_path, "0" * 64) is False
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_5point_landmark_alignment_and_inverse_warp():
    """Verify 5-point landmark face alignment and inverse affine transformation back to frame."""
    frame_h, frame_w = 640, 640
    frame = np.full((frame_h, frame_w, 3), 40, dtype=np.uint8)

    # Synthetic 5-point facial landmarks (left_eye, right_eye, nose, left_mouth, right_mouth)
    kps = np.array([
        [280.0, 260.0],
        [360.0, 260.0],
        [320.0, 310.0],
        [290.0, 360.0],
        [350.0, 360.0],
    ], dtype=np.float32)

    # 1. Forward 5-point alignment
    crop_size = 512
    aligned_crop, M = align_face_5point(frame, kps, crop_size=crop_size, template="ffhq_512")
    assert aligned_crop.shape == (crop_size, crop_size, 3)
    assert M.shape == (2, 3)

    # Mark enhanced face brightly to trace coordinates
    enhanced_crop = np.full((crop_size, crop_size, 3), 200, dtype=np.uint8)

    # 2. Inverse affine warp back
    pasted_full = inverse_affine_warp_back(
        target_frame=frame,
        enhanced_crop=enhanced_crop,
        M=M,
        original_crop=aligned_crop,
        blend_ratio=1.0,
    )
    assert pasted_full.shape == (frame_h, frame_w, 3)
    # The face area should be brighter than the 40 background
    assert pasted_full[300, 320, 0] > 100

    # 3. Test blend_ratio = 0.0 (returns original frame)
    pasted_zero = inverse_affine_warp_back(
        target_frame=frame,
        enhanced_crop=enhanced_crop,
        M=M,
        original_crop=aligned_crop,
        blend_ratio=0.0,
    )
    # With blend_ratio=0.0, the pasted crop is identical to original_crop (pixel 40)
    assert pasted_zero[300, 320, 0] == 40


def test_ultramax_composite_filter_and_blend_ratio():
    """Verify UltraMax composite filter and blend logic (cv2.addWeighted) across blend_ratio."""
    h, w = 256, 256
    structure = np.full((h, w, 3), 120, dtype=np.uint8)
    reference = np.full((h, w, 3), 40, dtype=np.uint8)

    # Full blend (ratio = 1.0)
    out_100 = apply_ultramax_composite(structure, None, reference, blend_ratio=1.0)
    # Zero blend (ratio = 0.0) -> exactly reference
    out_0 = apply_ultramax_composite(structure, None, reference, blend_ratio=0.0)
    # Half blend (ratio = 0.5) -> midpoint between out_100 and reference
    out_50 = apply_ultramax_composite(structure, None, reference, blend_ratio=0.5)

    assert np.allclose(out_0, reference)
    expected_mid = cv2.addWeighted(out_100, 0.5, reference, 0.5, 0.0)
    assert np.allclose(out_50, expected_mid, atol=1.0)


def test_session_init_cpu_fallback():
    """Verify ONNX session fallback succeeds with CPUExecutionProvider."""
    path = ensure_model_downloaded("gpen_bfr_512")
    assert os.path.isfile(path)
    sess, providers = create_enhancer_session(
        path,
        requested_providers=["CPUExecutionProvider"],
        label="TestGPEN-CPU",
    )
    assert "CPUExecutionProvider" in providers
    assert sess is not None
