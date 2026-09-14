"""Core Face Enhancer module (face_enhancer.py).

Provides:
1. ONNX session initialization with automatic execution provider fallback
   (TensorRT -> CUDAExecutionProvider -> CPUExecutionProvider).
2. 5-point landmark face alignment, forward inference, and inverse affine transformation
   (cv2.warpAffine / cv2.invertAffineTransform) back to the target frame.
3. UltraMax multi-pass composite filter and blending logic (cv2.addWeighted)
   driven by a normalized blend_ratio parameter (0.0 to 1.0).
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Optional, Sequence, Tuple

import cv2
import numpy as np
import onnxruntime
from skimage import transform as trans

import roop.globals
from roop.model_registry import MODEL_REGISTRY, ensure_model_downloaded
from roop.processors.enhance_common import (
    looks_collapsed,
    luma_only_recolour,
    sized,
)
from roop.processors.frequency_split import frequency_split_luma
from roop.provider_fallback import build_cuda_fallback_providers

_LOGGER = logging.getLogger(__name__)

# Standard 5-point alignment templates (normalized to [0, 1])
WARP_TEMPLATES = {
    "ffhq_512": np.array(
        [
            [0.37691676, 0.46864664],
            [0.62285697, 0.46912813],
            [0.50123859, 0.61331904],
            [0.39308822, 0.72541100],
            [0.61150205, 0.72490465],
        ],
        dtype=np.float32,
    ),
    "arcface_112_v1": np.array(
        [
            [0.35473214, 0.45658929],
            [0.64526786, 0.45658929],
            [0.50000000, 0.61154464],
            [0.37913393, 0.77687500],
            [0.62086607, 0.77687500],
        ],
        dtype=np.float32,
    ),
}


# ── ONNX Session Initialization with Multi-Tier EP Fallback ──────────────────

def create_enhancer_session(
    model_path: str,
    requested_providers: Optional[Sequence[Any]] = None,
    session_options: Optional[onnxruntime.SessionOptions] = None,
    label: str = "FaceEnhancer",
) -> Tuple[onnxruntime.InferenceSession, list[str]]:
    """Initialize an ONNX Runtime InferenceSession with graceful fallback.

    Fallback chain:
        1. Requested Providers (e.g. TensorRT + CUDA + CPU)
        2. CUDAExecutionProvider + CPUExecutionProvider
        3. CPUExecutionProvider
    """
    opts = session_options or onnxruntime.SessionOptions()
    opts.log_severity_level = 2

    providers_to_try = list(requested_providers or roop.globals.execution_providers or [])
    available_fn = getattr(onnxruntime, "get_available_providers", None)
    available = set(list(available_fn()) if callable(available_fn) else ["CPUExecutionProvider"])

    # Tier 1: User-requested providers filtered by what is installed
    tier1 = [
        p for p in providers_to_try
        if (p[0] if isinstance(p, (tuple, list)) else str(p)) in available
    ]
    if not tier1:
        tier1 = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in available else ["CPUExecutionProvider"]

    try:
        sess = onnxruntime.InferenceSession(model_path, opts, providers=tier1)
        active = list(sess.get_providers())
        _LOGGER.info("[%s] Initialized with providers: %s", label, active)
        return sess, active
    except Exception as exc_tier1:
        _LOGGER.warning(
            "[%s] Primary provider configuration failed: %s. Attempting CUDA fallback...",
            label,
            exc_tier1,
        )

    # Tier 2: Pure CUDA execution provider fallback
    if "CUDAExecutionProvider" in available:
        tier2 = build_cuda_fallback_providers(tier1)
        try:
            sess = onnxruntime.InferenceSession(model_path, opts, providers=tier2)
            active = list(sess.get_providers())
            _LOGGER.info("[%s] Fallback to CUDA succeeded: %s", label, active)
            return sess, active
        except Exception as exc_tier2:
            _LOGGER.warning(
                "[%s] CUDA fallback failed: %s. Attempting CPU fallback...",
                label,
                exc_tier2,
            )

    # Tier 3: Guaranteed CPU fallback
    tier3 = ["CPUExecutionProvider"]
    try:
        sess = onnxruntime.InferenceSession(model_path, opts, providers=tier3)
        active = list(sess.get_providers())
        _LOGGER.warning("[%s] Running on CPUExecutionProvider as fallback.", label)
        return sess, active
    except Exception as exc_tier3:
        raise RuntimeError(
            f"[{label}] Fatal error: all execution provider tiers failed to build session for {model_path}! "
            f"Errors: Tier1={exc_tier1}, Tier3={exc_tier3}"
        ) from exc_tier3


# ── 5-Point Landmark Face Alignment & Inverse Affine Transformation ─────────

def align_face_5point(
    img: np.ndarray,
    kps: np.ndarray,
    crop_size: int = 512,
    template: str = "ffhq_512",
) -> Tuple[np.ndarray, np.ndarray]:
    """Align and crop face using 5-point facial landmarks.

    Args:
        img: Input BGR image (H, W, 3).
        kps: 5 facial keypoints array with shape (5, 2).
        crop_size: Output square resolution in pixels (default 512).
        template: Alignment template name ('ffhq_512' or 'arcface_112_v1').

    Returns:
        (aligned_crop, M):
            aligned_crop: BGR image of shape (crop_size, crop_size, 3).
            M: 2x3 affine transformation matrix mapping from frame -> crop.
    """
    assert kps is not None and kps.shape == (5, 2), f"Expected kps shape (5, 2), got {getattr(kps, 'shape', None)}"

    ref_template = WARP_TEMPLATES.get(template, WARP_TEMPLATES["ffhq_512"])
    dst_points = ref_template * float(crop_size)

    # Use SimilarityTransform (scale, rotation, translation) without non-uniform shearing
    tform = trans.SimilarityTransform()
    tform.estimate(kps, dst_points)
    M = tform.params[0:2, :].astype(np.float32)

    # Warp affine with border replication to avoid dark perimeter artifacts
    warped = cv2.warpAffine(
        img,
        M,
        (crop_size, crop_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return warped, M


def inverse_affine_warp_back(
    target_frame: np.ndarray,
    enhanced_crop: np.ndarray,
    M: np.ndarray,
    original_crop: Optional[np.ndarray] = None,
    blend_ratio: float = 1.0,
    scale_factor: int = 1,
) -> np.ndarray:
    """Invert affine transformation to warp enhanced face back onto the target frame.

    Supports blending between the enhanced face and the original unenhanced face
    driven by normalized blend_ratio (0.0 to 1.0).

    Args:
        target_frame: Full-size target frame image (H, W, 3).
        enhanced_crop: Enhanced square crop (e.g. 512x512).
        M: 2x3 forward affine matrix used during alignment.
        original_crop: Optional unenhanced crop for blend_ratio blending.
        blend_ratio: Weight of enhanced face (0.0 = original, 1.0 = fully enhanced).
        scale_factor: Integer scaling factor if enhanced_crop is larger than aligned crop.

    Returns:
        Frame with face seamlessly transformed and composited back.
    """
    ratio = float(np.clip(blend_ratio, 0.0, 1.0))
    th, tw = target_frame.shape[:2]

    # Calculate inverse 2x3 affine matrix
    M_inv = cv2.invertAffineTransform(M)
    if scale_factor > 1:
        M_inv[:, 2] *= float(scale_factor)

    # Create soft elliptical blending mask in crop space to eliminate boundary seams
    ch, cw = enhanced_crop.shape[:2]
    crop_mask = np.zeros((ch, cw), dtype=np.float32)
    center = (cw // 2, ch // 2)
    axes = (int(cw * 0.46), int(ch * 0.46))
    cv2.ellipse(crop_mask, center, axes, 0, 0, 360, 1.0, -1)
    crop_mask = cv2.GaussianBlur(crop_mask, (0, 0), sigmaX=float(cw) * 0.05)

    # Blend enhanced crop with original crop if blend_ratio < 1.0
    if ratio < 1.0 and original_crop is not None:
        if original_crop.shape[:2] != (ch, cw):
            orig_resized = cv2.resize(original_crop, (cw, ch), interpolation=cv2.INTER_CUBIC)
        else:
            orig_resized = original_crop
        crop_to_paste = cv2.addWeighted(enhanced_crop, ratio, orig_resized, 1.0 - ratio, 0.0)
    else:
        crop_to_paste = enhanced_crop

    # Warp crop and mask back to full target frame coordinates
    warped_face = cv2.warpAffine(
        crop_to_paste,
        M_inv,
        (tw, th),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    warped_mask = cv2.warpAffine(
        crop_mask,
        M_inv,
        (tw, th),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )[:, :, np.newaxis]

    # Seamless composite
    out_frame = (warped_face.astype(np.float32) * warped_mask +
                 target_frame.astype(np.float32) * (1.0 - warped_mask))
    return np.clip(out_frame, 0, 255).astype(np.uint8)


# ── UltraMax Multi-Pass Composite Filter & Blend Logic ───────────────────────

def apply_ultramax_composite(
    structure_crop: np.ndarray,
    detail_crop: Optional[np.ndarray],
    reference_crop: np.ndarray,
    blend_ratio: float = 1.0,
    chroma_weight: float = 0.0,
    dual_stream: bool = False,
    gain: float = 1.25,
) -> np.ndarray:
    """Multi-pass composite filter and blend for UltraMax.

    Pass 1: Preserves reference chrominance while retaining CodeFormer structural luminance
            (eliminates CodeFormer's characteristic pale skin drift).
    Pass 2: (Optional Dual-Stream) Injects edge-guided high-frequency detail from GPEN-512
            via guided frequency split.
    Pass 3: Proportional blending (cv2.addWeighted) with reference crop driven by blend_ratio.
    """
    ratio = float(np.clip(blend_ratio, 0.0, 1.0))
    chroma_w = float(np.clip(chroma_weight, 0.0, 1.0))

    # Pass 1: Luma-only color fix
    if chroma_w < 1.0:
        pass1 = luma_only_recolour(structure_crop, reference_crop)
        if chroma_w > 0.0:
            pass1 = cv2.addWeighted(pass1, 1.0 - chroma_w, structure_crop, chroma_w, 0.0)
    else:
        pass1 = structure_crop

    # Pass 2: Dual-stream frequency-split detail injection
    if dual_stream and detail_crop is not None:
        try:
            pass2 = frequency_split_luma(
                pass1,
                detail_crop,
                radius=8,
                eps=0.04,
                gain=gain,
                clamp=24.0,
            )
        except Exception as e:
            _LOGGER.warning("[UltraMax] Frequency split detail pass failed (%s); retaining Pass 1.", e)
            pass2 = pass1
    else:
        pass2 = pass1

    # Pass 3: Blend with original reference crop
    if ratio < 1.0:
        final_output = cv2.addWeighted(pass2, ratio, reference_crop, 1.0 - ratio, 0.0)
    else:
        final_output = pass2

    return final_output


# ── Standalone FaceEnhancer Pipeline Engine ──────────────────────────────────

class FaceEnhancer:
    """Unified Face Enhancer pipeline orchestrator."""

    def __init__(self, enhancer_type: str = "gpen_realistic", model_size: int = 512):
        self.enhancer_type = enhancer_type.lower()
        self.model_size = model_size
        self.session: Optional[onnxruntime.InferenceSession] = None
        self.active_providers: list[str] = []
        self._lock = threading.Lock()
        self._in_name: str = ""
        self._out_name: str = ""

    def initialize(self, providers: Optional[Sequence[Any]] = None) -> None:
        """Download model if needed and initialize ONNX session with fallback."""
        with self._lock:
            if self.session is not None:
                return

            if "gpen" in self.enhancer_type:
                model_key = "gpen_bfr_512" if self.model_size == 512 else "gpen_bfr_256"
            else:
                model_key = "codeformer_fp16"

            model_path = ensure_model_downloaded(model_key)
            self.session, self.active_providers = create_enhancer_session(
                model_path,
                requested_providers=providers,
                label=f"{self.enhancer_type.upper()}-{self.model_size}",
            )
            self._in_name = self.session.get_inputs()[0].name
            self._out_name = self.session.get_outputs()[0].name

    def enhance_crop(
        self,
        aligned_crop: np.ndarray,
        blend_ratio: float = 1.0,
    ) -> np.ndarray:
        """Run forward enhancement on an aligned face crop."""
        if self.session is None:
            self.initialize()

        S = self.model_size
        if aligned_crop.shape[:2] != (S, S):
            input_bgr = cv2.resize(aligned_crop, (S, S), interpolation=cv2.INTER_CUBIC)
        else:
            input_bgr = aligned_crop

        # Preprocessing: BGR -> normalized float32 [-1, 1] RGB NCHW
        tensor = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
        x = np.ascontiguousarray(tensor.transpose(2, 0, 1)[None, ...])

        with self._lock:
            ort_outs = self.session.run([self._out_name], {self._in_name: x})

        raw_out = ort_outs[0][0]
        if not np.isfinite(raw_out.sum()):
            _LOGGER.warning("[%s] Non-finite model output; using unenhanced crop.", self.enhancer_type)
            return aligned_crop

        hwc = np.ascontiguousarray(raw_out[::-1].transpose(1, 2, 0), dtype=np.float32)
        np.maximum(hwc, -1.0, out=hwc)
        restored = cv2.convertScaleAbs(hwc, alpha=127.5, beta=127.5)

        if looks_collapsed(restored):
            _LOGGER.warning("[%s] Model output collapsed; using unenhanced crop.", self.enhancer_type)
            return aligned_crop

        if "gpen" in self.enhancer_type:
            # GPEN Realistic: apply chroma-from-swapper recolouring and blend
            fixed = luma_only_recolour(restored, input_bgr)
            ratio = float(np.clip(blend_ratio, 0.0, 1.0))
            if ratio < 1.0:
                return cv2.addWeighted(fixed, ratio, input_bgr, 1.0 - ratio, 0.0)
            return fixed
        else:
            # UltraMax composite logic
            return apply_ultramax_composite(
                structure_crop=restored,
                detail_crop=None,
                reference_crop=input_bgr,
                blend_ratio=blend_ratio,
            )

    def enhance_frame(
        self,
        frame: np.ndarray,
        kps: np.ndarray,
        blend_ratio: float = 1.0,
    ) -> np.ndarray:
        """Full pipeline: 5-point alignment -> inference -> inverse affine warp back."""
        aligned_crop, M = align_face_5point(frame, kps, crop_size=self.model_size)
        enhanced_crop = self.enhance_crop(aligned_crop, blend_ratio=1.0)
        return inverse_affine_warp_back(
            target_frame=frame,
            enhanced_crop=enhanced_crop,
            M=M,
            original_crop=aligned_crop,
            blend_ratio=blend_ratio,
        )

    def release(self) -> None:
        """Free ONNX session resources."""
        with self._lock:
            self.session = None
            self.active_providers = []
