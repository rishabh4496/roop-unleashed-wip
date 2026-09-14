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
from roop.model_loader import (
    get_tensorrt_provider_options,
    get_tensorrt_cache_dir,
    calculate_dynamic_trt_workspace_size,
    build_provider_priority_stack,
)

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


# ── VRAM Safety & CUDA OOM Utilities ─────────────────────────────────────────

class CudaOOMError(RuntimeError):
    """Raised when GPU / CUDA runs out of memory during face enhancement."""
    pass


def clear_cuda_cache() -> None:
    """Safely flush garbage collection and GPU memory caches."""
    try:
        import gc
        gc.collect()
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass


def is_cuda_oom(exc: BaseException) -> bool:
    """Detect whether an exception is caused by GPU / CUDA Out Of Memory."""
    if isinstance(exc, CudaOOMError):
        return True
    try:
        import torch
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass

    msg = str(exc).lower()
    oom_patterns = (
        "out of memory",
        "cuda oom",
        "cudamalloc failed",
        "cuda failure 2",
        "failed to allocate",
        "device memory allocation",
        "resource_exhausted",
    )
    return any(p in msg for p in oom_patterns)


# ── ONNX Session Initialization with Multi-Tier EP Fallback ──────────────────

def _warmup_enhancer_session(
    sess: onnxruntime.InferenceSession,
    label: str,
    explicit_shape: Optional[Tuple[int, ...]] = None,
    batch_size: int = 1,
) -> bool:
    """Implement an automated warm-up inference pass (np.zeros((1, 3, H, W), dtype=np.float32)).

    Triggers TensorRT engine compilation during startup before the first video frame is processed.
    """
    try:
        inputs = sess.get_inputs()
        feed = {}
        bs = max(1, int(batch_size or 1))
        sz = 1024 if "1024" in label else (256 if "256" in label else 512)

        for inp in inputs:
            name = inp.name
            shape = list(inp.shape) if inp.shape else []
            if explicit_shape is not None and name in ("input", "x"):
                c = explicit_shape[1] if len(explicit_shape) > 1 else 3
                h = explicit_shape[2] if len(explicit_shape) > 2 else sz
                w = explicit_shape[3] if len(explicit_shape) > 3 else sz
                res_shape = [bs, c, h, w]
            else:
                if len(shape) == 4:
                    c = shape[1] if (shape[1] is not None and not isinstance(shape[1], str) and shape[1] > 0) else 3
                    h = shape[2] if (shape[2] is not None and not isinstance(shape[2], str) and shape[2] > 0) else sz
                    w = shape[3] if (shape[3] is not None and not isinstance(shape[3], str) and shape[3] > 0) else sz
                    res_shape = [bs, c, h, w]
                elif len(shape) == 1:
                    res_shape = [1]
                elif len(shape) == 0:
                    res_shape = []
                else:
                    res_shape = [1] * len(shape)
            dtype = np.float16 if "float16" in str(inp.type).lower() else (np.float64 if "double" in str(inp.type).lower() else np.float32)
            feed[name] = np.zeros(res_shape, dtype=dtype)

        sess.run(None, feed)
        return True
    except Exception as exc:
        _LOGGER.debug("[%s] Warmup run exception: %s", label, exc)
        raise


def create_enhancer_session(
    model_path: str,
    requested_providers: Optional[Sequence[Any]] = None,
    session_options: Optional[onnxruntime.SessionOptions] = None,
    label: str = "FaceEnhancer",
    warmup: bool = True,
    explicit_shape: Optional[Tuple[int, ...]] = None,
    precision: Optional[str] = None,
    batch_size: int = 1,
) -> Tuple[onnxruntime.InferenceSession, list[str]]:
    """Initialize an ONNX Runtime InferenceSession with an explicit fallback hierarchy.

    Provider Priority Stack:
        providers = [
            ('TensorrtExecutionProvider', trt_options),
            ('CUDAExecutionProvider', cuda_options),
            'CPUExecutionProvider'
        ]

    Session Health Check:
        Executes automated warm-up inference pass (np.zeros((1, 3, H, W))) during startup.
        Catches compilation errors, logs actionable diagnostic output to terminal,
        and falls back cleanly to CUDAExecutionProvider without crashing video batches.
    """
    opts = session_options or onnxruntime.SessionOptions()
    device_id = int(getattr(roop.globals, "cuda_device_id", 0) or 0)
    available_fn = getattr(onnxruntime, "get_available_providers", None)
    available = set(list(available_fn()) if callable(available_fn) else ["CPUExecutionProvider"])

    if requested_providers is None:
        if roop.globals.execution_providers == ["CPUExecutionProvider"]:
            raw_providers = ["CPUExecutionProvider"]
        elif "TensorrtExecutionProvider" in available:
            raw_providers = build_provider_priority_stack(device_id=device_id)
        else:
            raw_providers = list(roop.globals.execution_providers or ["CUDAExecutionProvider", "CPUExecutionProvider"])
    else:
        raw_providers = list(requested_providers)

    from roop.precision_policy import providers_for
    from roop.trt_shape_profile import is_engine_cached
    from roop.trt_events import TensorRTCompilationMonitor

    # Specialize providers with precision policy, 4GB dynamic workspace, and static profile binding
    specialized_providers, effective_precision = providers_for(
        model_tag=label,
        execution_providers=raw_providers,
        model_path=model_path,
        requested_precision=precision,
        explicit_shape=explicit_shape,
        batch_size=batch_size,
    )

    # Ensure CUDA options are present in fallback hierarchy
    cuda_options = {
        "device_id": device_id,
        "cudnn_conv_algo_search": os.environ.get("ROOP_CUDNN_CONV_ALGO", "DEFAULT"),
        "do_copy_in_default_stream": True,
        "arena_extend_strategy": os.environ.get("ROOP_CUDA_ARENA_STRATEGY", "kSameAsRequested"),
    }

    # Build Tier 1 priority stack
    has_trt_requested = any(
        "tensorrt" in (p[0] if isinstance(p, (tuple, list)) else str(p)).lower()
        for p in specialized_providers
    )

    if has_trt_requested and "TensorrtExecutionProvider" in available:
        tier1: List[Any] = []
        for p in specialized_providers:
            name = p[0] if isinstance(p, (tuple, list)) else str(p)
            if "tensorrt" in name.lower():
                tier1.append(p)
                break
        if "CUDAExecutionProvider" in available:
            tier1.append(("CUDAExecutionProvider", cuda_options))
        tier1.append("CPUExecutionProvider")
    else:
        tier1 = [
            p for p in specialized_providers
            if (p[0] if isinstance(p, (tuple, list)) else str(p)) in available
        ]
        if not tier1:
            tier1 = [("CUDAExecutionProvider", cuda_options), "CPUExecutionProvider"] if "CUDAExecutionProvider" in available else ["CPUExecutionProvider"]

    has_trt = any(
        "tensorrt" in (p[0] if isinstance(p, (tuple, list)) else str(p)).lower()
        for p in tier1
    )

    # Check cache status for TensorRT to avoid freezing and display progress
    cache_path = None
    if has_trt:
        for p in tier1:
            if isinstance(p, (tuple, list)) and len(p) == 2 and "tensorrt" in str(p[0]).lower():
                cache_path = p[1].get("trt_engine_cache_path")
                break

    cached = is_engine_cached(cache_path) if (has_trt and cache_path) else False

    try:
        if has_trt and not cached and warmup:
            with TensorRTCompilationMonitor(label=label, cache_dir=cache_path):
                sess = onnxruntime.InferenceSession(model_path, opts, providers=tier1)
                _warmup_enhancer_session(sess, label, explicit_shape, batch_size=batch_size)
        else:
            sess = onnxruntime.InferenceSession(model_path, opts, providers=tier1)
            if warmup:
                _warmup_enhancer_session(sess, label, explicit_shape, batch_size=batch_size)

        active = list(sess.get_providers())
        _LOGGER.info("[%s] Initialized with providers: %s (precision=%s)", label, active, effective_precision)
        return sess, active
    except Exception as exc_tier1:
        # Actionable diagnostic output and clean fallback
        print(f"\n[{label}] [Session Health Check] TensorRT compilation / warmup aborted: {exc_tier1}", flush=True)
        print(f"[{label}] [Actionable Diagnostic] Falling back cleanly to CUDAExecutionProvider without crashing video batch.", flush=True)
        _LOGGER.warning(
            "[%s] Primary provider configuration failed: %s. Falling back to CUDAExecutionProvider...",
            label,
            exc_tier1,
        )
        clear_cuda_cache()

    # Tier 2: Clean CUDA execution provider fallback
    if "CUDAExecutionProvider" in available:
        tier2 = [
            ("CUDAExecutionProvider", cuda_options),
            "CPUExecutionProvider",
        ]
        tier2 = [p for p in tier2 if (p[0] if isinstance(p, (tuple, list)) else p) in available]
        try:
            sess = onnxruntime.InferenceSession(model_path, opts, providers=tier2)
            if warmup:
                try:
                    _warmup_enhancer_session(sess, label, explicit_shape, batch_size=batch_size)
                except Exception as exc_cuda_warmup:
                    _LOGGER.debug("[%s] CUDA warm-up bypassed: %s", label, exc_cuda_warmup)
            active = list(sess.get_providers())
            print(f"[{label}] Clean fallback to CUDAExecutionProvider succeeded: {active}", flush=True)
            _LOGGER.info("[%s] Fallback to CUDA succeeded: %s", label, active)
            return sess, active
        except Exception as exc_tier2:
            print(f"[{label}] CUDA fallback failed: {exc_tier2}. Attempting CPU fallback...", flush=True)
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
        print(f"[{label}] Warning: Running on CPUExecutionProvider as fallback: {active}", flush=True)
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


# ── Zero-Copy Tensor Handling & Memory Contiguity ────────────────────────────

def prepare_zero_copy_tensor(
    img: np.ndarray,
    target_size: int = 512,
    normalization_mode: str = "symmetric",
    dtype: Any = np.float32,
) -> np.ndarray:
    """Prepare normalized NCHW tensor with guaranteed C-contiguous memory layout for zero-copy DMA.

    Normalization modes:
    - 'symmetric': ((img / 255.0 - 0.5) / 0.5) mapping [0, 255] -> [-1.0, 1.0]
    - 'unit': (img / 255.0) mapping [0, 255] -> [0.0, 1.0]

    Transposition (transpose(2, 0, 1)) is forced to be C-contiguous via np.ascontiguousarray,
    eliminating strided host memory copies and maximizing TensorRT DMA transfer speeds.
    """
    if img.shape[0] != target_size or img.shape[1] != target_size:
        resized = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
    else:
        resized = img

    # Convert BGR to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # Normalization: ((img / 255.0 - 0.5) / 0.5) or (img / 255.0)
    target_dt = np.float16 if "float16" in str(dtype).lower() else np.float32
    if normalization_mode == "unit":
        norm = rgb.astype(target_dt) / 255.0
    else:  # symmetric: maps [0, 255] -> [-1.0, 1.0]
        norm = (rgb.astype(target_dt) / 255.0 - 0.5) / 0.5

    # Transpose HWC -> CHW and add batch axis NCHW (1, 3, H, W)
    # np.ascontiguousarray guarantees contiguous linear layout for zero-copy DMA to TensorRT
    tensor = np.ascontiguousarray(norm.transpose(2, 0, 1)[None, ...], dtype=target_dt)
    return tensor


# ── Decoupled Composite Visual Filters & Post-Inference Blending ─────────────

def apply_unsharp_mask(
    img: np.ndarray,
    amount: float = 0.35,
    sigma: float = 1.0,
) -> np.ndarray:
    """Vectorized unsharp masking in OpenCV to boost high-frequency clarity.

    Formula:
        blur = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
        unsharp = cv2.addWeighted(img, 1.0 + amount, blur, -amount, 0.0)
    """
    if amount <= 0.0 or img is None:
        return img
    sigma = max(0.5, float(sigma))
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return cv2.addWeighted(img, 1.0 + float(amount), blur, -float(amount), 0.0)


def apply_bilateral_edge_preservation(
    img: np.ndarray,
    d: int = 5,
    sigma_color: float = 25.0,
    sigma_space: float = 25.0,
    blend: float = 0.85,
) -> np.ndarray:
    """Bilateral filtering to preserve sharp edges and facial geometry while removing noise.

    Uses vectorized cv2.addWeighted to combine edge-smoothed plate with original.
    """
    if blend <= 0.0 or img is None:
        return img
    bilateral = cv2.bilateralFilter(img, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    if blend >= 1.0:
        return bilateral
    return cv2.addWeighted(bilateral, float(blend), img, 1.0 - float(blend), 0.0)


def apply_adaptive_sharpening(
    img: np.ndarray,
    strength: float = 0.30,
    radius: int = 1,
    limit: float = 2.5,
) -> np.ndarray:
    """Adaptive edge-aware sharpening with local min/max bounding to eliminate halos.

    Operates on the Luminance (L) channel in LAB space using vectorized OpenCV operations.
    Suppresses noise in flat facial areas while enhancing edges (eyes, lips, nostrils).
    """
    if strength <= 0.0 or img is None:
        return img
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * radius + 1, 2 * radius + 1))
    L_min = cv2.erode(L, kernel)
    L_max = cv2.dilate(L, kernel)

    blur = cv2.GaussianBlur(L, (0, 0), sigmaX=float(radius))
    high = L - blur

    mag = np.abs(high)
    np.maximum(mag - 1.0, 0.0, out=mag)
    high_clamped = np.copysign(mag, high)

    sharpened = L + float(strength) * high_clamped
    sharpened = np.clip(sharpened, np.maximum(0.0, L_min - limit), np.minimum(255.0, L_max + limit))

    lab[:, :, 0] = np.clip(sharpened, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def post_inference_blend(
    enhanced_crop: np.ndarray,
    reference_crop: np.ndarray,
    blend_ratio: float = 1.0,
) -> np.ndarray:
    """Vectorized in-place blending in OpenCV (cv2.addWeighted) to prevent CPU bottlenecks."""
    ratio = float(np.clip(blend_ratio, 0.0, 1.0))
    if ratio >= 1.0:
        return enhanced_crop
    if ratio <= 0.0:
        return reference_crop
    if reference_crop.shape[:2] != enhanced_crop.shape[:2]:
        h, w = enhanced_crop.shape[:2]
        reference_crop = cv2.resize(reference_crop, (w, h), interpolation=cv2.INTER_CUBIC)
    return cv2.addWeighted(enhanced_crop, ratio, reference_crop, 1.0 - ratio, 0.0)


def apply_ultramax_composite_filters(
    enhanced_chip: np.ndarray,
    reference_chip: Optional[np.ndarray] = None,
    unsharp_amount: float = 0.25,
    bilateral_strength: float = 0.35,
    adaptive_sharpen_strength: float = 0.25,
    blend_ratio: float = 1.0,
) -> np.ndarray:
    """Execute decoupled composite visual filters and post-inference blending.

    Filters are applied sequentially:
    1. Unsharp mask (high-frequency clarity boost)
    2. Bilateral edge preservation (contour preservation & noise attenuation)
    3. Adaptive sharpening (anti-halo edge enhancement)
    4. Vectorized post-inference blending with reference chip via cv2.addWeighted
    """
    out = enhanced_chip

    # 1. Unsharp mask
    if unsharp_amount > 0.0:
        out = apply_unsharp_mask(out, amount=unsharp_amount)

    # 2. Bilateral edge preservation
    if bilateral_strength > 0.0:
        out = apply_bilateral_edge_preservation(out, blend=bilateral_strength)

    # 3. Adaptive sharpening
    if adaptive_sharpen_strength > 0.0:
        out = apply_adaptive_sharpening(out, strength=adaptive_sharpen_strength)

    # 4. Vectorized post-inference blending with reference crop
    if reference_chip is not None and blend_ratio < 1.0:
        out = post_inference_blend(out, reference_chip, blend_ratio=blend_ratio)

    return out


# ── UltraMax Multi-Pass Composite Filter & Blend Logic ───────────────────────

def apply_ultramax_composite(
    structure_crop: np.ndarray,
    detail_crop: Optional[np.ndarray],
    reference_crop: np.ndarray,
    blend_ratio: float = 1.0,
    chroma_weight: float = 0.0,
    dual_stream: bool = False,
    gain: float = 1.25,
    unsharp_amount: float = 0.0,
    bilateral_strength: float = 0.0,
    adaptive_sharpen_strength: float = 0.0,
) -> np.ndarray:
    """Multi-pass composite filter and blend for UltraMax.

    Pass 1: Preserves reference chrominance while retaining CodeFormer structural luminance
            (eliminates CodeFormer's characteristic pale skin drift).
    Pass 2: (Optional Dual-Stream) Injects edge-guided high-frequency detail from GPEN-512
            via guided frequency split.
    Pass 3: Decoupled composite visual filters (unsharp mask, bilateral edge preservation, adaptive sharpening).
    Pass 4: Vectorized post-inference blending (cv2.addWeighted) with reference crop driven by blend_ratio.
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

    # Pass 3: Decoupled Composite Visual Filters
    u_amt = unsharp_amount if unsharp_amount > 0.0 else float(os.environ.get("ROOP_ULTRAMAX_UNSHARP", "0.0") or 0.0)
    b_str = bilateral_strength if bilateral_strength > 0.0 else float(os.environ.get("ROOP_ULTRAMAX_BILATERAL", "0.0") or 0.0)
    s_str = adaptive_sharpen_strength if adaptive_sharpen_strength > 0.0 else float(os.environ.get("ROOP_ULTRAMAX_SHARPEN", "0.0") or 0.0)

    if u_amt > 0.0 or b_str > 0.0 or s_str > 0.0:
        pass3 = apply_ultramax_composite_filters(
            enhanced_chip=pass2,
            reference_chip=None,
            unsharp_amount=u_amt,
            bilateral_strength=b_str,
            adaptive_sharpen_strength=s_str,
            blend_ratio=1.0,
        )
    else:
        pass3 = pass2

    # Pass 4: Vectorized post-inference blend with original reference crop
    return post_inference_blend(pass3, reference_crop, blend_ratio=ratio)


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
            inputs = self.session.get_inputs()
            self._in_name = inputs[0].name
            self._has_w = len(inputs) > 1
            self._w_name = inputs[1].name if self._has_w else ""
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

        # Preprocessing: Zero-copy contiguous memory normalization & RGB transposition
        x = prepare_zero_copy_tensor(
            input_bgr,
            target_size=S,
            normalization_mode="symmetric",
            dtype=np.float32,
        )

        feed = {self._in_name: x}
        if getattr(self, "_has_w", False) and self._w_name:
            fidelity = float(getattr(roop.globals, "codeformer_fidelity", 0.5) or 0.5)
            feed[self._w_name] = np.array([fidelity], dtype=np.float64)

        try:
            with self._lock:
                ort_outs = self.session.run([self._out_name], feed)
        except Exception as exc:
            if is_cuda_oom(exc):
                _LOGGER.error("[%s] CUDA Out of Memory during inference: %s", self.enhancer_type, exc)
                clear_cuda_cache()
                # Attempt graceful fallback to CPUExecutionProvider
                try:
                    _LOGGER.warning("[%s] Attempting CPU fallback session...", self.enhancer_type)
                    opts = onnxruntime.SessionOptions()
                    opts.log_severity_level = 2
                    opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
                    model_key = (
                        "gpen_bfr_512"
                        if (self.model_size == 512 and "gpen" in self.enhancer_type)
                        else ("gpen_bfr_256" if "gpen" in self.enhancer_type else "codeformer_fp16")
                    )
                    model_path = ensure_model_downloaded(model_key)
                    cpu_sess = onnxruntime.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
                    with self._lock:
                        self.session = cpu_sess
                        self.active_providers = ["CPUExecutionProvider"]
                        cpu_inputs = self.session.get_inputs()
                        self._in_name = cpu_inputs[0].name
                        self._has_w = len(cpu_inputs) > 1
                        self._w_name = cpu_inputs[1].name if self._has_w else ""
                        self._out_name = self.session.get_outputs()[0].name
                        cpu_feed = {self._in_name: x}
                        if self._has_w and self._w_name:
                            fidelity = float(getattr(roop.globals, "codeformer_fidelity", 0.5) or 0.5)
                            cpu_feed[self._w_name] = np.array([fidelity], dtype=np.float64)
                        ort_outs = self.session.run([self._out_name], cpu_feed)
                except Exception as cpu_exc:
                    clear_cuda_cache()
                    raise CudaOOMError(
                        f"CUDA Out of Memory in {self.enhancer_type} ({exc}); CPU fallback also failed: {cpu_exc}"
                    ) from exc
            else:
                raise

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
            # GPEN Realistic: apply chroma-from-swapper recolouring and vectorized blend
            fixed = luma_only_recolour(restored, input_bgr)
            return post_inference_blend(fixed, input_bgr, blend_ratio=blend_ratio)
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
        try:
            aligned_crop, M = align_face_5point(frame, kps, crop_size=self.model_size)
            enhanced_crop = self.enhance_crop(aligned_crop, blend_ratio=1.0)
            return inverse_affine_warp_back(
                target_frame=frame,
                enhanced_crop=enhanced_crop,
                M=M,
                original_crop=aligned_crop,
                blend_ratio=blend_ratio,
            )
        except Exception as exc:
            if is_cuda_oom(exc):
                clear_cuda_cache()
                if not isinstance(exc, CudaOOMError):
                    raise CudaOOMError(f"CUDA Out of Memory during enhance_frame: {exc}") from exc
            raise

    def release(self) -> None:
        """Free ONNX session resources."""
        with self._lock:
            self.session = None
            self.active_providers = []
