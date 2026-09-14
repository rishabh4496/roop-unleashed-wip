"""TensorRT & ONNX model loader and provider options configuration.

Provides:
1. get_tensorrt_provider_options(): Robust TensorRT execution provider configuration.
2. get_tensorrt_cache_dir(): Dedicated engine/timing cache directory under models/cache/tensorrt/.
3. calculate_dynamic_trt_workspace_size(): Dynamic allocation up to 4GB (4 * 1024 * 1024 * 1024).
4. create_enhancer_session: Forwarded from face_enhancer for unified model loading.
"""

from __future__ import annotations

import logging
import os
import pathlib
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import roop.globals

_LOGGER = logging.getLogger(__name__)

MAX_4GB = 4 * 1024 * 1024 * 1024  # 4GB in bytes


def get_models_directory() -> str:
    """Resolve the base models directory."""
    # 1. app/models
    app_models = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    if os.path.isdir(app_models):
        return app_models
    # 2. repo-root models
    root_models = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))
    if os.path.isdir(root_models):
        return root_models
    # Fallback to app_models and create it if needed
    os.makedirs(app_models, exist_ok=True)
    return app_models


def get_tensorrt_cache_dir(subdir: Optional[str] = None) -> str:
    """Return the dedicated TensorRT engine cache directory under models/cache/tensorrt/.

    Parameters:
        subdir: Optional subfolder (e.g. precision label 'fp16', 'fp32', 'mixed').
    """
    base_models = get_models_directory()
    cache_dir = os.path.join(base_models, "cache", "tensorrt")
    if subdir:
        cache_dir = os.path.join(cache_dir, str(subdir).strip("/\\"))
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def calculate_dynamic_trt_workspace_size(
    device_id: int = 0,
    max_cap: int = MAX_4GB,
) -> int:
    """Calculate dynamic TensorRT workspace size up to 4GB (4 * 1024 * 1024 * 1024).

    Inspects total VRAM on the requested CUDA device if available, dynamically allocating
    up to 50% of available memory capped at max_cap (default: 4GB).
    """
    # Check environment override if present
    env_bytes = os.environ.get("ROOP_TRT_MAX_WORKSPACE_BYTES")
    if env_bytes:
        try:
            return max(256 * 1024 * 1024, min(int(env_bytes), max_cap))
        except (ValueError, TypeError):
            pass

    env_mb = os.environ.get("ROOP_TRT_WORKSPACE_MB")
    if env_mb:
        try:
            return max(256 * 1024 * 1024, min(int(float(env_mb) * 1024 * 1024), max_cap))
        except (ValueError, TypeError):
            pass

    try:
        import torch
        if torch.cuda.is_available() and device_id < torch.cuda.device_count():
            total_vram = torch.cuda.get_device_properties(device_id).total_memory
            if total_vram > 0:
                # Dynamic allocation: 50% of VRAM up to 4GB max
                dynamic_alloc = int(total_vram * 0.5)
                return max(512 * 1024 * 1024, min(dynamic_alloc, max_cap))
    except Exception as exc:
        _LOGGER.debug("Could not query GPU properties for dynamic workspace: %s", exc)

    return max_cap


def get_tensorrt_provider_options(
    device_id: Optional[int] = None,
    max_workspace_size: Optional[int] = None,
    fp16_enable: bool = True,
    engine_cache_enable: bool = True,
    engine_cache_path: Optional[str] = None,
    timing_cache_enable: bool = True,
    timing_cache_path: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Return robust TensorRT provider options for ONNX Runtime sessions.

    Parameters:
        device_id: Configurable GPU index (default: 0).
        max_workspace_size: Dynamic allocation up to 4GB (4 * 1024 * 1024 * 1024).
        fp16_enable: Boolean flag (default: True).
        engine_cache_enable: Set to True.
        engine_cache_path: Dedicated directory under models/cache/tensorrt/.
        timing_cache_enable: Set to True for faster subsequent rebuilds.
        timing_cache_path: Directory for timing cache (defaults to engine_cache_path).
        **kwargs: Additional provider option overrides.
    """
    # 1. Resolve device_id (default: 0)
    if device_id is None:
        try:
            device_id = int(getattr(roop.globals, "cuda_device_id", 0) or 0)
        except Exception:
            device_id = 0
    else:
        device_id = int(device_id)

    # 2. Dynamic workspace allocation up to 4GB (4 * 1024 * 1024 * 1024)
    if max_workspace_size is not None:
        workspace_bytes = min(int(max_workspace_size), MAX_4GB)
    else:
        workspace_bytes = calculate_dynamic_trt_workspace_size(device_id=device_id, max_cap=MAX_4GB)

    # 3. Dedicated engine cache directory under models/cache/tensorrt/
    if not engine_cache_path:
        engine_cache_path = get_tensorrt_cache_dir()
    else:
        os.makedirs(engine_cache_path, exist_ok=True)

    # 4. Timing cache path (defaults to engine cache directory)
    if not timing_cache_path:
        timing_cache_path = engine_cache_path
    else:
        os.makedirs(timing_cache_path, exist_ok=True)

    opts: Dict[str, Any] = {
        "device_id": device_id,
        "trt_max_workspace_size": workspace_bytes,
        "trt_fp16_enable": bool(fp16_enable),
        "trt_engine_cache_enable": bool(engine_cache_enable),
        "trt_engine_cache_path": engine_cache_path,
        "trt_timing_cache_enable": bool(timing_cache_enable),
        "trt_timing_cache_path": timing_cache_path,
        "trt_context_memory_sharing_enable": True,
    }

    # Merge any extra kwargs (e.g. profile shapes or partition iterations)
    for k, v in kwargs.items():
        if v is not None:
            opts[k] = v

    return opts


def build_provider_priority_stack(
    trt_options: Optional[Dict[str, Any]] = None,
    cuda_options: Optional[Dict[str, Any]] = None,
    device_id: Optional[int] = None,
    fp16_enable: bool = True,
    max_workspace_size: Optional[int] = None,
    engine_cache_path: Optional[str] = None,
    model_key: Optional[str] = None,
    model_path: Optional[str] = None,
    explicit_shape: Optional[Tuple[int, ...]] = None,
    batch_size: int = 1,
) -> List[Any]:
    """Initialize the explicit provider fallback hierarchy:

    [
        ('TensorrtExecutionProvider', trt_options),
        ('CUDAExecutionProvider', cuda_options),
        'CPUExecutionProvider'
    ]
    """
    if device_id is None:
        try:
            device_id = int(getattr(roop.globals, "cuda_device_id", 0) or 0)
        except Exception:
            device_id = 0

    if trt_options is None:
        trt_options = get_tensorrt_provider_options(
            device_id=device_id,
            max_workspace_size=max_workspace_size,
            fp16_enable=fp16_enable,
            engine_cache_path=engine_cache_path,
        )

    if cuda_options is None:
        cuda_options = {
            "device_id": device_id,
            "cudnn_conv_algo_search": os.environ.get("ROOP_CUDNN_CONV_ALGO", "DEFAULT"),
            "do_copy_in_default_stream": True,
            "arena_extend_strategy": os.environ.get("ROOP_CUDA_ARENA_STRATEGY", "kSameAsRequested"),
        }

    stack: List[Any] = [
        ("TensorrtExecutionProvider", trt_options),
        ("CUDAExecutionProvider", cuda_options),
        "CPUExecutionProvider",
    ]

    # Attach static optimization profile if model metadata is available
    if model_key or model_path or explicit_shape is not None or (batch_size and batch_size > 1):
        try:
            from roop.trt_shape_profile import apply_shape_profile
            stack = apply_shape_profile(
                stack,
                model_key=model_key,
                model_path=model_path,
                explicit_shape=explicit_shape,
                batch_size=batch_size,
            )
        except Exception as exc:
            _LOGGER.debug("Could not attach shape profile to provider stack: %s", exc)

    return stack


