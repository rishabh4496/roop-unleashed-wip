"""Model-specific precision and TensorRT execution provider policy.

Provides model-aware precision routing (FP16 vs FP32 vs mixed) and binds explicit
static TensorRT optimization profiles to guarantee:
1. Maximum inference throughput (~2x vs CUDAExecutionProvider).
2. Elimination of white speckle and color clipping / NaN artifacts through selectable precision.
3. Segregated persistent engine caches (.engine / .timing_cache) per precision and model geometry.
"""

from __future__ import annotations

import logging
import os
import pathlib
from typing import Any, Iterable, List, Optional, Tuple

import roop.globals
from roop.processors.enhance_common import fp32_trt_providers
from roop.trt_shape_profile import apply_shape_profile

_LOGGER = logging.getLogger(__name__)

PRECISIONS = ("fp32", "fp16", "mixed")

# Models known to suffer numerical overflow or NaN collapse in TRT FP16.
# By default, these route to FP32 unless explicitly overridden by user policy.
_FORCE_FP32 = frozenset({
    "gpen_1024",
    "gpen_2048",
    "gfpgan",
    "frame_upscaler",
})


def _is_trt_provider(p: Any) -> bool:
    name = p[0] if isinstance(p, (tuple, list)) else str(p)
    return "tensorrt" in name.lower()


def providers_for(
    model_tag: str,
    execution_providers: Iterable[Any],
    model_path: str = "",
    requested_precision: Optional[str] = None,
    explicit_shape: Optional[Tuple[int, ...]] = None,
) -> Tuple[List[Any], str]:
    """Return configured ``(providers, effective_precision)`` for a model session.

    Handles:
    1. Precision selection (fp16, fp32, mixed) with FP32 safety override for GPEN-1024.
    2. Static optimization profile binding (min/opt/max shapes) to prevent TRT dynamic axis rejection.
    3. Engine and timing cache path segregation under `models/trt_cache/{precision}/`.
    """
    providers = list(execution_providers or [])
    tag_lower = str(model_tag or "").lower().replace(" ", "_")

    # 1. Determine base precision
    if requested_precision and str(requested_precision).lower() in PRECISIONS:
        base_precision = str(requested_precision).lower()
    else:
        # Check global config or environment override
        env_prec = os.environ.get("ROOP_GPEN_PRECISION", "").strip().lower()
        if env_prec in PRECISIONS:
            base_precision = env_prec
        else:
            cfg_prec = getattr(getattr(roop.globals, "CFG", None), "trt_precision", None)
            if not cfg_prec:
                cfg_prec = getattr(roop.globals, "trt_precision", "mixed")
            base_precision = str(cfg_prec).lower() if cfg_prec in PRECISIONS else "mixed"

    # 2. Apply model safety policies
    # GPEN 1024 / 2048 models overflow in FP16, causing white speckles / NaN / black faces
    if tag_lower in _FORCE_FP32 or "1024" in tag_lower or "2048" in tag_lower:
        # Allow explicit ROOP_GPEN_FP16=1 or user request for experimental FP16
        if requested_precision == "fp16" or os.environ.get("ROOP_GPEN_FP16", "0") == "1":
            effective_precision = "fp16"
        else:
            effective_precision = "fp32"
    else:
        effective_precision = base_precision

    # 3. Configure TensorRT options if present in execution providers
    has_trt = any(_is_trt_provider(p) for p in providers)
    if has_trt:
        # Set up default cache paths if not already defined
        base_cache = str(pathlib.Path(__file__).parent.parent / "models" / "trt_cache" / effective_precision)
        os.makedirs(base_cache, exist_ok=True)

        updated: List[Any] = []
        for p in providers:
            if _is_trt_provider(p):
                name = p[0] if isinstance(p, (tuple, list)) else p
                opts = dict(p[1]) if (isinstance(p, (tuple, list)) and len(p) == 2 and isinstance(p[1], dict)) else {}

                opts.setdefault("device_id", int(getattr(roop.globals, "cuda_device_id", 0) or 0))
                opts["trt_engine_cache_enable"] = True
                opts.setdefault("trt_engine_cache_path", base_cache)
                opts["trt_timing_cache_enable"] = True
                opts.setdefault("trt_timing_cache_path", base_cache)
                opts.setdefault("trt_context_memory_sharing_enable", True)

                # Set precision flags
                if effective_precision == "fp32":
                    opts["trt_fp16_enable"] = False
                    opts["trt_layer_norm_fp32_fallback"] = True
                    # Direct to fp32 cache path to avoid engine collision
                    fp32_cache = base_cache + "_fp32"
                    os.makedirs(fp32_cache, exist_ok=True)
                    opts["trt_engine_cache_path"] = fp32_cache
                    opts["trt_timing_cache_path"] = fp32_cache
                elif effective_precision == "fp16":
                    opts["trt_fp16_enable"] = True
                    opts["trt_layer_norm_fp32_fallback"] = False
                else:  # mixed
                    opts["trt_fp16_enable"] = True
                    opts["trt_layer_norm_fp32_fallback"] = True

                updated.append((name, opts))
            else:
                updated.append(p)
        providers = updated

        # 4. Attach explicit static optimization profiles (e.g. 1x3x512x512)
        providers = apply_shape_profile(
            providers,
            model_key=tag_lower,
            model_path=model_path,
            explicit_shape=explicit_shape,
        )

    return providers, effective_precision
