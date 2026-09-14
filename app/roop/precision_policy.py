"""Model-specific precision and provider policy — compatibility shim.

This is a reduced port of roop-ultimate's precision_policy.py that does not
depend on backend_manager.cache_namespace (which does not exist in
roop-unleashed-wip). It exposes only `providers_for`, the one function that
Enhance_GPENRealistic and Enhance_UltraMax actually call.

Full precision audit and caching from the original module is omitted; this
shim simply passes the global execution_providers through, applying the
conservative FP32 override for any model known to overflow in FP16.
"""

from __future__ import annotations

import os
from typing import Iterable, Any

import roop.globals
from roop.processors.enhance_common import fp32_trt_providers

# Models known to overflow or collapse in TRT FP16.  These always get FP32.
_FORCE_FP32 = frozenset({
    'gpen_1024',
    'gpen_2048',
    'gfpgan',
    'frame_upscaler',
})


def providers_for(model_tag: str, execution_providers: Iterable[Any],
                  model_path: str = '') -> tuple[list, str]:
    """Return ``(providers, precision_label)`` for the given model tag.

    The precision label is 'fp32', 'fp16', or 'mixed' — used only for
    logging; no ORT/TRT configuration is derived from it in this shim.

    Models in ``_FORCE_FP32`` always get the FP32 TensorRT engine cache
    via ``fp32_trt_providers``.  All others pass through unchanged.
    """
    providers = list(execution_providers or [])
    tag_lower = model_tag.lower().replace(' ', '_')

    if tag_lower in _FORCE_FP32:
        providers = fp32_trt_providers(providers, tag_lower)
        precision = 'fp32'
    else:
        # Honour the user's trt_precision setting if present.  This mirrors
        # the behaviour the TRT provider options already carry through globals.
        trt_precision = getattr(roop.globals, 'trt_precision', 'mixed')
        precision = trt_precision if trt_precision else 'mixed'

    return providers, precision
