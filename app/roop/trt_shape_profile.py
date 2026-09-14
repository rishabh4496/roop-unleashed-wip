"""TensorRT optimization profiles derived from model graphs and explicit static geometries.

TensorRT requires explicit min/opt/max optimization profiles for any model with
dynamic axes (e.g. batch or spatial dimensions). When models are fed fixed geometries
(such as 1x3x512x512 for GPEN-512/CodeFormer, or 1x3x1024x1024 for GPEN-1024), binding
an explicit static profile ensures:
1. Optimal TensorRT kernel tactic selection without dynamic resolution compilation overhead.
2. Prevention of TensorRT engine build failures due to unconstrained dynamic axes.
3. Stable persistent engine cache segregation by profile namespace.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import re
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

_LOGGER = logging.getLogger(__name__)

# Default spatial configurations for known model families (min, opt, max)
_SPATIAL_BANDS: Dict[str, Tuple[int, int, int]] = {
    "gpen_realistic": (512, 512, 512),
    "gpen_512": (512, 512, 512),
    "gpen_1024": (1024, 1024, 1024),
    "gpen_256": (256, 256, 256),
    "ultramax": (512, 512, 512),
    "codeformer": (512, 512, 512),
    "codeformer_fp16": (512, 512, 512),
    "gfpgan": (512, 512, 512),
    "restoreformer_pp": (512, 512, 512),
}

_lock = threading.Lock()
_spec_cache: Dict[str, Tuple["InputSpec", ...]] = {}


@dataclass(frozen=True)
class InputSpec:
    """One graph input; dims holds an int for static, None for dynamic."""
    name: str
    dims: Tuple[Optional[int], ...]

    @property
    def dynamic(self) -> bool:
        return any(d is None for d in self.dims)


@dataclass(frozen=True)
class ShapeProfile:
    """Resolved ONNX Runtime TensorRT profile options plus cache namespace."""
    min_shapes: str
    opt_shapes: str
    max_shapes: str
    namespace: str

    def as_options(self) -> Dict[str, str]:
        return {
            "trt_profile_min_shapes": self.min_shapes,
            "trt_profile_opt_shapes": self.opt_shapes,
            "trt_profile_max_shapes": self.max_shapes,
        }


def _cache_path(model_path: str) -> str:
    root = os.path.join(os.path.dirname(__file__), "..", "models",
                        "runtime_profiles", "shapes")
    try:
        stat = os.stat(model_path)
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_",
                      f"{os.path.basename(model_path)}_{stat.st_size}_{int(stat.st_mtime)}")
    except OSError:
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", os.path.basename(model_path))
    return os.path.join(root, stem + ".json")


def _read_graph_inputs(model_path: str) -> Tuple[InputSpec, ...]:
    """Parse non-initializer feed inputs from an ONNX model file."""
    try:
        import onnx
        model = onnx.load(model_path, load_external_data=False)
        initializers = {t.name for t in model.graph.initializer}
        specs: List[InputSpec] = []
        for entry in model.graph.input:
            if entry.name in initializers:
                continue
            dims: List[Optional[int]] = []
            for dim in entry.type.tensor_type.shape.dim:
                if dim.HasField("dim_value") and dim.dim_value > 0:
                    dims.append(int(dim.dim_value))
                else:
                    dims.append(None)
            specs.append(InputSpec(entry.name, tuple(dims)))
        return tuple(specs)
    except Exception as exc:
        _LOGGER.debug("Could not read graph inputs for %s: %s", model_path, exc)
        return ()


def graph_inputs(model_path: Optional[str]) -> Tuple[InputSpec, ...]:
    """Return model feed inputs, cached in-process and on disk."""
    if not model_path or not os.path.isfile(model_path):
        return ()
    key = os.path.abspath(model_path)
    with _lock:
        if key in _spec_cache:
            return _spec_cache[key]

    specs: Tuple[InputSpec, ...] = ()
    path = None
    try:
        path = _cache_path(model_path)
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            specs = tuple(
                InputSpec(item["name"],
                          tuple(None if d is None else int(d) for d in item["dims"]))
                for item in payload["inputs"]
            )
    except Exception:
        specs = ()

    if not specs:
        specs = _read_graph_inputs(model_path)
        if specs and path:
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                tmp = path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as handle:
                    json.dump({
                        "model": os.path.basename(model_path),
                        "inputs": [{"name": s.name, "dims": list(s.dims)} for s in specs]
                    }, handle, indent=2)
                os.replace(tmp, path)
            except Exception:
                pass

    with _lock:
        _spec_cache[key] = specs
    return specs


def enabled() -> bool:
    """Whether shape profiling is enabled (default yes)."""
    return os.environ.get("ROOP_TRT_SHAPE_PROFILE", "1").strip().lower() not in (
        "0", "off", "false", "no"
    )


def is_engine_cached(cache_path: Optional[str], prefix: Optional[str] = None) -> bool:
    """Check whether a compiled TensorRT engine file exists in the cache directory."""
    if not cache_path or not os.path.isdir(cache_path):
        return False
    pattern = f"{prefix}*.engine" if prefix else "*.engine"
    matches = glob.glob(os.path.join(cache_path, pattern))
    return len(matches) > 0


def resolve_profile(
    model_key: Optional[str] = None,
    model_path: Optional[str] = None,
    explicit_shape: Optional[Tuple[int, ...]] = None,
    batch_size: int = 1,
) -> Optional[ShapeProfile]:
    """Resolve min/opt/max shape profile options for an ONNX model.

    Detects model input dimensions dynamically. For GPEN-512 with batch_size=1:
        min_shapes: 'input:1x3x512x512'
        opt_shapes: 'input:1x3x512x512'
        max_shapes: 'input:1x3x512x512'
    When batch enhancement (batch_size > 1) is enabled:
        min_shapes: 'input:1x3x512x512'
        opt_shapes: 'input:{batch_size}x3x512x512'
        max_shapes: 'input:{batch_size}x3x512x512'
    """
    if not enabled():
        return None

    key_normalized = str(model_key or "").lower().replace(" ", "_")
    specs = graph_inputs(model_path)
    bs = max(1, int(batch_size or 1))

    # Determine default spatial size from spec, model_path or key
    size = 512
    if "1024" in key_normalized or (model_path and "1024" in model_path):
        size = 1024
    elif "256" in key_normalized or (model_path and "256" in model_path):
        size = 256

    # 1. Explicit shape provided by caller (e.g. (1, 3, 512, 512))
    if explicit_shape is not None:
        explicit_b = explicit_shape[0] if (len(explicit_shape) > 0 and explicit_shape[0] is not None and explicit_shape[0] > 0) else 1
        c = explicit_shape[1] if len(explicit_shape) > 1 else 3
        h = explicit_shape[2] if len(explicit_shape) > 2 else size
        w = explicit_shape[3] if len(explicit_shape) > 3 else size
        if explicit_b > 1:
            bs = explicit_b

        # Determine main input name
        in_name = "input"
        if specs and len(specs) > 0:
            in_name = specs[0].name
        elif "codeformer" in key_normalized or "ultra" in key_normalized:
            in_name = "x"

        min_parts = [f"{in_name}:1x{c}x{h}x{w}"]
        opt_parts = [f"{in_name}:{bs}x{c}x{h}x{w}"]
        max_parts = [f"{in_name}:{bs}x{c}x{h}x{w}"]

        # Check for secondary scalar/vector inputs like CodeFormer's 'w'
        if specs and len(specs) > 1:
            for s in specs[1:]:
                sub_dim = "x".join(str(d) for d in s.dims if d is not None) or "1"
                min_parts.append(f"{s.name}:{sub_dim}")
                opt_parts.append(f"{s.name}:{sub_dim}")
                max_parts.append(f"{s.name}:{sub_dim}")
        elif "codeformer" in key_normalized or "ultra" in key_normalized:
            min_parts.append("w:1")
            opt_parts.append("w:1")
            max_parts.append("w:1")

        min_shapes = ",".join(min_parts)
        opt_shapes = ",".join(opt_parts)
        max_shapes = ",".join(max_parts)
        ns = f"_sp_{in_name}_{bs}_{h}x{w}"
        return ShapeProfile(min_shapes=min_shapes, opt_shapes=opt_shapes, max_shapes=max_shapes, namespace=ns)

    # 2. Derive from model graph specs
    if not specs:
        in_name = "x" if ("codeformer" in key_normalized or "ultra" in key_normalized) else "input"
        min_shapes = f"{in_name}:1x3x{size}x{size}"
        opt_shapes = f"{in_name}:{bs}x3x{size}x{size}"
        max_shapes = f"{in_name}:{bs}x3x{size}x{size}"
        if in_name == "x":
            min_shapes += ",w:1"
            opt_shapes += ",w:1"
            max_shapes += ",w:1"
        ns = f"_sp_{in_name}_{bs}_{size}"
        return ShapeProfile(min_shapes=min_shapes, opt_shapes=opt_shapes, max_shapes=max_shapes, namespace=ns)

    # 3. Dynamic graph inspection from parsed model input specs
    min_parts: List[str] = []
    opt_parts: List[str] = []
    max_parts: List[str] = []

    for spec in specs:
        if any(":" in spec.name for spec in specs):
            return None
        # Build 4D shape for image inputs, 1D/0D for weights
        if len(spec.dims) == 4:
            c = spec.dims[1] if spec.dims[1] is not None else 3
            h = spec.dims[2] if spec.dims[2] is not None else size
            w = spec.dims[3] if spec.dims[3] is not None else size
            min_parts.append(f"{spec.name}:1x{c}x{h}x{w}")
            opt_parts.append(f"{spec.name}:{bs}x{c}x{h}x{w}")
            max_parts.append(f"{spec.name}:{bs}x{c}x{h}x{w}")
        elif len(spec.dims) == 1:
            dim_val = spec.dims[0] if spec.dims[0] is not None else 1
            min_parts.append(f"{spec.name}:{dim_val}")
            opt_parts.append(f"{spec.name}:{dim_val}")
            max_parts.append(f"{spec.name}:{dim_val}")
        elif len(spec.dims) == 0:
            min_parts.append(f"{spec.name}:1")
            opt_parts.append(f"{spec.name}:1")
            max_parts.append(f"{spec.name}:1")
        else:
            resolved_dims = [str(d if d is not None else 1) for d in spec.dims]
            dim_s = "x".join(resolved_dims)
            min_parts.append(f"{spec.name}:{dim_s}")
            opt_parts.append(f"{spec.name}:{dim_s}")
            max_parts.append(f"{spec.name}:{dim_s}")

    if not min_parts:
        return None

    min_shapes = ",".join(min_parts)
    opt_shapes = ",".join(opt_parts)
    max_shapes = ",".join(max_parts)
    sanitized = re.sub(r"[^A-Za-z0-9]+", "_", opt_shapes)[-20:]
    namespace = f"_sp{sanitized}"
    return ShapeProfile(min_shapes=min_shapes, opt_shapes=opt_shapes, max_shapes=max_shapes, namespace=namespace)


def apply_shape_profile(
    providers: Sequence[Any],
    model_key: Optional[str] = None,
    model_path: Optional[str] = None,
    explicit_shape: Optional[Tuple[int, ...]] = None,
    batch_size: int = 1,
) -> List[Any]:
    """Return providers list with static TensorRT profile options attached.

    Ensures that any TensorrtExecutionProvider entry receives explicit static
    optimization profiles and an isolated persistent engine cache directory.
    Supports multi-profile if batch enhancement (batch_size > 1) is enabled.
    """
    try:
        profile = resolve_profile(model_key, model_path, explicit_shape, batch_size=batch_size)
    except Exception as exc:
        _LOGGER.debug("Failed to resolve shape profile: %s", exc)
        profile = None

    result: List[Any] = []
    for provider in (providers or []):
        if isinstance(provider, (tuple, list)) and len(provider) == 2 and "tensorrt" in str(provider[0]).lower():
            name, opts = provider[0], dict(provider[1])
            if profile is not None:
                opts.update(profile.as_options())
                cache_dir = opts.get("trt_engine_cache_path")
                if cache_dir:
                    scoped = cache_dir + profile.namespace
                    try:
                        os.makedirs(scoped, exist_ok=True)
                        opts["trt_engine_cache_path"] = scoped
                        if opts.get("trt_timing_cache_path"):
                            opts["trt_timing_cache_path"] = scoped
                    except OSError:
                        pass
            result.append((name, opts))
        elif isinstance(provider, str) and "tensorrt" in provider.lower():
            # Convert bare string to configured tuple
            opts: Dict[str, Any] = {
                "trt_engine_cache_enable": True,
                "trt_timing_cache_enable": True,
            }
            if profile is not None:
                opts.update(profile.as_options())
            result.append((provider, opts))
        else:
            result.append(provider)

    return result
