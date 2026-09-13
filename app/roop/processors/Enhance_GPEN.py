"""GPEN face restoration with fail-fast GPU execution and reusable I/O buffers.

The ONNX GPEN exports used by roop have a fixed batch dimension of one.  GPU
throughput therefore comes from independent TensorRT execution contexts (one
per worker slot), not from concatenating face crops into a batch that the graph
cannot accept.
"""

from __future__ import annotations

import gc
import os
import threading
from collections.abc import Iterable
from typing import Any

import cv2
import numpy as np
import onnxruntime

import roop.globals
from roop import session_pool
from roop.processors.enhance_common import (
    enhance_gpen_ultimate,
    is_usable,
    sized,
)
from roop.typing import Face, FaceSet, Frame
from roop.utilities import conditional_download, resolve_relative_path
from roop.model_lifecycle import model_lifecycle_manager
from roop.provider_fallback import is_trt_error, build_cuda_fallback_providers

_GPU_PROVIDERS = frozenset({
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
})


def _provider_name(provider: Any) -> str:
    if isinstance(provider, (tuple, list)):
        return str(provider[0])
    return str(provider)


def _provider_names(providers: Iterable[Any]) -> list[str]:
    return [_provider_name(provider) for provider in (providers or [])]


def _has_gpu_provider(providers: Iterable[Any]) -> bool:
    return any(name in _GPU_PROVIDERS for name in _provider_names(providers))


def _provider_device_id(providers: Iterable[Any]) -> int:
    for provider in providers or []:
        if not isinstance(provider, (tuple, list)) or len(provider) != 2:
            continue
        if _provider_name(provider) not in _GPU_PROVIDERS:
            continue
        try:
            return int(dict(provider[1]).get("device_id", 0))
        except (TypeError, ValueError):
            return 0
    return 0


def _session_options(strict_gpu: bool):
    """Make unsupported CPU node placement a session-construction error."""
    options_type = getattr(onnxruntime, "SessionOptions", None)
    if options_type is None:  # Minimal test doubles.
        return None
    options = options_type()
    # Warnings remain visible while normal ORT graph chatter stays quiet.
    options.log_severity_level = 2
    if strict_gpu:
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    return options


def create_gpen_session(
    model_path: str,
    providers: Iterable[Any],
    device_name: str,
    label: str = "GPEN",
):
    """Create and validate a GPEN ONNX Runtime session.

    If the selected configuration contains a GPU provider, CPU node placement
    is forbidden.  Missing CUDA/cuDNN/TensorRT libraries and unsupported nodes
    consequently fail here with a descriptive error instead of degrading an
    entire render to CPU speed.
    """
    requested = list(providers or [])
    requested_names = _provider_names(requested)
    strict_gpu = device_name == "cuda" or _has_gpu_provider(requested)

    available_fn = getattr(onnxruntime, "get_available_providers", None)
    available = list(available_fn()) if callable(available_fn) else requested_names

    if strict_gpu:
        # Preserve the user's TensorRT/CUDA ordering, but never put the CPU EP in
        # a strict GPU session. ORT rejects disable_cpu_ep_fallback when CPU is
        # also explicitly registered.
        selected = [
            provider for provider in requested
            if _provider_name(provider) in _GPU_PROVIDERS
            and _provider_name(provider) in available
        ]
        if not selected:
            raise RuntimeError(
                f"[{label}] GPU acceleration was requested, but none of "
                f"{requested_names!r} is available. ONNX Runtime reports "
                f"{available!r}. Install a matching onnxruntime-gpu/CUDA/cuDNN "
                "stack or explicitly select the CPU provider."
            )
    else:
        selected = [
            provider for provider in requested
            if _provider_name(provider) in available
        ] or ["CPUExecutionProvider"]

    selected_names = _provider_names(selected)
    if strict_gpu and selected_names[0] not in _GPU_PROVIDERS:
        raise RuntimeError(
            f"[{label}] invalid execution-provider order {selected_names!r}; "
            "the GPU provider must be index 0."
        )

    options = _session_options(strict_gpu)
    try:
        session = onnxruntime.InferenceSession(
            model_path,
            options,
            providers=selected,
        )
    except Exception as exc:
        if strict_gpu and is_trt_error(exc):
            print(f"[{label}] TensorRT engine creation failed ({exc}). Falling back to CUDAExecutionProvider...")
            cuda_providers = build_cuda_fallback_providers()
            try:
                session = onnxruntime.InferenceSession(
                    model_path,
                    options,
                    providers=cuda_providers,
                )
            except Exception as cuda_exc:
                exc = cuda_exc
            else:
                return session
        mode = "strict GPU" if strict_gpu else "CPU"
        fallback_note = (
            "No CPU fallback was allowed."
            if strict_gpu
            else "Explicit CPU mode was selected."
        )
        raise RuntimeError(
            f"[{label}] failed to create {mode} ONNX session for "
            f"{model_path!r}; requested={selected_names!r}, "
            f"available={available!r}. {fallback_note} "
            f"Original error: {exc}"
        ) from exc

    disable_runtime_fallback = getattr(session, "disable_fallback", None)
    if strict_gpu and callable(disable_runtime_fallback):
        disable_runtime_fallback()

    get_providers = getattr(session, "get_providers", None)
    active = list(get_providers()) if callable(get_providers) else selected_names
    if strict_gpu and (not active or active[0] not in _GPU_PROVIDERS):
        raise RuntimeError(
            f"[{label}] ONNX Runtime initialized unexpected providers "
            f"{active!r}; expected a GPU provider at index 0. "
            "Refusing silent CPU degradation."
        )

    print(
        f"[{label}] requested providers={selected_names}; active={active}; "
        f"CPU node fallback={'disabled' if strict_gpu else 'allowed'}"
    )
    return session


def _fp32_trt_providers(providers):
    """Force large GPEN graphs to a separate FP32 TensorRT engine cache."""
    if os.environ.get("ROOP_GPEN_FP16", "0") == "1":
        return providers
    patched = []
    for provider in providers:
        if (
            isinstance(provider, (tuple, list))
            and len(provider) == 2
            and "tensorrt" in str(provider[0]).lower()
        ):
            name, options = provider[0], dict(provider[1])
            options["trt_fp16_enable"] = False
            cache = options.get("trt_engine_cache_path")
            if cache:
                fp32_cache = cache + "_gpen_fp32"
                os.makedirs(fp32_cache, exist_ok=True)
                options["trt_engine_cache_path"] = fp32_cache
            patched.append((name, options))
        else:
            patched.append(provider)
    return patched


GPEN_MODELS = {
    256: {
        "file": "gpen_bfr_256.onnx",
        "url": (
            "https://huggingface.co/facefusion/models-3.0.0/resolve/main/"
            "gpen_bfr_256.onnx"
        ),
    },
    512: {
        "file": "GPEN-BFR-512.onnx",
        "url": (
            "https://huggingface.co/countfloyd/deepfake/resolve/main/"
            "GPEN-BFR-512.onnx"
        ),
    },
    1024: {
        "file": "gpen_bfr_1024.onnx",
        "url": (
            "https://huggingface.co/facefusion/models-3.0.0/resolve/main/"
            "gpen_bfr_1024.onnx"
        ),
    },
    2048: {
        "file": "gpen_bfr_2048.onnx",
        "url": (
            "https://huggingface.co/facefusion/models-3.0.0/resolve/main/"
            "gpen_bfr_2048.onnx"
        ),
    },
}


def _numpy_dtype(ort_type: str):
    return np.float16 if "float16" in str(ort_type).lower() else np.float32


def _static_shape(value_info) -> tuple[int, ...] | None:
    shape = getattr(value_info, "shape", None)
    if not shape:
        return None
    if any(not isinstance(dim, int) or dim <= 0 for dim in shape):
        return None
    return tuple(shape)


class _GPENSlot:
    """One session/context and its private reusable device buffers."""

    def __init__(
        self,
        session,
        input_name: str,
        output_name: str,
        device_name: str,
        providers: Iterable[Any],
    ):
        self.session = session
        self.input_name = input_name
        self.output_name = output_name
        self.device_name = device_name
        self.device_id = _provider_device_id(providers)
        self.serialized = any(
            "tensorrt" in name.lower() for name in _provider_names(providers)
        )
        self.lock = threading.Lock()
        self._thread_local = threading.local()
        self._can_reuse_device_buffers = False
        self.binding = None
        self.input_value = None
        self.output_value = None

        self.inputs = session.get_inputs()
        self.outputs = session.get_outputs()
        self.input_shape = _static_shape(self.inputs[0])
        self.output_shape = _static_shape(self.outputs[0])
        ort_value = getattr(onnxruntime, "OrtValue", None)
        self._can_reuse_device_buffers = (
            device_name == "cuda"
            and self.input_shape is not None
            and self.output_shape is not None
            and ort_value is not None
        )
        if self._can_reuse_device_buffers:
            resources = self._new_device_resources()
            if self.serialized:
                self.binding, self.input_value, self.output_value = resources
            else:
                # CUDA EP sessions can serve concurrent Run calls, but bindings
                # and their buffers are mutable. Keep one reusable set per
                # worker thread instead of turning I/O reuse into a new lock.
                self._thread_local.resources = resources
                self.binding, self.input_value, self.output_value = resources

    def _new_device_resources(self):
        ort_value = onnxruntime.OrtValue
        input_dtype = _numpy_dtype(
            getattr(self.inputs[0], "type", "tensor(float)")
        )
        output_dtype = _numpy_dtype(
            getattr(self.outputs[0], "type", "tensor(float)")
        )
        input_value = ort_value.ortvalue_from_shape_and_type(
            self.input_shape, input_dtype, "cuda", self.device_id
        )
        output_value = ort_value.ortvalue_from_shape_and_type(
            self.output_shape, output_dtype, "cuda", self.device_id
        )
        binding = self.session.io_binding()
        binding.bind_ortvalue_input(self.input_name, input_value)
        binding.bind_ortvalue_output(self.output_name, output_value)
        return binding, input_value, output_value

    @property
    def reuses_device_buffers(self) -> bool:
        return self._can_reuse_device_buffers

    def _run(self, input_array: np.ndarray) -> np.ndarray:
        if self._can_reuse_device_buffers:
            if self.serialized:
                binding = self.binding
                input_value = self.input_value
                output_value = self.output_value
            else:
                resources = getattr(self._thread_local, "resources", None)
                if resources is None:
                    resources = self._new_device_resources()
                    self._thread_local.resources = resources
                binding, input_value, output_value = resources

            input_value.update_inplace(input_array)
            self.session.run_with_iobinding(binding)
            sync = getattr(binding, "synchronize_outputs", None)
            if callable(sync):
                sync()
            return output_value.numpy()

        # CPU mode, dynamic graphs, and minimal test doubles use a private
        # per-call binding. This path is correct but performs pageable
        # host/device copies on GPU.
        binding = self.session.io_binding()
        binding.bind_cpu_input(self.input_name, input_array)
        binding.bind_output(self.output_name, self.device_name)
        self.session.run_with_iobinding(binding)
        sync = getattr(binding, "synchronize_outputs", None)
        if callable(sync):
            sync()
        return binding.copy_outputs_to_cpu()[0]

    def run(self, input_array: np.ndarray) -> np.ndarray:
        # A TensorRT execution context is not thread-safe. A pool lease makes
        # each pooled slot exclusive; the lock also protects direct callers and
        # the one-context fallback. CUDA EP uses thread-local mutable bindings
        # and does not take this lock.
        if self.serialized:
            with self.lock:
                return self._run(input_array)
        return self._run(input_array)


class Enhance_GPEN:
    plugin_options: dict | None = None
    model_gpen = None
    name = None
    devicename = None

    processorname = "gpen"
    type = "enhance"
    # Kept as a literal because the alignment contract test verifies that every
    # FFHQ-trained restorer declares its crop space in source.
    model_template = 'ffhq_512'

    def __init__(self):
        self.model_size = 512
        self.sessions: dict[int, Any] = {}
        self._slots: dict[int, _GPENSlot] = {}
        self._provider_fingerprint = None
        self.pool = None
        self._pool_model_size = None
        self.profile = None

    @staticmethod
    def _fingerprint(providers: Iterable[Any]) -> str:
        return repr(list(providers or []))

    def _build(self, model_size: int, providers, device_name: str) -> _GPENSlot:
        spec = GPEN_MODELS[model_size]
        model_dir = resolve_relative_path("../models")
        conditional_download(model_dir, [spec["url"]])
        model_path = os.path.join(model_dir, spec["file"])
        session = create_gpen_session(
            model_path,
            providers,
            device_name,
            label=f"GPEN-{model_size}",
        )
        slot = _GPENSlot(
            session,
            session.get_inputs()[0].name,
            session.get_outputs()[0].name,
            device_name,
            providers,
        )

        if device_name == "cuda" or _has_gpu_provider(providers):
            # Force lazy CUDA/TensorRT engine initialization now. Missing DLLs,
            # unsupported kernels, allocation failures, and non-finite FP16
            # output must abort before a long render begins.
            dummy = np.zeros((1, 3, model_size, model_size), dtype=np.float32)
            try:
                probe = slot.run(dummy)
            except Exception as exc:
                raise RuntimeError(
                    f"[GPEN-{model_size}] GPU warm-up inference failed. "
                    "Check CUDA/cuDNN/TensorRT compatibility and free VRAM. "
                    f"Original error: {exc}"
                ) from exc
            if not is_usable(probe):
                raise RuntimeError(
                    f"[GPEN-{model_size}] warm-up produced NaN/Inf output. "
                    "Use TensorRT FP32 for this model size or unset "
                    "ROOP_GPEN_FP16."
                )
            print(
                f"[GPEN-{model_size}] warm-up passed; "
                f"reusable CUDA I/O buffers={slot.reuses_device_buffers}"
            )
        else:
            print(f"[GPEN-{model_size}] CPU session ready (explicit CPU mode)")
        return slot

    def Initialize(self, plugin_options: dict):
        providers = list(roop.globals.execution_providers or [])
        fingerprint = self._fingerprint(providers)
        requested_device = plugin_options["devicename"].replace("mps", "cpu")

        if self.plugin_options is not None and (
            self.devicename != requested_device
            or self._provider_fingerprint != fingerprint
        ):
            self.Release()

        self.plugin_options = plugin_options
        self.profile = plugin_options.get("profile")
        self.devicename = requested_device
        self._provider_fingerprint = fingerprint

        size = int(plugin_options.get("size", 512))
        if size not in GPEN_MODELS:
            size = 512
        self.model_size = size

        model_providers = providers
        if size >= 1024:
            model_providers = _fp32_trt_providers(providers)

        if size not in self._slots:
            slot = self._build(size, model_providers, self.devicename)
            self._slots[size] = slot
            self.sessions[size] = slot.session

        primary = self._slots[size]
        self.model_gpen = primary.session
        self.name = primary.input_name
        self.output_name = primary.output_name

        # Register with ModelLifecycleManager for VRAM guard & offloading
        model_lifecycle_manager.register_model(
            name=f"gpen_{size}",
            unload_cb=self.Release,
            device="cuda" if self.devicename == "cuda" or _has_gpu_provider(model_providers) else "cpu"
        )

        # The fixed batch=1 model scales through independent TRT contexts.
        # Keep 1024/2048 single-context: their activation footprint is too large
        # for safe automatic multiplication on typical cards.
        wants_pool = (
            size <= 512
            and _has_gpu_provider(model_providers)
            and any("tensorrt" in name.lower() for name in _provider_names(model_providers))
            and session_pool.pooling_enabled()
        )
        if self.pool is not None and (
            not wants_pool or self._pool_model_size != size
        ):
            self.pool.release()
            self.pool = None
            self._pool_model_size = None

        if wants_pool and self.pool is None:
            requested_count = session_pool.pool_size()
            slots = [primary]
            failure = None
            for _ in range(requested_count - 1):
                try:
                    slots.append(self._build(size, model_providers, self.devicename))
                except Exception as exc:  # noqa: BLE001 - pool OOM/provider errors
                    failure = exc
                    break

            if len(slots) >= 2:
                self.pool = session_pool.SessionPool(
                    lambda index, resources=slots: resources[index],
                    len(slots),
                )
                self._pool_model_size = size
                print(
                    f"[GPEN-{size}] TensorRT context pool ready: "
                    f"{len(slots)}/{requested_count} slots"
                )
            if failure is not None:
                print(
                    f"[GPEN-{size}] stopped growing the TensorRT pool after "
                    f"{len(slots)} slot(s): {failure}"
                )
            if self.pool is None:
                # The validated primary remains strict-GPU and usable. This is a
                # throughput fallback, never a CPU execution fallback.
                print(
                    f"[GPEN-{size}] using one validated GPU context behind "
                    "the inference lock"
                )

    def Prepare(
        self,
        source_faceset: FaceSet,
        target_face: Face,
        temp_frame: Frame,
    ) -> dict[str, Any]:
        """CPU preprocessing; deliberately outside the global GPU lock."""
        input_size = temp_frame.shape[1]
        reference_bgr = temp_frame
        resized_bgr = cv2.resize(
            reference_bgr,
            (self.model_size, self.model_size),
            interpolation=cv2.INTER_CUBIC,
        )
        tensor = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        tensor *= 1.0 / 127.5
        tensor -= 1.0
        tensor = np.ascontiguousarray(tensor.transpose(2, 0, 1)[None, ...])
        return {
            "tensor": tensor,
            "input_size": input_size,
            "reference_bgr": reference_bgr,
            "fallback_bgr": resized_bgr,
            "target_face": target_face,
        }

    def Infer(self, prepared: dict[str, Any]) -> np.ndarray:
        """GPU-only portion; a pool lease isolates each TensorRT context."""
        if self.model_size not in self._slots or self._slots[self.model_size] is None:
            self.Initialize(self.plugin_options or {"devicename": self.devicename or "cuda", "size": self.model_size})

        with model_lifecycle_manager.execution_guard(f"gpen_{self.model_size}", required_gb=1.5):
            if self.pool is not None:
                with self.pool.lease() as slot:
                    return slot.run(prepared["tensor"])
            return self._slots[self.model_size].run(prepared["tensor"])

    def Finish(
        self,
        ort_output: np.ndarray,
        prepared: dict[str, Any],
        target_face: Face | None = None,
    ):
        """CPU postprocessing; deliberately outside the global GPU lock."""
        result = ort_output[0]
        if not is_usable(result):
            raise RuntimeError(
                f"[GPEN-{self.model_size}] inference produced NaN/Inf output. "
                "CPU fallback is disabled; for 1024/2048 use TensorRT FP32."
            )

        result = np.clip(result, -1.0, 1.0)
        result = (result + 1.0) * 127.5
        result = result.transpose(1, 2, 0)
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        result, scale_factor = sized(
            result.astype(np.uint8),
            prepared["input_size"],
        )

        if self.profile == "ultimate":
            reference = cv2.resize(
                prepared["reference_bgr"],
                (result.shape[1], result.shape[0]),
                interpolation=cv2.INTER_CUBIC,
            )
            finish_face = (
                target_face
                if target_face is not None
                else prepared["target_face"]
            )
            result = enhance_gpen_ultimate(
                result,
                reference,
                target_face=finish_face,
            )
        return result, scale_factor

    def Run(
        self,
        source_faceset: FaceSet,
        target_face: Face,
        temp_frame: Frame,
    ) -> Frame:
        """Compatibility wrapper for callers outside ProcessMgr."""
        prepared = self.Prepare(source_faceset, target_face, temp_frame)
        output = self.Infer(prepared)
        return self.Finish(output, prepared, target_face)

    def Release(self):
        if self.pool is not None:
            self.pool.release()
            self.pool = None
        self._pool_model_size = None
        self._slots.clear()
        self.sessions.clear()
        self.model_gpen = None
        model_lifecycle_manager.set_unloaded(f"gpen_{self.model_size}")
        gc.collect()
