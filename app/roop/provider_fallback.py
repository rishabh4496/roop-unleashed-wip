"""Graceful Execution Provider Fallback for ONNX Runtime & TensorRT.

Handles:
1. Engine compilation failures during `InferenceSession` construction by automatically
   downgrading to `CUDAExecutionProvider` (FP16) with an informative warning.
2. Dynamic input resolution rejections and shape verification failures during inference
   by immediately downgrading the session to `CUDAExecutionProvider` (FP16) and resuming
   execution without interrupting the batch.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import onnxruntime

_LOGGER = logging.getLogger(__name__)

# Lock ensuring thread-safe rebuild of sessions across worker threads
_FALLBACK_LOCK = threading.RLock()


def is_trt_provider(provider: Any) -> bool:
    """Check if a provider entry is TensorRT."""
    name = provider[0] if isinstance(provider, (tuple, list)) else provider
    return "tensorrt" in str(name).lower()


def is_trt_error(exc: BaseException) -> bool:
    """Check if an exception is related to TensorRT execution, compilation, or shape verification."""
    msg = str(exc).lower()
    keywords = (
        "tensorrt",
        "trt",
        "engine creation failed",
        "shape verification failed",
        "must not be dynamic",
        "invalid input shape",
        "unsupported dynamic",
        "not supported by tensorrt",
        "current shape:{",
        "cuda error 999",
        "failed to build cuda engine",
    )
    return any(k in msg for k in keywords)


def build_cuda_fallback_providers(
    original_providers: Optional[Sequence[Any]] = None,
    device_id: int = 0,
) -> List[Any]:
    """Return providers with TensorRT removed and CUDAExecutionProvider placed first with optimal FP16/HEURISTIC settings."""
    try:
        import roop.globals
        device_id = int(getattr(roop.globals, "cuda_device_id", device_id) or 0)
    except Exception:
        pass

    cuda_opts = {
        "device_id": device_id,
        "cudnn_conv_algo_search": "HEURISTIC",
        "do_copy_in_default_stream": True,
        "arena_extend_strategy": os.environ.get(
            "ROOP_CUDA_ARENA_STRATEGY", "kSameAsRequested"
        ),
    }

    result: List[Any] = [("CUDAExecutionProvider", cuda_opts)]

    if original_providers:
        for p in original_providers:
            if is_trt_provider(p):
                continue
            name = p[0] if isinstance(p, (tuple, list)) else p
            if name == "CUDAExecutionProvider":
                continue
            result.append(p)

    if "CPUExecutionProvider" not in [
        p[0] if isinstance(p, (tuple, list)) else p for p in result
    ]:
        result.append("CPUExecutionProvider")

    return result


class FallbackSessionResult(tuple):
    """A tuple subclass (session, downgraded: bool) that also acts directly as the session.

    This ensures full backwards- and forward-compatibility with both calling conventions:
    1. Direct assignment:
       session = create_fallback_session(...)
       session.get_inputs()
       session.run(...)
    2. Tuple unpacking:
       session, downgraded = create_fallback_session(...)
    """

    def __new__(cls, session: Any, downgraded: bool = False):
        return super().__new__(cls, (session, downgraded))

    @property
    def session(self) -> Any:
        return self[0]

    @property
    def downgraded(self) -> bool:
        return self[1]

    def __getattr__(self, name: str) -> Any:
        return getattr(self[0], name)

    def __repr__(self) -> str:
        return f"FallbackSessionResult(session={self[0]!r}, downgraded={self[1]!r})"


def create_fallback_session(
    model_arg: Union[str, bytes],
    sess_options: Optional[onnxruntime.SessionOptions] = None,
    providers: Optional[Sequence[Any]] = None,
    label: str = "model",
    session_name: Optional[str] = None,
    ort: Optional[Any] = None,
    **kwargs: Any,
) -> FallbackSessionResult:
    """Create an ONNX Runtime InferenceSession with automatic TensorRT-to-CUDA fallback.

    If TensorRT fails to compile, build an engine, or reject inputs, downgrades to
    CUDAExecutionProvider (FP16) and logs an informative warning.

    Returns:
        FallbackSessionResult(session, downgraded_flag)
    """
    effective_label = session_name or label
    requested_providers = list(providers or [])
    has_trt = any(is_trt_provider(p) for p in requested_providers)

    ort_impl = ort
    if ort_impl is None:
        try:
            import inspect
            caller_frame = inspect.currentframe().f_back
            if caller_frame and "onnxruntime" in caller_frame.f_globals:
                ort_impl = caller_frame.f_globals["onnxruntime"]
        except Exception:
            pass
    if ort_impl is None:
        ort_impl = onnxruntime

    try:
        session = ort_impl.InferenceSession(
            model_arg, sess_options, providers=requested_providers, **kwargs
        )
        return FallbackSessionResult(session, False)
    except Exception as exc:
        if has_trt or is_trt_error(exc):
            _LOGGER.warning(
                "[ProviderFallback] TensorRT compilation/initialization failed for '%s' (%s). "
                "Automatically downgrading session to CUDAExecutionProvider (FP16)...",
                effective_label,
                exc,
            )
            fallback_providers = build_cuda_fallback_providers(requested_providers)
            session = ort_impl.InferenceSession(
                model_arg, sess_options, providers=fallback_providers, **kwargs
            )
            print(
                f"[ProviderFallback] '{effective_label}' fell back from TensorRT to CUDAExecutionProvider (FP16)."
            )
            return FallbackSessionResult(session, True)
        raise


def safe_run_with_fallback(
    session: Any,
    feed: Optional[Dict[str, Any]] = None,
    model_arg: Optional[Union[str, bytes]] = None,
    sess_options: Optional[onnxruntime.SessionOptions] = None,
    label: str = "model",
    rebuild_fn: Optional[Any] = None,
    output_names: Optional[List[str]] = None,
    input_feed: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Any:
    """Execute inference session, automatically downgrading from TensorRT to CUDA (FP16)
    if dynamic input resolution rejection or runtime execution failure occurs.

    Supports both:
    1. Direct run with rebuild_fn: safe_run_with_fallback(session, rebuild_fn=..., output_names=..., input_feed=...)
    2. Model-arg run: safe_run_with_fallback(session, feed, model_arg, ...)
    """
    effective_feed = input_feed if input_feed is not None else feed
    if effective_feed is None:
        effective_feed = {}

    try:
        outs = session.run(output_names, effective_feed)
        return outs
    except Exception as exc:
        is_running_trt = False
        try:
            active_providers = session.get_providers()
            is_running_trt = any(is_trt_provider(p) for p in active_providers)
        except Exception:
            pass

        if not is_running_trt and not is_trt_error(exc):
            raise

        with _FALLBACK_LOCK:
            _LOGGER.warning(
                "[ProviderFallback] Runtime execution error under TensorRT for '%s' (%s). "
                "Downgrading to CUDAExecutionProvider (FP16) and resuming without interrupting batch...",
                label,
                exc,
            )
            print(
                f"[ProviderFallback] Runtime rejection under TensorRT for '{label}'; "
                "rebuilding on CUDAExecutionProvider (FP16) and resuming batch."
            )
            if rebuild_fn is not None:
                new_session = rebuild_fn()
                return new_session.run(output_names, effective_feed)
            elif model_arg is not None:
                try:
                    active_providers = session.get_providers()
                except Exception:
                    active_providers = None
                cuda_providers = build_cuda_fallback_providers(active_providers)
                new_session = onnxruntime.InferenceSession(
                    model_arg, sess_options, providers=cuda_providers
                )
                return new_session.run(output_names, effective_feed)
            else:
                raise
