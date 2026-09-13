"""Centralized Model Lifecycle Manager for Dynamic VRAM Guard and Model Offloading.

Provides:
1. Dynamic VRAM monitoring and safety guards (threshold: 1.5 GB free).
2. Sequential offloading or migration of inactive models to CPU when VRAM is constrained.
3. Try/except guards around inference runs catching `torch.cuda.OutOfMemoryError`
   and automatic batch-size reduction fallbacks.
"""

from __future__ import annotations

import gc
import logging
import os
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

_LOGGER = logging.getLogger(__name__)

# Default free VRAM threshold in GB below which inactive models are offloaded
DEFAULT_VRAM_THRESHOLD_GB = 1.5


class ModelMetadata:
    """Metadata descriptor for a tracked neural model / processor."""

    def __init__(
        self,
        name: str,
        category: str = "general",
        is_loaded: bool = True,
        is_active: bool = False,
        last_used: float = 0.0,
        device: str = "cuda",
    ):
        self.name = name
        self.category = category
        self.is_loaded = is_loaded
        self.is_active = is_active
        self.last_used = last_used
        self.device = device

    def __repr__(self) -> str:
        return (
            f"ModelMetadata(name={self.name!r}, category={self.category!r}, "
            f"loaded={self.is_loaded}, active={self.is_active}, device={self.device!r})"
        )


class ModelLifecycleManager:
    """Centralized manager tracking model memory residency, GPU allocation,

    and dynamic offloading across combined pipelines (e.g. Swapper + Enhancers).
    """

    _instance: Optional[ModelLifecycleManager] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs) -> ModelLifecycleManager:
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, vram_threshold_gb: Optional[float] = None) -> None:
        if getattr(self, "_initialized", False):
            if vram_threshold_gb is not None:
                self._default_threshold_gb = float(vram_threshold_gb)
            return
        self._registry: Dict[str, Dict[str, Any]] = {}
        self._registry_lock = threading.RLock()
        self._active_models: Set[str] = set()
        self._state_lock = threading.RLock()
        self._default_threshold_gb = (
            float(vram_threshold_gb)
            if vram_threshold_gb is not None
            else float(os.environ.get("ROOP_VRAM_THRESHOLD_GB", str(DEFAULT_VRAM_THRESHOLD_GB)))
        )
        self._initialized = True
        _LOGGER.info(
            "ModelLifecycleManager initialized (threshold=%.2f GB)",
            self._default_threshold_gb,
        )

    # ── VRAM Introspection ───────────────────────────────────────────────────

    @staticmethod
    def get_device_id() -> int:
        """Return the active CUDA device index."""
        try:
            import roop.globals
            return int(getattr(roop.globals, "cuda_device_id", 0) or 0)
        except Exception:
            return 0

    @classmethod
    def get_free_vram_gb(cls, device_id: Optional[int] = None) -> float:
        """Return free GPU memory in gigabytes (0.0 if CUDA unavailable)."""
        try:
            import torch
            if not torch.cuda.is_available():
                return 0.0
            idx = cls.get_device_id() if device_id is None else device_id
            free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
            return float(free_bytes) / (1024.0 ** 3)
        except Exception as exc:
            _LOGGER.debug("Could not query GPU memory info: %s", exc)
            return 0.0

    @classmethod
    def get_total_vram_gb(cls, device_id: Optional[int] = None) -> float:
        """Return total GPU memory in gigabytes (0.0 if CUDA unavailable)."""
        try:
            import torch
            if not torch.cuda.is_available():
                return 0.0
            idx = cls.get_device_id() if device_id is None else device_id
            props = torch.cuda.get_device_properties(idx)
            return float(props.total_memory) / (1024.0 ** 3)
        except Exception:
            return 0.0

    def is_vram_constrained(
        self, threshold_gb: Optional[float] = None, device_id: Optional[int] = None
    ) -> bool:
        """Check if free VRAM is below the given threshold (default 1.5 GB)."""
        limit = threshold_gb if threshold_gb is not None else self._default_threshold_gb
        free_gb = self.get_free_vram_gb(device_id)
        if free_gb <= 0.0:
            return False  # Non-CUDA or unable to inspect; do not trigger false alarm
        return free_gb < limit

    # ── Model Registration ───────────────────────────────────────────────────

    def register_model(
        self,
        name: str,
        category: Union[str, Callable[[], Any]] = "general",
        offload_fn: Optional[Callable[[], Any]] = None,
        reload_fn: Optional[Callable[[], Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        unload_cb: Optional[Callable[[], Any]] = None,
        device: str = "cuda",
        **kwargs,
    ) -> None:
        """Register a model or processor with offload/reload hooks."""
        actual_category = "general"
        actual_offload = offload_fn

        if callable(category) and actual_offload is None:
            actual_offload = category
            actual_category = kwargs.get("cat", "general")
        elif isinstance(category, str):
            actual_category = category

        if unload_cb is not None and actual_offload is None:
            actual_offload = unload_cb

        with self._registry_lock:
            self._registry[name] = {
                "name": name,
                "category": actual_category,
                "offload_fn": actual_offload,
                "reload_fn": reload_fn,
                "metadata": metadata or {},
                "device": device,
                "is_offloaded": False,
                "last_active": time.monotonic(),
            }
            _LOGGER.debug("Registered model in lifecycle manager: %s (%s)", name, actual_category)

    def unregister_model(self, name: str) -> None:
        """Unregister a model from tracking."""
        with self._registry_lock:
            self._registry.pop(name, None)
            self._active_models.discard(name)

    def set_unloaded(self, name: str) -> None:
        """Mark a model as unloaded."""
        with self._registry_lock:
            entry = self._registry.get(name)
            if entry:
                entry["is_offloaded"] = True

    def touch(self, name: str) -> None:
        """Update last_active timestamp for model."""
        with self._registry_lock:
            entry = self._registry.get(name)
            if entry:
                entry["last_active"] = time.monotonic()

    def get_model_meta(self, name: str) -> Optional[ModelMetadata]:
        """Get model metadata object."""
        with self._registry_lock:
            entry = self._registry.get(name)
            if entry is None:
                return None
            return ModelMetadata(
                name=entry["name"],
                category=entry["category"],
                is_loaded=not entry.get("is_offloaded", False),
                is_active=name in self._active_models,
                last_used=entry.get("last_active", 0.0),
                device=entry.get("device", "cuda"),
            )

    def mark_active(self, name: str) -> None:
        """Mark a model as currently executing inference."""
        with self._state_lock:
            self._active_models.add(name)
            with self._registry_lock:
                entry = self._registry.get(name)
                if entry:
                    entry["last_active"] = time.monotonic()

    def mark_inactive(self, name: str) -> None:
        """Mark a model as having finished inference."""
        with self._state_lock:
            self._active_models.discard(name)

    # ── Dynamic Offloading ───────────────────────────────────────────────────

    def ensure_vram(
        self,
        required_gb: Optional[float] = None,
        exclude: Optional[Union[str, Set[str], List[str]]] = None,
        target_category: Optional[str] = None,
    ) -> bool:
        """Ensure at least `required_gb` free VRAM (default 1.5 GB).

        Sequentially unloads or offloads inactive model weights to host CPU memory
        until free VRAM is above the threshold or all candidates are exhausted.
        Returns True if target threshold is satisfied.
        """
        threshold = required_gb if required_gb is not None else self._default_threshold_gb
        current_free = self.get_free_vram_gb()
        if current_free <= 0.0 or current_free >= threshold:
            return True

        _LOGGER.warning(
            "[ModelLifecycleManager] Low VRAM detected (%.2f GB free < %.2f GB threshold). "
            "Initiating sequential model offload/migration...",
            current_free,
            threshold,
        )

        exclude_set: Set[str] = set()
        if exclude is not None:
            if isinstance(exclude, str):
                exclude_set = {exclude}
            else:
                exclude_set = set(exclude)

        candidates = []
        with self._registry_lock, self._state_lock:
            for name, entry in self._registry.items():
                if name in exclude_set:
                    continue
                if name in self._active_models:
                    continue
                if entry.get("is_offloaded", False):
                    continue
                if target_category and entry.get("category") != target_category:
                    continue
                candidates.append((entry.get("last_active", 0.0), name, entry))

        # Offload LRU (least recently used) first
        candidates.sort(key=lambda item: item[0])

        for _, name, entry in candidates:
            if not self.is_vram_constrained(threshold):
                break
            offload_fn = entry.get("offload_fn")
            if callable(offload_fn):
                try:
                    _LOGGER.info(
                        "[ModelLifecycleManager] Offloading inactive model '%s' (%s) to CPU memory",
                        name,
                        entry.get("category"),
                    )
                    offload_fn()
                    entry["is_offloaded"] = True
                    self._clear_gpu_caches()
                except Exception as err:
                    _LOGGER.error(
                        "[ModelLifecycleManager] Failed to offload model '%s': %s",
                        name,
                        err,
                        exc_info=True,
                    )

        final_free = self.get_free_vram_gb()
        _LOGGER.info(
            "[ModelLifecycleManager] VRAM offloading complete: %.2f GB free",
            final_free,
        )
        return final_free >= threshold or final_free >= current_free

    def release_all(self) -> None:
        """Release all registered models and clear GPU memory."""
        with self._registry_lock:
            for name, entry in list(self._registry.items()):
                offload_fn = entry.get("offload_fn")
                if callable(offload_fn):
                    try:
                        offload_fn()
                        entry["is_offloaded"] = True
                    except Exception as exc:
                        _LOGGER.debug("Error during release of '%s': %s", name, exc)
        self._active_models.clear()
        self._clear_gpu_caches()

    @staticmethod
    def _clear_gpu_caches() -> None:
        """Garbage collect and empty Torch CUDA cache."""
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                idx = ModelLifecycleManager.get_device_id()
                with torch.cuda.device(idx):
                    torch.cuda.empty_cache()
                    if hasattr(torch.cuda, "ipc_collect"):
                        torch.cuda.ipc_collect()
        except Exception:
            pass

    # ── Safe Execution & Batch Halving ───────────────────────────────────────

    @contextmanager
    def execution_guard(
        self,
        name: str,
        required_gb: Optional[float] = None,
        retry_on_oom: bool = True,
    ):
        """Context manager guarding inference runs:

        1. Ensures VRAM headroom before execution by offloading inactive models.
        2. Marks model active during execution.
        3. Catches OutOfMemoryError and purges GPU caches.
        """
        req = required_gb if required_gb is not None else self._default_threshold_gb
        self.ensure_vram(required_gb=req, exclude=name)
        self.mark_active(name)
        try:
            yield
        except Exception as exc:
            if self._is_oom_exception(exc):
                _LOGGER.error(
                    "[ModelLifecycleManager] OOM encountered during execution of '%s'. Purging caches...",
                    name,
                )
                self._clear_gpu_caches()
                self.ensure_vram(required_gb=0.1, exclude=name)
            raise
        finally:
            self.mark_inactive(name)

    @classmethod
    def _is_oom_exception(cls, exc: BaseException) -> bool:
        """Check if an exception is a CUDA/Torch OOM error."""
        try:
            import torch
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                return True
        except Exception:
            pass
        msg = str(exc).lower()
        return "out of memory" in msg or "cuda error: out of memory" in msg or "cuda oom" in msg

    @classmethod
    def run_with_batch_fallback(
        cls,
        arg1: Any,
        arg2: Any = None,
        fallback_sequential_fn: Optional[Callable[[List[Any]], List[Any]]] = None,
        model_name: str = "batch_worker",
        initial_batch_size: Optional[int] = None,
        min_batch_size: int = 1,
    ) -> List[Any]:
        """Execute a batch with automatic recursive batch halving upon OOM errors."""
        if callable(arg1) and not callable(arg2):
            process_batch_fn = arg1
            items = list(arg2 or [])
        else:
            items = list(arg1 or [])
            process_batch_fn = arg2

        if not items:
            return []

        # If initial_batch_size is specified and smaller than items, chunk first
        if initial_batch_size is not None and initial_batch_size < len(items):
            out = []
            for i in range(0, len(items), initial_batch_size):
                chunk = items[i : i + initial_batch_size]
                out.extend(
                    cls.run_with_batch_fallback(
                        chunk,
                        process_batch_fn,
                        fallback_sequential_fn=fallback_sequential_fn,
                        model_name=model_name,
                        min_batch_size=min_batch_size,
                    )
                )
            return out

        try:
            return process_batch_fn(items)
        except Exception as exc:
            if not cls._is_oom_exception(exc):
                raise
            _LOGGER.warning(
                "[ModelLifecycleManager] Batch of size %d raised OOM in '%s'. "
                "Purging cache and splitting batch...",
                len(items),
                model_name,
            )
            cls._clear_gpu_caches()

            if len(items) <= min_batch_size:
                if fallback_sequential_fn is not None:
                    return fallback_sequential_fn(items)
                raise

            mid = max(1, len(items) // 2)
            left_chunk = items[:mid]
            right_chunk = items[mid:]

            out_left = cls.run_with_batch_fallback(
                left_chunk,
                process_batch_fn,
                fallback_sequential_fn=fallback_sequential_fn,
                model_name=model_name,
                min_batch_size=min_batch_size,
            )
            out_right = cls.run_with_batch_fallback(
                right_chunk,
                process_batch_fn,
                fallback_sequential_fn=fallback_sequential_fn,
                model_name=model_name,
                min_batch_size=min_batch_size,
            )
            return out_left + out_right


# Global singleton instance
model_lifecycle_manager = ModelLifecycleManager()
