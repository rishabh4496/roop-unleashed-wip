"""TensorRT asynchronous compilation events, engine cache monitoring, and UI status broadcasting.

During the initial TensorRT engine compilation (which takes 2 to 6 minutes),
this module emits periodic status events and logs through the backend API and
WebSocket/progress polling pipelines. This ensures that:
1. The React UI displays live compilation progress with elapsed timing.
2. Frontend HTTP and WebSocket connections do not time out or freeze.
3. The user has complete visibility into cold compilation versus warm cache hits.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional

_LOGGER = logging.getLogger(__name__)

# Active compilation event cache
_CURRENT_COMPILATION_EVENT: Optional[dict] = None


def get_compilation_event() -> Optional[dict]:
    """Retrieve the current or most recent TensorRT compilation event."""
    return _CURRENT_COMPILATION_EVENT


def normalize_model_label(label: str) -> str:
    """Normalize internal model names to canonical UI labels."""
    l_lower = (label or "").lower()
    if "gpen" in l_lower:
        return "GPEN-Realistic"
    if "ultra" in l_lower:
        return "UltraMax"
    if "codeformer" in l_lower:
        return "CodeFormer"
    if "restoreformer" in l_lower:
        return "RestoreFormer++"
    if "gfpgan" in l_lower:
        return "GFPGAN"
    return label or "Model"


def _broadcast_log(message: str, force: bool = True) -> None:
    """Safely forward log message to terminal and React UI console."""
    try:
        print(message, flush=True)
    except UnicodeEncodeError:
        try:
            # Fallback for Windows cp1252 consoles lacking emoji support
            print(message.encode("ascii", "replace").decode("ascii"), flush=True)
        except Exception:
            pass
    try:
        import sys
        for mod_name in ("api", "app.api"):
            api_mod = sys.modules.get(mod_name)
            if api_mod is not None and hasattr(api_mod, "_push_log"):
                api_mod._push_log(message, force=force)
                break
    except Exception:
        pass


def _broadcast_progress_desc(description: str) -> None:
    """Safely update live progress description polled by React frontend."""
    try:
        import sys
        for mod_name in ("api", "app.api"):
            api_mod = sys.modules.get(mod_name)
            if api_mod is not None and hasattr(api_mod, "_progress") and isinstance(api_mod._progress, dict):
                api_mod._progress["desc"] = description
                break
    except Exception:
        pass


def broadcast_compilation_event(event: dict) -> None:
    """Broadcast compilation event to REST polling state and WebSocket subscribers."""
    global _CURRENT_COMPILATION_EVENT
    _CURRENT_COMPILATION_EVENT = dict(event)
    try:
        import sys
        for mod_name in ("api", "app.api"):
            api_mod = sys.modules.get(mod_name)
            if api_mod is not None:
                if hasattr(api_mod, "_update_compilation_state"):
                    api_mod._update_compilation_state(event)
                elif hasattr(api_mod, "_progress") and isinstance(api_mod._progress, dict):
                    api_mod._progress["compilation"] = event
                    if event.get("status") == "compiling_engine":
                        api_mod._progress["compiling_engine"] = True
    except Exception:
        pass


class TensorRTCompilationMonitor:
    """Context manager and heartbeat monitor for TensorRT engine builds."""

    def __init__(self, label: str, cache_dir: Optional[str] = None, interval_sec: float = 4.0):
        self.label = label
        self.model_label = normalize_model_label(label)
        self.cache_dir = cache_dir or "models/trt_cache"
        self.interval_sec = interval_sec
        self.start_time = 0.0
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.wait(self.interval_sec):
            elapsed = time.time() - self.start_time
            msg = f"⏳ [TensorRT] Compiling {self.model_label} engine... ({int(elapsed)}s elapsed, please wait)"
            _broadcast_progress_desc(f"[TensorRT] Compiling {self.model_label}... ({int(elapsed)}s)")
            
            # Emit live compilation heartbeat event
            event = {
                "status": "compiling_engine",
                "model": self.model_label,
                "provider": "TensorRT",
                "estimated_time": "2-4 minutes",
                "elapsed_sec": int(elapsed),
            }
            broadcast_compilation_event(event)

            # Periodically log every ~16 seconds to console
            if int(elapsed) % 16 < int(self.interval_sec) + 1:
                _broadcast_log(msg, force=False)

    def start(self) -> None:
        self.start_time = time.time()
        # Stage 4 exact required compilation event structure
        event = {
            "status": "compiling_engine",
            "model": self.model_label,
            "provider": "TensorRT",
            "estimated_time": "2-4 minutes",
            "elapsed_sec": 0,
        }
        broadcast_compilation_event(event)

        _broadcast_log(
            f"⚡ [TensorRT] Compiling engine for '{self.model_label}'... "
            f"First-time compilation typically takes 2-4 minutes. "
            f"Cache target: {self.cache_dir}",
            force=True,
        )
        _broadcast_progress_desc(f"[TensorRT] Compiling {self.model_label} engine...")
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._heartbeat_loop, daemon=True, name=f"trt_compile_{self.label}")
        self._thread.start()

    def finish(self, success: bool = True, error_msg: Optional[str] = None) -> float:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        elapsed = time.time() - self.start_time

        if success:
            event = {
                "status": "engine_ready",
                "model": self.model_label,
                "provider": "TensorRT",
                "elapsed_sec": round(elapsed, 1),
            }
            broadcast_compilation_event(event)
            _broadcast_log(
                f"✓ [TensorRT] '{self.model_label}' engine compilation completed in {elapsed:.1f}s. Engine cached.",
                force=True,
            )
            _broadcast_progress_desc(f"[TensorRT] {self.model_label} ready")
        else:
            event = {
                "status": "compilation_failed",
                "model": self.model_label,
                "provider": "TensorRT",
                "elapsed_sec": round(elapsed, 1),
                "error": error_msg or "",
            }
            broadcast_compilation_event(event)
            _broadcast_log(
                f"⚠ [TensorRT] '{self.model_label}' engine build failed after {elapsed:.1f}s: {error_msg}. "
                f"Falling back to CUDAExecutionProvider without interrupting batch.",
                force=True,
            )
            _broadcast_progress_desc(f"[TensorRT] {self.model_label} failed -> CUDA fallback")
        return elapsed

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.finish(success=False, error_msg=str(exc_val))
        else:
            self.finish(success=True)
        return False
