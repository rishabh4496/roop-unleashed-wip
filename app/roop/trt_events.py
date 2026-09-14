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


def _broadcast_log(message: str, force: bool = True) -> None:
    """Safely forward log message to terminal and React UI console."""
    print(message, flush=True)
    try:
        from api import _push_log
        _push_log(message, force=force)
    except Exception:
        pass


def _broadcast_progress_desc(description: str) -> None:
    """Safely update live progress description polled by React frontend."""
    try:
        from api import _progress
        if isinstance(_progress, dict):
            _progress["desc"] = description
    except Exception:
        pass


class TensorRTCompilationMonitor:
    """Context manager and heartbeat monitor for TensorRT engine builds."""

    def __init__(self, label: str, cache_dir: Optional[str] = None, interval_sec: float = 4.0):
        self.label = label
        self.cache_dir = cache_dir or "models/trt_cache"
        self.interval_sec = interval_sec
        self.start_time = 0.0
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.wait(self.interval_sec):
            elapsed = time.time() - self.start_time
            msg = f"⏳ [TensorRT] Compiling {self.label} engine... ({int(elapsed)}s elapsed, please wait)"
            _broadcast_progress_desc(f"[TensorRT] Compiling {self.label}... ({int(elapsed)}s)")
            # Periodically log every ~16 seconds to console
            if int(elapsed) % 16 < int(self.interval_sec) + 1:
                _broadcast_log(msg, force=False)

    def start(self) -> None:
        self.start_time = time.time()
        _broadcast_log(
            f"⚡ [TensorRT] Compiling engine for '{self.label}'... "
            f"First-time compilation typically takes 2-5 minutes. "
            f"Cache target: {self.cache_dir}",
            force=True,
        )
        _broadcast_progress_desc(f"[TensorRT] Compiling {self.label} engine...")
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._heartbeat_loop, daemon=True, name=f"trt_compile_{self.label}")
        self._thread.start()

    def finish(self, success: bool = True, error_msg: Optional[str] = None) -> float:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        elapsed = time.time() - self.start_time

        if success:
            _broadcast_log(
                f"✓ [TensorRT] '{self.label}' engine compilation completed in {elapsed:.1f}s. Engine cached.",
                force=True,
            )
            _broadcast_progress_desc(f"[TensorRT] {self.label} ready")
        else:
            _broadcast_log(
                f"⚠ [TensorRT] '{self.label}' engine build failed after {elapsed:.1f}s: {error_msg}. "
                f"Falling back to CUDAExecutionProvider without interrupting batch.",
                force=True,
            )
            _broadcast_progress_desc(f"[TensorRT] {self.label} failed -> CUDA fallback")
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
