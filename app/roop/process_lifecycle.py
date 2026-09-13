"""Centralized Process Lifecycle & Temporary Storage Garbage Collection Manager.

Ensures that clicking "Stop" or encountering an unhandled exception immediately:
1. Terminates all child process trees (using `taskkill /F /T` on Windows or `SIGTERM`/`SIGKILL` on POSIX).
2. Releases all ONNX/CUDA sessions to clear VRAM.
3. Safely deletes temporary frame caches and incomplete output files.
"""

from __future__ import annotations

import gc
import logging
import os
import shutil
import signal
import subprocess
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Set

_LOGGER = logging.getLogger(__name__)


class ProcessLifecycleManager:
    """Thread-safe registry and supervisor for subprocesses, temp directories, and output files."""

    _instance: Optional[ProcessLifecycleManager] = None
    _lock = threading.Lock()

    def __new__(cls) -> ProcessLifecycleManager:
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._processes: Dict[int, subprocess.Popen] = {}
        self._temp_dirs: Set[str] = set()
        self._incomplete_files: Set[str] = set()
        self._release_hooks: List[Callable[[], Any]] = []
        self._registry_lock = threading.RLock()
        self._is_terminating = False
        self._initialized = True
        _LOGGER.info("ProcessLifecycleManager initialized")

    # ── Subprocess Tracking ──────────────────────────────────────────────────

    def register_release_hook(self, hook: Callable[[], Any]) -> None:
        """Register a callback to run during teardown."""
        if callable(hook):
            with self._registry_lock:
                self._release_hooks.append(hook)

    @property
    def active_processes(self) -> List[subprocess.Popen]:
        with self._registry_lock:
            return list(self._processes.values())

    def register_process(
        self,
        proc: subprocess.Popen,
        description: str = "",
        label: str = "",
        **kwargs,
    ) -> None:
        """Register a spawned subprocess for lifecycle tracking."""
        desc = description or label or kwargs.get("name", "subprocess")
        if proc is None or not hasattr(proc, "pid"):
            return
        with self._registry_lock:
            self._processes[proc.pid] = proc
            _LOGGER.debug(
                "Registered process %d (%s) in lifecycle manager",
                proc.pid,
                desc,
            )

    def unregister_process(self, proc: subprocess.Popen) -> None:
        """Unregister a completed subprocess."""
        if proc is None or not hasattr(proc, "pid"):
            return
        with self._registry_lock:
            self._processes.pop(proc.pid, None)

    # ── Temporary Directory Tracking ─────────────────────────────────────────

    def register_temp_dir(self, dir_path: str) -> None:
        """Register a temporary frames or cache directory for safe garbage collection."""
        if not dir_path:
            return
        with self._registry_lock:
            self._temp_dirs.add(os.path.abspath(dir_path))

    def unregister_temp_dir(self, dir_path: str) -> None:
        """Unregister a directory once intentionally cleaned or processed."""
        if not dir_path:
            return
        with self._registry_lock:
            self._temp_dirs.discard(os.path.abspath(dir_path))

    # ── Incomplete Output File Tracking ──────────────────────────────────────

    def register_incomplete_file(self, file_path: str) -> None:
        """Register a destination file currently being written to."""
        if not file_path:
            return
        with self._registry_lock:
            self._incomplete_files.add(os.path.abspath(file_path))

    def unregister_incomplete_file(self, file_path: str) -> None:
        """Unregister a destination file once finalized successfully."""
        if not file_path:
            return
        with self._registry_lock:
            self._incomplete_files.discard(os.path.abspath(file_path))

    # ── Master Termination and Garbage Collection Routine ────────────────────

    def terminate_all(self, reason: str = "stop") -> None:
        """Immediately terminate all child process trees, release VRAM sessions,

        and purge temporary frame caches and incomplete output files.
        """
        with self._registry_lock:
            if self._is_terminating:
                return
            self._is_terminating = True

        _LOGGER.warning(
            "[ProcessLifecycleManager] Emergency teardown triggered (reason='%s')",
            reason,
        )

        try:
            # 1. Terminate all tracked child process trees
            self._terminate_process_trees()

            # 2. Release ONNX and CUDA sessions to clear VRAM
            self._release_gpu_sessions()

            # 3. Clean temporary frame caches and delete incomplete output files
            self._clean_temporary_storage()

        finally:
            with self._registry_lock:
                self._is_terminating = False
            _LOGGER.info("[ProcessLifecycleManager] Teardown and cleanup completed.")

    def _terminate_process_trees(self) -> None:
        """Kill every active child process and its descendants."""
        with self._registry_lock:
            active_procs = list(self._processes.values())
            self._processes.clear()

        for proc in active_procs:
            if proc.poll() is not None:
                continue
            pid = proc.pid
            _LOGGER.info("[ProcessLifecycleManager] Terminating process tree for PID %d", pid)
            try:
                if os.name == "nt":
                    # taskkill /F /T terminates the specified process and any child processes started by it
                    cmd = ["taskkill", "/F", "/T", "/PID", str(pid)]
                    subprocess.run(
                        cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=5.0,
                        check=False,
                    )
                else:
                    # POSIX: Send SIGTERM to process group, followed by SIGKILL if needed
                    try:
                        pgid = os.getpgid(pid)
                        os.killpg(pgid, signal.SIGTERM)
                    except (OSError, ProcessLookupError):
                        try:
                            proc.terminate()
                        except Exception:
                            pass
            except Exception as exc:
                _LOGGER.error(
                    "[ProcessLifecycleManager] Error terminating PID %d: %s",
                    pid,
                    exc,
                )
                try:
                    proc.kill()
                except Exception:
                    pass

    def _release_gpu_sessions(self) -> None:
        """Safely release all ONNX/CUDA sessions and purge VRAM."""
        _LOGGER.info("[ProcessLifecycleManager] Releasing GPU sessions and clearing VRAM...")
        with self._registry_lock:
            hooks = list(self._release_hooks)
        for h in hooks:
            try:
                h()
            except Exception as exc:
                _LOGGER.debug("Error in release hook: %s", exc)

        try:
            from roop.model_lifecycle import model_lifecycle_manager
            model_lifecycle_manager.release_all()
        except Exception as exc:
            _LOGGER.debug("ModelLifecycleManager release error: %s", exc)

        try:
            from roop import core
            core.release_resources()
        except Exception as exc:
            _LOGGER.debug("core.release_resources error: %s", exc)

        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
        except Exception:
            pass

    def _clean_temporary_storage(self) -> None:
        """Delete temporary frame caches and incomplete output files."""
        with self._registry_lock:
            dirs_to_clean = list(self._temp_dirs)
            files_to_clean = list(self._incomplete_files)
            self._temp_dirs.clear()
            self._incomplete_files.clear()

        # Delete temp directories
        for dir_path in dirs_to_clean:
            if os.path.isdir(dir_path):
                _LOGGER.info("[ProcessLifecycleManager] Purging temp directory: %s", dir_path)
                try:
                    shutil.rmtree(dir_path, ignore_errors=True)
                except Exception as exc:
                    _LOGGER.error("Failed to remove temp dir %s: %s", dir_path, exc)

        # Delete incomplete output files
        for file_path in files_to_clean:
            if os.path.isfile(file_path):
                _LOGGER.info(
                    "[ProcessLifecycleManager] Removing incomplete output file: %s",
                    file_path,
                )
                for attempt in range(5):
                    try:
                        os.remove(file_path)
                        break
                    except OSError:
                        time.sleep(0.2)


# Global singleton lifecycle manager instance
process_lifecycle_manager = ProcessLifecycleManager()
