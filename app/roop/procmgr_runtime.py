"""Runtime primitives shared by ProcessMgr and its mixins.

The GPU serialisation guard, the per-stage timing probe, the progress-bar
format and the pause gate. These sat in ProcessMgr.py, which meant any mixin
extracted from that file could not use them without an import cycle — so they
live one layer down, where both the class and its mixins can import them.

Moved verbatim from ProcessMgr.py; the tuning constants keep their env vars and
their original comments explaining the measured values behind them.
"""

import contextlib
import os
import time
from threading import Lock
from collections import defaultdict as _defaultdict

import roop.globals


# Serialises GPU inference across worker threads ONLY when required.
#
# onnxruntime's InferenceSession.run() is thread-safe for the CPU and CUDA
# execution providers, so multiple worker threads can run frames concurrently
# and actually use the GPU + CPU in parallel. The TensorRT EP's execution
# context is NOT thread-safe (concurrent enqueue corrupts the CUDA context →
# error 999), so for TensorRT we serialise GPU work with this lock.
#
# Net effect: CUDA/CPU → full multi-thread throughput; TensorRT → serialised
# (switch to the CUDA provider for parallelism).
_gpu_lock = Lock()


_PROFILE = os.environ.get('ROOP_PROFILE', '0') == '1'


# ── Identity-lock source veto ────────────────────────────────────────────────
# Guards against a tracked source being applied to the wrong face (people
# crossing, an ID switch, or an unselected bystander standing where a track
# was). Deliberately LOOSER than the face-distance threshold: this is a veto on
# clear mismatches, not a re-selection, so a blurred or turned frame of the
# right person still swaps. Same-person frames measured up to ~0.66 on a hard
# clip while different people sat at ~0.93-1.07, so 0.85 separates them.
_TRACK_VETO_DIST = float(os.environ.get('ROOP_TRACK_VETO', '0.85'))


# Reject when a DIFFERENT selected person explains the face this much better.
_TRACK_VETO_MARGIN = float(os.environ.get('ROOP_TRACK_VETO_MARGIN', '0.15'))


# ROOP_TRACK_VETO=0 disables the veto entirely (pre-fix behavior: a tracked
# source is applied wherever the spatial association points).
# Fraction of a track's frames that must overlap an already-assigned track of the
# same person before the track is treated as a genuinely concurrent second body
# (and so refused that person's source) rather than an occlusion handoff.
_TRACK_OVERLAP_FRAC = float(os.environ.get('ROOP_TRACK_OVERLAP_FRAC', '0.15'))


_prof_lock = Lock()


_prof_times = _defaultdict(float)


_prof_counts = _defaultdict(int)


@contextlib.contextmanager
def _prof(stage):
    if not _PROFILE:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        with _prof_lock:
            _prof_times[stage] += dt
            _prof_counts[stage] += 1


def _prof_report():
    if not _PROFILE or not _prof_times:
        return
    total = sum(_prof_times.values()) or 1.0
    print("\n==== STAGE TIMING (ROOP_PROFILE) — wall-clock summed across worker threads ====", flush=True)
    print(f"  {'stage':16s} {'total':>9s} {'share':>7s} {'calls':>8s} {'ms/call':>9s}", flush=True)
    for k in sorted(_prof_times, key=lambda x: -_prof_times[x]):
        t = _prof_times[k]
        c = _prof_counts[k]
        print(f"  {k:16s} {t:8.2f}s {100 * t / total:6.1f}% {c:8d} {1000 * t / max(c, 1):8.2f}", flush=True)
    print("=============================================================================\n", flush=True)


def _gpu_guard(pooled=False):
    """Return the GPU lock only when the active provider needs serialising
    (TensorRT); otherwise a no-op context so threads run concurrently.

    `pooled=True` marks a stage that leases from a pool of INDEPENDENT sessions /
    contexts (the swapper's SessionPool, the FaceAnalysis pool, or a mask
    SessionPool). Each lease hands one thread its own context, so the work is
    already safely concurrent and must NOT also take the global lock or it would
    re-serialise — return a no-op context instead. Callers pass pooled=True only
    when that pool actually exists, so this is safe regardless of which pool knob
    (ROOP_TRT_POOL for the swapper, ROOP_DETMASK_POOL for detect/mask) enabled it."""
    if pooled:
        return contextlib.nullcontext()
    needs_lock = any('tensorrt' in str(p).lower() for p in roop.globals.execution_providers)
    return _gpu_lock if needs_lock else contextlib.nullcontext()


# ANSI escape codes for terminal coloring
COLOR_RESET = "\033[0m"


COLOR_ACCENT = "\033[38;5;205m"  # Pink/Red matching UI #E94560


COLOR_CYAN = "\033[36m"          # Cyan for counts


COLOR_GREEN = "\033[32m"         # Green for times


COLOR_GRAY = "\033[90m"          # Gray for separators


COLOR_YELLOW = "\033[33m"        # Yellow for stats


PROGRESS_BAR_FORMAT = (
    f"{COLOR_ACCENT}{{desc}}{COLOR_RESET}: "
    f"{COLOR_GRAY}|{{bar}}|{COLOR_RESET} "
    f"{COLOR_CYAN}{{n_fmt}}/{{total_fmt}}{COLOR_RESET} "
    f"[{COLOR_GREEN}{{elapsed}}{COLOR_RESET}<{COLOR_GREEN}{{remaining}}{COLOR_RESET}, "
    f"{COLOR_YELLOW}{{rate_fmt}}{COLOR_RESET}{{postfix}}]"
)


def wait_while_paused():
    """Block while a pause has been requested so processing can later resume
    from the exact same frame. Returns immediately if a stop was requested
    instead (roop.globals.processing == False), so abort always wins."""
    while getattr(roop.globals, 'pause', False) and roop.globals.processing:
        time.sleep(0.1)


