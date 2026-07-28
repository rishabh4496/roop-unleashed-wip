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


# Absolute veto for the SINGLE-selected-person case, which _TRACK_VETO_DIST
# deliberately skips (see the reasoning at its use site: with nobody else's
# faceset to protect, an absolute gate can only take swaps away, and a profile
# or motion-blurred frame of the right person routinely exceeds 0.85).
#
# That reasoning holds for a lone subject, but it leaves one hole: if the
# TRACKER itself switches identity — two people cross, or one leaves and another
# stands where the track was — the track keeps its source and every face on it is
# swapped, with no identity check at swap time at all. That reads as the swap
# jumping to the wrong person for a run of frames.
#
# OFF by default (0) because it is the same class of change as the Re-ID gates
# reverted in f7cbdb6, which broke recognition on hard poses. Set it to catch
# UNAMBIGUOUS mismatches only — different people measured ~0.93-1.07 on the clip
# these constants were tuned on, so ~1.0 vetoes strangers while leaving even a
# full profile of the right person (up to ~1.0 from a frontal capture) alone.
# Anything near the match threshold will make hard frames blink instead.
_TRACK_VETO_SINGLE = float(os.environ.get('ROOP_TRACK_VETO_SINGLE', '0'))


# Verbose match diagnostics ([TRACKASSIGN] / [TRACKMATCH]).
# Read once, and treat '0'/'false'/'off' as OFF: the call sites used a bare
# os.environ.get(), which is truthy for the STRING "0", so the documented way to
# turn it back off left it running.
_DEBUG_MATCH = os.environ.get('ROOP_DEBUG_MATCH', '').strip().lower() not in ('', '0', 'false', 'off')


# Appearance gate for track association. The tracking SCAN has always gated on
# this (procmgr_tracking: `if cos_dist > EMB_MAX: continue`) — a detection that
# looks wrong is refused outright rather than merely made expensive, which is
# standard tracking-by-detection practice (BoT-SORT/ByteTrack `appearance_thresh`).
#
# The swap-time re-association did NOT. It scored entries with
#     cost = d_spatial * (1.0 + 2.5 * d_cosine)
# and took the best one however bad the appearance match was, so a large identity
# mismatch could always be outweighed by a small spatial distance. Same tracker,
# two different rules. This constant is now the single source for both.
#
# 0 disables the swap-time gate (restores the pre-fix behaviour).
_TRACK_EMB_MAX = float(os.environ.get('ROOP_TRACK_EMB_MAX', '0.7'))


# Reject when a DIFFERENT selected person explains the face this much better.
_TRACK_VETO_MARGIN = float(os.environ.get('ROOP_TRACK_VETO_MARGIN', '0.15'))


# ── Gap-fill continuity ──────────────────────────────────────────────────────
# The temporal pre-pass fills a track's detection misses by LINEARLY
# INTERPOLATING between the two observations either side of the gap, and the
# only condition was that the gap be short enough (ROOP_TEMPORAL_GAP frames).
# Nothing checked that the two anchors were in the same PLACE.
#
# They need not be. The scan's primary association is IoU-gated so it cannot
# teleport, but the Re-ID fallback matches on embedding ALONE, with no spatial
# constraint at all — by design, since it exists to reconnect a face that left
# and came back. So a track can legitimately jump across the frame between two
# consecutive observations, and the gap-filler would then manufacture a face for
# every frame in between, sliding across whatever the background happens to be.
#
# Those manufactured faces are invisible to every identity check downstream:
# _interp_face sets their embedding to the TRACK MEAN, so their distance to the
# track is 0 (passes the appearance gate) and their distance to the captured
# target is the track's own (which already passed the assignment gate). They are
# swapped unconditionally, wherever they were placed — and because a source can
# only be used once per frame, the real face in that frame is then refused.
#
# So bridge a gap only when the face could plausibly have travelled between the
# anchors: at most this many face-widths per skipped frame, with a bounded size
# change. Generous by construction — a head crossing half its own width every
# frame is already fast motion. 0 disables the guard (pre-fix behaviour).
_INTERP_MAX_TRAVEL = float(os.environ.get('ROOP_INTERP_MAX_TRAVEL', '0.5'))
_INTERP_MAX_SCALE = float(os.environ.get('ROOP_INTERP_MAX_SCALE', '2.0'))


# ── Track → source assignment gate ───────────────────────────────────────────
# Binding a track to a source is a DURABLE decision: every face on that track,
# for as long as it runs, is swapped with no further identity check beyond the
# vetoes. It is also made from the track's MEAN embedding over every accepted
# observation — a far cleaner measurement than any single frame.
#
# It was gated on max_face_distance, the same threshold as per-frame matching.
# That threshold is deliberately loose because it has to carry one bad frame of
# the right person; applied to a mean it lets a track that merely resembles the
# target own the source for a whole stretch of frames. Measured: a real person's
# track mean sat at 0.36 while background/blur false detections clustered at
# 0.85-1.0 — i.e. exactly where a 0.75-0.85 threshold sits. A run of a 33k-frame
# clip bound 16 of its 81 tracks to the one selected person.
#
# A track refused here is not dropped: its frames fall through to per-frame
# matching at the full threshold, so a real face still swaps, just without
# identity locking. 0 restores the old behaviour (gate == max_face_distance).
_TRACK_ASSIGN_MAX = float(os.environ.get('ROOP_TRACK_ASSIGN_MAX', '0.6'))


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


