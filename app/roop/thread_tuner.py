"""Empirical thread auto-tuner.

The optimal number of worker threads for video processing depends on the active
execution provider (TensorRT serialises GPU work unless pooled, CUDA runs
concurrently), the TensorRT context-pool size (ROOP_TRT_POOL), whether an
enhancer is active, and the frame resolution. A static VRAM-based guess (see
settings.py) cannot capture all of that, so this module *measures* throughput.

On the first video run for a given hardware/pipeline signature, it pushes a
small sample of real frames through the actual per-frame pipeline at several
candidate thread counts, times each, and keeps the fastest. The winner is
cached on disk keyed by that signature, so later runs reuse it for free.

The "Max. Number of Threads" setting is treated as a ceiling: the tuner only
ever picks a value <= that, so users can still cap concurrency.

Disable with env ROOP_AUTOTUNE=0. Wipe the cache with ROOP_AUTOTUNE_RESET=1.
"""

import os
import json
import time
import queue
import threading

import roop.globals

# app/.thread_tune_cache.json  (this file is app/roop/thread_tuner.py)
_CACHE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           '.thread_tune_cache.json')

_lock = threading.Lock()
_cache = None  # lazily loaded dict


def enabled() -> bool:
    return os.environ.get('ROOP_AUTOTUNE', '1') != '0'


def _load_cache() -> dict:
    global _cache
    if _cache is not None:
        return _cache
    if os.environ.get('ROOP_AUTOTUNE_RESET', '0') == '1':
        _cache = {}
        try:
            if os.path.exists(_CACHE_PATH):
                os.remove(_CACHE_PATH)
        except OSError:
            pass
        return _cache
    try:
        with open(_CACHE_PATH, 'r', encoding='utf-8') as fp:
            _cache = json.load(fp)
            if not isinstance(_cache, dict):
                _cache = {}
    except (OSError, ValueError):
        _cache = {}
    return _cache


def _save_cache() -> None:
    try:
        with open(_CACHE_PATH, 'w', encoding='utf-8') as fp:
            json.dump(_cache, fp, indent=2)
    except OSError:
        pass


def _provider_tag() -> str:
    provs = ' '.join(str(p).lower() for p in (roop.globals.execution_providers or []))
    if 'tensorrt' in provs:
        return 'trt'
    if 'cuda' in provs:
        return 'cuda'
    if 'dml' in provs:
        return 'dml'
    if 'rocm' in provs:
        return 'rocm'
    return 'cpu'


def _res_bucket(width: int, height: int) -> int:
    # Bucket by longest edge rounded up to the nearest 360 so e.g. 1920x1080
    # and 1920x800 share a tuned value but 720p and 4K do not.
    longest = max(int(width), int(height))
    return ((longest + 359) // 360) * 360


def make_key(width: int, height: int, enhancer_on: bool, max_threads: int) -> str:
    pool = os.environ.get('ROOP_TRT_POOL', '0') or '0'
    return (f"{_provider_tag()}|enh={int(bool(enhancer_on))}"
            f"|res={_res_bucket(width, height)}|pool={pool}|max={int(max_threads)}")


def get_cached(key: str):
    return _load_cache().get(key)


def set_cached(key: str, value: int) -> None:
    cache = _load_cache()
    with _lock:
        cache[key] = int(value)
        _save_cache()


def candidate_counts(max_threads: int) -> list:
    base = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32]
    cands = {c for c in base if 1 <= c <= max_threads}
    cands.add(1)
    cands.add(int(max_threads))
    return sorted(cands)


def _measure(process_fn, frames, threads: int) -> float:
    """Return frames-per-second processing `frames` across `threads` workers."""
    q: "queue.Queue" = queue.Queue()
    for fr in frames:
        q.put(fr)
    for _ in range(threads):
        q.put(None)

    def worker():
        while True:
            fr = q.get()
            if fr is None:
                return
            try:
                process_fn(fr)
            except Exception:
                # A failing frame must not skew or crash calibration.
                pass

    # daemon so a stuck GPU call during calibration can't block app exit/restart
    workers = [threading.Thread(target=worker, name=f'tune{threads}', daemon=True) for _ in range(threads)]
    t0 = time.perf_counter()
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    dt = time.perf_counter() - t0
    return (len(frames) / dt) if dt > 0 else 0.0


def calibrate(process_fn, frames, max_threads: int, log=print):
    """Time `process_fn` over `frames` at each candidate thread count.

    Returns (best_thread_count, {thread_count: fps}). `frames` should be a list
    of representative BGR frames (copies — workers may mutate). Caller is
    responsible for snapshotting/restoring any per-run state `process_fn` mutates.
    """
    cands = candidate_counts(max_threads)
    if len(cands) <= 1:
        return (cands[0] if cands else 1), {}

    # Warm up: trigger lazy model loads / CUDA context so the first timed
    # candidate isn't unfairly penalised.
    for fr in frames[:min(4, len(frames))]:
        try:
            process_fn(fr.copy())
        except Exception:
            pass

    results = {}
    best_fps = -1.0
    for c in cands:
        # Hand each candidate its own copies so earlier runs can't affect later.
        sample = [fr.copy() for fr in frames]
        fps = _measure(process_fn, sample, c)
        results[c] = fps
        log(f"[AutoTune]   threads={c:>2}  ->  {fps:6.2f} fps")
        if fps > best_fps:
            best_fps = fps

    # Prefer the *smallest* thread count that already reaches ~97% of peak
    # throughput: once the GPU is saturated, extra threads only burn VRAM and
    # add contention for no real speedup.
    knee = 0.97 * best_fps
    best_c = max(results, key=lambda k: results[k])
    for c in sorted(results):
        if results[c] >= knee:
            best_c = c
            break
    return best_c, results
