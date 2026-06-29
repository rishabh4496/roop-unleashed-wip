"""Optional per-model session pooling to break TensorRT's single-context
serialization.

TensorRT's execution context is NOT thread-safe (concurrent enqueue on one
context corrupts the CUDA context -> error 999), so the pipeline normally
serialises *all* GPU inference behind one global lock (see ProcessMgr._gpu_guard).
That caps GPU utilisation well below 100% even with many worker threads.

A SessionPool holds N independent onnxruntime sessions for the same model, each
with its own TensorRT engine + execution context. Because the contexts are
distinct, N worker threads can run that model concurrently and safely. Different
models running on different contexts across threads is also safe; only reuse of
the *same* context must be serialised, which the lease/return queue guarantees.

Enable with the env var ROOP_TRT_POOL=<N> (N>=2). Default (unset / <2) keeps the
original single-session behaviour byte-for-byte, so this is a no-op unless opted
in. VRAM cost scales ~N x per pooled model, so keep N small on limited GPUs.
"""
import os
import contextlib
from queue import Queue


def _detect_vram_gb() -> float:
    """Best-effort total VRAM of the active CUDA device, in GB (0 if unknown).

    Used to auto-tune the pool sizes so the same install runs on cards of very
    different capacity. Detection is deferred to first use (not import time) so
    torch's CUDA context is already initialised and import-order is irrelevant.
    """
    try:
        import torch
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            return torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
    except Exception:
        pass
    return 0.0


def _auto_pool_defaults():
    """VRAM-tiered defaults for (swapper pool, detmask pool).

    Each pooled instance holds its OWN TensorRT engine + context, so VRAM scales
    ~Nx per pooled model. The pools raise GPU concurrency/throughput (the swapper
    pool was validated at +46% fps on a large card) but small cards can't afford
    them — on a 6GB card the extra engines OOM / trigger an endless engine-build
    thrash that drops throughput below 1fps. So: pools OFF on small cards, the
    validated multi-context settings on large cards.

        < 7 GB    (e.g. RTX 3060 6GB)       -> 0 / 0  (single context + lock)
        7-11.5 GB (e.g. 3080 10GB)          -> 2 / 2
        >= 11.5 GB (e.g. RTX 4070 12GB)     -> 4 / 2  (validated)

    Note the 11.5 boundary: a nominal "12GB" card reports ~11.99GB to torch
    (RTX 4070 = 12282 MiB), so the large-card tier must sit just below 12 to
    catch them — otherwise a 12GB card gets demoted to 2/2 and loses throughput.
    """
    gb = _detect_vram_gb()
    if gb <= 0:
        return 0, 0          # unknown / CPU-only -> safest
    if gb < 7:
        return 0, 0
    if gb < 11.5:
        return 2, 2
    return 4, 2


def _resolve(env_name, auto_value) -> int:
    """Explicit env var wins (manual override); otherwise use the auto value."""
    raw = os.environ.get(env_name)
    if raw is not None and raw != '':
        try:
            return max(0, int(raw))
        except ValueError:
            pass
    return auto_value


_pool_cache = {}


def _resolve_pools():
    if not _pool_cache:
        auto_trt, auto_detmask = _auto_pool_defaults()
        trt = _resolve('ROOP_TRT_POOL', auto_trt)
        detmask = _resolve('ROOP_DETMASK_POOL', auto_detmask)
        _pool_cache['trt'] = trt
        _pool_cache['detmask'] = detmask
        gb = _detect_vram_gb()
        print(f"[SessionPool] detected {gb:.1f}GB VRAM -> "
              f"ROOP_TRT_POOL={trt}, ROOP_DETMASK_POOL={detmask} "
              f"(env override wins if set)")
    return _pool_cache


def pool_size() -> int:
    return _resolve_pools()['trt']


def providers_without_tensorrt(providers):
    """Return *providers* with the TensorRT EP removed (keeping CUDA/CPU).

    Some ONNX models don't run under the TensorRT execution provider — e.g. the
    MobileSAM decoder feeds a float `orig_im_size`, which TRT treats as a shape
    tensor and rejects ("must have type Int32 or Int64. Type is Float"). Routing
    such models to CUDA avoids the breakage; TRT's benefit on these small per-crop
    models is marginal anyway. Each entry may be a string or an (name, opts) tuple.
    """
    out = []
    for p in (providers or []):
        name = p[0] if isinstance(p, (list, tuple)) else p
        if 'Tensorrt' in name or 'TensorRT' in name:
            continue
        out.append(p)
    return out or ['CPUExecutionProvider']


def pooling_enabled() -> bool:
    return pool_size() >= 2


# Separate, opt-in pool for the face-analysis (detection/landmark/recognition)
# and mask models. Profiling showed those two stages are ~90% of video time and
# run single-threaded behind the global GPU lock (their per-call cost is dominated
# by lock-wait, not compute). Giving each its OWN pool of independent TensorRT
# contexts lets N worker threads run them concurrently — keeping TRT's fast FP16
# per-call (CUDA FP32 was benchmarked slower) while removing the serialisation.
#
# Kept distinct from ROOP_TRT_POOL (the swapper pool) so it can be tuned / turned
# off independently: each FaceAnalysis instance loads 5 small models, so VRAM
# scales with the pool size. The default is auto-tuned by VRAM (see
# _auto_pool_defaults): 0 on small cards = original single-instance + global lock
# behaviour, byte-for-byte. Set ROOP_DETMASK_POOL explicitly to override.
def detmask_pool_size() -> int:
    return _resolve_pools()['detmask']


def detmask_pooling_enabled() -> bool:
    return detmask_pool_size() >= 2


class SessionPool:
    """A fixed set of interchangeable per-model resources (e.g. an onnxruntime
    session, optionally paired with its own io_binding). `lease()` hands one
    resource to exactly one thread for the duration of a GPU call, then returns
    it to the pool, so each underlying TensorRT context is only ever touched by
    one thread at a time."""

    def __init__(self, build_fn, size):
        self._items = [build_fn(i) for i in range(size)]
        self._q = Queue()
        for it in self._items:
            self._q.put(it)

    @contextlib.contextmanager
    def lease(self):
        item = self._q.get()
        try:
            yield item
        finally:
            self._q.put(item)

    def release(self):
        items, self._items = self._items, []
        try:
            while True:
                self._q.get_nowait()
        except Exception:
            pass
        items.clear()
