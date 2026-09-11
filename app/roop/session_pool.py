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
import logging
import threading
import time
from queue import Empty, Queue


_LOGGER = logging.getLogger(__name__)


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
        11.5-15.5 GB (e.g. RTX 4070 12GB)   -> 4 / 4
        >= 15.5 GB (e.g. RTX 3090 24GB)     -> 8 / 8

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
    if gb < 15.5:
        return 4, 4          # 12GB cards (e.g. RTX 4070): 4 swapper, 4 detmask
    return 8, 8              # 16GB+ cards (e.g. RTX 3090/4080/4090): 8 swapper, 8 detmask


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
_pool_cache_lock = threading.Lock()

# ── Single-context mode ───────────────────────────────────────────────────────
# The pools exist so N worker threads can run one model concurrently. The live
# preview has no such threads: core.live_swap runs one frame at a time under
# _preview_lock, on a ProcessMgr of its own. Yet that ProcessMgr built every
# processor through the same Initialize() code as the render's, so it held
# ROOP_TRT_POOL copies of the swapper, the enhancer and each mask model — four
# TensorRT engines apiece on a 12GB card — and could only ever drive one of
# them. Measured on an RTX 4070 with hyperswap + Restore Ultra + BiSeNet + XSeg-3:
# 8.6GB dedicated plus 2GB of the process's allocations demoted to system RAM
# (WDDM oversubscription; the card is shared with the compositor and the
# browser). The demoted engines ran over PCIe: with a FIFO pool one slot in
# four was the slow one, so every fourth preview took 5-6s against ~0.8s, and
# once the card started paging in earnest single previews reached 16-70s and
# the UI's 15-minute preview deadline became reachable by scrubbing.
# (That figure was taken while the swapper's TensorRT profile was sized to
# threads x tiles — see swap_batch_capacity — which alone was ~5 GB of it.
# The render was over the card for the same reason; the argument here, that
# a one-frame-at-a-time consumer should not hold N contexts, stands on its own.)
#
# Inside single_context() every per-ProcessMgr pool query answers 1, so the
# processors build one session each and go through _gpu_guard's lock like an
# unpooled install. It is thread-local: the render's worker threads never see
# it, and a render cannot overlap a preview anyway (api.py's _swap_start_lock).
#
# The process-wide pools are deliberately NOT collapsed. The FaceAnalysis pool
# (face_util) and the hybrid detectors (retinaface/yoloface/yunet) outlive any
# one ProcessMgr and are shared with auto-angles, the clip advisor and the
# render's tracking pre-pass, all of which lease from them N-wide. A preview is
# the first thing to build them after boot, and _ensure_face_analyser does not
# rebuild on a pool-size mismatch, so a 1-wide pool built here would serialise
# every later consumer. Those callers pass shared=True and get the configured
# size regardless.
_override = threading.local()


@contextlib.contextmanager
def single_context():
    """Make every per-ProcessMgr pool on this thread one session wide."""
    previous = getattr(_override, 'single', False)
    _override.single = True
    try:
        yield
    finally:
        _override.single = previous


def _single_context() -> bool:
    return bool(getattr(_override, 'single', False))


def _resolve_pools():
    with _pool_cache_lock:
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
    if _single_context():
        return 1
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
def detmask_pool_size(shared: bool = False) -> int:
    """`shared=True` is for the process-wide FaceAnalysis and detector pools,
    which keep the configured size inside single_context() (see above)."""
    if not shared and _single_context():
        return 1
    return _resolve_pools()['detmask']


def detmask_pooling_enabled(shared: bool = False) -> bool:
    return detmask_pool_size(shared) >= 2


# The mask engines share the detmask knob but not its curve. They are one crop
# per face per call and cheap enough that two contexts already run ahead of
# what a render can feed them. Measured on an RTX 4070, TRT FP16, N threads
# each hammering its own context:
#
#     BiSeNet (resnet18 @512)   1: 82.9   2: 98.1   4: 97.9  calls/s   192 MB each
#     XSeg-3  (@256)            1: 412.6  2: 535.5  4: 462.2 calls/s    ~80 MB each
#
# The second context is the whole gain; the third and fourth cost VRAM and,
# for XSeg, throughput. A render at 18 fps with two faces needs ~36 of these
# per second per engine against ~200/s at two contexts. And this is paid once
# per SELECTED engine: mask_engine + mask_engine_2 on the 4/4 tier held
# 2 x 4 contexts, 1.1 GB for the BiSeNet + XSeg pair, for the throughput of
# 2 x 2. ROOP_MASK_POOL overrides; 0 or 1 means one context behind the lock.
_MASK_POOL_CAP = 2


def mask_pool_size() -> int:
    if _single_context():
        return 1
    raw = os.environ.get('ROOP_MASK_POOL')
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return min(_MASK_POOL_CAP, detmask_pool_size())


def mask_pooling_enabled() -> bool:
    return mask_pool_size() >= 2


def detector_pool_size() -> int:
    """How many independent instances of the SELECTED detector to build.

    The hybrid engines (retinaface / yoloface / yunet) bring their own detector
    and only borrow buffalo_l's aux models, so widening ROOP_DETMASK_POOL alone
    parallelises the aux models while the detector itself stays single-file. One
    instance per detect worker is what actually removes that serialisation.

    ROOP_DETECTOR_POOL overrides, and is the knob to turn DOWN first when VRAM is
    tight: retinaface_r50.onnx is ~104MB per instance (yoloface_8n is ~9MB and
    yunet ~350KB, so those are close to free).
    """
    raw = os.environ.get('ROOP_DETECTOR_POOL')
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    try:
        return max(1, detmask_pool_size(shared=True))
    except Exception:
        return 1


# Separate, opt-in pool for the LivePortrait expression restorer. It is the only
# GPU stage still running one-wide while the swapper, mask and detect stages run
# N-wide, so on a chunk where it is enabled its cost adds almost entirely in
# series. Measured on an RTX 4070: a 192-frame chunk went 9.00s with expression
# off to 13.53s with it on, i.e. ~4.5s of serialised work.
#
# This is now the LAST of the serialisation fixes, not the first to reach for.
# The restorer already (a) holds the GPU lock around its session runs only, not
# its CPU conversion, (b) skips the global lock entirely, since its own
# lock/lease is what keeps its contexts exclusive, and (c) overlaps its three
# independent front-half calls (ROOP_EXPR_PARALLEL). Those cost no VRAM. A pool
# is what remains for making two whole restores concurrent, and it is the only
# one that has to be paid for in engines.
#
# Measured on an RTX 4070, 100 restores over 4 threads, 256px crops, TRT FP16
# (all four produce bit-identical pixels):
#
#     old: global lock over the whole call   29.2 faces/s   34.23 ms   baseline
#     (a)+(b)+(c), no pool                   33.3 faces/s   30.06 ms      +14%
#     ROOP_EXPR_POOL=2                       37.3 faces/s   26.84 ms      +28%   +654 MiB
#     ROOP_EXPR_POOL=2 + PARALLEL=2          37.8 faces/s   26.44 ms      +29%   +899 MiB
#
# Note where that stops. A third slot measured no faster than the second: the
# stage is GPU-bound, not lock-bound — warping_spade alone is 23.4 ms of the
# 34 ms (68%), one 421 MB generator producing 512x512, and no amount of
# concurrency makes the card do that work faster. Scheduling was worth ~29% and
# that is the ceiling of it; the rest would have to come out of the model.
#
# VRAM-tiered like the other two, and for the same reason: a fixed value is
# wrong on somebody else's card. This started as a hardcoded ROOP_EXPR_POOL=2 in
# start_react.js, which is tracked and ships to every install — so a 6GB card,
# where _auto_pool_defaults deliberately turns the other two pools OFF because
# extra engines OOM or thrash below 1fps, would still have been handed two extra
# restorer contexts. Tiering it puts that decision back on the machine running
# it.
#
#     < 11.5 GB   -> 0   single context; the measured +28% is not worth being
#                        the allocation that pushes a mid-size card into an
#                        engine-rebuild thrash, and these are the LARGEST models
#                        of any pool here (~537 MB of weights per slot).
#     >= 11.5 GB  -> 2   the configuration measured above. 3 was no faster.
#
# The boundary is 11.5 for the same reason as _auto_pool_defaults': a nominal
# 12GB card reports ~11.99GB. Costs nothing on a card that never enables
# expression restore — ProcessMgr builds the restorer lazily, only once a run
# actually asks for a non-zero strength. ROOP_EXPR_POOL overrides either way.
def _auto_expression_pool() -> int:
    return 2 if _detect_vram_gb() >= 11.5 else 0


def expression_pool_size() -> int:
    if _single_context():
        return 1
    return _resolve('ROOP_EXPR_POOL', _auto_expression_pool())


def expression_pooling_enabled() -> bool:
    return expression_pool_size() >= 2


class SessionPool:
    """A fixed set of interchangeable per-model resources (e.g. an onnxruntime
    session, optionally paired with its own io_binding). `lease()` hands one
    resource to exactly one thread for the duration of a GPU call, then returns
    it to the pool, so each underlying TensorRT context is only ever touched by
    one thread at a time."""

    _CLOSE_POLL_SECONDS = 0.1

    def __init__(self, build_fn, size):
        size = int(size)
        if size < 1:
            raise ValueError("SessionPool size must be at least 1")
        self._items = [build_fn(i) for i in range(size)]
        self._q = Queue()
        self._state_lock = threading.Lock()
        self._released = False
        self._leased = 0
        for it in self._items:
            self._q.put(it)

    @contextlib.contextmanager
    def lease(self, timeout=None):
        """Lease one resource, aborting promptly if the pool is released.

        The old unbounded ``Queue.get`` left waiters parked forever when a
        cancellation drained the pool.  Polling only the close flag keeps the
        steady-state queue semantics while making shutdown deterministic.
        """
        deadline = None if timeout is None else time.monotonic() + max(0.0, float(timeout))
        item = None
        while item is None:
            with self._state_lock:
                if self._released:
                    raise RuntimeError("SessionPool has been released")
            wait_for = self._CLOSE_POLL_SECONDS
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Timed out waiting for a SessionPool resource")
                wait_for = min(wait_for, remaining)
            try:
                item = self._q.get(timeout=wait_for)
            except Empty:
                continue

        with self._state_lock:
            if self._released:
                # release() may have won the race after Queue.get().  Do not
                # publish an item from a closed pool to new GPU work.
                raise RuntimeError("SessionPool was released while acquiring a resource")
            self._leased += 1
        try:
            yield item
        finally:
            with self._state_lock:
                self._leased -= 1
                # Queue.put_nowait is performed under the state lock so
                # release() cannot drain the queue and then lose a late return.
                if not self._released:
                    self._q.put_nowait(item)

    def release(self):
        """Close the pool idempotently and discard every idle resource.

        In-flight leases keep their local reference until inference returns;
        they are deliberately not put back into the closed queue.  This avoids
        destroying an active CUDA context while also guaranteeing it becomes
        collectible immediately after the call completes.
        """
        with self._state_lock:
            if self._released:
                return
            self._released = True
            items, self._items = self._items, []
            try:
                while True:
                    self._q.get_nowait()
            except Empty:
                pass
            leased = self._leased
        items.clear()
        if leased:
            _LOGGER.info(
                "Released SessionPool with %d active lease(s); resources will be "
                "collected when those inference calls return", leased,
            )
