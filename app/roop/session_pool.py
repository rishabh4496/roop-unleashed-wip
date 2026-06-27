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

try:
    _POOL_SIZE = max(0, int(os.environ.get('ROOP_TRT_POOL', '0') or '0'))
except ValueError:
    _POOL_SIZE = 0


def pool_size() -> int:
    return _POOL_SIZE


def pooling_enabled() -> bool:
    return _POOL_SIZE >= 2


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
