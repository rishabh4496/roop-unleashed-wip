"""Regression tests for SwapBatcher's release contract.

The batcher accepts crops from N worker threads and each worker blocks in
wait() until its request is fulfilled. The invariant that keeps the render
alive is simple: every request the batcher ACCEPTS must have its event set
exactly once, on every code path.

_run_batch used to fulfil results with `zip(batch, outs)`, which truncates to
the shorter side. A run_fn returning fewer outputs than requests — reachable
via FaceSwapInsightFace.RunBatchMulti when a model's graph collapses the batch
dimension, i.e. the HyperSwap [1,-1,1,1] reshape — left the surplus workers
blocked in wait() forever, stalling the whole encode with nothing logged.
"""
import os
import sys
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import contextlib

from roop.swap_batcher import SwapBatcher


@contextlib.contextmanager
def _noop_guard():
    yield


def _submit_and_collect(b, n, timeout=20):
    """Submit n crops from n threads; return (results, errors, timed_out)."""
    results, errors = [None] * n, [None] * n
    done = threading.Event()
    remaining = {'n': n}
    lock = threading.Lock()

    def worker(i):
        req = b.submit(f'src{i}', f'tgt{i}', f'blob{i}')
        try:
            results[i] = b.wait(req)
        except Exception as e:      # noqa: BLE001 - asserted on by the caller
            errors[i] = e
        with lock:
            remaining['n'] -= 1
            if remaining['n'] == 0:
                done.set()

    for i in range(n):
        threading.Thread(target=worker, args=(i,), daemon=True).start()
    finished = done.wait(timeout)
    return results, errors, not finished


class TestShortResultList(unittest.TestCase):
    def test_short_result_list_does_not_strand_workers(self):
        """The bug: 4 requests in, 1 result back, 3 workers hung forever."""
        def collapsing_run_fn(reqs):
            # Exactly what an ONNX graph with a baked batch-1 reshape returns.
            return [f'out-for-{reqs[0][0]}']

        b = SwapBatcher(collapsing_run_fn, _noop_guard, max_batch=4,
                        max_wait_ms=30)
        try:
            results, errors, timed_out = _submit_and_collect(b, 4)
            self.assertFalse(
                timed_out,
                "workers were left blocked in wait() — the release contract "
                "is broken",
            )
            # Every worker must be told what happened; none may silently
            # receive a result that was never computed for it.
            self.assertTrue(all(e is not None for e in errors),
                            f"expected an error per worker, got {errors}")
            self.assertIn('collapsed the batch dimension', str(errors[0]))
        finally:
            b.stop()

    def test_long_result_list_is_also_rejected(self):
        """More outputs than inputs means the mapping is wrong too."""
        b = SwapBatcher(lambda reqs: ['a'] * (len(reqs) + 2), _noop_guard,
                        max_batch=4, max_wait_ms=30)
        try:
            _r, errors, timed_out = _submit_and_collect(b, 2)
            self.assertFalse(timed_out)
            self.assertTrue(all(e is not None for e in errors))
        finally:
            b.stop()


class TestNormalOperationUnchanged(unittest.TestCase):
    def test_every_worker_gets_its_own_result(self):
        b = SwapBatcher(lambda reqs: [f'out-{s}' for s, _t, _bl in reqs],
                        _noop_guard, max_batch=4, max_wait_ms=30)
        try:
            results, errors, timed_out = _submit_and_collect(b, 8)
            self.assertFalse(timed_out)
            self.assertEqual(errors, [None] * 8, f"unexpected errors: {errors}")
            # Correct pairing, not just correct count.
            self.assertEqual(sorted(results),
                             sorted(f'out-src{i}' for i in range(8)))
        finally:
            b.stop()

    def test_run_fn_exception_reaches_every_waiter(self):
        def boom(_reqs):
            raise ValueError('inference exploded')

        b = SwapBatcher(boom, _noop_guard, max_batch=4, max_wait_ms=30)
        try:
            _r, errors, timed_out = _submit_and_collect(b, 4)
            self.assertFalse(timed_out)
            self.assertTrue(all(isinstance(e, ValueError) for e in errors),
                            f"got {errors}")
        finally:
            b.stop()

    def test_submit_after_stop_runs_inline(self):
        b = SwapBatcher(lambda reqs: [f'out-{s}' for s, _t, _bl in reqs],
                        _noop_guard, max_batch=4, max_wait_ms=30)
        b.stop()
        req = b.submit('srcX', 'tgtX', 'blobX')
        self.assertEqual(b.wait(req), 'out-srcX')


if __name__ == '__main__':
    unittest.main(verbosity=2)
