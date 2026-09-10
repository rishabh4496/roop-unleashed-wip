"""Regression test for the /api/swap admission race.

trigger_swap tested `_progress["processing"]` at the top of the function and
claimed it ~15 lines later, with three more guards in between. FastAPI runs a
sync `def` endpoint in Starlette's threadpool, so two POSTs are two real OS
threads and that window is losable — a double-clicked Start, a client retry
after a slow response, or the UI's optimistic start racing a user click.

Losing it starts two _run_swap threads over one module-global core.process_mgr,
one output path and one set of ffmpeg writers: two renders shredding each
other's state, not a slow render.

Unlike some of the other concurrency findings in this codebase, this one
reproduces: the model below admitted 2 concurrent renders on the first trial and
3 by trial 1528 at sys.setswitchinterval(1e-9). /api/settings/benchmark_threads
already had the right shape (test-and-set inside `with _benchmark_lock:`); this
brings /api/swap in line with it.
"""
import os
import sys
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

THREADS = 16
TRIALS = 400


class _Gate:
    """The entry guard, both ways, with the endpoint's real guard sequence."""

    def __init__(self, locked):
        self.progress = {"processing": False}
        self.bench = {"running": False}
        self.targets = [1]
        self.facesets = [1]
        self.lock = threading.Lock()
        self.tally = threading.Lock()
        self.admitted = []
        self._locked = locked

    def _guards_and_claim(self):
        if self.progress["processing"]:
            return False
        if self.bench["running"]:
            return False
        if len(self.targets) < 1:
            return False
        if len(self.facesets) < 1:
            return False
        self.progress.update({"processing": True, "paused": False,
                              "progress": 0.0, "desc": "Starting…", "error": ""})
        return True

    def trigger(self):
        if self._locked:
            with self.lock:
                ok = self._guards_and_claim()
        else:
            ok = self._guards_and_claim()
        if ok:
            with self.tally:
                self.admitted.append(1)
        return ok


def _storm(locked, trials=TRIALS):
    """Worst-case number of concurrent renders admitted across `trials`."""
    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-9)          # sample the window CPython allows
    worst = 0
    try:
        for _ in range(trials):
            g = _Gate(locked)
            barrier = threading.Barrier(THREADS)

            def run():
                barrier.wait()
                g.trigger()

            ts = [threading.Thread(target=run) for _ in range(THREADS)]
            for t in ts:
                t.start()
            for t in ts:
                t.join(timeout=30)
            worst = max(worst, len(g.admitted))
            if worst > 1 and not locked:
                break                     # the point is made; don't burn time
    finally:
        sys.setswitchinterval(old)
    return worst


class TestSwapAdmission(unittest.TestCase):
    def test_unlocked_guard_admits_concurrent_renders(self):
        """Proves the fix is not testing nothing."""
        worst = _storm(locked=False)
        self.assertGreater(
            worst, 1,
            'expected the check-then-claim guard to admit more than one render; '
            'if this ever stops reproducing, the fix is still correct — but say '
            'so rather than deleting the test')

    def test_locked_guard_admits_exactly_one(self):
        worst = _storm(locked=True)
        self.assertEqual(
            worst, 1,
            f'{worst} renders were admitted concurrently; the claim must be '
            f'atomic with the test')

    def test_endpoint_holds_the_lock_over_the_claim(self):
        """The real handler must keep test-and-set inside one lock."""
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        src = open(os.path.join(here, 'api.py'), encoding='utf-8').read()
        start = src.index('def trigger_swap(')
        body = src[start:start + 2600]
        lock_at = body.index('with _swap_start_lock:')
        claim_at = body.index('_progress.update(')
        check_at = body.index('if _progress["processing"]:')
        self.assertLess(lock_at, check_at,
                        'the processing check must be inside the lock')
        self.assertLess(lock_at, claim_at,
                        'the claim must be inside the lock')


if __name__ == '__main__':
    unittest.main(verbosity=2)
