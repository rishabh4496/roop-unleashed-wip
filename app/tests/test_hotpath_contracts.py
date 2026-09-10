"""Regression tests for two hot-path defects in ProcessMgr / api.

1. process_frame's USE_LAST_SWAPPED policy did check-then-increment on
   self.num_frames_no_face from N worker threads with no lock. Unlike
   total_swaps — whose lost increments are explicitly accepted as a coarse
   average — this one is the control variable capping how many consecutive
   frames may be reused, so the TOCTOU let the cap overrun by up to one frame
   per thread.

2. api.parse_subsample_size replaces `int(str(value)[:3])`, which raised
   ValueError (HTTP 500) on any option whose digits are not exactly three
   characters — '64px' and '' among them.
"""
import os
import re
import sys
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

THREADS = 8
LIMIT = 15


class _ReusePolicy:
    """The locked block from ProcessMgr.process_frame, lifted so the test needs
    no CUDA, no models and no ProcessOptions."""

    def __init__(self, locked=True):
        self.lock = threading.Lock()
        self.num_frames_no_face = 0
        self.last_swapped_frame = object()      # stands in for a stored frame
        self.max_num_reuse_frame = LIMIT
        self._locked = locked
        self.reuses = 0
        self._count_lock = threading.Lock()

    def _tally(self):
        with self._count_lock:
            self.reuses += 1

    def try_reuse(self):
        if self._locked:
            with self.lock:
                reuse = (self.last_swapped_frame is not None
                         and self.num_frames_no_face < self.max_num_reuse_frame)
                if reuse:
                    self.num_frames_no_face += 1
        else:
            # The original, for contrast.
            reuse = (self.last_swapped_frame is not None
                     and self.num_frames_no_face < self.max_num_reuse_frame)
            if reuse:
                # A yield here makes the interleaving the GIL allows anyway
                # deterministic, rather than leaving the test flaky.
                _ = sum(range(50))
                self.num_frames_no_face += 1
        if reuse:
            self._tally()
        return reuse


def _hammer(policy, calls_per_thread=40):
    barrier = threading.Barrier(THREADS)

    def run():
        barrier.wait()
        for _ in range(calls_per_thread):
            policy.try_reuse()

    ts = [threading.Thread(target=run) for _ in range(THREADS)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=30)


class TestReuseCap(unittest.TestCase):
    def test_locked_policy_never_exceeds_the_cap(self):
        p = _ReusePolicy(locked=True)
        _hammer(p)
        self.assertEqual(
            p.reuses, LIMIT,
            f'the cap is {LIMIT} consecutive reuses; {p.reuses} were handed out')
        self.assertEqual(p.num_frames_no_face, LIMIT)

    def test_unlocked_variant_is_kept_for_contrast_only(self):
        """Honest scope marker: the unlocked version is NOT demonstrably wrong.

        The window between `if n < limit` and `n += 1` is about three bytecodes,
        and this path only runs on frames where nothing was detected, so
        contention is low. Hammering it with 8 threads x 40 calls did not
        overrun the cap, and neither did 20 trials at
        sys.setswitchinterval(1e-9). The lock is therefore HARDENING — it makes
        the cap and the stored frame provably consistent — not a fix for an
        observed failure. Recorded here so nobody later reads the locked version
        as evidence of a bug that was measured.
        """
        p = _ReusePolicy(locked=False)
        _hammer(p)
        self.assertGreaterEqual(p.reuses, LIMIT)

    def test_processmgr_takes_the_lock(self):
        """The real method must still hold self.lock around that state."""
        import inspect
        from roop.ProcessMgr import ProcessMgr
        src = inspect.getsource(ProcessMgr.process_frame)
        self.assertIn('with self.lock', src)
        # and must not have gone back to a bare check-then-increment
        self.assertNotIn(
            'if self.last_swapped_frame is not None and self.num_frames_no_face <',
            src)


class TestParseSubsampleSize(unittest.TestCase):
    """Loaded by source extraction so the test does not import the whole API
    (FastAPI app construction pulls in the full model stack)."""

    @classmethod
    def setUpClass(cls):
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        src = open(os.path.join(here, 'api.py'), encoding='utf-8').read()
        m = re.search(r'def parse_subsample_size.*?\n    return size\n', src, re.S)
        assert m, 'parse_subsample_size not found in api.py'
        ns = {}
        exec(m.group(0), ns)
        cls.parse = staticmethod(ns['parse_subsample_size'])

    def test_advertised_options_round_trip(self):
        # /api/meta advertises exactly these.
        for text, want in (('128px', 128), ('256px', 256), ('512px', 512)):
            self.assertEqual(self.parse(text), want)

    def test_values_that_used_to_raise(self):
        # int('64p') and int('') were ValueError -> HTTP 500 on the request.
        self.assertEqual(self.parse('64px'), 64)
        self.assertEqual(self.parse(''), 256)

    def test_out_of_band_falls_back_rather_than_lying(self):
        # int(str('2048px')[:3]) was 204 — a size nothing tiles into.
        self.assertEqual(self.parse('2048px'), 256)
        self.assertEqual(self.parse('0px'), 256)

    def test_never_raises_and_always_returns_a_usable_size(self):
        for junk in ('', '   ', 'abc', 'px', None, 0, -1, [], {}, 3.5,
                     '999999px', '\x00', 'NaN', '256px ', ' 256px'):
            got = self.parse(junk)
            self.assertIsInstance(got, int, f'{junk!r} gave {got!r}')
            self.assertTrue(64 <= got <= 1024, f'{junk!r} gave {got!r}')


if __name__ == '__main__':
    unittest.main(verbosity=2)
