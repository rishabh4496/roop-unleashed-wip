"""Regression tests for teardown completeness and frame-rate validation.

1. ProcessMgr.release_resources walked past the two lazily-built restorers
   (_expr_restorer / _lipsync_restorer_inst). They are not in self.processors,
   so nothing ever called their Release() — which is what joins LivePortrait's
   ThreadPoolExecutor and disposes its SessionPool (~537 MB of weights per slot).
   api.py's upscale-after-swap pass calls release_resources() specifically to
   hand the upscale a clear card, and records a 26 s/frame crawl when the VRAM
   is not actually free.

2. utilities.detect_fps assigned cap.get(CAP_PROP_FPS) over its own 24.0
   default, so a container OpenCV cannot read a rate from (VFR MKV/WebM,
   damaged header) returned 0.0 — which reaches FFMPEG_VideoWriter as '-r 0.0'
   and makes ffmpeg exit before a frame is written.

Neither test needs a GPU, a model, or a media file.
"""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop.utilities import _plausible_fps, detect_fps


class _FakeRestorer:
    """Stands in for Expression_LivePortrait / Lipsync_MuseTalk."""

    def __init__(self, explode=False):
        self.released = False
        self._explode = explode

    def Release(self):
        if self._explode:
            raise RuntimeError('CUDA context already gone')
        self.released = True


class _BareMgr:
    """The teardown loop under test, lifted off ProcessMgr's __init__ so the
    test needs no CUDA, no models and no options object. Kept byte-identical to
    the block in ProcessMgr.release_resources."""

    def __init__(self, expr=None, lipsync=None):
        if expr is not None:
            self._expr_restorer = expr
        if lipsync is not None:
            self._lipsync_restorer_inst = lipsync

    def release_restorers(self):
        for attr in ('_expr_restorer', '_lipsync_restorer_inst'):
            inst = getattr(self, attr, None)
            if inst is None:
                continue
            try:
                inst.Release()
            except Exception as e:
                print(f"[release] {attr}.Release() failed: {e!r}")
            setattr(self, attr, None)


class TestRestorerTeardown(unittest.TestCase):
    def test_both_restorers_are_released_and_cleared(self):
        expr, lip = _FakeRestorer(), _FakeRestorer()
        m = _BareMgr(expr, lip)
        m.release_restorers()
        self.assertTrue(expr.released, 'expression restorer was not released')
        self.assertTrue(lip.released, 'lipsync restorer was not released')
        # Cleared, so the lazy getters rebuild instead of handing back an
        # instance whose sessions have been disposed.
        self.assertIsNone(getattr(m, '_expr_restorer', None))
        self.assertIsNone(getattr(m, '_lipsync_restorer_inst', None))

    def test_a_failing_release_does_not_strand_the_other(self):
        expr, lip = _FakeRestorer(explode=True), _FakeRestorer()
        m = _BareMgr(expr, lip)
        m.release_restorers()                      # must not raise
        self.assertTrue(lip.released,
                        'the second restorer was skipped because the first threw')
        self.assertIsNone(getattr(m, '_expr_restorer', None))

    def test_absent_restorers_are_a_no_op(self):
        m = _BareMgr()                             # never built one
        m.release_restorers()                      # must not raise

    def test_processmgr_release_covers_the_restorers(self):
        """Guard against the real method drifting away from the block above."""
        import inspect
        from roop.ProcessMgr import ProcessMgr
        src = inspect.getsource(ProcessMgr.release_resources)
        self.assertIn('_expr_restorer', src)
        self.assertIn('_lipsync_restorer_inst', src)


class TestPlausibleFps(unittest.TestCase):
    def test_rejects_unusable_values(self):
        for bad in (None, '', 'abc', 0, 0.0, -5, -0.001,
                    float('nan'), float('inf'), 1e6, 1001):
            self.assertFalse(_plausible_fps(bad), f'{bad!r} should be rejected')

    def test_accepts_real_frame_rates(self):
        for good in (1, 10.0, 23.976, 24, 25, '29.97', 30, 59.94, 120, 1000):
            self.assertTrue(_plausible_fps(good), f'{good!r} should be accepted')


class TestDetectFpsFallback(unittest.TestCase):
    """detect_fps must never hand a caller a value ffmpeg would reject."""

    def _detect_with(self, cv2_value, probe):
        class _Cap:
            def isOpened(self):
                return True

            def get(self, _prop):
                return cv2_value

            def release(self):
                pass

        with mock.patch('roop.utilities.cv2.VideoCapture', return_value=_Cap()), \
             mock.patch('roop.capturer._probe_video', return_value=probe):
            return detect_fps('clip.mkv')

    def test_uses_cv2_when_it_is_sane(self):
        self.assertAlmostEqual(self._detect_with(29.97, None), 29.97, places=3)

    def test_falls_back_to_ffprobe_when_cv2_returns_zero(self):
        # The VFR case: cv2 reports 0, ffprobe knows the real avg_frame_rate.
        self.assertAlmostEqual(
            self._detect_with(0.0, {'fps': 30000 / 1001.0}), 29.97, places=2)

    def test_falls_back_to_ffprobe_when_cv2_returns_nan(self):
        self.assertAlmostEqual(self._detect_with(float('nan'), {'fps': 25.0}), 25.0)

    def test_defaults_to_24_when_nothing_knows(self):
        self.assertEqual(self._detect_with(0.0, None), 24.0)

    def test_ignores_an_absurd_ffprobe_answer_too(self):
        self.assertEqual(self._detect_with(0.0, {'fps': 1e9}), 24.0)

    def test_never_returns_something_ffmpeg_would_reject(self):
        for cv2_value in (0.0, -1.0, float('nan'), float('inf'), 1e9):
            for probe in (None, {'fps': 0.0}, {'fps': float('nan')}, {'fps': 30.0}):
                got = detect_fps and self._detect_with(cv2_value, probe)
                self.assertTrue(
                    _plausible_fps(got),
                    f'detect_fps returned {got!r} for cv2={cv2_value!r} '
                    f'probe={probe!r}',
                )


if __name__ == '__main__':
    unittest.main(verbosity=2)
