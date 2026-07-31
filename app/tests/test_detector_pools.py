"""The hybrid detectors must not go back to one shared instance behind a mutex.

Every hybrid engine (retinaface / yoloface / yunet) brings its OWN detector and
only borrows buffalo_l's aux models. So a mutex around its detect() makes the
whole engine single-file no matter how wide ROOP_DETMASK_POOL is set — widening
the pool then adds pure queue time. That was measured on retinaface before
7a1d345 fixed it: a constant 17.4ms serial section at pool 4 AND pool 6, both
landing on 57-58 f/s, with the GPU at only ~51%. (A real GPU ceiling shows ~100%
util instead; a constant serial section that does not shrink with pool width is
the signature of a lock.)

yunet and yoloface kept that exact pattern for another week. They now use the
same per-worker leasing, and this pins the shape so it cannot quietly come back:
a `with self._lock` re-added around a session run would restore the bug while
every other test still passed.

Concurrency itself is verified out-of-band rather than here — importing these
modules pulls roop.globals and torch (~4s), which would nearly double a suite
that runs in five seconds. The out-of-band check runs N threads into detect()
against a threading.Barrier(N): they must ALL be inside at once, which is
exactly what a mutex makes impossible. It was confirmed to fail (4/4 broken
barriers) when the pool is forced to 1, so it is not vacuous.
"""

import os
import re
import sys
import unittest
from pathlib import Path

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)

ENGINES = ('retinaface', 'yoloface', 'yunet')
SRC = {name: Path(APP, 'roop', f'{name}.py').read_text(encoding='utf-8')
       for name in ENGINES}


def _code(text):
    """Source with comments and docstring prose stripped — these files DISCUSS
    the old lock at length, so a plain substring search would match the very
    comments explaining why the lock is gone."""
    text = re.sub(r'""".*?"""', '', text, flags=re.S)
    return '\n'.join(re.sub(r'#.*$', '', ln) for ln in text.splitlines())


class DetectorsAreNotSerialised(unittest.TestCase):

    def test_sources_were_found(self):
        """Guard the parsing — empty sources would pass everything below."""
        for name in ENGINES:
            self.assertGreater(len(_code(SRC[name])), 500, name)

    def test_no_detect_time_mutex(self):
        """The regression: any lock held across the actual detect call."""
        for name in ENGINES:
            code = _code(SRC[name])
            with self.subTest(engine=name):
                # Construction may still be guarded; a detect-time lock may not.
                for bad in ('with _detect_lock', 'with self._lock'):
                    self.assertNotIn(bad, code,
                                     f'{name}.py re-serialises detection with '
                                     f'"{bad}" — the pool width becomes queue time')
                # The only Lock left must be the pool-construction one.
                locks = re.findall(r'with\s+(_?\w*lock\w*)\s*:', code, re.I)
                self.assertTrue(
                    set(locks) <= {'_detector_lock'},
                    f'{name}.py holds an unexpected lock: {sorted(set(locks))}')

    def test_each_engine_leases_from_a_queue(self):
        for name in ENGINES:
            code = _code(SRC[name])
            with self.subTest(engine=name):
                self.assertIn('def lease_detector', code,
                              f'{name}.py has no lease — instances cannot be '
                              f'owned by one thread for the length of a call')
                self.assertIn('Queue()', code,
                              f'{name}.py must hand instances out from a Queue, '
                              f'which is what caps concurrency at the pool size')

    def test_one_pool_sizing_rule_for_all_three(self):
        """All three must answer to the same ROOP_DETECTOR_POOL.

        Three private copies of the sizing rule would drift, and the user would
        have no single knob to turn down when VRAM is tight — which matters
        because retinaface_r50 is ~104MB per instance while the other two are
        nearly free.
        """
        for name in ENGINES:
            with self.subTest(engine=name):
                self.assertIn('detector_pool_size', _code(SRC[name]),
                              f'{name}.py does not use the shared sizing rule')
        pool_src = Path(APP, 'roop', 'session_pool.py').read_text(encoding='utf-8')
        self.assertIn('def detector_pool_size', pool_src)
        self.assertIn('ROOP_DETECTOR_POOL', pool_src)

    def test_the_hybrid_callers_go_through_the_leasing_entry_point(self):
        """A caller that grabs get_detector() and calls .detect() on it shares
        one instance across every worker again — the lease is bypassed, and the
        pool silently stops doing anything."""
        fu = _code(Path(APP, 'roop', 'face_util.py').read_text(encoding='utf-8'))
        for hybrid in ('_hybrid_yolo_faces', '_hybrid_yunet_faces',
                       '_hybrid_retinaface_faces'):
            body = re.search(r'def %s\(.*?\n(?=\n\ndef |\Z)' % hybrid, fu, re.S)
            self.assertIsNotNone(body, f'{hybrid} not found')
            with self.subTest(caller=hybrid):
                self.assertNotIn('get_detector()', body.group(0),
                                 f'{hybrid} calls detect() on an unleased '
                                 f'instance, bypassing the pool')


if __name__ == '__main__':
    unittest.main()
