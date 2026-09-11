"""A per-frame failure costs the frame; a streak of them costs the run.

face_util.get_all_faces raises on a detector failure instead of reporting an
empty frame, so ProcessMgr has to decide what a RuntimeError means. One is a
transient (write the original frame through); thirty in a row is a dead CUDA
context or a broken model, and rendering the rest of the clip unswapped under
a green status line is the failure this guards against.
"""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from roop.ProcessMgr import ProcessMgr  # noqa: E402


class FrameErrorBudget(unittest.TestCase):

    def setUp(self):
        self.mgr = ProcessMgr(None)

    def test_gpu_and_detector_errors_are_recoverable_until_the_streak_cap(self):
        with mock.patch('roop.ProcessMgr.bar_write'):
            for kind in ('CUDA error 999', 'onnxruntime: Fail',
                         'Face detector failed: boom',
                         'High-resolution face detector failed: boom'):
                self.assertTrue(self.mgr._frame_error_is_recoverable(RuntimeError(kind), 'f'))
            self.mgr._frame_ok()
            self.assertEqual(self.mgr._consecutive_frame_errors, 0)
            cap = self.mgr._MAX_CONSECUTIVE_FRAME_ERRORS
            for _ in range(cap):
                self.assertTrue(self.mgr._frame_error_is_recoverable(RuntimeError('CUDA x'), 'f'))
            self.assertFalse(self.mgr._frame_error_is_recoverable(RuntimeError('CUDA x'), 'f'),
                             'the frame after the cap must abort the run')

    def test_other_runtime_errors_propagate(self):
        with mock.patch('roop.ProcessMgr.bar_write'):
            self.assertFalse(self.mgr._frame_error_is_recoverable(RuntimeError('shape mismatch'), 'f'))
            self.assertEqual(self.mgr._consecutive_frame_errors, 0)


if __name__ == '__main__':
    unittest.main()
