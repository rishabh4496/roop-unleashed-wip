"""Safety tests for the motion-gated temporal detector stride."""

import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop.procmgr_tracking import (  # noqa: E402
    _adaptive_motion_is_stable,
    _motion_signature,
)


class AdaptiveTemporalMotionTest(unittest.TestCase):
    def setUp(self):
        self.shape = (720, 1280, 3)
        self.box = np.asarray([400, 180, 880, 620], dtype=np.float32)
        self.base = np.zeros(self.shape, dtype=np.uint8)
        self.previous = _motion_signature(self.base)

    def test_identical_frame_with_an_active_track_can_be_filled(self):
        current = _motion_signature(self.base)
        self.assertTrue(_adaptive_motion_is_stable(
            self.previous, current, [self.box], self.shape))

    def test_face_motion_forces_a_fresh_detection(self):
        frame = self.base.copy()
        cv2.rectangle(frame, (500, 260), (780, 540), (255, 255, 255), -1)
        current = _motion_signature(frame)
        self.assertFalse(_adaptive_motion_is_stable(
            self.previous, current, [self.box], self.shape))

    def test_scene_motion_outside_face_forces_a_fresh_detection(self):
        frame = self.base.copy()
        cv2.rectangle(frame, (20, 20), (180, 180), (255, 255, 255), -1)
        current = _motion_signature(frame)
        self.assertFalse(_adaptive_motion_is_stable(
            self.previous, current, [self.box], self.shape))

    def test_no_active_track_is_never_treated_as_safe_to_skip(self):
        current = _motion_signature(self.base)
        self.assertFalse(_adaptive_motion_is_stable(
            self.previous, current, [], self.shape))


if __name__ == '__main__':
    unittest.main()
