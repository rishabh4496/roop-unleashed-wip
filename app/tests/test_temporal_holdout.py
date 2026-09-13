"""Forward detector-miss holdout for temporal face swapping.

The temporal pre-pass must be able to keep a confirmed face alive when there is
no later detection to interpolate from. These tests exercise the geometry and
identity contract without loading an InsightFace model.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop.procmgr_tracking import TrackingMixin  # noqa: E402


class _Face:
    def __init__(self, x, y, step=0.0):
        self.bbox = np.array([x, y, x + 100.0, y + 100.0], dtype=np.float32)
        self.kps = np.array([[x + 25 + step, y + 35],
                             [x + 75 + step, y + 35],
                             [x + 50 + step, y + 55],
                             [x + 32 + step, y + 78],
                             [x + 68 + step, y + 78]], dtype=np.float32)
        self.landmark_2d_106 = np.zeros((106, 2), dtype=np.float32)
        self.landmark_3d_68 = np.zeros((68, 3), dtype=np.float32)
        self.embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.det_score = 0.9


class TemporalHoldoutTest(unittest.TestCase):

    def test_predicts_bbox_and_landmarks_from_last_two_observations(self):
        first = _Face(100, 200, step=0)
        last = _Face(120, 210, step=20)
        track_embedding = np.array([0.8, 0.6, 0.0], dtype=np.float32)
        track = {
            'obs': {0: first, 2: last},
            'emb_mean': track_embedding,
        }

        held = TrackingMixin._coast_face(track, 3)

        self.assertIsNotNone(held)
        np.testing.assert_allclose(held.bbox, [130, 215, 230, 315])
        np.testing.assert_allclose(held.kps, last.kps + (last.kps - first.kps) / 2.0)
        np.testing.assert_allclose(held.embedding, track_embedding)
        self.assertTrue(held._interpolated)
        self.assertTrue(held._temporal_hold)
        self.assertEqual(held._hold_frames, 1)

    def test_missing_previous_observation_keeps_last_landmarks(self):
        last = _Face(120, 210)
        held = TrackingMixin._coast_face(
            {'obs': {12: last}, 'emb_mean': last.embedding}, 15)

        np.testing.assert_allclose(held.bbox, last.bbox)
        np.testing.assert_allclose(held.kps, last.kps)

    def test_observed_frame_gate_excludes_predictions(self):
        track = {'id': 7, 'observed_frames': {10, 12}}
        frames_of = {7: {10, 11, 12}}
        self.assertEqual(TrackingMixin._track_frames(track, frames_of), {10, 12})


if __name__ == '__main__':
    unittest.main()
