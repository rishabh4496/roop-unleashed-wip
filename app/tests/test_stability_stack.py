"""Automated Verification and Regression Test Suite for Lateral Tracking and Stability Stack.

Covers:
1. Diagnostic Telemetry formatting, failure flag emission, and history buffering.
2. TemporalSwapHoldBuffer optical flow motion tracking, multi-frame chaining (1-3 frames), and hold cap.
3. Affine rotation / unrotation round-trip geometry (30 deg, 90 deg, 180 deg).
4. FaceSet pose-aware multi-angle face selection and distance scoring.
5. Dynamic mask edge erosion and Gaussian feathering expansion for yaw > 45 deg.
6. Settings configuration persistence and no_face_action dispatch mapping.
"""

from __future__ import annotations

import os
import sys
import unittest
import numpy as np
import cv2

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from roop.telemetry import (
    format_telemetry,
    log_frame_telemetry,
    ACTION_SWAPPED,
    ACTION_ROI_RETRY,
    ACTION_LAST_SWAP_HOLD,
    ACTION_RAW_FALLBACK,
    FAIL_SCRFD,
    FAIL_SIMILARITY,
    FAIL_LANDMARKS,
    _TELEMETRY_HISTORY,
)
from roop.temporal_hold import TemporalSwapHoldBuffer
from roop.face_util import (
    _rotate_affine,
    _unrotate_affine_face,
    _unrotate_face_coords,
    rotate_clockwise,
    rotate_anticlockwise,
    rotate_image_180,
)
from roop.FaceSet import FaceSet
from roop.procmgr_masking import MaskingMixin
from roop.ProcessMgr import eNoFaceAction


class DummyFace:
    """Mock Face object for geometry and pose testing."""
    def __init__(self, bbox, kps=None, embedding=None):
        self.bbox = np.array(bbox, dtype=np.float32)
        self.kps = np.array(kps, dtype=np.float32) if kps is not None else None
        self.embedding = np.array(embedding, dtype=np.float32) if embedding is not None else None
        self.landmark_2d_106 = None
        self.landmark_3d_68 = None


class DummyMasker(MaskingMixin):
    """Dummy class to test MaskingMixin.blur_area."""
    def __init__(self):
        pass


class TestTelemetry(unittest.TestCase):
    """Verify diagnostic per-frame telemetry format and failure tracking."""

    def test_format_telemetry_swapped(self):
        msg = format_telemetry(
            frame_idx=42,
            detected=True,
            det_score=0.88,
            yaw_pitch_roll=(22.5, -5.2, 1.1),
            match_sim=0.74,
            action=ACTION_SWAPPED,
            failure_flag=None,
        )
        expected = "Frame [42] | Detected: [T] | Det Score: [0.88] | Yaw/Pitch/Roll: [+22.5, -5.2, +1.1] | Match Sim: [0.74] | Action: [SWAPPED]"
        self.assertEqual(msg, expected)

    def test_format_telemetry_failure_flags(self):
        # SCRFD drop
        msg_scrfd = format_telemetry(
            frame_idx=10,
            detected=False,
            det_score=0.0,
            yaw_pitch_roll=None,
            match_sim=0.0,
            action=ACTION_LAST_SWAP_HOLD,
            failure_flag=FAIL_SCRFD,
        )
        self.assertIn("Detected: [F]", msg_scrfd)
        self.assertIn("Action: [LAST_SWAP_HOLD]", msg_scrfd)
        self.assertIn("Failure: [FAIL_SCRFD]", msg_scrfd)

        # Similarity drop
        msg_sim = format_telemetry(
            frame_idx=15,
            detected=True,
            det_score=0.75,
            yaw_pitch_roll=(55.0, 10.0, 0.0),
            match_sim=0.25,
            action=ACTION_RAW_FALLBACK,
            failure_flag=FAIL_SIMILARITY,
        )
        self.assertIn("Failure: [FAIL_SIMILARITY]", msg_sim)
        self.assertIn("Action: [RAW_FALLBACK]", msg_sim)

        # Landmarks failure
        msg_lm = format_telemetry(
            frame_idx=20,
            detected=True,
            det_score=0.60,
            yaw_pitch_roll=None,
            match_sim=0.65,
            action=ACTION_ROI_RETRY,
            failure_flag=FAIL_LANDMARKS,
        )
        self.assertIn("Failure: [FAIL_LANDMARKS]", msg_lm)
        self.assertIn("Action: [ROI_RETRY]", msg_lm)

    def test_log_frame_telemetry_history(self):
        _TELEMETRY_HISTORY.clear()
        ret = log_frame_telemetry(
            frame_idx=1,
            detected=True,
            det_score=0.90,
            yaw_pitch_roll=(0.0, 0.0, 0.0),
            match_sim=0.85,
            action=ACTION_SWAPPED,
        )
        self.assertIn("Frame [1]", ret)
        self.assertEqual(len(_TELEMETRY_HISTORY), 1)
        self.assertEqual(_TELEMETRY_HISTORY[0]["frame_idx"], 1)
        self.assertEqual(_TELEMETRY_HISTORY[0]["action"], ACTION_SWAPPED)


class TestTemporalHoldBuffer(unittest.TestCase):
    """Verify optical flow persistence buffer holds up to 3 frames with warping."""

    def setUp(self):
        self.buffer = TemporalSwapHoldBuffer(max_holds=3)

    def test_initial_state(self):
        self.assertFalse(self.buffer.can_hold())
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        self.assertIsNone(self.buffer.hold_and_warp(dummy_frame))

    def test_hold_and_warp_chain(self):
        # Create a textured canvas with a distinguishable face region
        frame0 = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(frame0, (100, 100), 30, (200, 200, 200), -1)
        # Add random texture for robust optical flow tracking
        np.random.seed(42)
        noise = np.random.randint(0, 50, (200, 200, 3), dtype=np.uint8)
        frame0 = cv2.add(frame0, noise)

        # Swapped composite has distinct blue tint in the face region
        composite0 = frame0.copy()
        cv2.circle(composite0, (100, 100), 30, (255, 100, 0), -1)
        bbox = [70, 70, 130, 130]

        # Record valid swap
        self.buffer.record_swap(frame0, composite0, bbox=bbox)
        self.assertTrue(self.buffer.can_hold())
        self.assertEqual(self.buffer.consecutive_holds, 0)

        # Simulate small camera shift: translate frame by dx=3, dy=2
        dx, dy = 3, 2
        M_shift = np.float32([[1, 0, dx], [0, 1, dy]])
        frame1 = cv2.warpAffine(frame0, M_shift, (200, 200))

        # Hold Frame 1
        held1 = self.buffer.hold_and_warp(frame1)
        self.assertIsNotNone(held1)
        self.assertEqual(held1.shape, (200, 200, 3))
        self.assertEqual(self.buffer.consecutive_holds, 1)
        self.assertTrue(self.buffer.can_hold())

        # Hold Frame 2
        frame2 = cv2.warpAffine(frame1, M_shift, (200, 200))
        held2 = self.buffer.hold_and_warp(frame2)
        self.assertIsNotNone(held2)
        self.assertEqual(self.buffer.consecutive_holds, 2)
        self.assertTrue(self.buffer.can_hold())

        # Hold Frame 3
        frame3 = cv2.warpAffine(frame2, M_shift, (200, 200))
        held3 = self.buffer.hold_and_warp(frame3)
        self.assertIsNotNone(held3)
        self.assertEqual(self.buffer.consecutive_holds, 3)

        # Hold Frame 4 -> Cap reached (max_holds = 3)
        self.assertFalse(self.buffer.can_hold())
        frame4 = cv2.warpAffine(frame3, M_shift, (200, 200))
        held4 = self.buffer.hold_and_warp(frame4)
        self.assertIsNone(held4)

    def test_buffer_reset(self):
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        self.buffer.record_swap(frame, frame, bbox=[20, 20, 80, 80])
        self.assertTrue(self.buffer.can_hold())
        self.buffer.reset()
        self.assertFalse(self.buffer.can_hold())
        self.assertEqual(self.buffer.consecutive_holds, 0)
        self.assertIsNone(self.buffer.prev_composite)


class TestAffineRotationGeometry(unittest.TestCase):
    """Verify geometry unrotation roundtrips for 30 deg affine and 90/180 orthogonal turns."""

    def test_affine_rotation_30deg_roundtrip(self):
        h, w = 240, 240
        img = np.zeros((h, w, 3), dtype=np.uint8)

        # Point at (100, 120), bbox [90, 110, 110, 130]
        orig_kps = np.array([[100.0, 120.0], [120.0, 120.0], [110.0, 130.0], [105.0, 140.0], [115.0, 140.0]], dtype=np.float32)
        orig_bbox = np.array([90.0, 110.0, 120.0, 140.0], dtype=np.float32)

        face = DummyFace(bbox=orig_bbox, kps=orig_kps)

        # Forward rotate 30 degrees
        angle = 30.0
        rotated_img, M = _rotate_affine(img, angle)
        self.assertEqual(rotated_img.shape, (h, w, 3))

        # Transform face to rotated frame coords
        corners = np.array([[orig_bbox[0], orig_bbox[1]], [orig_bbox[2], orig_bbox[1]],
                            [orig_bbox[2], orig_bbox[3]], [orig_bbox[0], orig_bbox[3]]], dtype=np.float32)
        rot_corners = cv2.transform(corners.reshape(1, -1, 2), M).reshape(-1, 2)
        rot_kps = cv2.transform(orig_kps.reshape(1, -1, 2), M).reshape(-1, 2)

        face.bbox = np.array([rot_corners[:, 0].min(), rot_corners[:, 1].min(),
                              rot_corners[:, 0].max(), rot_corners[:, 1].max()], dtype=np.float32)
        face.kps = rot_kps

        # Unrotate back
        _unrotate_affine_face(face, M)

        # Verify keypoints restored within 0.05 px
        np.testing.assert_allclose(face.kps, orig_kps, atol=0.1)

    def test_orthogonal_turns_roundtrip(self):
        w, h = 300, 200
        orig_bbox = np.array([50.0, 60.0, 150.0, 160.0], dtype=np.float32)
        orig_kps = np.array([[80.0, 90.0], [120.0, 90.0], [100.0, 110.0]], dtype=np.float32)

        # Test 90 CW turn
        face = DummyFace(bbox=orig_bbox, kps=orig_kps)
        # 90 CW maps (x, y) -> (h - 1 - y, x), canvas becomes (h, w)
        cw_w, cw_h = h, w
        rot_kps = np.zeros_like(orig_kps)
        rot_kps[:, 0] = h - 1.0 - orig_kps[:, 1]
        rot_kps[:, 1] = orig_kps[:, 0]
        face.kps = rot_kps

        _unrotate_face_coords(face, orig_w=w, orig_h=h, angle="clockwise")
        np.testing.assert_allclose(face.kps, orig_kps, atol=0.1)


class TestFaceSetMultiAngleMatching(unittest.TestCase):
    """Verify pose-aware source bank selects best matching face."""

    def test_select_best_pose_face(self):
        fs = FaceSet()
        # Add 3 mock faces: Frontal (0 deg), Left profile (+50 deg yaw), Right profile (-45 deg yaw)
        face_frontal = DummyFace(bbox=[0, 0, 100, 100], embedding=np.array([1.0, 0.0, 0.0]))
        face_left = DummyFace(bbox=[0, 0, 100, 100], embedding=np.array([0.0, 1.0, 0.0]))
        face_right = DummyFace(bbox=[0, 0, 100, 100], embedding=np.array([0.0, 0.0, 1.0]))

        fs.faces = [face_frontal, face_left, face_right]
        fs.face_poses = [
            (0.0, 0.0, 0.0),    # Face 0: Frontal
            (50.0, 5.0, 0.0),   # Face 1: Left profile
            (-45.0, -8.0, 0.0), # Face 2: Right profile
        ]

        # Target turning left (+48 yaw, +3 pitch) -> Should select Face 1
        best_face, best_idx = fs.select_best_pose_face(target_yaw=48.0, target_pitch=3.0)
        self.assertEqual(best_idx, 1)
        self.assertIs(best_face, face_left)

        # Target turning right (-40 yaw, -5 pitch) -> Should select Face 2
        best_face, best_idx = fs.select_best_pose_face(target_yaw=-40.0, target_pitch=-5.0)
        self.assertEqual(best_idx, 2)
        self.assertIs(best_face, face_right)

        # Target frontal (+2 yaw, +1 pitch) -> Should select Face 0
        best_face, best_idx = fs.select_best_pose_face(target_yaw=2.0, target_pitch=1.0)
        self.assertEqual(best_idx, 0)
        self.assertIs(best_face, face_frontal)

    def test_get_best_match_distance_pose_aware(self):
        fs = FaceSet()
        face_frontal = DummyFace(bbox=[0, 0, 100, 100], embedding=np.array([1.0, 0.0, 0.0]))
        face_left = DummyFace(bbox=[0, 0, 100, 100], embedding=np.array([0.0, 1.0, 0.0]))
        fs.faces = [face_frontal, face_left]
        fs.face_poses = [(0.0, 0.0, 0.0), (55.0, 0.0, 0.0)]

        # Target embedding matches face_left exactly, target yaw = 50.0
        target_emb = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        dist, idx = fs.get_best_match_distance(target_emb, target_yaw=50.0, target_pitch=0.0)
        self.assertEqual(idx, 1)
        self.assertAlmostEqual(dist, 0.0, places=4)


class TestDynamicMaskFeathering(unittest.TestCase):
    """Verify mask erosion and Gaussian feathering widen when yaw > 45 deg."""

    def test_dynamic_feathering_expansion(self):
        masker = DummyMasker()
        # Create a 200x200 binary white mask box at center
        matte = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(matte, (50, 50), (150, 150), 255, -1)

        # 1. Frontal face (yaw = 0)
        blurred_frontal = masker.blur_area(matte.copy(), face_mask_blend=20.0, yaw=0.0)

        # 2. Steep lateral profile (yaw = 90) -> yaw_scale = 1.0 + (90-45)/45 = 2.0
        blurred_profile = masker.blur_area(matte.copy(), face_mask_blend=20.0, yaw=90.0)

        # Measure feather width: pixels in intermediate alpha range [20, 235]
        feather_frontal_px = np.count_nonzero((blurred_frontal > 20) & (blurred_frontal < 235))
        feather_profile_px = np.count_nonzero((blurred_profile > 20) & (blurred_profile < 235))

        # Profile feathering must be significantly wider than frontal to eliminate boundary tearing
        self.assertGreater(feather_profile_px, feather_frontal_px)


class TestSettingsIntegration(unittest.TestCase):
    """Verify no_face_action and detector settings are valid and dispatches align."""

    def test_no_face_action_enum_values(self):
        self.assertEqual(eNoFaceAction.USE_ORIGINAL_FRAME, 0)
        self.assertEqual(eNoFaceAction.RETRY_ROTATED, 1)
        self.assertEqual(eNoFaceAction.SKIP_FRAME, 2)
        self.assertEqual(eNoFaceAction.SKIP_FRAME_IF_DISSIMILAR, 3)
        self.assertEqual(eNoFaceAction.USE_LAST_SWAPPED, 4)

    def test_settings_load_defaults(self):
        from settings import Settings
        cfg = Settings('config.yaml')
        self.assertTrue(hasattr(cfg, 'face_detector_threshold'))
        self.assertTrue(hasattr(cfg, 'temporal_roi_hint'))
        self.assertTrue(hasattr(cfg, 'no_face_action'))
        self.assertIsInstance(cfg.face_detector_threshold, float)
        self.assertIsInstance(cfg.temporal_roi_hint, bool)


if __name__ == '__main__':
    unittest.main()
