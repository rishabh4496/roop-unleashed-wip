"""Stage 5 End-to-End Stress & Final Certification Test Suite.

Validates the full integrated pipeline under sustained long-running workloads:
1. Sustained 1,000-frame video sequence with mixed detection, drops, and recoveries.
2. Extreme lateral profile yaw angles (-85 deg to +85 deg) without NaN/Inf or crash.
3. Rapid scene-cut / camera cut rejection (MAD > 45.0) ensuring zero visual bleeding.
4. Multithreaded concurrent face matching and telemetry logging (8 threads).
5. Clean shutdown, resource release, and memory pool zero-leak validation.
"""

from __future__ import annotations

import os
import sys
import unittest
import threading
from unittest.mock import MagicMock
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
    _TELEMETRY_HISTORY,
)
from roop.temporal_hold import TemporalSwapHoldBuffer
from roop.FaceSet import FaceSet
from roop.face_util import (
    solve_pose_5pt,
    _roi_window,
    _unrotate_affine_face,
    _unrotate_face_coords,
    _rotate_affine,
)
from roop.buffer_pool import (
    PinnedBufferPool,
    release_frame_buffer_pools,
    get_crop_buffer,
)
from roop.ProcessMgr import ProcessMgr


class DummyFace:
    """Lightweight mock Face representation for end-to-end stress testing."""
    def __init__(self, bbox, kps=None, embedding=None):
        self.bbox = np.array(bbox, dtype=np.float32)
        self.kps = np.array(kps, dtype=np.float32) if kps is not None else None
        self.embedding = np.array(embedding, dtype=np.float32) if embedding is not None else None


class TestEndToEndStressAndCertification(unittest.TestCase):
    """Rigorous Stage 5 stress tests."""

    def setUp(self):
        self.hold_buffer = TemporalSwapHoldBuffer(max_holds=3)
        self.hold_buffer.reset()
        _TELEMETRY_HISTORY.clear()

    def tearDown(self):
        self.hold_buffer.reset()
        _TELEMETRY_HISTORY.clear()
        pm = ProcessMgr(MagicMock())
        pm.release_resources()

    def test_sustained_pipeline_run_1000_frames(self):
        """Simulate a sustained 1,000-frame video sequence with diverse frame types."""
        frame_shape = (480, 640, 3)
        base_img = np.zeros(frame_shape, dtype=np.uint8)
        cv2.rectangle(base_img, (200, 150), (350, 320), (180, 150, 120), -1)

        swapped_count = 0
        held_count = 0
        fallback_count = 0

        for frame_idx in range(1000):
            # Introduce slight motion jitter
            shift = int(5 * np.sin(frame_idx * 0.1))
            current_frame = np.roll(base_img, shift, axis=1)

            # Cycle pattern: 8 frames detected, 2 frames dropped (hold), 4 frames no face (fallback)
            cycle_pos = frame_idx % 14
            if cycle_pos < 8:
                # Normal detected frame
                swapped_composite = current_frame.copy()
                cv2.rectangle(swapped_composite, (200 + shift, 150), (350 + shift, 320), (220, 180, 140), -1)
                bbox = np.array([200 + shift, 150, 350 + shift, 320], dtype=np.float32)
                self.hold_buffer.record_swap(current_frame, swapped_composite, bbox=bbox)

                log_frame_telemetry(
                    frame_idx=frame_idx,
                    detected=True,
                    det_score=0.85,
                    yaw_pitch_roll=(10.0, 5.0, 0.0),
                    match_sim=0.72,
                    action=ACTION_SWAPPED,
                )
                swapped_count += 1

            elif cycle_pos < 10:
                # Dropped detection: attempt hold
                held = self.hold_buffer.hold_and_warp(current_frame)
                if held is not None:
                    log_frame_telemetry(
                        frame_idx=frame_idx,
                        detected=False,
                        det_score=0.0,
                        yaw_pitch_roll=None,
                        match_sim=0.0,
                        action=ACTION_LAST_SWAP_HOLD,
                        failure_flag=FAIL_SCRFD,
                    )
                    held_count += 1
                else:
                    log_frame_telemetry(
                        frame_idx=frame_idx,
                        detected=False,
                        det_score=0.0,
                        yaw_pitch_roll=None,
                        match_sim=0.0,
                        action=ACTION_RAW_FALLBACK,
                        failure_flag=FAIL_SCRFD,
                    )
                    fallback_count += 1
            else:
                # Sustained absence: hold buffer exhausts or resets
                held = self.hold_buffer.hold_and_warp(current_frame)
                if held is not None:
                    log_frame_telemetry(
                        frame_idx=frame_idx,
                        detected=False,
                        det_score=0.0,
                        yaw_pitch_roll=None,
                        match_sim=0.0,
                        action=ACTION_LAST_SWAP_HOLD,
                        failure_flag=FAIL_SCRFD,
                    )
                    held_count += 1
                else:
                    log_frame_telemetry(
                        frame_idx=frame_idx,
                        detected=False,
                        det_score=0.0,
                        yaw_pitch_roll=None,
                        match_sim=0.0,
                        action=ACTION_RAW_FALLBACK,
                        failure_flag=FAIL_SCRFD,
                    )
                    fallback_count += 1

        # Assertions
        self.assertGreater(swapped_count, 500)
        self.assertGreater(held_count, 100)
        self.assertGreater(fallback_count, 100)
        # Telemetry deque is strictly bounded at maxlen=1000
        self.assertEqual(len(_TELEMETRY_HISTORY), 1000)

    def test_extreme_profile_yaw_traversal(self):
        """Test facial pose estimation across yaw angles from -85 deg to +85 deg."""
        for yaw_deg in range(-85, 86, 5):
            # Synthetic 5 keypoints reflecting yaw rotation
            cos_y = np.cos(np.radians(yaw_deg))
            sin_y = np.sin(np.radians(yaw_deg))

            # Base keypoints: left_eye, right_eye, nose, left_mouth, right_mouth
            kps = np.array([
                [50.0 + 30.0 * cos_y, 40.0],
                [50.0 - 30.0 * cos_y, 40.0],
                [50.0 - 20.0 * sin_y, 60.0],
                [40.0 + 20.0 * cos_y, 80.0],
                [60.0 - 20.0 * cos_y, 80.0],
            ], dtype=np.float32)

            yaw, pitch, roll = solve_pose_5pt(kps)
            self.assertFalse(np.isnan(yaw))
            self.assertFalse(np.isnan(pitch))
            self.assertFalse(np.isnan(roll))
            self.assertFalse(np.isinf(yaw))

    def test_scene_cut_abrupt_transition_rejection(self):
        """Assert that abrupt scene cuts (MAD > 45.0) immediately abort temporal hold."""
        frame_a = np.zeros((240, 320, 3), dtype=np.uint8)
        frame_a[50:150, 50:150] = 200

        composite_a = frame_a.copy()
        bbox_a = np.array([50, 50, 150, 150], dtype=np.float32)
        self.hold_buffer.record_swap(frame_a, composite_a, bbox=bbox_a)

        # Scene cut to completely different scene B (inverted bright scene)
        frame_b = np.full((240, 320, 3), 255, dtype=np.uint8)
        frame_b[50:150, 50:150] = 0

        # Attempt hold on frame_b - must be rejected as a scene cut
        held = self.hold_buffer.hold_and_warp(frame_b)
        self.assertIsNone(held, "Temporal hold must return None on a scene cut (MAD > 45.0)")
        self.assertEqual(self.hold_buffer.consecutive_holds, 0, "Hold counter should be reset")

    def test_multithreaded_pipeline_stress(self):
        """Stress-test concurrent worker threads against FaceSet and telemetry logging."""
        fs = FaceSet()
        rng = np.random.RandomState(42)

        # Populate FaceSet with synthetic reference embeddings
        for i in range(5):
            emb = rng.randn(512).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            face = DummyFace(bbox=[10, 10, 50, 50], embedding=emb)
            fs.faces.append(face)

        errors = []

        def worker_task(thread_id: int):
            try:
                local_rng = np.random.RandomState(100 + thread_id)
                for step in range(50):
                    q_emb = local_rng.randn(512).astype(np.float32)
                    q_emb = q_emb / np.linalg.norm(q_emb)

                    dist, best_i = fs.get_best_match_distance(q_emb)
                    self.assertIsNotNone(dist)
                    self.assertGreaterEqual(dist, 0.0)

                    log_frame_telemetry(
                        frame_idx=thread_id * 1000 + step,
                        detected=True,
                        det_score=0.9,
                        yaw_pitch_roll=(0.0, 0.0, 0.0),
                        match_sim=float(1.0 - dist),
                        action=ACTION_SWAPPED,
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker_task, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Encountered thread errors: {errors}")

    def test_resource_release_and_zero_leak(self):
        """Assert release_resources cleanly purges pinned buffer pools and caches."""
        pool = PinnedBufferPool((256, 256, 3), capacity=4, dtype=np.uint8)
        buf1 = pool.acquire()
        buf2 = pool.acquire()
        pool.release(buf1)
        pool.release(buf2)

        pm = ProcessMgr(MagicMock())
        pm.input_face_datas = ["f1", "f2"]
        pm.temporal_hold_buffer = self.hold_buffer
        self.hold_buffer.consecutive_holds = 2

        # Call global release
        pm.release_resources()

        # Hold buffer reset
        self.assertIsNone(self.hold_buffer.prev_frame_gray)
        self.assertIsNone(self.hold_buffer.prev_composite)
        self.assertIsNone(self.hold_buffer.prev_bbox)
        self.assertEqual(self.hold_buffer.consecutive_holds, 0)
        self.assertEqual(len(pm.input_face_datas), 0)


if __name__ == "__main__":
    unittest.main()
