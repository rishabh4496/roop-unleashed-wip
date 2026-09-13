"""Profile admissions must not alternate with detector/interpolation cadence."""

import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from insightface.app.common import Face

from roop.face_identity import GeometryFilter
from roop.procmgr_tracking import TrackingMixin, _adaptive_tracks_allow_skip
from tests.facegeom import project_kps


def face(yaw=0, roll=0, scale=60, cx=160, interpolated=False):
    kps = project_kps(yaw, roll_deg=roll, scale=scale, cx=cx, cy=160)
    # Include forehead and chin, like a detector box; keep its size stable
    # through the turn so this test isolates foreshortened eye separation.
    return Face(bbox=np.array([cx - 65, 80, cx + 65, 240], np.float32),
                kps=kps, det_score=0.95, embedding=np.array([1., 0., 0.]),
                _interpolated=interpolated)


class ProfileGeometryTest(unittest.TestCase):
    def setUp(self):
        self.guard = GeometryFilter(min_det_score=0, min_box_px=48,
                                    min_box_frac=0, min_interocular_px=24)

    def test_large_face_survives_yaw_and_roll(self):
        for yaw in (-90, -85, -70, -45, 0, 45, 70, 85, 90):
            for roll in (-180, -90, -30, 0, 30, 90, 180):
                with self.subTest(yaw=yaw, roll=roll):
                    self.assertIsNone(self.guard.reject_reason(face(yaw, roll)))

    def test_collapsed_points_cannot_use_a_large_box_to_pass(self):
        f = face()
        f.kps[:] = [160, 160]
        self.assertIsNotNone(self.guard.reject_reason(f))

    def test_tiny_feature_cluster_still_fails(self):
        for yaw in (0, 85):
            self.assertIsNotNone(self.guard.reject_reason(face(yaw, scale=15)))

    def test_small_boxes_are_not_exempted_for_profiles(self):
        f = face(85)
        f.bbox = np.array([150, 150, 170, 170])
        self.assertIn('box', self.guard.reject_reason(f))

    def test_invalid_geometry_is_refused(self):
        for value in (np.full((5, 2), np.nan), np.zeros((2, 2)),
                      np.full((5, 2), np.inf), np.full((5, 2), 10000)):
            f = face(85)
            f.kps = value
            self.assertIsNotNone(self.guard.reject_reason(f))
        for value in ([0, 0, np.nan, 100], [0, 0, 100]):
            f = face()
            f.bbox = value
            self.assertIsNotNone(self.guard.reject_reason(f))

    def test_detector_and_prediction_get_identical_verdicts(self):
        for yaw in (0, 60, 85, 90):
            for scale in (15, 60):
                self.assertEqual(
                    self.guard.reject_reason(face(yaw, scale=scale)),
                    self.guard.reject_reason(face(yaw, scale=scale, interpolated=True)))


class AdaptiveProfileTest(unittest.TestCase):
    @staticmethod
    def track(f, seen=10):
        return {'last_seen': seen, 'obs': {seen: f}}

    def test_isolated_frontal_face_can_skip(self):
        self.assertTrue(_adaptive_tracks_allow_skip([self.track(face())], 10))

    def test_profile_needs_detection_even_on_a_quiet_thumbnail(self):
        for yaw in (-90, -70, 70, 90):
            for roll in (0, 90, 180):
                self.assertFalse(_adaptive_tracks_allow_skip(
                    [self.track(face(yaw, roll))], 10))

    def test_adjacent_faces_need_detection(self):
        self.assertFalse(_adaptive_tracks_allow_skip(
            [self.track(face(cx=160)), self.track(face(cx=300))], 10))

    def test_separated_frontal_faces_can_skip(self):
        self.assertTrue(_adaptive_tracks_allow_skip(
            [self.track(face(cx=160)), self.track(face(cx=550))], 10))

    def test_known_detection_miss_cannot_be_skipped(self):
        self.assertFalse(_adaptive_tracks_allow_skip([self.track(face())], 11))
        self.assertFalse(_adaptive_tracks_allow_skip([], 11))


class SwapAdmissionIntegrationTest(unittest.TestCase):
    def test_real_and_interpolated_profiles_both_reach_compositor(self):
        from roop.ProcessMgr import ProcessMgr
        import roop.globals

        mgr = ProcessMgr(None)
        mgr.options = SimpleNamespace(swap_mode='all', selected_index=0)
        mgr._temporal_mode = True
        mgr._temporal_covered = 4
        mgr._temporal_faces = {
            i: [face(85, interpolated=bool(i % 2))] for i in range(4)}
        calls = []

        def composite(pending, plate, output, others):
            calls.append(len(pending))
            return output

        mgr._composite_faces = composite
        frame = np.zeros((320, 640, 3), np.uint8)
        with patch.object(roop.globals, 'vr_mode', False):
            counts = [mgr.swap_faces(frame, frame.copy(), frame_idx=i)[0]
                      for i in range(4)]
        self.assertEqual(counts, [1, 1, 1, 1])
        self.assertEqual(calls, [1, 1, 1, 1])


class RescueCoverageTest(unittest.TestCase):
    def test_adaptive_prepass_observes_every_profile_frame_with_detector_pool(self):
        import roop.face_util as fu
        import roop.globals
        from roop import procmgr_tracking as tracking

        class Manager(TrackingMixin):
            options = SimpleNamespace(face_distance_threshold=0.75)
            target_face_datas = []
            target_face_groups = []
            progress_gradio = None

            def _publish_live(self, frame):
                pass

        frames = [np.zeros((320, 720, 3), np.uint8) for _ in range(20)]
        with (patch.object(roop.globals, 'processing', True),
              patch.object(tracking.session_pool, 'detmask_pooling_enabled', return_value=True),
              patch.object(tracking.session_pool, 'detmask_pool_size', return_value=2),
              patch.object(fu, 'get_all_faces', side_effect=lambda _: [face(85)]) as detect,
              contextlib.redirect_stdout(io.StringIO())):
            mgr = Manager()
            tracks = mgr._precompute_tracks(None, 0, 20, 20, awebp_frames=frames,
                                             step=2, adaptive=True, collect_obs=True)
        self.assertEqual(detect.call_count, 20)
        self.assertEqual(set(tracks[0]['obs']), set(range(20)))

    def test_new_arrival_does_not_hide_missing_target_at_equal_count(self):
        """Run the real pre-pass with scripted detector results, no GPU models."""
        import roop.face_util as fu
        import roop.globals
        from roop import procmgr_tracking as tracking

        class Manager(TrackingMixin):
            options = SimpleNamespace(face_distance_threshold=0.75)
            target_face_datas = []
            target_face_groups = []
            progress_gradio = None

            def _publish_live(self, frame):
                pass

        target = face(cx=160)
        newcomer = face(cx=550)
        newcomer.embedding = np.array([0., 1., 0.])
        frames = [np.zeros((320, 720, 3), np.uint8) for _ in range(2)]
        with (patch.object(roop.globals, 'processing', True),
              patch.object(tracking.session_pool, 'detmask_pooling_enabled', return_value=False),
              patch.object(tracking, '_TRACK_ROI_RESCUE', True),
              patch.object(fu, 'get_all_faces', side_effect=[[target], [newcomer]]),
              patch.object(fu, 'get_all_faces_in_roi', return_value=[face(cx=160)]) as rescue,
              contextlib.redirect_stdout(io.StringIO())):
            mgr = Manager()
            tracks = mgr._precompute_tracks(None, 0, 2, 2, awebp_frames=frames,
                                             step=1, collect_obs=True)
        self.assertEqual(rescue.call_count, 1)
        recovered = [t for t in tracks if 0 in t.get('obs', {})]
        self.assertEqual(len(recovered), 1)
        self.assertIn(1, recovered[0]['obs'])
        self.assertTrue(recovered[0]['obs'][1].get('_roi_rescue'))


if __name__ == '__main__':
    unittest.main()
