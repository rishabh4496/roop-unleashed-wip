"""estimate_norm — the 5-point crop alignment, and the opt-in profile variant.

The headline guarantee here is that `yaw_align` off is a BIT-EXACT no-op. It is
opt-in precisely because it changes the crop, and therefore the swap output, for
high-yaw faces; if it ever leaked into the default path it would silently change
every render. test_off_is_bit_exact_noop is what makes that claim checkable
instead of a sentence in a commit message.
"""

import os
import sys
import unittest

import numpy as np
from skimage import transform as trans

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import roop.globals                                              # noqa: E402
from roop.face_util import (arcface_dst, estimate_norm,          # noqa: E402
                            kps_pose_ratios, WARP_TEMPLATES,
                            YAW_ALIGN_RATIO)
from tests.facegeom import decompose, fit_residual, project_kps  # noqa: E402

SIZES = (112, 128, 256, 512, 320)
MODES = ("arcface", "arcface_112_v1", "arcface_112_v2", "ffhq_512", "mtcnn_512")


def reference_template(image_size, mode):
    """The destination points estimate_norm targets, rebuilt independently so a
    change to its template maths shows up as a test failure."""
    if mode in WARP_TEMPLATES:
        return WARP_TEMPLATES[mode] * float(image_size)
    if image_size % 112 == 0:
        ratio, shift = float(image_size) / 112.0, 0.0
    elif image_size % 128 == 0:
        ratio, shift = float(image_size) / 128.0, 8.0 * float(image_size) / 128.0
    elif image_size % 512 == 0:
        ratio, shift = float(image_size) / 512.0, 32.0 * float(image_size) / 512.0
    else:
        ratio, shift = float(image_size) / 112.0, 0.0
    dst = arcface_dst * ratio
    dst[:, 0] += shift
    return dst


class YawAlignCase(unittest.TestCase):
    """Restores the global so one test can never leak into another."""

    def setUp(self):
        self._saved = roop.globals.yaw_align

    def tearDown(self):
        roop.globals.yaw_align = self._saved


class TestDefaultPath(YawAlignCase):
    def test_off_is_bit_exact_noop(self):
        """With the toggle off, estimate_norm must equal the plain similarity
        fit exactly — not approximately — for every size, template and pose."""
        roop.globals.yaw_align = False
        checked = 0
        for size in SIZES:
            for mode in MODES:
                for yaw in (0, 30, 60, 75, 85, 90):
                    for pitch in (-25, 0, 25):
                        for roll in (-20, 0, 20):
                            kps = project_kps(yaw, pitch, roll)
                            got = estimate_norm(kps, size, mode)
                            ref = trans.SimilarityTransform()
                            ref.estimate(kps, reference_template(size, mode))
                            np.testing.assert_array_equal(
                                got, ref.params[0:2, :],
                                f"size={size} mode={mode} yaw={yaw} "
                                f"pitch={pitch} roll={roll}")
                            checked += 1
        self.assertEqual(checked, 1350)

    def test_toggling_back_off_restores_exactly(self):
        kps = project_kps(90, 10)
        roop.globals.yaw_align = False
        before = estimate_norm(kps, 512, "arcface")
        roop.globals.yaw_align = True
        estimate_norm(kps, 512, "arcface")
        roop.globals.yaw_align = False
        np.testing.assert_array_equal(estimate_norm(kps, 512, "arcface"), before)

    def test_reads_the_global_live(self):
        """Must not capture the setting at import time, or the UI toggle would
        need an app restart to take effect."""
        kps = project_kps(90)
        roop.globals.yaw_align = False
        off = estimate_norm(kps, 512, "arcface")
        roop.globals.yaw_align = True
        self.assertFalse(np.array_equal(estimate_norm(kps, 512, "arcface"), off))


class TestProfileAlignment(YawAlignCase):
    def setUp(self):
        super().setUp()
        roop.globals.yaw_align = True

    def test_gate_leaves_frontal_and_mid_angles_untouched(self):
        """Only near-profile faces may be affected; everything else must be
        bit-identical to the default fit."""
        roop.globals.yaw_align = False
        baseline = {y: estimate_norm(project_kps(y), 512, "arcface")
                    for y in (0, 15, 30, 45, 55)}
        roop.globals.yaw_align = True
        for yaw, expected in baseline.items():
            self.assertGreaterEqual(kps_pose_ratios(project_kps(yaw))[0],
                                    YAW_ALIGN_RATIO)
            np.testing.assert_array_equal(
                estimate_norm(project_kps(yaw), 512, "arcface"), expected,
                f"profile alignment leaked into yaw={yaw}")

    def test_removes_nod_coupled_rotation_swing(self):
        """The defect being fixed: with the plain fit the collapsed eye pair
        makes rotation ill-conditioned, so pitch leaks into in-plane roll — a
        ~30 deg swing at 90 deg yaw as the head nods."""
        for yaw in (75, 90):
            roop.globals.yaw_align = False
            plain = [decompose(estimate_norm(project_kps(yaw, p), 512, "arcface"))[1]
                     for p in (-25, 25)]
            roop.globals.yaw_align = True
            fixed = [decompose(estimate_norm(project_kps(yaw, p), 512, "arcface"))[1]
                     for p in (-25, 25)]
            plain_swing = abs(plain[1] - plain[0])
            fixed_swing = abs(fixed[1] - fixed[0])
            self.assertGreater(plain_swing, 20.0,
                               f"expected a large plain-fit swing at yaw={yaw}")
            self.assertLess(fixed_swing, 2.0,
                            f"swing still {fixed_swing:.1f} deg at yaw={yaw}")

    def test_output_is_always_finite_and_invertible(self):
        import cv2
        for yaw in (70, 75, 80, 85, 90):
            for pitch in (-30, 0, 30):
                for roll in (-30, 0, 30):
                    m = estimate_norm(project_kps(yaw, pitch, roll), 512, "arcface")
                    self.assertTrue(np.isfinite(m).all())
                    scale, _ = decompose(m)
                    self.assertGreater(scale, 1e-6)
                    cv2.invertAffineTransform(m)   # must not raise

    def test_residual_tradeoff_is_bounded(self):
        """Constraining the rotation costs some least-squares fit accuracy —
        expected, because the plain fit was buying a lower residual by rotating
        the face. Guard that the cost stays modest rather than exploding."""
        roop.globals.yaw_align = False
        kps = project_kps(90, 25)
        dst = reference_template(512, "arcface")
        plain = fit_residual(estimate_norm(kps, 512, "arcface"), kps, dst)
        roop.globals.yaw_align = True
        fixed = fit_residual(estimate_norm(kps, 512, "arcface"), kps, dst)
        self.assertLess(fixed, plain * 1.25,
                        f"residual grew from {plain:.1f} to {fixed:.1f} px")


class TestModeSelection(YawAlignCase):
    """yaw_align accepts booleans (legacy) and mode names. Anything
    unrecognised must fall back to 'off' rather than silently enabling a mode
    that changes render output."""

    def test_normalisation(self):
        from roop.face_util import _yaw_align_mode
        for value, expected in (
            (False, 'off'), (True, 'stabilize'), (None, 'off'), ('', 'off'),
            ('off', 'off'), ('stabilize', 'stabilize'), ('pose', 'pose'),
            ('POSE', 'pose'), ('  Pose  ', 'pose'),
            ('nonsense', 'off'), ('1', 'off'),
        ):
            roop.globals.yaw_align = value
            self.assertEqual(_yaw_align_mode(), expected, f"value={value!r}")

    def test_unknown_mode_is_bit_exact_off(self):
        kps = project_kps(90, 10)
        roop.globals.yaw_align = 'off'
        expected = estimate_norm(kps, 512, "arcface")
        for value in ('nonsense', '1', 0, None):
            roop.globals.yaw_align = value
            np.testing.assert_array_equal(estimate_norm(kps, 512, "arcface"),
                                          expected, f"value={value!r}")


class TestPoseTemplate(YawAlignCase):
    def setUp(self):
        super().setUp()
        roop.globals.yaw_align = 'pose'

    def test_gated_off_for_frontal_and_mid_angles(self):
        roop.globals.yaw_align = 'off'
        baseline = {y: estimate_norm(project_kps(y), 512, "arcface")
                    for y in (0, 15, 30, 45, 55)}
        roop.globals.yaw_align = 'pose'
        for yaw, expected in baseline.items():
            np.testing.assert_array_equal(
                estimate_norm(project_kps(yaw), 512, "arcface"), expected,
                f"pose template leaked into yaw={yaw}")

    def test_cuts_the_fit_residual_at_high_yaw(self):
        """The point of the mode: a template congruent to the input makes the
        similarity fit well posed instead of a shear/rotation compromise."""
        for yaw in (75, 85, 90):
            kps = project_kps(yaw, 20)
            dst = reference_template(512, "arcface")
            roop.globals.yaw_align = 'off'
            plain = fit_residual(estimate_norm(kps, 512, "arcface"), kps, dst)
            roop.globals.yaw_align = 'pose'
            posed = estimate_norm(kps, 512, "arcface")
            # Residual is measured against the POSE template the fit targeted.
            from roop.face_util import _pose_template, _yaw_from_ratio
            target = _pose_template(_yaw_from_ratio(kps_pose_ratios(kps)[0]), dst)
            improved = fit_residual(posed, kps, target)
            self.assertLess(improved, plain * 0.6,
                            f"yaw={yaw}: residual {plain:.1f} -> {improved:.1f} px")

    def test_exact_fit_at_zero_pitch(self):
        """At the pose the template was built for, input and template are
        congruent, so the residual should collapse to ~0."""
        from roop.face_util import _pose_template, _yaw_from_ratio
        dst = reference_template(512, "arcface")
        for yaw in (60, 75, 90):
            kps = project_kps(yaw)
            m = estimate_norm(kps, 512, "arcface")
            target = _pose_template(_yaw_from_ratio(kps_pose_ratios(kps)[0]), dst)
            self.assertLess(fit_residual(m, kps, target), 1.0, f"yaw={yaw}")

    def test_face_size_stays_constant_through_a_turn(self):
        """Scale is taken from the FRONTAL reference so the head does not appear
        to zoom as it turns."""
        scales = [decompose(estimate_norm(project_kps(y), 512, "arcface"))[0]
                  for y in (70, 75, 80, 85, 90)]
        self.assertLess(max(scales) / min(scales), 1.15,
                        f"crop scale swings through a turn: {scales}")

    def test_output_is_always_finite_and_invertible(self):
        import cv2
        for yaw in (70, 80, 90):
            for pitch in (-30, 0, 30):
                for roll in (-30, 0, 30):
                    m = estimate_norm(project_kps(yaw, pitch, roll), 512, "arcface")
                    self.assertTrue(np.isfinite(m).all())
                    self.assertGreater(decompose(m)[0], 1e-6)
                    cv2.invertAffineTransform(m)


class TestYawLookup(unittest.TestCase):
    def test_inverts_accurately_where_it_is_actually_used(self):
        """Only the near-profile range matters: the pose template is gated to
        yaw_ratio < YAW_ALIGN_RATIO, i.e. roughly 65 deg and above."""
        from roop.face_util import _yaw_from_ratio
        for yaw in (65, 70, 75, 80, 85, 90):
            ratio = kps_pose_ratios(project_kps(yaw))[0]
            self.assertLess(ratio, YAW_ALIGN_RATIO)      # confirms it is in range
            self.assertAlmostEqual(_yaw_from_ratio(ratio), yaw, delta=1.0)

    def test_is_ill_conditioned_near_frontal_by_construction(self):
        """Eye separation falls off as cos(yaw), so the ratio's derivative
        vanishes at 0 deg and the inversion cannot be sharp there. Harmless —
        that range is gated out — but asserted so nobody 'fixes' the lookup or
        starts trusting it for frontal poses."""
        from roop.face_util import _yaw_from_ratio
        error = abs(_yaw_from_ratio(kps_pose_ratios(project_kps(0))[0]) - 0.0)
        self.assertGreater(error, 1.0, "inversion is now sharp at frontal — if "
                                       "that is real, this test can go")
        self.assertLess(error, 10.0)

    def test_clamps_outside_the_table(self):
        from roop.face_util import _yaw_from_ratio
        self.assertGreaterEqual(_yaw_from_ratio(-5.0), 0.0)
        self.assertLessEqual(_yaw_from_ratio(-5.0), 90.0)
        self.assertGreaterEqual(_yaw_from_ratio(99.0), 0.0)
        self.assertLessEqual(_yaw_from_ratio(99.0), 90.0)


class TestAlignmentIsIllConditionedAtProfile(unittest.TestCase):
    """Records WHY the profile path exists, so the motivation survives even if
    the implementation is rewritten."""

    def test_eye_separation_collapses(self):
        sep = lambda y: float(np.linalg.norm(project_kps(y)[1] - project_kps(y)[0]))
        self.assertGreater(sep(0), 130.0)
        self.assertAlmostEqual(sep(90), 0.0, places=4)

    def test_fit_residual_explodes_with_yaw(self):
        dst = reference_template(512, "arcface")
        saved = roop.globals.yaw_align
        roop.globals.yaw_align = False
        try:
            frontal = fit_residual(estimate_norm(project_kps(0), 512, "arcface"),
                                   project_kps(0), dst)
            profile = fit_residual(estimate_norm(project_kps(90), 512, "arcface"),
                                   project_kps(90), dst)
        finally:
            roop.globals.yaw_align = saved
        self.assertLess(frontal, 15.0)
        self.assertGreater(profile, 50.0)
        self.assertGreater(profile, frontal * 4)



class TestRotatedFaceHandling(unittest.TestCase):
    def test_unrotate_face_coords_clockwise(self):
        from roop.face_util import _unrotate_face_coords
        from insightface.app.common import Face

        face = Face(
            bbox=np.array([799.0, 700.0, 899.0, 800.0], dtype=np.float32),
            kps=np.array([[799.0, 700.0]], dtype=np.float32)
        )
        orig_w, orig_h = 800, 1000
        _unrotate_face_coords(face, orig_w, orig_h, "clockwise")
        self.assertAlmostEqual(float(face.kps[0][0]), 700.0, places=4)
        self.assertAlmostEqual(float(face.kps[0][1]), 200.0, places=4)

    def test_unrotate_face_coords_anticlockwise(self):
        from roop.face_util import _unrotate_face_coords
        from insightface.app.common import Face

        face = Face(
            bbox=np.array([200.0, 99.0, 300.0, 199.0], dtype=np.float32),
            kps=np.array([[200.0, 99.0]], dtype=np.float32)
        )
        orig_w, orig_h = 800, 1000
        _unrotate_face_coords(face, orig_w, orig_h, "anticlockwise")
        self.assertAlmostEqual(float(face.kps[0][0]), 700.0, places=4)
        self.assertAlmostEqual(float(face.kps[0][1]), 200.0, places=4)

    def test_rotation_action_90deg_clockwise_roll(self):
        from roop.ProcessMgr import ProcessMgr
        from insightface.app.common import Face

        mgr = ProcessMgr(None)
        # Head lying on right cheek (nose points right)
        # Left eye: (100, 100), Right eye: (100, 150), Nose: (130, 125)
        # mx = 100, my = 125 -> vx = 30, vy = 0 (vx > |vy| -> rotate_anticlockwise)
        face = Face(
            bbox=np.array([80, 80, 160, 160], dtype=np.float32),
            kps=np.array([[100, 100], [100, 150], [130, 125], [90, 100], [90, 150]], dtype=np.float32)
        )
        dummy_frame = np.zeros((400, 400, 3), dtype=np.uint8)
        action = mgr.rotation_action(face, dummy_frame)
        self.assertEqual(action, "rotate_anticlockwise")

    def test_rotation_action_90deg_anticlockwise_roll(self):
        from roop.ProcessMgr import ProcessMgr
        from insightface.app.common import Face

        mgr = ProcessMgr(None)
        # Head lying on left cheek (nose points left)
        # Left eye: (100, 150), Right eye: (100, 100), Nose: (70, 125)
        # mx = 100, my = 125 -> vx = -30, vy = 0 (-vx > |vy| -> rotate_clockwise)
        face = Face(
            bbox=np.array([60, 80, 140, 160], dtype=np.float32),
            kps=np.array([[100, 150], [100, 100], [70, 125], [90, 150], [90, 100]], dtype=np.float32)
        )
        dummy_frame = np.zeros((400, 400, 3), dtype=np.uint8)
        action = mgr.rotation_action(face, dummy_frame)
        self.assertEqual(action, "rotate_clockwise")


if __name__ == "__main__":
    unittest.main()
