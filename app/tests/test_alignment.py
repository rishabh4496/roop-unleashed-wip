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

    def test_every_rescue_angle_round_trips_through_the_real_helpers(self):
        """The two tests above check hand-worked numbers, which is how an angle
        can be added with an un-map that was never exercised. This one asks the
        shipped rotate helper where a pixel actually WENT, then requires the
        un-rotation to bring it back -- so a new angle cannot ship half-wired."""
        from roop.face_util import _unrotate_face_coords, _rescue_rotated
        from roop.ProcessMgr import ProcessMgr
        from insightface.app.common import Face
        import inspect

        h, w = 13, 9
        marker = np.zeros((h, w), dtype=np.int32)
        for y in range(h):
            for x in range(w):
                marker[y, x] = y * w + x

        angles = {"clockwise": "rotate_clockwise",
                  "anticlockwise": "rotate_anticlockwise",
                  "180": "rotate_180"}
        # Whatever the rescue tries, it must be un-mapped here.
        src = inspect.getsource(_rescue_rotated)
        for name in angles:
            self.assertIn(f'"{name}"', src, f"{name} is not part of the rescue")

        for name, action in angles.items():
            turned = ProcessMgr.apply_rotation(marker, action)
            for (sy, sx) in ((0, 0), (4, 7), (h - 1, w - 2)):
                ty, tx = [int(v) for v in np.argwhere(turned == marker[sy, sx])[0]]
                face = Face(bbox=np.array([tx, ty, tx + 1, ty + 1], dtype=np.float32),
                            kps=np.array([[tx, ty]], dtype=np.float32))
                _unrotate_face_coords(face, w, h, name)
                self.assertAlmostEqual(float(face.kps[0][0]), sx, places=4,
                                       msg=f"{name}: x came back wrong")
                self.assertAlmostEqual(float(face.kps[0][1]), sy, places=4,
                                       msg=f"{name}: y came back wrong")


class RotationActionCase(unittest.TestCase):
    """The orientation call is only ever asserted through its CONSEQUENCE: does
    turning the frame that way actually stand the face up?

    Asserting the returned string against a hand-worked convention is what let
    a sign inversion ship -- the test agreed with the code because it was
    derived from it. So these tests apply the project's real rotate helpers and
    measure the face afterwards.
    """

    @staticmethod
    def _face(kps):
        from insightface.app.common import Face
        kps = np.asarray(kps, dtype=np.float32)
        return Face(bbox=np.array([kps[:, 0].min(), kps[:, 1].min(),
                                   kps[:, 0].max(), kps[:, 1].max()],
                                  dtype=np.float32),
                    kps=kps)

    @staticmethod
    def _rotate_dir(action, vec):
        """Push a direction through the SHIPPED rotate helper's real point map."""
        from roop.ProcessMgr import ProcessMgr
        h, w = 9, 7
        img = np.zeros((h, w, 3), dtype=np.int32)
        for y in range(h):
            for x in range(w):
                img[y, x] = (x, y, 0)
        out = ProcessMgr.apply_rotation(img, action)
        pos = {}
        for iy in range(out.shape[0]):
            for ix in range(out.shape[1]):
                pos[(int(out[iy, ix, 0]), int(out[iy, ix, 1]))] = np.array([ix, iy])
        origin = pos[(3, 4)]
        ex, ey = pos[(4, 4)] - origin, pos[(3, 5)] - origin
        return vec[0] * ex + vec[1] * ey

    def _tilt_after(self, action, kps):
        """Tilt of the face's down axis once the frame is turned, in degrees."""
        from roop.face_util import face_down_axis
        axis = np.array(face_down_axis(self._face(kps)))
        axis /= np.linalg.norm(axis)
        moved = self._rotate_dir(action, axis)
        return abs(float(np.degrees(np.arctan2(moved[0], moved[1]))))


class TestRotationStandsTheFaceUp(RotationActionCase):
    def test_rolled_faces_end_up_upright(self):
        """A face lying on its side must be offered a rotation, and that
        rotation must leave it within 45 deg of upright -- not upside down."""
        from roop.face_util import face_rotation_action
        acted = 0
        for roll in (-120, -100, -90, -80, -70, 70, 80, 90, 100, 120):
            for yaw in (0, 45, 90):
                kps = project_kps(yaw, 0, roll)
                action = face_rotation_action(self._face(kps), (512, 512))
                self.assertIsNotNone(
                    action, f"no rotation offered at roll={roll} yaw={yaw}")
                tilt = self._tilt_after(action, kps)
                self.assertLess(tilt, 45.0,
                                f"roll={roll} yaw={yaw}: {action} left the face "
                                f"{tilt:.1f} deg off upright")
                acted += 1
        self.assertEqual(acted, 30)

    def test_rotation_strictly_improves_uprightness(self):
        from roop.face_util import face_roll_tilt, face_rotation_action
        for roll in (-110, -90, -70, 70, 90, 110):
            kps = project_kps(0, 0, roll)
            action = face_rotation_action(self._face(kps), (512, 512))
            self.assertLess(self._tilt_after(action, kps),
                            abs(face_roll_tilt(self._face(kps))),
                            f"roll={roll} got no more upright")

    def test_inverted_faces_get_the_half_turn_a_quarter_turn_cannot_do(self):
        """The case a quarter turn cannot reach. An upside-down face is 90 deg
        off upright after EITHER quarter turn -- still lying on its side -- so
        it needs the half turn, and it must actually be offered one rather than
        declined."""
        from roop.face_util import face_rotation_action
        for roll in (-180, -170, -160, -150, -140, 140, 150, 160, 170, 180):
            kps = project_kps(0, 0, roll)
            action = face_rotation_action(self._face(kps), (512, 512))
            self.assertEqual(action, "rotate_180",
                             f"roll={roll} was not treated as inverted")
            self.assertLess(self._tilt_after(action, kps), 45.0,
                            f"roll={roll}: the half turn left it off upright")

    def test_a_half_turn_is_never_offered_to_an_upright_face(self):
        """Reaching the inverted band takes 135 deg of error against a 9.1 deg
        axis, so no pose of an upright head may land there."""
        from roop.face_util import face_rotation_action
        for yaw in (0, 30, 45, 60, 75, 90):
            for pitch in (-30, -15, 0, 15, 30):
                for roll in (-40, -20, 0, 20, 40):
                    self.assertNotEqual(
                        face_rotation_action(
                            self._face(project_kps(yaw, pitch, roll)), (512, 512)),
                        "rotate_180",
                        f"upside-down call at yaw={yaw} pitch={pitch} roll={roll}")

    def test_the_two_branches_meet_without_a_seam(self):
        """At the boundary the quarter and half turns are equivalent -- both
        land the face at 45 deg -- so a noisy reading either side of 135 costs
        nothing. That is why the upper edge carries no safety margin."""
        from roop.face_util import FACE_ROLL_UPPER
        kps = project_kps(0, 0, FACE_ROLL_UPPER)
        self.assertAlmostEqual(self._tilt_after("rotate_clockwise", kps),
                               self._tilt_after("rotate_180", kps), places=3)

    def test_the_two_edges_are_not_symmetric(self):
        """The lower edge carries a safety margin over 45 deg because a false
        positive there corrupts an upright face; the upper edge must NOT
        inherit it, because nothing near 135 deg risks being upright and the
        margin would only cost recall -- a face at 120 deg is 90 deg better off
        for being turned."""
        from roop.face_util import (face_rotation_action,
                                    FACE_ROLL_LOWER, FACE_ROLL_UPPER)
        self.assertGreater(FACE_ROLL_LOWER, 45.0)
        self.assertEqual(FACE_ROLL_UPPER, 135.0)
        for roll in (115, 120, 125, 130, -115, -120, -125, -130):
            kps = project_kps(0, 0, roll)
            action = face_rotation_action(self._face(kps), (512, 512))
            self.assertIsNotNone(action, f"declined a rescuable roll={roll}")
            self.assertLess(self._tilt_after(action, kps), 45.0)


class TestRotationLeavesUprightFacesAlone(RotationActionCase):
    def test_no_rotation_for_any_upright_pose(self):
        """The regression that made this file necessary: an upright head that is
        merely TURNED must never be treated as a head lying on its side."""
        from roop.face_util import face_rotation_action
        for yaw in (0, 15, 30, 45, 50, 60, 70, 75, 80, 85, 90):
            for pitch in (-30, -15, 0, 15, 30):
                kps = project_kps(yaw, pitch, 0)
                self.assertIsNone(
                    face_rotation_action(self._face(kps), (512, 512)),
                    f"upright face rotated at yaw={yaw} pitch={pitch}")

    def test_small_and_mid_roll_is_left_alone(self):
        """Below the boundary a 90 deg turn would make things worse. Swept over
        the full pose grid, because it is yaw stacking onto real roll that
        pushes a merely-tilted head over the line."""
        from roop.face_util import face_rotation_action
        for roll in (-44, -40, -30, -15, 0, 15, 30, 40, 44):
            for yaw in (0, 15, 30, 45, 60, 75, 90):
                for pitch in (-30, 0, 30):
                    self.assertIsNone(
                        face_rotation_action(
                            self._face(project_kps(yaw, pitch, roll)), (512, 512)),
                        f"rotated at roll={roll} yaw={yaw} pitch={pitch}")

    def test_the_axis_it_uses_is_the_yaw_robust_one(self):
        """Guards the actual finding behind the fix. The nose tip swings up to
        ~45 deg under a turn of a perfectly upright head -- indistinguishable
        from real roll -- while the mouth midline stays under ~10 deg. If a
        future edit reaches for the nose again, this fails."""
        from roop.face_util import face_down_axis
        worst_used, worst_nose = 0.0, 0.0
        for yaw in (0, 30, 50, 60, 70, 80, 90):
            for pitch in (-30, -15, 0, 15, 30):
                kps = project_kps(yaw, pitch, 0)
                ax = face_down_axis(self._face(kps))
                worst_used = max(worst_used,
                                 abs(np.degrees(np.arctan2(ax[0], ax[1]))))
                nose = kps[2] - (kps[0] + kps[1]) / 2.0
                worst_nose = max(worst_nose,
                                 abs(np.degrees(np.arctan2(nose[0], nose[1]))))
        self.assertLess(worst_used, 12.0,
                        f"axis in use lies by {worst_used:.1f} deg under yaw")
        self.assertGreater(worst_nose, 40.0,
                           "nose axis no longer misbehaves -- re-check the note "
                           "in face_util if this ever fails")


class TestRotationVerificationGuard(unittest.TestCase):
    """rotation_improves_upright is the backstop that keeps a wrong orientation
    call cheap: it rejects the rotation instead of swapping an upside-down crop."""

    @staticmethod
    def _face_at(roll):
        from insightface.app.common import Face
        kps = project_kps(0, 0, roll)
        return Face(bbox=np.array([0, 0, 1, 1], dtype=np.float32), kps=kps)

    def test_accepts_a_rotation_that_stood_the_face_up(self):
        from roop.face_util import rotation_improves_upright
        self.assertTrue(rotation_improves_upright(self._face_at(90),
                                                  self._face_at(3)))

    def test_rejects_an_inverted_rotation(self):
        from roop.face_util import rotation_improves_upright
        self.assertFalse(rotation_improves_upright(self._face_at(90),
                                                   self._face_at(179)))

    def test_rejects_a_rotation_that_changed_nothing(self):
        from roop.face_util import rotation_improves_upright
        self.assertFalse(rotation_improves_upright(self._face_at(90),
                                                   self._face_at(88)))

    def test_passes_through_when_there_is_nothing_to_judge(self):
        from insightface.app.common import Face
        from roop.face_util import rotation_improves_upright
        blind = Face(bbox=np.array([0, 0, 1, 1], dtype=np.float32), kps=None)
        self.assertTrue(rotation_improves_upright(blind, self._face_at(0)))


class TestRotationActionIsSharedNotDuplicated(unittest.TestCase):
    """Render and Frame-Editor preview must agree on orientation; they did not
    while each kept its own copy of the heuristic."""

    def test_processmgr_delegates_to_face_util(self):
        import inspect
        from roop.ProcessMgr import ProcessMgr
        self.assertIn("face_rotation_action",
                      inspect.getsource(ProcessMgr.rotation_action))

    def test_core_has_no_private_copy(self):
        import inspect
        import roop.core
        src = inspect.getsource(roop.core.get_face_crop_from_frame)
        self.assertNotIn("def _rotation_action", src)
        self.assertIn("face_rotation_action", src)


if __name__ == "__main__":
    unittest.main()
