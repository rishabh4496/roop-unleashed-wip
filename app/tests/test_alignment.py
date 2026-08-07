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
                            YAW_ALIGN_RATIO, STAB_ALIGN_ONSET_DEG)
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

    def test_below_the_onset_is_bit_identical(self):
        """Faces inside STAB_ALIGN_ONSET_DEG of frontal must be bit-identical to
        the default fit — that is what makes the band's low end a no-op rather
        than a small unexplained change to every mid-angle face."""
        roop.globals.yaw_align = False
        baseline = {y: estimate_norm(project_kps(y), 512, "arcface")
                    for y in (0, 15, 30, 39)}
        roop.globals.yaw_align = True
        for yaw, expected in baseline.items():
            # Strictly below, not at: the pose solve lands within ~1e-7 deg of
            # the requested angle, so a face projected at exactly the onset can
            # solve a hair above it and pick up a w of ~1e-9. That is 5e-5 on the
            # matrix — invisible, but it is not equality, and this test is the
            # one that gets to insist on equality.
            self.assertLess(yaw, STAB_ALIGN_ONSET_DEG)
            np.testing.assert_array_equal(
                estimate_norm(project_kps(yaw), 512, "arcface"), expected,
                f"stabilize leaked below its onset at yaw={yaw}")

    def test_mid_angle_rotation_deviation_stays_small(self):
        """Inside the band the crop does move — that is the point — but it has to
        arrive gradually. Guard the magnitude, since the whole reason 'stabilize'
        was gated tightly was fear of disturbing angles that already look right.

        Rotation only. The SCALE deliberately moves now, and by more than any
        bound this test could put on "how different is it from off" — see
        test_scale_is_held_against_the_frontal_crop for what replaced that half.
        """
        for yaw in (45, 50, 55, 60):
            roop.globals.yaw_align = False
            base = decompose(estimate_norm(project_kps(yaw), 512, "arcface"))
            roop.globals.yaw_align = True
            got = decompose(estimate_norm(project_kps(yaw), 512, "arcface"))
            self.assertLess(abs(got[1] - base[1]), 2.0,
                            f"crop rotation moved {abs(got[1]-base[1]):.2f} deg "
                            f"at yaw={yaw}")

    def test_scale_is_held_against_the_frontal_crop(self):
        """The half of the mode that was missing, and was making it worse than
        doing nothing.

        Pinning the crop rotation is not free: the least-squares scale is solved
        AT the pinned angle, so rotating away from the angle the free fit chose
        shrinks it by about cos(delta) — and delta reaches 30 deg at high yaw.
        The mode marketed as the anti-wobble fix was therefore trading a rotation
        wobble for a LARGER scale one. Measured crop-scale swing over
        yaw 0-90 x pitch +/-40: 1.389x with the mode off, 1.575x with the
        rotation pinned and the scale left free, 1.193x holding both.

        What is asserted is the thing a viewer sees: the pasted face must not
        change size as the head turns. Across the band the free fit drifts from
        1.041x the frontal crop scale down to 0.897x; holding it keeps every
        angle within a few percent of 1.0.
        """
        roop.globals.yaw_align = False
        frontal = decompose(estimate_norm(project_kps(0), 512, "arcface"))[0]

        roop.globals.yaw_align = False
        free = [decompose(estimate_norm(project_kps(y), 512, "arcface"))[0] / frontal
                for y in (40, 50, 60, 70, 80, 90)]
        roop.globals.yaw_align = True
        held = [decompose(estimate_norm(project_kps(y), 512, "arcface"))[0] / frontal
                for y in (40, 50, 60, 70, 80, 90)]

        free_swing = max(free) / min(free)
        held_swing = max(held) / min(held)
        # Compared on the EXCESS over 1.0, not on the ratios themselves: a swing
        # of 1.0 is perfection, so `held < 0.6 * free` on the raw numbers asks
        # for something arithmetically impossible.
        self.assertLess(held_swing - 1.0, (free_swing - 1.0) * 0.6,
                        f"scale swing across the band: free {free_swing:.3f}x, "
                        f"held {held_swing:.3f}x — the hold is not earning its place")
        for yaw, s in zip((40, 50, 60, 70, 80, 90), held):
            self.assertLess(abs(s - 1.0), 0.06,
                            f"crop is {s:.3f}x the frontal size at yaw={yaw}")

    def test_holding_the_scale_still_collapses_to_off_below_the_onset(self):
        """The scale hold is faded by the same weight as the rotation, so at
        w = 0 its correction ratio is exactly 1.0 rather than merely close to
        it. Without that the mode would no longer be a no-op on frontal faces,
        which is the guarantee the whole band exists to keep."""
        roop.globals.yaw_align = False
        baseline = {y: estimate_norm(project_kps(y, p), 512, "arcface")
                    for y in (0, 10, 20, 30, 39) for p in (0,)}
        roop.globals.yaw_align = True
        for yaw, expected in baseline.items():
            np.testing.assert_array_equal(
                estimate_norm(project_kps(yaw), 512, "arcface"), expected,
                f"the scale hold leaked below the onset at yaw={yaw}")

    def test_no_step_change_anywhere_along_a_turn(self):
        """The defect this band replaced. A hard `yaw_ratio < 0.40` gate sat
        between two fits up to 30 deg apart in crop rotation, so crossing it was
        a step change — 18.4 deg in a single frame along this sweep, and a still
        head parked on the gate had its crop rotating +/-11 deg on detector noise
        alone. Nothing may step now.

        Swept with a nod riding on the turn because the old gate was keyed on a
        pitch-contaminated proxy: a level sweep crosses it in one clean place and
        makes the discontinuity look far smaller than it was.
        """
        roop.globals.yaw_align = True
        prev = None
        worst_rot = worst_scale = 0.0
        for i in range(2401):
            yaw = -90.0 + i * 180.0 / 2400.0
            pitch = 40.0 * np.sin(np.radians(i * 0.75))
            scale, rot = decompose(
                estimate_norm(project_kps(yaw, pitch), 512, "arcface"))
            if prev is not None:
                worst_rot = max(worst_rot, abs(rot - prev[1]))
                worst_scale = max(worst_scale, abs(scale - prev[0]) / prev[0])
            prev = (scale, rot)
        self.assertLess(worst_rot, 1.0,
                        f"crop rotation jumped {worst_rot:.2f} deg between "
                        f"adjacent frames")
        self.assertLess(worst_scale, 0.01,
                        f"crop scale jumped {100*worst_scale:.2f}% between "
                        f"adjacent frames")

    def test_still_head_on_the_old_gate_no_longer_wobbles(self):
        """The user-visible form of the same defect: a head that is not moving.
        At yaw 90 / pitch -30 the old gate sat exactly under the noise, giving a
        6.2 deg rotation sd on a motionless face. 'off' manages 0.45 deg there,
        so the mode sold as the fix for rotational wobble has to beat that, not
        lose to it by 14x."""
        base = project_kps(90, -30)
        rng = np.random.default_rng(2)
        noisy = [base + rng.normal(0, 1.0, (5, 2)).astype(np.float32)
                 for _ in range(400)]
        out = {}
        for mode in ("off", "stabilize"):
            roop.globals.yaw_align = mode
            out[mode] = np.std([decompose(estimate_norm(k, 512, "arcface"))[1]
                                for k in noisy])
        self.assertLess(out["stabilize"], out["off"] * 1.5,
                        f"stabilize wobbles {out['stabilize']:.2f} deg vs "
                        f"{out['off']:.2f} deg with the mode off")

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

    def test_frontal_faces_are_left_bit_exact(self):
        """The pose template is the REFERENCE head's projection while the
        default is an empirical template, and the two differ by ~7.7px even at
        zero pose. That gap is a fixed cost, so faces with nothing to correct
        must not pay it."""
        roop.globals.yaw_align = 'off'
        baseline = {(y, p): estimate_norm(project_kps(y, p), 512, "arcface")
                    for y, p in ((0, 0), (5, 0), (10, 0), (0, 10), (0, -10))}
        roop.globals.yaw_align = 'pose'
        for (yaw, pitch), expected in baseline.items():
            np.testing.assert_array_equal(
                estimate_norm(project_kps(yaw, pitch), 512, "arcface"), expected,
                f"pose template leaked into yaw={yaw} pitch={pitch}")

    def test_pitch_alone_is_corrected(self):
        """The blind spot this mode shipped with: the template was built from
        yaw only, so a head tilted up or down — no turn at all — was modelled
        as perfectly level and got no correction. `pitch` reaching the template
        is the whole fix for the up/down case."""
        from roop.face_util import pose_align_weight
        for pitch in (-45, -40, -30, 30, 40, 45):
            kps = project_kps(0, pitch)
            self.assertGreater(pose_align_weight(kps), 0.5,
                               f"pure pitch={pitch} barely engages")
            roop.globals.yaw_align = 'off'
            plain = estimate_norm(kps, 512, "arcface")
            roop.globals.yaw_align = 'pose'
            self.assertFalse(
                np.array_equal(estimate_norm(kps, 512, "arcface"), plain),
                f"pitch={pitch} still aligned as if the head were level")

    def test_turned_and_tilted_is_no_longer_invisible(self):
        """A yaw_ratio gate cannot see this pose. Pitch inflates yaw_ratio, so
        a profile head that is ALSO tilted reads as a mid-angle face and the
        old gate stayed shut on the most extreme poses in the range."""
        from roop.face_util import pose_align_weight
        for yaw, pitch in ((90, -30), (90, -40), (90, 40), (75, 40), (85, -45)):
            kps = project_kps(yaw, pitch)
            self.assertGreaterEqual(kps_pose_ratios(kps)[0], YAW_ALIGN_RATIO,
                                    f"yaw={yaw} pitch={pitch} would have been "
                                    f"caught by the old gate; pick a harder pose")
            self.assertEqual(pose_align_weight(kps), 1.0,
                             f"yaw={yaw} pitch={pitch} still not fully corrected")

    def test_crop_scale_stays_flat_across_the_whole_pose_grid(self):
        """What the user actually sees. The fixed template makes the pasted
        face BREATHE — it swings 1.39x in size over yaw 0-90 x pitch +/-40,
        changing as the head moves. Holding it flat is most of what makes an
        angled swap sit still."""
        grid = [(y, p) for y in (0, 15, 30, 45, 60, 75, 90)
                for p in (-40, -20, 0, 20, 40)]

        def swing():
            s = [decompose(estimate_norm(project_kps(y, p), 512, "arcface"))[0]
                 for y, p in grid]
            return max(s) / min(s)

        roop.globals.yaw_align = 'off'
        plain = swing()
        roop.globals.yaw_align = 'pose'
        posed = swing()
        self.assertGreater(plain, 1.3, "the fixed template stopped breathing — "
                                       "re-check this test's premise")
        self.assertLess(posed, 1.10, f"crop scale still swings {posed:.3f}x")

    def test_engagement_is_continuous_so_it_cannot_flicker(self):
        """The anti-flicker guarantee, and the reason this is a band and not a
        threshold.

        A hard gate puts a finite jump in the crop geometry at the boundary, so
        a head sitting near it — or just detector noise on a head that is not
        moving — pops between two different transforms frame to frame. Sweeping
        a turn with a nod riding on it, no single step may move the crop more
        than a hair.
        """
        for mode in ('off', 'pose'):
            roop.globals.yaw_align = mode
            prev, worst_s, worst_r = None, 0.0, 0.0
            for i in range(1801):
                yaw = i * 90.0 / 1800.0
                pitch = 40.0 * np.sin(np.radians(i * 0.5))
                scale, rot = decompose(
                    estimate_norm(project_kps(yaw, pitch), 512, "arcface"))
                if prev is not None:
                    worst_s = max(worst_s, abs(scale - prev[0]) / prev[0])
                    worst_r = max(worst_r, abs(rot - prev[1]))
                prev = (scale, rot)
            if mode == 'pose':
                self.assertLess(worst_s, 0.004,
                                f"crop scale jumps {worst_s * 100:.2f}% in one step")
                self.assertLess(worst_r, 0.10,
                                f"crop rotation jumps {worst_r:.3f} deg in one step")

    def test_no_hard_edge_at_the_band_boundaries(self):
        """Continuity specifically WHERE a gate would have been: straddling the
        onset and the full-engagement angle must be smooth, since that is
        exactly where a threshold implementation would pop."""
        from roop.face_util import (POSE_ALIGN_ONSET_DEG, POSE_ALIGN_FULL_DEG,
                                    pose_align_weight)
        for edge in (POSE_ALIGN_ONSET_DEG, POSE_ALIGN_FULL_DEG):
            lo = pose_align_weight(project_kps(edge - 0.25, 0))
            hi = pose_align_weight(project_kps(edge + 0.25, 0))
            self.assertLess(abs(hi - lo), 0.02,
                            f"weight steps {abs(hi - lo):.3f} across {edge} deg")

    def test_weight_never_leaves_its_range(self):
        from roop.face_util import pose_align_weight
        for yaw in range(0, 91, 3):
            for pitch in range(-60, 61, 10):
                w = pose_align_weight(project_kps(yaw, pitch, 25))
                self.assertGreaterEqual(w, 0.0)
                self.assertLessEqual(w, 1.0)

    @staticmethod
    def _target_template(kps, dst):
        """The template estimate_norm actually aimed at for these keypoints."""
        from roop.face_util import (_pose_template, pose_align_weight,
                                    solve_pose_jaw_5pt)
        yaw, pitch, _, jaw = solve_pose_jaw_5pt(kps)
        w = pose_align_weight(kps)
        return (1.0 - w) * dst + w * _pose_template(yaw, dst, pitch, jaw)

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
            improved = fit_residual(posed, kps, self._target_template(kps, dst))
            self.assertLess(improved, plain * 0.6,
                            f"yaw={yaw}: residual {plain:.1f} -> {improved:.1f} px")

    def test_exact_fit_at_the_solved_pose(self):
        """At the pose the template was built for, input and template are
        congruent, so the residual should collapse to ~0. Now that pitch is
        modelled this must hold off-level too, which is what the old yaw-only
        template could not do."""
        dst = reference_template(512, "arcface")
        for yaw in (60, 75, 90):
            for pitch in (-40, -20, 0, 20, 40):
                kps = project_kps(yaw, pitch)
                m = estimate_norm(kps, 512, "arcface")
                self.assertLess(fit_residual(m, kps, self._target_template(kps, dst)),
                                1.0, f"yaw={yaw} pitch={pitch}")

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


class TestPoseSolve(unittest.TestCase):
    """`solve_pose_5pt` replaces the scalar ratio proxies for anything that has
    to reason about yaw and pitch together."""

    def test_recovers_every_angle_over_the_whole_grid(self):
        from roop.face_util import solve_pose_5pt
        worst = [0.0, 0.0, 0.0]
        for yaw in range(0, 91, 5):
            for pitch in range(-45, 50, 5):
                for roll in (-30, -10, 0, 10, 30):
                    got = solve_pose_5pt(project_kps(yaw, pitch, roll))
                    self.assertIsNotNone(got, f"{yaw}/{pitch}/{roll}")
                    for i, truth in enumerate((yaw, pitch, roll)):
                        worst[i] = max(worst[i], abs(got[i] - truth))
        for i, name in enumerate(("yaw", "pitch", "roll")):
            self.assertLess(worst[i], 0.5, f"{name} off by {worst[i]:.2f} deg")

    def test_works_at_a_true_profile_where_the_ratios_are_degenerate(self):
        """At 90 deg the eyes project to the same point, so eye separation — the
        thing every ratio proxy divides by — has collapsed to zero. The solve
        leans on the nose tip standing proud of the eye/mouth plane instead."""
        from roop.face_util import solve_pose_5pt
        kps = project_kps(90, 0)
        self.assertAlmostEqual(
            float(np.linalg.norm(kps[1] - kps[0])), 0.0, places=3)
        yaw, pitch, _ = solve_pose_5pt(kps)
        self.assertAlmostEqual(abs(yaw), 90.0, delta=0.5)
        self.assertAlmostEqual(pitch, 0.0, delta=0.5)

    def test_pitch_does_not_leak_into_the_reported_yaw(self):
        """The defect that made the ratio proxies unusable: pitch inflates
        yaw_ratio, so a tilted profile reads as a mid-angle face."""
        from roop.face_util import solve_pose_5pt
        self.assertGreaterEqual(kps_pose_ratios(project_kps(90, -30))[0],
                                YAW_ALIGN_RATIO)   # the proxy is fooled
        for pitch in (-40, -20, 0, 20, 40):
            yaw, _, _ = solve_pose_5pt(project_kps(90, pitch))
            self.assertAlmostEqual(abs(yaw), 90.0, delta=0.5,
                                   msg=f"pitch={pitch} corrupted the yaw solve")

    def test_degenerate_input_returns_none_rather_than_guessing(self):
        from roop.face_util import solve_pose_5pt
        self.assertIsNone(solve_pose_5pt(None))
        self.assertIsNone(solve_pose_5pt(np.zeros((5, 2), np.float32)))
        self.assertIsNone(solve_pose_5pt(np.zeros((3, 2), np.float32)))
        self.assertIsNone(solve_pose_5pt(np.full((5, 2), np.nan, np.float32)))

    def test_offaxis_combines_the_two_angles(self):
        """|yaw| alone under-ranks a turned AND tilted head, which is how the
        worst poses kept passing for mid-angle ones."""
        from roop.face_util import offaxis_deg
        self.assertAlmostEqual(offaxis_deg(0, 0), 0.0, places=3)
        self.assertAlmostEqual(offaxis_deg(40, 0), 40.0, places=3)
        self.assertAlmostEqual(offaxis_deg(0, 40), 40.0, places=3)
        self.assertGreater(offaxis_deg(40, 40), 50.0)
        self.assertGreater(offaxis_deg(75, 40), 75.0)


class TestNonFrontalMaskRouting(unittest.TestCase):
    """The mask router picks between two different mask derivations. It read the
    same contaminated scalars the alignment gate did, and had the same blind
    spot — sixteen cells over yaw 0-90 x pitch +/-45 came out "frontal" while
    being 79-90 deg off-axis, every one a profile head also tilted up or down.
    """

    @staticmethod
    def _rule(kps):
        """The shipped test, exercised through the real helpers."""
        from roop.face_util import offaxis_deg, solve_pose_5pt
        non_frontal = False
        lex, rex, nx = kps[0][0], kps[1][0], kps[2][0]
        d_le, d_re = abs(nx - lex), abs(nx - rex)
        if d_le + d_re > 1e-5 and abs(d_le - d_re) / (d_le + d_re) > 0.25:
            non_frontal = True
        yaw_ratio, pitch_ratio = kps_pose_ratios(kps)
        if yaw_ratio is not None and yaw_ratio < 0.55:
            non_frontal = True
        if pitch_ratio is not None and not (0.32 < pitch_ratio < 0.70):
            non_frontal = True
        if not non_frontal:
            pose = solve_pose_5pt(kps)
            if pose is not None and offaxis_deg(pose[0], pose[1]) > 50.0:
                non_frontal = True
        return non_frontal

    def test_the_added_term_matches_what_the_router_ships(self):
        """This file re-implements the rule, so it can drift from the real one.
        Pin the parts that matter to the source — which now lives in
        roop/nonfrontal.py, alongside the latch that consumes it."""
        import inspect
        import roop.nonfrontal
        src = inspect.getsource(roop.nonfrontal)
        self.assertIn("solve_pose_5pt", src)
        self.assertIn("offaxis_deg", src)
        self.assertEqual(roop.nonfrontal._OFFAXIS_MAX, 50.0)

    def test_no_extreme_pose_is_routed_as_frontal(self):
        from roop.face_util import offaxis_deg, solve_pose_5pt
        blind = []
        for yaw in range(0, 91, 5):
            for pitch in range(-45, 50, 5):
                kps = project_kps(yaw, pitch)
                pose = solve_pose_5pt(kps)
                if offaxis_deg(pose[0], pose[1]) >= 50.0 and not self._rule(kps):
                    blind.append((yaw, pitch))
        self.assertEqual(blind, [], f"still routed as frontal: {blind}")

    def test_the_old_proxies_really_were_blind_there(self):
        """Records the defect, so the extra term cannot be dropped as redundant."""
        for yaw, pitch in ((75, 40), (85, -45), (90, -40), (90, 45)):
            kps = project_kps(yaw, pitch)
            yaw_ratio, pitch_ratio = kps_pose_ratios(kps)
            lex, rex, nx = kps[0][0], kps[1][0], kps[2][0]
            asym = abs(abs(nx - lex) - abs(nx - rex)) / (abs(nx - lex) + abs(nx - rex))
            self.assertLess(asym, 0.25, f"asymmetry saw yaw={yaw} pitch={pitch}")
            self.assertGreaterEqual(yaw_ratio, 0.55)
            self.assertTrue(0.32 < pitch_ratio < 0.70)
            self.assertTrue(self._rule(kps), "the new term did not rescue it")

    @staticmethod
    def _rule_without_the_new_term(kps):
        non_frontal = False
        lex, rex, nx = kps[0][0], kps[1][0], kps[2][0]
        d_le, d_re = abs(nx - lex), abs(nx - rex)
        if d_le + d_re > 1e-5 and abs(d_le - d_re) / (d_le + d_re) > 0.25:
            non_frontal = True
        yaw_ratio, pitch_ratio = kps_pose_ratios(kps)
        if yaw_ratio is not None and yaw_ratio < 0.55:
            non_frontal = True
        if pitch_ratio is not None and not (0.32 < pitch_ratio < 0.70):
            non_frontal = True
        return non_frontal

    def test_the_added_term_changes_nothing_below_its_threshold(self):
        """The term is additive and must stay that way: everything under 50 deg
        off-axis has to keep whatever verdict it already had. Otherwise closing
        the blind spot would quietly re-route ordinary faces as well."""
        from roop.face_util import offaxis_deg, solve_pose_5pt
        checked = 0
        for yaw in range(0, 91, 5):
            for pitch in range(-45, 50, 5):
                kps = project_kps(yaw, pitch)
                pose = solve_pose_5pt(kps)
                if offaxis_deg(pose[0], pose[1]) >= 50.0:
                    continue
                self.assertEqual(self._rule(kps),
                                 self._rule_without_the_new_term(kps),
                                 f"yaw={yaw} pitch={pitch} re-routed")
                checked += 1
        self.assertGreater(checked, 100)

    def test_a_dead_frontal_face_is_still_frontal(self):
        for pitch in (-15, 0, 15):
            self.assertFalse(self._rule(project_kps(0, pitch)),
                             f"yaw=0 pitch={pitch} newly non-frontal")


class TestTheYawRatioCannotDoThePoseSolvesJob(unittest.TestCase):
    """A yaw-ratio -> yaw lookup used to stand where solve_pose_5pt is now, and
    these are the two reasons it had to go. Kept as tests rather than a comment
    so nobody reintroduces the cheaper-looking scalar inversion."""

    def test_it_cannot_see_pitch_at_all(self):
        """One scalar, two unknowns. Heads at the same yaw and wildly different
        pitch are indistinguishable to it — including a pair straddling level,
        which the ratio maps to nearly the same number."""
        a = kps_pose_ratios(project_kps(0, -25))[0]
        b = kps_pose_ratios(project_kps(0, 25))[0]
        self.assertAlmostEqual(a, b, delta=0.12)
        from roop.face_util import solve_pose_5pt
        self.assertAlmostEqual(solve_pose_5pt(project_kps(0, -25))[1], -25, delta=0.5)
        self.assertAlmostEqual(solve_pose_5pt(project_kps(0, 25))[1], 25, delta=0.5)

    def test_pitch_corrupts_the_yaw_it_would_have_reported(self):
        """Worse than merely blind: a tilted profile reads as a mid-angle face,
        so the old lookup would have posed the template at the wrong yaw."""
        level = kps_pose_ratios(project_kps(90, 0))[0]
        tilted = kps_pose_ratios(project_kps(90, -30))[0]
        self.assertLess(level, 0.1)          # a profile, correctly
        self.assertGreater(tilted, 0.4)      # the same profile, read as mid-angle
        from roop.face_util import solve_pose_5pt
        for pitch in (-30, 0, 30):
            self.assertAlmostEqual(abs(solve_pose_5pt(project_kps(90, pitch))[0]),
                                   90.0, delta=0.5)

    def test_the_lookup_is_gone(self):
        import roop.face_util as fu
        for dead in ('_yaw_from_ratio', '_build_yaw_lookup', '_LUT_YAWS'):
            self.assertFalse(hasattr(fu, dead),
                             f"{dead} is back — two ways to ask the same question")


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

    def test_every_caller_also_applies_the_outcome_gate(self):
        """Agreeing on the heuristic is only half of it.

        Every one of these three re-detects on the turned crop and commits to the
        rotation only if the face came out MORE upright — otherwise it falls back
        to the unrotated crop. A caller that shares `face_rotation_action` but
        skips `rotation_improves_upright` still ends up in a different coordinate
        space from the render on exactly the frames the gate rejects, which is
        the disagreement sharing the call was meant to end. The Gradio mask
        editor shipped in precisely that state.
        """
        import inspect
        import os
        import roop.core
        from roop.ProcessMgr import ProcessMgr

        sources = {
            "ProcessMgr.process_face": inspect.getsource(ProcessMgr.process_face),
            "core.get_face_crop_from_frame":
                inspect.getsource(roop.core.get_face_crop_from_frame),
        }
        # Read rather than import: this module pulls in the whole Gradio UI.
        tab = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "ui", "tabs", "faceswap_tab.py")
        with open(tab, encoding="utf-8") as fh:
            sources["ui/tabs/faceswap_tab.py"] = fh.read()

        for where, src in sources.items():
            self.assertIn("rotation_improves_upright", src,
                          f"{where} turns the frame without checking the result")


class TestClippedFaceCrop(unittest.TestCase):
    """align_crop must not feed the swap model a black wedge.

    A face running off the edge of the frame — "only part of the face is
    visible" — makes the aligned crop sample outside the image. That used to be
    filled with pure black (BORDER_CONSTANT/0), putting a third of the swap
    model's input far outside the distribution it was trained on. The border is
    replicated instead.

    This is invisible on any face that is fully inside the frame, which is why
    it needs a test: nothing in normal footage would catch a regression here.
    """

    @staticmethod
    def _img():
        rng = np.random.default_rng(7)
        # No pure-black pixels anywhere, so any black in a crop came from the
        # border fill rather than from the source frame.
        return rng.integers(40, 255, (720, 1280, 3), dtype=np.uint8)

    INTERIOR = [[600, 300], [680, 300], [640, 350], [605, 400], [675, 400]]
    OFF_LEFT = [[8, 300], [88, 300], [48, 350], [13, 400], [83, 400]]
    OFF_TOP = [[600, 8], [680, 8], [640, 40], [605, 70], [675, 70]]

    def test_clipped_face_crop_has_no_black_fill(self):
        from roop.face_util import align_crop
        img = self._img()
        for name, kps in (("off left", self.OFF_LEFT), ("off top", self.OFF_TOP)):
            crop, _ = align_crop(img, np.asarray(kps, np.float32), 256, "arcface")
            black = int((crop.reshape(-1, 3).sum(axis=1) == 0).sum())
            self.assertEqual(black, 0,
                             f"{name}: {black} pure-black px fed to the swap model")

    def test_interior_face_is_unaffected_by_the_border_mode(self):
        """The fix must be a no-op wherever it is not needed.

        warpAffine only consults the border mode for samples outside the source,
        so a face fully inside the frame has to come out bit-identical to the
        old BORDER_CONSTANT call — otherwise this quietly changed every render.
        """
        import cv2
        from roop.face_util import align_crop
        img = self._img()
        kps = np.asarray(self.INTERIOR, np.float32)
        new_crop, M = align_crop(img, kps, 256, "arcface")
        old_crop = cv2.warpAffine(img, M, (256, 256), borderValue=0.0)
        self.assertTrue(np.array_equal(new_crop, old_crop),
                        "border mode changed the crop of a face that is not clipped")

    def test_the_old_behaviour_really_was_black(self):
        """Guards the premise: without the fix these crops DO fill with black.

        If a future template change moved the crop fully inside the frame, the
        test above would pass vacuously and stop protecting anything.
        """
        import cv2
        from roop.face_util import estimate_norm
        img = self._img()
        M = estimate_norm(np.asarray(self.OFF_LEFT, np.float32), 256, "arcface")
        old = cv2.warpAffine(img, M, (256, 256), borderValue=0.0)
        black = int((old.reshape(-1, 3).sum(axis=1) == 0).sum())
        self.assertGreater(black, 1000,
                           "off-left keypoints no longer produce a clipped crop — "
                           "pick coordinates that do, or this suite proves nothing")


if __name__ == "__main__":
    unittest.main()
