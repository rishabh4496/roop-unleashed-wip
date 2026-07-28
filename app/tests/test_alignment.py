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


if __name__ == "__main__":
    unittest.main()
