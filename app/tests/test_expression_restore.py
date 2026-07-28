"""The expression-restore geometry, tested without the models.

Expression_LivePortrait keeps its maths in module-level functions precisely so
this suite can run: no 537 MB download, no GPU, no ONNX session. What is asserted
here is the part that would be silently wrong rather than loudly broken — a sign
error in the rotation, a strength that does not reduce to a no-op, or a transfer
that quietly moves the head instead of only the mouth.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop.processors.Expression_LivePortrait import (  # noqa: E402
    EYE_INDICES, LIP_INDICES, NUM_BINS, blend_expression, concat_feat,
    driving_keypoints, get_rotation_matrix, headpose_pred_to_degree,
    transform_keypoint,
)

RNG = np.random.default_rng(11)


def _kp(n=21):
    return RNG.normal(size=(1, n, 3)).astype(np.float32)


class TestHeadposeDecoding(unittest.TestCase):
    """The pose heads emit a 66-bin distribution, not an angle."""

    def test_range_matches_the_bin_scheme(self):
        lo = headpose_pred_to_degree(np.eye(NUM_BINS)[0] * 50)[0]
        hi = headpose_pred_to_degree(np.eye(NUM_BINS)[NUM_BINS - 1] * 50)[0]
        self.assertAlmostEqual(lo, -97.5, places=3)
        self.assertAlmostEqual(hi, NUM_BINS * 3 - 3 - 97.5, places=3)

    def test_monotonic_in_the_argmax_bin(self):
        angles = [headpose_pred_to_degree(np.eye(NUM_BINS)[i] * 50)[0]
                  for i in range(0, NUM_BINS, 8)]
        self.assertEqual(angles, sorted(angles))

    def test_softmax_is_overflow_safe(self):
        """Raw logits can be large; a naive exp() would return NaN and poison
        the rotation matrix."""
        out = headpose_pred_to_degree(np.full((1, NUM_BINS), 10000.0))
        self.assertTrue(np.isfinite(out).all())


class TestRotationMatrix(unittest.TestCase):
    def test_identity_at_zero(self):
        np.testing.assert_allclose(get_rotation_matrix(0, 0, 0)[0],
                                   np.eye(3), atol=1e-6)

    def test_orthonormal_with_unit_determinant(self):
        for angles in ((30, -20, 15), (-75, 60, -40), (10, 90, 0)):
            r = get_rotation_matrix(*angles)[0]
            np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-5)
            self.assertAlmostEqual(float(np.linalg.det(r)), 1.0, places=5)

    def test_preserves_lengths(self):
        pts = _kp()[0]
        r = get_rotation_matrix(25, -35, 12)[0]
        np.testing.assert_allclose(np.linalg.norm(pts @ r, axis=1),
                                   np.linalg.norm(pts, axis=1), atol=1e-5)


class TestKeypointTransform(unittest.TestCase):
    def test_drops_the_z_translation(self):
        kp, exp = _kp(), np.zeros((1, 21, 3), np.float32)
        t = np.array([[0.1, 0.2, 999.0]], np.float32)
        out = transform_keypoint(kp, exp, np.ones((1, 1), np.float32), t,
                                 get_rotation_matrix(0, 0, 0))
        np.testing.assert_allclose(out[0, :, 2], kp[0, :, 2], atol=1e-5)

    def test_scale_and_translation_apply(self):
        kp = _kp()
        zeros = np.zeros_like(kp)
        base = transform_keypoint(kp, zeros, np.ones((1, 1), np.float32),
                                  np.zeros((1, 3), np.float32),
                                  get_rotation_matrix(0, 0, 0))
        scaled = transform_keypoint(kp, zeros, np.full((1, 1), 2.0, np.float32),
                                    np.zeros((1, 3), np.float32),
                                    get_rotation_matrix(0, 0, 0))
        np.testing.assert_allclose(scaled, base * 2, atol=1e-5)


class TestExpressionBlend(unittest.TestCase):
    def test_zero_strength_is_an_exact_no_op(self):
        """The setting's off position must not perturb a single value."""
        src, dst = _kp(), _kp()
        np.testing.assert_array_equal(blend_expression(src, dst, 0.0), src)

    def test_full_strength_adopts_the_target_expression(self):
        src, dst = _kp(), _kp()
        np.testing.assert_allclose(blend_expression(src, dst, 1.0), dst, atol=1e-6)

    def test_above_one_exaggerates_past_the_target(self):
        src, dst = _kp(), _kp()
        out = blend_expression(src, dst, 2.0)
        np.testing.assert_allclose(out, src + 2 * (dst - src), atol=1e-6)

    def test_region_gating_touches_only_its_indices(self):
        src, dst = _kp(), _kp()
        for region, idx in (("lips", LIP_INDICES), ("eyes", EYE_INDICES)):
            out = blend_expression(src, dst, 1.0, region)
            moved = {i for i in range(src.shape[1])
                     if not np.allclose(out[0, i], src[0, i])}
            self.assertTrue(moved.issubset(set(idx)), f"{region} moved {moved - set(idx)}")
            self.assertTrue(moved, f"{region} moved nothing at all")


class TestDrivingKeypoints(unittest.TestCase):
    """The property the whole design rests on: expression moves, pose cannot."""

    def _setup(self):
        kp, exp_s, exp_d = _kp(), _kp(), _kp()
        scale = np.full((1, 1), 1.7, np.float32)
        t = np.array([[0.3, -0.2, 5.0]], np.float32)
        rot = get_rotation_matrix(21, -33, 9)
        x_s = transform_keypoint(kp, exp_s, scale, t, rot)
        return kp, exp_s, exp_d, scale, t, rot, x_s

    def test_zero_strength_returns_the_source_exactly(self):
        _, exp_s, exp_d, scale, _, _, x_s = self._setup()
        np.testing.assert_allclose(
            driving_keypoints(x_s, scale, exp_s, exp_d, 0.0), x_s, atol=1e-6)

    def test_matches_the_full_transform_it_short_circuits(self):
        """driving_keypoints skips rebuilding the rotation. It must agree with
        the long form for any strength, or the shortcut is not equivalent."""
        kp, exp_s, exp_d, scale, t, rot, x_s = self._setup()
        for strength in (0.25, 0.5, 1.0, 1.75):
            mixed = blend_expression(exp_s, exp_d, strength)
            expected = transform_keypoint(kp, mixed, scale, t, rot)
            got = driving_keypoints(x_s, scale, exp_s, exp_d, strength)
            np.testing.assert_allclose(got, expected, atol=1e-5,
                                       err_msg=f"strength={strength}")

    def test_head_pose_cannot_drift(self):
        """Rotation and translation cancel in the difference, so the centroid
        shift comes only from the expression delta — never from pose."""
        _, exp_s, exp_d, scale, _, _, x_s = self._setup()
        x_d = driving_keypoints(x_s, scale, exp_s, exp_d, 1.0)
        expected_shift = (scale.reshape(-1, 1, 1) * (exp_d - exp_s)).mean(axis=1)
        np.testing.assert_allclose((x_d - x_s).mean(axis=1), expected_shift, atol=1e-5)

    def test_region_restriction_survives_into_the_keypoints(self):
        _, exp_s, exp_d, scale, _, _, x_s = self._setup()
        x_d = driving_keypoints(x_s, scale, exp_s, exp_d, 1.0, region="lips")
        moved = {i for i in range(x_s.shape[1])
                 if not np.allclose(x_d[0, i], x_s[0, i], atol=1e-6)}
        self.assertTrue(moved.issubset(set(LIP_INDICES)))


class TestStitchingInput(unittest.TestCase):
    def test_concat_feat_shape_and_order(self):
        a, b = _kp(), _kp()
        out = concat_feat(a, b)
        self.assertEqual(out.shape, (1, 21 * 3 * 2))
        np.testing.assert_allclose(out[0, :63], a.reshape(-1), atol=1e-6)
        np.testing.assert_allclose(out[0, 63:], b.reshape(-1), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
