"""An open mouth reads as head pitch, and the 5 keypoints cannot tell them apart.

Two of the five arcface keypoints are the MOUTH CORNERS, and the jaw is a moving
part. Everything that solves head pose from those five points is therefore
solving for a rigid head using two points that are not rigidly attached to it.

Measured on a dead-frontal synthetic head as the jaw drops, expressed as a
fraction of the eye-to-mouth distance:

    jaw drop     0%     15%     30%     45%     60%
    solved pitch 0.0   -11.0   -18.7   -24.1   -28.0 deg
    crop scale  1.000   0.937   0.878   0.821   0.699

So a wide-open mouth reports up to 28 degrees of pitch that is not there, and —
because the alignment fits all five points — shrinks the crop by up to 30%. On
video that is the face BREATHING in size as someone talks, which reads as the
swap sliding around and not sitting on the head. It is also why an open-mouth
frame looks bad in every swapper at once: the model never sees a correctly
scaled crop, whichever model it is.

WHY IT IS NOT FIXED HERE, which is the point of this file.

The obvious repair is to add a jaw parameter and solve for it alongside the
pose. It does not work, and the reason is exact rather than approximate: after
centring — which any pose solve must do, translation not being a pose — the jaw
displacement direction lies EXACTLY in the span of the reference head's own
coordinates. Fitting the centred jaw indicator from the centred reference
columns leaves a residual of 4e-16. Look at the reference's y column against
the jaw pattern and it is obvious: both eyes sit at one y, both mouth corners at
another, and dropping the jaw is indistinguishable from a linear reweighting of
that axis, which is what a pitch change is.

A joint linear solve for (pose, jaw) therefore returns exactly the shipped
answer — verified below, to the digit.

What DOES break the tie is the rotation constraint: a jaw drop leaves a residual
no rigid rotation can produce. Iterating the solve and re-projecting through the
true non-linear rotation converges to the right answer (phantom pitch at 60%
jaw: -28.0 -> -4.0 at 8 iterations, -0.5 at 16). It costs 585 us/call against
11.4, which is 50x, on a function that runs per face per frame. A constrained
Gauss-Newton over (yaw, pitch, roll, scale, jaw) would be far cheaper and is the
route if this is ever worth fixing.

And it may not be. The alignment that shrinks the crop is standard arcface
alignment, which is what the swap models were TRAINED with — their open-mouth
training faces got exactly the same treatment. A "corrected" crop would be
further from the training distribution, not closer. Anything done here has to be
argued on output, not on geometry.
"""

import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import roop.globals                                              # noqa: E402
from roop.face_util import (_project_reference, solve_pose_5pt,  # noqa: E402
                            estimate_norm, _REF5_IMAGE)


def kps(yaw, pitch, roll=0.0, jaw=0.0, scale=180.0, cx=400.0, cy=300.0):
    """5 keypoints of the reference head, with the jaw dropped by `jaw` x the
    eye-to-mouth distance."""
    p = _project_reference(yaw, pitch).copy()
    eye = (p[0] + p[1]) / 2.0
    mouth = (p[3] + p[4]) / 2.0
    d = mouth - eye
    p[3] = p[3] + d * jaw
    p[4] = p[4] + d * jaw
    c, s = math.cos(math.radians(roll)), math.sin(math.radians(roll))
    p = p @ np.array([[c, -s], [s, c]]).T
    p = p - p.mean(axis=0)
    return (p * scale + np.array([cx, cy])).astype(np.float64)


def crop_scale(k, size=256):
    M = estimate_norm(k, size)
    return math.sqrt(abs(M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]))


class TheEffectIsReal(unittest.TestCase):
    def setUp(self):
        self._saved = roop.globals.yaw_align
        roop.globals.yaw_align = 'off'

    def tearDown(self):
        roop.globals.yaw_align = self._saved

    def test_a_dropped_jaw_is_reported_as_pitch(self):
        """Frontal head, mouth opening, nothing else moving."""
        pitches = [solve_pose_5pt(kps(0, 0, jaw=j))[1]
                   for j in (0.0, 0.15, 0.30, 0.45, 0.60)]
        self.assertAlmostEqual(pitches[0], 0.0, delta=0.05)
        self.assertLess(pitches[-1], -20.0,
                        f'expected a large phantom pitch, got {pitches[-1]:.1f}')
        # Monotonic, so it tracks the mouth rather than being noise.
        for a, b in zip(pitches, pitches[1:]):
            self.assertLess(b, a + 1e-6)

    def test_the_crop_shrinks_as_the_mouth_opens(self):
        """The part a viewer sees: the pasted face changes size while talking."""
        scales = [crop_scale(kps(0, 0, jaw=j)) for j in (0.0, 0.3, 0.6)]
        self.assertGreater(scales[0] / scales[-1], 1.25,
                           f'crop scale swing only {scales[0]/scales[-1]:.3f}x')

    def test_pose_alignment_does_not_rescue_it(self):
        """'pose' mode projects the reference at the SOLVED pose — so it acts on
        the phantom pitch and corrects toward a head position that is not real.
        On an open mouth it is measurably worse than leaving it alone, which is
        worth knowing before recommending it for angled faces."""
        spread = {}
        for mode in ('off', 'pose'):
            roop.globals.yaw_align = mode
            s = [crop_scale(kps(0, 0, jaw=j)) for j in (0.0, 0.3, 0.55)]
            spread[mode] = max(s) / min(s)
        self.assertGreater(spread['pose'], spread['off'],
                           f"expected 'pose' to be worse on an open mouth: "
                           f"off {spread['off']:.3f}x, pose {spread['pose']:.3f}x")


class WhyItIsNotSeparable(unittest.TestCase):
    """The degeneracy, stated as arithmetic so nobody re-derives it by trying."""

    def test_the_jaw_direction_lies_in_the_reference_span(self):
        X = np.asarray(_REF5_IMAGE, dtype=np.float64)
        jaw = np.array([0.0, 0.0, 0.0, 1.0, 1.0])      # mouth corners only
        # Centred, because a pose solve removes translation first.
        Xc = X - X.mean(axis=0)
        jc = jaw - jaw.mean()
        coef, *_ = np.linalg.lstsq(Xc, jc, rcond=None)
        residual = float(np.linalg.norm(Xc @ coef - jc))
        self.assertLess(residual, 1e-9,
                        f'the jaw direction is NOT in the reference span '
                        f'(residual {residual:.2e}) — if this ever becomes true, '
                        f'a joint linear solve for pose+jaw becomes possible')

    def test_so_a_joint_linear_solve_adds_nothing(self):
        """Built the way it would be built, and shown to return the shipped
        answer to the digit — so the idea is closed off, not merely untried."""
        X = np.asarray(_REF5_IMAGE, dtype=np.float64)
        delta = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
        D = np.zeros((10, 8))
        for i in range(5):
            D[2 * i, 0:3] = X[i]
            D[2 * i, 6] = delta[i]
            D[2 * i + 1, 3:6] = X[i]
            D[2 * i + 1, 7] = delta[i]
        dpinv = np.linalg.pinv(D)

        for jaw in (0.0, 0.3, 0.6):
            k = kps(0, 0, jaw=jaw)
            x = np.asarray(k) - np.asarray(k).mean(axis=0)
            th = dpinv @ x.reshape(-1)
            a0x, a0y, a0z = th[0], th[1], th[2]
            a1x, a1y, a1z = th[3], th[4], th[5]
            n1 = math.hypot(math.hypot(a0x, a0y), a0z)
            r1 = (a0x / n1, a0y / n1, a0z / n1)
            dot = a1x * r1[0] + a1y * r1[1] + a1z * r1[2]
            r2 = (a1x - dot * r1[0], a1y - dot * r1[1], a1z - dot * r1[2])
            n2 = math.hypot(math.hypot(*r2[:2]), r2[2])
            r2 = (r2[0] / n2, r2[1] / n2, r2[2] / n2)
            # r3 = r1 x r2; pitch = asin(-r3y), as in solve_pose_5pt.
            r3y = r1[2] * r2[0] - r1[0] * r2[2]
            joint_pitch = math.degrees(math.asin(max(-1.0, min(1.0, -r3y))))
            self.assertAlmostEqual(
                joint_pitch, solve_pose_5pt(k)[1], delta=0.05,
                msg=f'the joint solve differs at jaw={jaw} — if so, revisit')


if __name__ == '__main__':
    unittest.main()
