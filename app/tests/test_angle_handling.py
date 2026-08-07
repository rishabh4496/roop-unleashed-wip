"""Angle handling — the three shared layers that all 13 swap models go through.

The bug this suite exists for: lateral and down-lateral faces distort, and they
distort DIFFERENTLY per swap model, which sends people looking for the fix in
whichever swapper they happen to be using. It is not there. Every swapper reaches
the same `align_crop`, the same `paste_upscale`, and the same crop-space fade, so
all three corrections live in shared code:

  1. pose-matched alignment (`yaw_align='pose'`) — keeps the crop's scale and
     geometry constant so the model is handed something inside its training
     distribution and the crop does not breathe frame to frame.
  2. `pose_visibility_polygon` — trims the paste matte to the face surface still
     facing the camera, so a profile stops pasting swap pixels over hair.
  3. `angle_fade_weight` — fades the swapped crop back toward the plate past the
     pose range the models can serve. The only layer that bounds how wrong an
     out-of-distribution swap can look.

What is deliberately NOT asserted: that any of this makes a swap "look better".
That is a judgement on footage. What is asserted is the geometry each layer
claims, the invariants that make them safe to leave on by default (frontal faces
untouched, real face never trimmed), and that turning each one off restores the
previous behaviour exactly.
"""

import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import roop.globals as g                                        # noqa: E402
from roop.face_util import (                                    # noqa: E402
    ANGLE_FADE_FULL_DEG, ANGLE_FADE_ONSET_DEG, VIS_POLY_MARGIN,
    _pose_placement, _pose_template, _project_reference, arcface_dst,
    angle_fade_weight, estimate_norm, offaxis_deg, pose_visibility_polygon,
    pose_weight_for, solve_pose_jaw_5pt, yaw_align_mode,
)
from roop.procmgr_masking import MaskingMixin, landmark_hull    # noqa: E402
from tests.facegeom import head_106, project_points, rotation   # noqa: E402

SIZE = 512
DST = arcface_dst * (SIZE / 112.0)

# facegeom.head_106 returns 101 landmarks with the 5 arcface keypoints appended,
# so the two have to be sliced apart — using all 106 as "landmarks" would feed
# the keypoints into the hull as well.
_H = head_106()
LM3D, KPS3D = _H[:101], _H[101:]

# Outward-normal proxy for the ground truth "is this landmark actually visible":
# the direction from inside the head to the point. Independent of the ellipsoid
# the polygon itself is built from, so this is a real check and not the polygon
# agreeing with itself.
_TRUTH_CENTRE = np.array([0.0, -0.05, 0.0])
_TRUTH_N = LM3D - _TRUTH_CENTRE
_TRUTH_N = _TRUTH_N / np.linalg.norm(_TRUTH_N, axis=1, keepdims=True)

POSES = [(yaw, pitch)
         for pitch in (0, -20, -40, 20)
         for yaw in (0, 15, 30, 55, 70, 80, 88)]


def _project(yaw, pitch, scale=110.0):
    return (project_points(LM3D, yaw, pitch, scale=scale),
            project_points(KPS3D, yaw, pitch, scale=scale))


def _raster(poly, dilate=0.0, size=SIZE):
    m = np.zeros((size, size), np.uint8)
    if poly is not None:
        cv2.fillPoly(m, [np.asarray(poly, np.int32)], 255)
    if dilate > 0:
        r = max(1, int(size * dilate))
        m = cv2.dilate(m, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (r * 2 + 1, r * 2 + 1)), iterations=1)
    return m


def _crop_shapes(yaw, pitch):
    """(matte hull, visibility polygon) rasterised in the same crop space the
    matte is really built in, registered by the real `estimate_norm`."""
    lm, kp = _project(yaw, pitch)
    old = g.yaw_align
    g.yaw_align = 'pose'
    try:
        M = estimate_norm(np.asarray(kp, np.float32), SIZE, "arcface")
    finally:
        g.yaw_align = old
    Mh = np.vstack([np.asarray(M), [0, 0, 1]])

    hull, _, _ = landmark_hull(lm, kp)
    hc = (np.hstack([hull.reshape(-1, 2), np.ones((len(hull), 1))]) @ Mh.T)[:, :2]
    m_hull = _raster(cv2.convexHull(hc.astype(np.int32)))

    pose = solve_pose_jaw_5pt(np.asarray(kp, np.float64))
    jaw = pose[3] if pose is not None else 0.0
    m_vis = _raster(pose_visibility_polygon(yaw, pitch, DST, jaw), VIS_POLY_MARGIN)
    return m_hull, m_vis, lm, Mh


class PosePlacement(unittest.TestCase):
    """`_pose_placement` must place the reference head exactly where
    `_pose_template` does.

    This is the join the whole visibility layer rests on: the polygon is only in
    the same coordinates as the face because it is placed by the same scale and
    shift as the alignment template. `_pose_template` is deliberately left
    un-refactored (it is the tested alignment path and re-associating its
    arithmetic would move its last bits), so this test is what stops the two
    drifting apart.
    """

    def test_placement_reproduces_the_template(self):
        for yaw, pitch in POSES:
            for jaw in (0.0, 0.25):
                with self.subTest(yaw=yaw, pitch=pitch, jaw=jaw):
                    want = _pose_template(yaw, DST, pitch, jaw)
                    s, shift = _pose_placement(yaw, pitch, DST, jaw)
                    got = _project_reference(yaw, pitch, jaw) * s + shift
                    np.testing.assert_allclose(got, want, atol=1e-9)


class VisibilityPolygonSafety(unittest.TestCase):
    """The trim must never eat real face.

    A matte that clips visible skin is a worse artefact than the over-coverage it
    is removing — a hard edge across a cheek, rather than a soft wash over hair.
    So this is the property that has to hold everywhere, not on average, and it
    is what the safety margin is sized from: without the margin 1-2 truly-visible
    landmarks are lost at yaw 88.
    """

    def test_no_visible_landmark_is_ever_trimmed(self):
        for yaw, pitch in POSES:
            with self.subTest(yaw=yaw, pitch=pitch):
                m_hull, m_vis, lm, Mh = _crop_shapes(yaw, pitch)
                R = rotation(yaw, pitch, 0.0)
                vis = (_TRUTH_N @ R.T)[:, 2] > 0.05
                if vis.sum() < 3:
                    continue
                pts = (np.hstack([lm[vis], np.ones((int(vis.sum()), 1))]) @ Mh.T)[:, :2]
                lost = []
                for x, y in pts:
                    ix, iy = int(x), int(y)
                    if not (0 <= ix < SIZE and 0 <= iy < SIZE):
                        continue
                    if m_hull[iy, ix] > 0 and m_vis[iy, ix] == 0:
                        lost.append((ix, iy))
                self.assertEqual(lost, [], f"trimmed {len(lost)} visible landmark(s)")

    def test_frontal_faces_are_left_alone(self):
        """Not merely 'barely trimmed' — the layer must not engage at all near
        frontal, because that is the overwhelming majority of frames and the one
        place the existing matte is already correct. The gate is the same
        crossfade weight the alignment uses, so this is really asserting that the
        two layers share a band."""
        for yaw, pitch in ((0, 0), (10, 0), (0, 10), (8, -8)):
            with self.subTest(yaw=yaw, pitch=pitch):
                self.assertEqual(pose_weight_for(yaw, pitch), 0.0)

    def test_polygon_is_a_closed_usable_outline(self):
        for yaw, pitch in POSES:
            with self.subTest(yaw=yaw, pitch=pitch):
                poly = pose_visibility_polygon(yaw, pitch, DST)
                self.assertIsNotNone(poly)
                self.assertGreaterEqual(len(poly), 6)
                self.assertEqual(poly.shape[1], 2)
                self.assertGreater(int((_raster(poly) > 0).sum()), 0)

    def test_degenerate_input_returns_none_rather_than_raising(self):
        """It runs per face per frame; a bad pose must cost a skipped trim, not
        a dead frame."""
        for bad in (np.zeros((5, 2)), np.full((5, 2), np.nan)):
            self.assertIsNone(pose_visibility_polygon(0.0, 0.0, bad))


class VisibilityPolygonEffect(unittest.TestCase):
    """...and it must actually remove something, or it is dead weight."""

    def test_it_trims_more_as_the_head_turns_away(self):
        def trimmed(yaw, pitch):
            m_hull, m_vis, _, _ = _crop_shapes(yaw, pitch)
            a0 = int((m_hull > 0).sum())
            a1 = int((cv2.bitwise_and(m_hull, m_vis) > 0).sum())
            return 1.0 - a1 / max(1, a0)

        frontal = trimmed(0, 0)
        for yaw, pitch in ((55, -40), (70, -40), (88, -40), (88, -20)):
            with self.subTest(yaw=yaw, pitch=pitch):
                self.assertGreater(trimmed(yaw, pitch), frontal + 0.02,
                                   'trim is no larger off-axis than frontal — '
                                   'the layer is not doing anything')

    def test_the_visible_surface_shrinks_monotonically_with_yaw(self):
        """A property, not a threshold: whatever the absolute areas are, turning
        further away can never expose MORE face."""
        for pitch in (0, -20, -40):
            areas = [int((_raster(pose_visibility_polygon(y, pitch, DST)) > 0).sum())
                     for y in (0, 30, 55, 70, 80, 88)]
            with self.subTest(pitch=pitch):
                for a, b in zip(areas, areas[1:]):
                    self.assertLessEqual(b, a + 1)      # +1 for rasterisation


class AngleFade(unittest.TestCase):
    """The off-axis confidence fade — layer 3."""

    def test_strength_zero_is_exactly_off(self):
        for yaw, pitch in POSES:
            with self.subTest(yaw=yaw, pitch=pitch):
                self.assertEqual(angle_fade_weight(yaw, pitch, 0.0), 0.0)

    def test_frontal_is_never_faded(self):
        for yaw, pitch in ((0, 0), (20, 0), (0, 30), (30, 20)):
            with self.subTest(yaw=yaw, pitch=pitch):
                if offaxis_deg(yaw, pitch) <= ANGLE_FADE_ONSET_DEG:
                    self.assertEqual(angle_fade_weight(yaw, pitch, 100.0), 0.0)

    def test_it_reaches_the_ceiling_at_full_off_axis(self):
        for strength in (25.0, 65.0, 100.0):
            with self.subTest(strength=strength):
                self.assertAlmostEqual(
                    angle_fade_weight(ANGLE_FADE_FULL_DEG, 0.0, strength),
                    strength / 100.0, places=6)

    def test_it_never_exceeds_the_ceiling(self):
        for yaw, pitch in POSES:
            with self.subTest(yaw=yaw, pitch=pitch):
                self.assertLessEqual(angle_fade_weight(yaw, pitch, 65.0), 0.65 + 1e-9)

    def test_it_grows_with_off_axis_angle(self):
        """Monotone, because the thing it tracks — how far outside its training
        distribution the model is — only gets worse as the head turns."""
        prev = -1.0
        for yaw in range(0, 91, 5):
            w = angle_fade_weight(yaw, 0.0, 65.0)
            self.assertGreaterEqual(w, prev - 1e-12)
            prev = w

    def test_pitch_counts_too(self):
        """A head turned AND tilted is further off-axis than either alone, and
        the down-lateral case in the original bug report is exactly that. Keying
        on |yaw| would have missed it."""
        self.assertGreater(angle_fade_weight(55, -40, 65.0),
                           angle_fade_weight(55, 0, 65.0))

    def test_garbage_strength_does_not_raise(self):
        for bad in (None, '', 'lots'):
            self.assertEqual(angle_fade_weight(80, -30, bad), 0.0)


class PipelineWiring(unittest.TestCase):
    """The layers have to be reachable, gated, and defaulted as documented.

    Read from source for the gating, because the alternative is standing up a
    whole ProcessMgr with models loaded. The risk this covers is the one from
    experience: a new control that is sent, saved, shown — and read by nothing.
    """

    def setUp(self):
        here = os.path.dirname(os.path.abspath(__file__))
        self.procmgr = open(os.path.join(here, '..', 'roop', 'ProcessMgr.py'),
                            encoding='utf-8').read()
        self.masking = open(os.path.join(here, '..', 'roop', 'procmgr_masking.py'),
                            encoding='utf-8').read()

    def test_defaults_are_on(self):
        self.assertEqual(yaw_align_mode(), 'pose')
        self.assertIs(getattr(g, 'angle_visibility_mask'), True)
        self.assertGreater(float(getattr(g, 'angle_fade_strength')), 0.0)

    def test_the_fade_reads_its_global(self):
        self.assertIn('angle_fade_weight(tgt_yaw_deg, tgt_pitch_deg', self.procmgr)
        self.assertIn("'angle_fade_strength'", self.procmgr)

    def test_the_fade_is_applied_to_both_paste_surfaces(self):
        """`fake_frame` is blended into the result by `blend_ratio` even when an
        enhancer ran, so fading only the enhanced copy would leave un-faded swap
        in the output."""
        body = self.procmgr[self.procmgr.index('_fade = angle_fade_weight'):]
        body = body[:body.index('upscale = 512')]
        self.assertIn('fake_frame = _toward_plate(fake_frame)', body)
        self.assertIn('enhanced_frame = _toward_plate(enhanced_frame)', body)

    def test_the_trim_is_gated_on_pose_alignment(self):
        """The polygon is placed by the pose template. With the alignment off,
        the crop is somewhere else and the polygon would trim the wrong pixels —
        so this gate is a correctness requirement, not a preference."""
        body = self.procmgr[self.procmgr.index('vis_poly, vis_weight = None, 0.0'):]
        body = body[:body.index('if enhanced_frame is None:')]
        self.assertIn("angle_visibility_mask", body)
        self.assertIn("yaw_align_mode() == 'pose'", body)

    def test_the_trim_uses_the_jaw_aware_solve(self):
        """The alignment template is built from `solve_pose_jaw_5pt`. Placing the
        polygon from the jaw-blind solve would put it at a different pose than
        the template that decides where the face lands — and the disagreement is
        largest on a talking head, which is most footage."""
        body = self.procmgr[self.procmgr.index('vis_poly, vis_weight = None, 0.0'):]
        body = body[:body.index('if enhanced_frame is None:')]
        self.assertIn('solve_pose_jaw_5pt(target_face.kps)', body)

    def test_both_paste_calls_pass_the_polygon(self):
        """There are two `paste_upscale` call sites — enhanced and not. Missing
        one would make the trim silently depend on whether an enhancer is on."""
        calls = [ln for ln in self.procmgr.splitlines()
                 if 'self.paste_upscale(' in ln]
        self.assertEqual(len(calls), 2, 'paste_upscale call sites moved')
        for ln in calls:
            self.assertIn('vis_poly=vis_poly', ln)
            self.assertIn('vis_weight=vis_weight', ln)

    def test_the_matte_trim_is_applied_after_the_feather(self):
        """Order is load-bearing twice over.

        `blur_area` sizes its feather from the matte's extent, so trimming first
        narrows the feather everywhere the trim shrank the matte — measured 29px
        -> 18px, a 38% tighter seam, on exactly the high-pose faces where a seam
        shows most. And as with the overlap trim right above it, the feather is
        what would otherwise spread the swap back over the hair being removed.
        """
        feather = self.masking.index('img_matte = self.blur_area(img_matte')
        trim = self.masking.index('img_matte *= vis_matte')
        self.assertLess(feather, trim)

    def test_paste_upscale_defaults_to_no_trim(self):
        """So every other caller — virtualcam, the preview paths — is unchanged
        rather than quietly trimmed with a polygon it never computed."""
        import inspect
        from roop.procmgr_masking import MaskingMixin
        sig = inspect.signature(MaskingMixin.paste_upscale)
        self.assertIsNone(sig.parameters['vis_poly'].default)
        self.assertEqual(sig.parameters['vis_weight'].default, 0.0)


class _Options:
    """The three fields paste_upscale reads. Standing up a real ProcessOptions
    would drag in the whole processor-definition machinery for no extra coverage.
    """
    blend_ratio = 0.8
    show_face_area_overlay = False


class _Paster(MaskingMixin):
    def __init__(self):
        self.options = _Options()


class PasteUpscaleExecutes(unittest.TestCase):
    """Actually RUN the trim through paste_upscale.

    Everything above this either checks the polygon's geometry or reads source
    text. Neither would catch the failure that actually costs a render: a dtype
    or shape mismatch inside the paste, which only shows up when the code runs.
    """

    def setUp(self):
        self.paster = _Paster()
        rng = np.random.default_rng(7)
        self.target = rng.integers(0, 255, (400, 400, 3), dtype=np.uint8)
        self.face = rng.integers(0, 255, (512, 512, 3), dtype=np.uint8)
        _, kp = _project(70, -30)
        old = g.yaw_align
        g.yaw_align = 'pose'
        try:
            self.M = np.asarray(estimate_norm(np.asarray(kp, np.float32), SIZE,
                                              "arcface"), dtype=np.float64)
        finally:
            g.yaw_align = old
        pose = solve_pose_jaw_5pt(np.asarray(kp, np.float64))
        poly = pose_visibility_polygon(70, -30, DST, pose[3] if pose else 0.0)
        self.poly = np.asarray(poly, np.float64) / float(SIZE)

    def _paste(self, **kw):
        return self.paster.paste_upscale(
            self.face, self.face, self.M, self.target, 1,
            [0, 0, 0, 0, 20.0, 10.0], **kw)

    def test_no_trim_is_the_previous_behaviour_exactly(self):
        """The default path has to be untouched — this is what makes the layer
        safe to add to a pipeline everything else already depends on."""
        base = self._paste()
        np.testing.assert_array_equal(self._paste(vis_poly=None, vis_weight=0.0), base)
        np.testing.assert_array_equal(self._paste(vis_poly=self.poly, vis_weight=0.0), base)

    def test_a_trim_changes_the_result_and_only_toward_the_plate(self):
        """Trimming may only ever REVEAL the original frame. It removes matte, so
        no pixel may end up further from the plate than it already was.

        This is the test that caught the ordering bug: with the trim applied
        before `blur_area`, 4519 pixels moved AWAY from the plate by up to 67
        levels, because shrinking the matte shrank the feather and locally
        sharpened the seam. Strictly monotone is only achievable by trimming
        after the feather, which is why the code does.
        """
        base = self._paste()
        trimmed = self._paste(vis_poly=self.poly, vis_weight=1.0)
        self.assertFalse(np.array_equal(trimmed, base), 'the trim did nothing')
        d_base = np.abs(base.astype(np.int32) - self.target.astype(np.int32))
        d_trim = np.abs(trimmed.astype(np.int32) - self.target.astype(np.int32))
        worse = int((d_trim > d_base + 1).sum())
        self.assertEqual(worse, 0,
                         f'{worse} channel values moved AWAY from the original '
                         f'frame (max +{int((d_trim - d_base).max())} levels)')

    def test_partial_weight_lands_between_the_two(self):
        """The crossfade has to be a fade, not a switch, or the pose band cannot
        stop it flickering."""
        base = self._paste().astype(np.float64)
        full = self._paste(vis_poly=self.poly, vis_weight=1.0).astype(np.float64)
        half = self._paste(vis_poly=self.poly, vis_weight=0.5).astype(np.float64)
        self.assertLess(np.abs(half - base).mean(), np.abs(full - base).mean())
        self.assertLess(np.abs(half - full).mean(), np.abs(base - full).mean() * 1.05)

    def test_it_survives_a_polygon_that_misses_the_crop(self):
        """A wild pose solve must cost a wrong trim, not an exception."""
        for poly in (self.poly * 50.0, self.poly - 10.0,
                     np.zeros((4, 2)), np.full((4, 2), 0.5)):
            with self.subTest(poly=str(np.asarray(poly)[:1])):
                out = self._paste(vis_poly=poly, vis_weight=0.7)
                self.assertEqual(out.shape, self.target.shape)
                self.assertEqual(out.dtype, np.uint8)


class SettingsSurface(unittest.TestCase):
    """The env override has to keep working, or there is no way to A/B this."""

    def test_env_can_still_force_the_old_behaviour(self):
        from settings import initial_yaw_align
        old = os.environ.get('ROOP_YAW_ALIGN')
        try:
            for raw, want in (('off', 'off'), ('0', 'off'), ('false', 'off'),
                              ('stabilize', 'stabilize'), ('1', 'stabilize'),
                              ('pose', 'pose'), ('', 'pose'), ('nonsense', 'pose')):
                os.environ['ROOP_YAW_ALIGN'] = raw
                with self.subTest(raw=raw):
                    self.assertEqual(initial_yaw_align(), want)
        finally:
            if old is None:
                os.environ.pop('ROOP_YAW_ALIGN', None)
            else:
                os.environ['ROOP_YAW_ALIGN'] = old

    def test_the_two_new_settings_round_trip_through_the_config(self):
        """Saved as well as loaded. A setting that loads but is dropped on save
        resets itself the next time anything else on the panel is touched, which
        looks like the feature randomly turning itself off.

        Note the config is YAML despite being named .json — hence the yaml load
        below, and why the JSON fixture above parses (YAML is a superset).
        """
        import json
        import tempfile

        import yaml

        from settings import Settings
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'config.json')
            with open(path, 'w', encoding='utf-8') as fh:
                json.dump({'angle_visibility_mask': False,
                           'angle_fade_strength': 30.0}, fh)
            s = Settings(path)
            self.assertIs(s.angle_visibility_mask, False)
            self.assertEqual(float(s.angle_fade_strength), 30.0)
            s.save()
            with open(path, encoding='utf-8') as fh:
                saved = yaml.safe_load(fh)
            self.assertIs(saved['angle_visibility_mask'], False)
            self.assertEqual(float(saved['angle_fade_strength']), 30.0)


if __name__ == '__main__':
    unittest.main()
