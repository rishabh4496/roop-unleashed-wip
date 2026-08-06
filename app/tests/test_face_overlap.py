"""face_overlap — who owns a pixel when two swapped faces meet.

Each face is pasted under a matte built from its own geometry alone: a hull
that has been dilated, given a forehead extension reaching 60 % of the face
height past the brow, and feathered outward. None of that knows there is
somebody standing next to it, so on interacting faces the mattes overlap and
each face paints a band of its own swap over the other. Which band survives is
decided by whoever happens to be pasted last.

Properties asserted here:

  * SEPARATE FACES COST NOTHING. Faces that are not near each other return no
    regions at all, so the frames that make up the overwhelming majority of any
    render do not pay for this and cannot be changed by it.

  * NOTHING IS LEFT UNPAINTED. The faces composite in sequence, so the test is
    not that the two fields sum to 1 — it is that the SEQUENTIAL result covers
    the contested area completely. Two mattes at 0.5 sum to 1 and still leave
    a quarter of the plate showing down the join.

  * THE BOUNDARY IS BETWEEN THEM. Deep inside a face, that face owns
    everything; the hand-over happens in a narrow band around the midline.

  * IT DOES NOT DEPEND ON ORDER. The same pair of faces produces the same
    boundary whichever order they arrive in — which is the whole point, since
    match order changes frame to frame and used to decide the join.

  * NEARER WINS. A face twice the size on screen is in front, and takes the
    contested pixels rather than being painted over by the one behind it.

The compositing tests at the bottom cover the other half: that ProcessMgr
actually pastes far-to-near, and that it reads every face from the untouched
plate rather than from the half-finished composite.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop import face_overlap                           # noqa: E402
from roop.face_overlap import build_regions             # noqa: E402
from roop.ProcessMgr import ProcessMgr                  # noqa: E402

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SHAPE = (400, 800, 3)


class _Face:
    """A face with only the geometry face_overlap reads."""

    def __init__(self, cx, cy, size, landmarks=True):
        self.bbox = np.array([cx - size / 2, cy - size / 2,
                              cx + size / 2, cy + size / 2], dtype=np.float32)
        self.kps = np.array([[cx - size * 0.2, cy - size * 0.1],
                             [cx + size * 0.2, cy - size * 0.1],
                             [cx, cy + size * 0.05],
                             [cx - size * 0.15, cy + size * 0.25],
                             [cx + size * 0.15, cy + size * 0.25]],
                            dtype=np.float32)
        self.landmark_2d_106 = _ring(cx, cy, size) if landmarks else None


def _ring(cx, cy, size):
    """106 points around an oval face, ordered so indices 33-52 land on the
    brow line the way the real landmark model's do."""
    pts = np.zeros((106, 2), dtype=np.float32)
    # 0-32: jaw / silhouette, lower half of the oval.
    a = np.linspace(np.pi, 2 * np.pi, 33)
    pts[:33, 0] = cx + np.cos(a) * size * 0.5
    pts[:33, 1] = cy - np.sin(a) * size * 0.5
    # 33-52: brow line across the top of the face.
    pts[33:53, 0] = np.linspace(cx - size * 0.4, cx + size * 0.4, 20)
    pts[33:53, 1] = cy - size * 0.25
    # 53-105: interior features, well inside the hull.
    rng = np.random.RandomState(0)
    pts[53:, 0] = cx + rng.uniform(-size * 0.2, size * 0.2, 53)
    pts[53:, 1] = cy + rng.uniform(-size * 0.1, size * 0.2, 53)
    return pts


def _own_at(region, x, y):
    return float(region.own[int(y) - region.y0, int(x) - region.x0])


class Separated(unittest.TestCase):
    def test_one_face_has_nobody_to_argue_with(self):
        self.assertIsNone(build_regions([_Face(200, 200, 120)], SHAPE))

    def test_faces_across_the_frame_are_left_alone(self):
        faces = [_Face(120, 200, 120), _Face(680, 200, 120)]
        self.assertIsNone(
            build_regions(faces, SHAPE),
            'faces nowhere near each other must not pay for the demarcation, '
            'nor have their mattes touched by it')

    def test_adjacent_faces_still_get_a_boundary(self):
        """Not overlapping, but close enough that the dilation and the feather
        reach across — the case that produces the smear along the join."""
        faces = [_Face(320, 200, 120), _Face(440, 200, 120)]
        self.assertIsNotNone(build_regions(faces, SHAPE))

    def test_a_face_with_no_geometry_is_skipped(self):
        class _Blank:
            bbox = None
            landmark_2d_106 = None
        self.assertIsNone(build_regions([_Blank(), _Face(200, 200, 120)], SHAPE))

    def test_the_switch_turns_it_off(self):
        faces = [_Face(340, 200, 120), _Face(420, 200, 120)]
        self.assertIsNotNone(build_regions(faces, SHAPE))
        old = face_overlap.ENABLED
        try:
            face_overlap.ENABLED = False
            self.assertIsNone(build_regions(faces, SHAPE))
        finally:
            face_overlap.ENABLED = old


class Partition(unittest.TestCase):
    def setUp(self):
        # Face 0 is painted first (it is the one further away), face 1 on top.
        self.faces = [_Face(340, 200, 140), _Face(460, 200, 140)]
        self.regions = build_regions(self.faces, SHAPE, order=[0, 1])
        self.assertEqual(sorted(self.regions), [0, 1])

    def test_the_join_never_shows_the_plate(self):
        """The mattes are applied in sequence, so what has to reach 1 is the
        accumulated coverage — `b + (1-b)·a`, not `a + b`. Getting this wrong
        replaces the smear with a hairline of untouched footage down the join,
        which is just as visible and harder to explain."""
        a, b = self.regions[0].own, self.regions[1].own
        covered = b + (1.0 - b) * a
        contested = (a > 0) | (b > 0)
        worst = float(covered[contested].min())
        self.assertGreater(
            worst, 0.999,
            f'{(covered[contested] < 0.999).sum()} px of the contested area are '
            f'only {worst:.3f} covered — the plate shows through there')

    def test_a_face_owns_its_own_middle(self):
        self.assertEqual(_own_at(self.regions[0], 340, 200), 1.0)
        self.assertEqual(_own_at(self.regions[1], 460, 200), 1.0)

    def test_the_face_on_top_takes_none_of_the_one_below(self):
        self.assertEqual(_own_at(self.regions[1], 340, 200), 0.0)

    def test_the_face_below_stops_just_past_the_boundary(self):
        """It carries on a little way to back the ramp above it, and no further
        — the whole point is that it no longer paints across its neighbour."""
        self.assertEqual(_own_at(self.regions[0], 460, 200), 0.0)

    def test_the_hand_over_is_narrow(self):
        """A wide transition is a band where two different faces are mixed."""
        r = self.regions[1]
        row = r.own[200 - r.y0]
        soft = int(((row > 0.02) & (row < 0.98)).sum())
        self.assertLess(soft, 20, f'{soft}px hand-over across the join is a band, '
                                  'not a demarcation')
        self.assertGreater(soft, 0, 'a hard step would alias along the join')

    def test_the_boundary_sits_between_the_faces(self):
        r = self.regions[1]
        row = r.own[200 - r.y0]
        mid = int(np.argmin(np.abs(row - 0.5))) + r.x0
        self.assertGreater(mid, 340)
        self.assertLess(mid, 460)


class OrderIndependence(unittest.TestCase):
    """Match order changes frame to frame — in identity-lock mode the faces are
    re-sorted by identity distance every frame. If the boundary moved with it,
    a pair standing close together would hand the join back and forth and
    flicker, which is exactly the reported symptom."""

    def test_the_same_pair_gives_the_same_boundary_either_way(self):
        """Same faces, same paint order, listed the other way round."""
        a, b = _Face(340, 200, 140), _Face(460, 200, 140)
        fwd = build_regions([a, b], SHAPE, order=[0, 1])
        rev = build_regions([b, a], SHAPE, order=[1, 0])
        np.testing.assert_array_equal(fwd[0].own, rev[1].own)
        np.testing.assert_array_equal(fwd[1].own, rev[0].own)

    def test_equal_faces_meet_halfway(self):
        a, b = _Face(340, 200, 140), _Face(460, 200, 140)
        r = build_regions([a, b], SHAPE, order=[0, 1])
        row = r[1].own[200 - r[1].y0]
        mid = int(np.argmin(np.abs(row - 0.5))) + r[1].x0
        self.assertAlmostEqual(mid, 400, delta=4,
                               msg='two identical faces side by side must meet '
                                   'halfway; anything else is a bias nobody asked for')


class Depth(unittest.TestCase):
    """Index 0 is the near (larger) face throughout, painted last."""

    def _boundary(self, near_size, far_size):
        r = build_regions([_Face(340, 200, near_size), _Face(470, 200, far_size)],
                          SHAPE, order=[1, 0])
        row = r[0].own[200 - r[0].y0]
        return int(np.argmin(np.abs(row - 0.5))) + r[0].x0

    def test_the_bigger_face_is_in_front(self):
        """On-screen size is the only depth cue available here, and a head twice
        the size of another is in front of it, not behind."""
        self.assertGreater(
            self._boundary(220, 110), self._boundary(220, 220),
            'the nearer (larger) face must take MORE of the contested area')

    def test_size_alone_does_not_swallow_a_face(self):
        """The signed distance is normalised by each face's own size for this
        reason: without it a big face is deeper everywhere just for being big,
        and would take the whole of a small face standing in front of it."""
        r = build_regions([_Face(300, 200, 260), _Face(470, 200, 90)],
                          SHAPE, order=[1, 0])
        self.assertEqual(_own_at(r[0], 470, 200), 0.0,
                         'the big face must not claim the small one\'s centre')


class Trimming(unittest.TestCase):
    def setUp(self):
        self.faces = [_Face(340, 200, 140), _Face(460, 200, 140)]
        self.regions = build_regions(self.faces, SHAPE, order=[0, 1])

    def test_trim_frame_clears_the_neighbours_half(self):
        matte = np.ones(SHAPE[:2], dtype=np.float32)
        self.regions[0].trim_frame(matte)
        self.assertEqual(matte[200, 460], 0.0)   # over face 1
        self.assertEqual(matte[200, 340], 1.0)   # over face 0
        self.assertEqual(matte[200, 40], 1.0)    # outside the ROI entirely

    def test_trim_frame_refuses_a_mismatched_matte(self):
        small = np.ones((50, 50), dtype=np.float32)
        self.regions[0].trim_frame(small)
        self.assertTrue((small == 1.0).all(),
                        'a matte at another resolution must be left alone '
                        'rather than silently mis-registered')

    def test_crop_is_none_when_the_box_is_all_ours(self):
        self.assertIsNone(self.regions[0].crop(0, 0, 30, 30))

    def test_crop_fills_the_part_outside_the_roi_with_ownership(self):
        r = self.regions[0]
        out = r.crop(r.x0 - 20, 190, r.x0 + 20, 210)
        self.assertEqual(out.shape, (20, 40))
        self.assertTrue((out[:, :20] == 1.0).all(),
                        'outside the contested rectangle the face owns everything')


class CompositeOrder(unittest.TestCase):
    """ProcessMgr._composite_faces decides paint order for the whole frame."""

    def _mgr(self):
        mgr = ProcessMgr.__new__(ProcessMgr)
        mgr.painted = []

        def _stub(face_index, target_face, frame, plate=None, region=None):
            mgr.painted.append((face_index, target_face, plate is frame, region))
            return frame

        mgr.process_face = _stub
        return mgr

    def test_faces_are_pasted_far_to_near(self):
        mgr = self._mgr()
        big, small = _Face(340, 200, 220), _Face(600, 200, 90)
        plate = np.zeros(SHAPE, dtype=np.uint8)
        mgr._composite_faces([(0, big), (1, small)], plate, plate.copy())
        self.assertEqual([p[0] for p in mgr.painted], [1, 0],
                         'the smaller (further) face must be painted first so '
                         'the nearer one ends up on top')

    def test_paint_order_ignores_match_order(self):
        mgr_a, mgr_b = self._mgr(), self._mgr()
        big, small = _Face(340, 200, 220), _Face(600, 200, 90)
        plate = np.zeros(SHAPE, dtype=np.uint8)
        mgr_a._composite_faces([(0, big), (1, small)], plate, plate.copy())
        mgr_b._composite_faces([(1, small), (0, big)], plate, plate.copy())
        self.assertEqual([p[0] for p in mgr_a.painted],
                         [p[0] for p in mgr_b.painted])

    def test_every_face_reads_the_untouched_plate(self):
        mgr = self._mgr()
        plate = np.zeros(SHAPE, dtype=np.uint8)
        mgr._composite_faces([(0, _Face(340, 200, 140)), (1, _Face(460, 200, 140))],
                             plate, plate.copy())
        self.assertEqual([p[2] for p in mgr.painted], [False, False],
                         'reading the running composite is how one face ends up '
                         'swapping its neighbour\'s already-swapped pixels')

    def test_overlapping_faces_get_regions_and_lone_ones_do_not(self):
        mgr = self._mgr()
        plate = np.zeros(SHAPE, dtype=np.uint8)
        mgr._composite_faces([(0, _Face(340, 200, 140)), (1, _Face(460, 200, 140))],
                             plate, plate.copy())
        self.assertTrue(all(p[3] is not None for p in mgr.painted))

        mgr = self._mgr()
        mgr._composite_faces([(0, _Face(120, 200, 120)), (1, _Face(680, 200, 120))],
                             plate, plate.copy())
        self.assertTrue(all(p[3] is None for p in mgr.painted))

    def test_near_equal_faces_keep_a_stable_order(self):
        """Detection noise moves a bbox by a pixel or two every frame. Sorting on
        raw area lets that flip which of two similarly sized faces is on top, so
        the join hands back and forth — the flicker the ordering exists to stop."""
        plate = np.zeros(SHAPE, dtype=np.uint8)
        seen = set()
        for jitter in (-2, -1, 0, 1, 2):
            mgr = self._mgr()
            left = _Face(340, 200, 140)
            right = _Face(460, 200, 140 + jitter)
            mgr._composite_faces([(0, left), (1, right)], plate, plate.copy())
            seen.add(tuple(p[0] for p in mgr.painted))
        self.assertEqual(len(seen), 1,
                         f'paint order changed under bbox noise: {seen}')

    def test_a_clearly_bigger_face_still_wins(self):
        """The quantisation must not be so coarse that a real depth difference
        stops registering."""
        mgr = self._mgr()
        plate = np.zeros(SHAPE, dtype=np.uint8)
        mgr._composite_faces([(0, _Face(340, 200, 140)), (1, _Face(460, 200, 175))],
                             plate, plate.copy())
        self.assertEqual([p[0] for p in mgr.painted], [0, 1])

    def test_nothing_pending_is_a_no_op(self):
        mgr = self._mgr()
        plate = np.zeros(SHAPE, dtype=np.uint8)
        temp = plate.copy()
        self.assertIs(mgr._composite_faces([], plate, temp), temp)


class SwapFacesPaintsOnlyAtTheEnd(unittest.TestCase):
    """A source guard, because the regression is invisible at runtime: pasting
    inside a match loop still produces a plausible frame, it just restores the
    order-dependent join. The match loops must only ever APPEND."""

    def test_the_match_loops_do_not_paste(self):
        src = open(os.path.join(APP, 'roop', 'ProcessMgr.py'), encoding='utf-8').read()
        body = src.split('def swap_faces(', 1)[1].split('\n    def ', 1)[0]
        calls = [ln.strip() for ln in body.splitlines()
                 if 'self.process_face(' in ln]
        self.assertEqual(
            len(calls), 1,
            'swap_faces may only call process_face once — the "first" mode, '
            'which has a single face. Every other mode collects its matches and '
            f'hands them to _composite_faces. Found: {calls}')
        self.assertIn('plate=frame', calls[0],
                      'even the single-face path must name its plate, or the '
                      'next reader assumes frame is both')


if __name__ == '__main__':
    unittest.main()
