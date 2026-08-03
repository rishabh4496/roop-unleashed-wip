"""The non-frontal mask-routing score and its temporal latch.

Two guarantees are load-bearing here:

1. `nonfrontal_score(...) > 1.0` is EXACTLY the verdict the previous
   OR-of-four-booleans gave. That is what makes replacing them a refactor
   rather than a behaviour change, and it is why the latch can be reasoned
   about as a single number.

2. Noise strictly inside the hysteresis band cannot flip the verdict, in ANY
   arrival order. The video path hands adjacent frames to different worker
   threads, so "any order" is not hypothetical.
"""

import os
import sys
import threading
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop.face_util import kps_pose_ratios, offaxis_deg, solve_pose_5pt  # noqa: E402
from roop.nonfrontal import (NONFRONTAL_MARGIN, NonFrontalRouter,        # noqa: E402
                             nonfrontal_score)
from tests.facegeom import project_kps                                   # noqa: E402


def legacy_or(kps, tgt_pitch_deg=0.0):
    """The rule as it shipped before the score existed, transcribed."""
    non_frontal = False
    left_eye_x, right_eye_x, nose_x = kps[0][0], kps[1][0], kps[2][0]
    d_left, d_right = abs(nose_x - left_eye_x), abs(nose_x - right_eye_x)
    if d_left + d_right > 1e-5 and abs(d_left - d_right) / (d_left + d_right) > 0.25:
        non_frontal = True
    yaw_ratio, pitch_ratio = kps_pose_ratios(kps)
    if yaw_ratio is not None and yaw_ratio < 0.55:
        non_frontal = True
    if pitch_ratio is not None and not (0.32 < pitch_ratio < 0.70):
        non_frontal = True
    pose = solve_pose_5pt(kps)
    if pose is not None and offaxis_deg(pose[0], pose[1]) > 50.0:
        non_frontal = True
    if (kps[0][1] + kps[1][1]) / 2.0 > (kps[3][1] + kps[4][1]) / 2.0 + 5.0:
        non_frontal = True
    if abs(tgt_pitch_deg) > 30.0:
        non_frontal = True
    return non_frontal


class TestScoreReproducesTheOldRule(unittest.TestCase):
    def test_exact_over_the_pose_grid(self):
        checked = 0
        for yaw in range(0, 91, 2):
            for pitch in range(-50, 51, 4):
                for roll in (-20, 0, 20):
                    kps = project_kps(yaw, pitch, roll)
                    self.assertEqual(nonfrontal_score(kps) > 1.0, legacy_or(kps),
                                     f"yaw={yaw} pitch={pitch} roll={roll}")
                    checked += 1
        self.assertGreater(checked, 3000)

    def test_exact_under_noise_too(self):
        """Grid poses are tidy; real keypoints are not. Noise is what pushes a
        face onto a threshold, which is where an inexact normalisation would
        show up."""
        rng = np.random.default_rng(0)
        for _ in range(4000):
            kps = project_kps(rng.uniform(0, 90), rng.uniform(-50, 50),
                              rng.uniform(-30, 30))
            kps = kps + rng.normal(0, 2.0, (5, 2)).astype(np.float32)
            self.assertEqual(nonfrontal_score(kps) > 1.0, legacy_or(kps))

    def test_the_exact_pitch_term_still_counts(self):
        """tgt_pitch_deg comes from landmark_3d_68 and is only present when that
        model is loaded, so it is easy to drop by accident."""
        frontal = project_kps(0, 0)
        self.assertFalse(nonfrontal_score(frontal) > 1.0)
        self.assertTrue(nonfrontal_score(frontal, tgt_pitch_deg=45.0) > 1.0)
        self.assertTrue(nonfrontal_score(frontal, tgt_pitch_deg=-45.0) > 1.0)

    def test_unreadable_keypoints_route_frontal(self):
        for bad in (None, np.zeros((3, 2), np.float32),
                    np.full((5, 2), np.nan, np.float32)):
            self.assertEqual(nonfrontal_score(bad), 0.0)


class TestHysteresisKillsTheChatter(unittest.TestCase):
    """The defect: a binary routing decision driven by a noisy per-frame score.
    The two mask paths derive the mask differently, so flipping between them
    moves the mask boundary on a head that is not moving."""

    HOT = [(0, 30), (5, 30), (10, 30), (75, -35), (75, -40)]

    @staticmethod
    def _flips(seq):
        return sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1])

    def test_a_still_head_stops_flipping(self):
        for yaw, pitch in self.HOT:
            base = project_kps(yaw, pitch)
            rng = np.random.default_rng(4)
            frames = [base + rng.normal(0, 1.0, (5, 2)).astype(np.float32)
                      for _ in range(400)]
            bare = self._flips([nonfrontal_score(k) > 1.0 for k in frames])
            router = NonFrontalRouter()
            latched = self._flips([router.verdict(k, 0.0, t)
                                   for t, k in enumerate(frames)])
            if bare > 10:      # only the poses that actually chatter
                self.assertEqual(latched, 0,
                                 f"yaw={yaw} pitch={pitch}: {bare} -> {latched}")

    def test_the_worst_case_really_was_bad(self):
        """Records the defect so the latch cannot be dropped as unnecessary."""
        base = project_kps(0, 30)
        rng = np.random.default_rng(4)
        seq = [nonfrontal_score(base + rng.normal(0, 1.0, (5, 2)).astype(np.float32)) > 1.0
               for _ in range(400)]
        self.assertGreater(self._flips(seq), 50,
                           "a still head tilted up no longer chatters without "
                           "the latch — re-check this test's premise")

    def test_a_real_turn_still_registers(self):
        """Hysteresis must not become a face that never re-routes. A head nodding
        through the boundary has to switch every time it genuinely crosses."""
        router = NonFrontalRouter()
        rng = np.random.default_rng(1)
        verdicts = []
        for t in range(400):
            pitch = 45.0 * np.sin(2 * np.pi * t / 200.0)
            kps = project_kps(0, pitch) + rng.normal(0, 1.0, (5, 2)).astype(np.float32)
            verdicts.append(router.verdict(kps, 0.0, t))
        # Two full nod cycles cross the +band and the -band twice each.
        self.assertEqual(self._flips(verdicts), 8,
                         f"{self._flips(verdicts)} transitions, expected 8")

    def test_margin_zero_is_the_bare_threshold(self):
        router = NonFrontalRouter(margin=0.0)
        rng = np.random.default_rng(2)
        for t in range(200):
            kps = project_kps(0, 30) + rng.normal(0, 1.0, (5, 2)).astype(np.float32)
            self.assertEqual(router.verdict(kps, 0.0, t),
                             nonfrontal_score(kps) > 1.0)

    def test_a_single_image_behaves_exactly_as_before(self):
        """No frame index means no temporal dimension to exploit."""
        router = NonFrontalRouter()
        for yaw in range(0, 91, 10):
            for pitch in (-40, 0, 40):
                kps = project_kps(yaw, pitch)
                self.assertEqual(router.verdict(kps, 0.0, None), legacy_or(kps))


class TestLatchIsAQuery(unittest.TestCase):
    """The identity the whole threading design rests on: a hysteresis latch is
    the same thing as "the value of the most recent decisive frame at or before
    t". A latch is an evolving state and cannot be shared across out-of-order
    workers; a query over an index-keyed event log can."""

    @staticmethod
    def _sequential_latch(scores, margin=NONFRONTAL_MARGIN):
        lo, hi = 1.0 - margin, 1.0 + margin
        latch = scores[0] > 1.0
        out = []
        for value in scores:
            latch = True if value > hi else (False if value < lo else latch)
            out.append(latch)
        return out

    def test_the_router_matches_a_sequential_latch_frame_for_frame(self):
        rng = np.random.default_rng(21)
        for yaw, pitch, amp in ((0, 30, 12.0), (75, -35, 10.0), (45, 0, 25.0)):
            frames, scores = [], []
            for t in range(400):
                p = pitch + amp * np.sin(2 * np.pi * t / 90.0)
                kps = project_kps(yaw, p) + rng.normal(0, 1.0, (5, 2)).astype(np.float32)
                frames.append(kps)
                scores.append(nonfrontal_score(kps))
            router = NonFrontalRouter()
            got = [router.verdict(k, 0.0, t) for t, k in enumerate(frames)]
            self.assertEqual(got, self._sequential_latch(scores),
                             f"diverged from a sequential latch at yaw={yaw}")


class TestOrderIndependence(unittest.TestCase):
    """Adjacent frames go to different worker threads (`frame % num_threads`),
    so scores reach the router out of order. The guarantee that has to survive
    that is the one that kills the chatter."""

    def test_in_band_noise_cannot_flip_the_verdict_in_any_order(self):
        base = project_kps(0, 30)
        rng = np.random.default_rng(9)
        frames = [(t, base + rng.normal(0, 1.0, (5, 2)).astype(np.float32))
                  for t in range(200)]
        # Every score here sits inside the band, which is the chatter case.
        scores = [nonfrontal_score(k) for _, k in frames]
        self.assertTrue(
            all(abs(s - 1.0) < NONFRONTAL_MARGIN for s in scores),
            "premise broken: these scores are not all in-band")

        outcomes = set()
        for seed in range(6):
            shuffled = list(frames)
            np.random.default_rng(seed).shuffle(shuffled)
            router = NonFrontalRouter()
            outcomes.add(tuple(router.verdict(k, 0.0, t) for t, k in shuffled))
        for got in outcomes:
            self.assertEqual(len(set(got)), 1,
                             "verdict changed mid-sequence on in-band noise")

    def test_repeating_a_frame_is_a_no_op(self):
        """process_mask runs once per mask processor, so the same face/frame can
        be scored several times. A latch is idempotent; a consecutive-frame
        debounce would not have been."""
        router = NonFrontalRouter()
        kps = project_kps(75, -40)
        first = router.verdict(kps, 0.0, 5)
        for _ in range(10):
            self.assertEqual(router.verdict(kps, 0.0, 5), first)

    def test_out_of_order_frames_do_not_orphan_a_track(self):
        """The bug an earlier version of this shipped with: matching on the
        frame index made a late frame look like a face that had been missing for
        ages, so it forked a fresh unlatched track and the chatter returned."""
        router = NonFrontalRouter()
        kps = project_kps(0, 30)
        router.verdict(kps, 0.0, 190)
        self.assertEqual(len(router._tracks), 1)
        for late in (3, 7, 40, 12):
            router.verdict(kps, 0.0, late)
            self.assertEqual(len(router._tracks), 1,
                             f"frame {late} forked the track")

    # A moving face whose score repeatedly crosses the band — the case that
    # rippled. Deliberately driven through SIMULATED arrival orders rather than
    # real threads: the residual publication race is genuinely nondeterministic,
    # so asserting a flip count against real threads is a FLAKY test (the first
    # version of this was, and failed at 11 against a tolerance of 10). Real
    # threads are exercised for SAFETY in test_concurrent_access_is_safe.
    @staticmethod
    def _moving_clip(n=400):
        rng = np.random.default_rng(7)
        out = []
        for t in range(n):
            pitch = 30.0 + 12.0 * np.sin(2 * np.pi * t / 100.0)
            out.append(project_kps(0, pitch)
                       + rng.normal(0, 1.0, (5, 2)).astype(np.float32))
        return out

    @staticmethod
    def _flips(seq):
        return sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1])

    @staticmethod
    def _arrival_orders(n, num_threads):
        """Two shapes of reordering the real pipeline can produce. Both are
        BOUNDED — a frame is never more than a round out of place — which is the
        regime `frame % threads` actually creates."""
        worker_major = [t for i in range(num_threads)
                        for t in range(i, n, num_threads)]
        round_reversed = [t for k in range(0, n, num_threads)
                          for t in reversed(range(k, min(k + num_threads, n)))]
        return {"worker-major": worker_major, "round-reversed": round_reversed}

    def _play(self, frames, order, observe_first=False):
        router = NonFrontalRouter()
        out = [None] * len(frames)
        if observe_first:
            for t in order:
                router.observe(frames[t], 0.0, t)
        for t in order:
            out[t] = router.verdict(frames[t], 0.0, t)
        return out

    def test_out_of_order_arrival_is_never_worse_than_no_latch(self):
        """The regression that killed the previous design. With a state machine
        per worker a MOVING face rippled to 22 flips against a no-latch baseline
        of 16 and a correct answer of 8 — the latch actively made things worse.
        Whatever the arrival order, that must not happen again."""
        frames = self._moving_clip()
        bare = self._flips([nonfrontal_score(k) > 1.0 for k in frames])
        for num_threads in (2, 4, 8, 16):
            for name, order in self._arrival_orders(len(frames), num_threads).items():
                got = self._flips(self._play(frames, order))
                self.assertLessEqual(
                    got, bare,
                    f"{name} @{num_threads} threads: {got} flips vs {bare} unlatched")

    def test_reordering_stays_benign_across_the_real_queue_depth(self):
        """How far out of order frames can actually get, from the pipeline's own
        plumbing: one reader deals round-robin into per-thread `Queue(3)` and
        blocks when any fills, so drift is bounded to roughly three rounds — 24
        frames at 8 threads. Swept well past that, the latch stays better than
        no latch.

        The window matters. Left unbounded (a microbenchmark with no per-frame
        work, where workers free-run) the same clip degrades to 28 flips, worse
        than the 16 of no latch at all. That regime cannot happen behind bounded
        queues, but it is why this test pins a window rather than shuffling
        freely.
        """
        frames = self._moving_clip()
        bare = self._flips([nonfrontal_score(k) > 1.0 for k in frames])

        def windowed(n, width, seed):
            gen = np.random.default_rng(seed)
            order, pending = [], list(range(n))
            while pending:
                order.append(pending.pop(int(gen.integers(0, min(len(pending), width)))))
            return order

        for width in (4, 8, 16, 24, 48):
            for seed in range(6):
                got = self._flips(self._play(frames, windowed(len(frames), width, seed)))
                self.assertLessEqual(
                    got, bare,
                    f"reorder window {width}, seed {seed}: {got} vs {bare} unlatched")

    def test_publishing_ahead_makes_the_result_order_independent(self):
        """What `observe()` buys, and why ProcessMgr calls it at detection time
        rather than at mask time. Once a frame's neighbours are already in the
        log the query cannot race them, so every arrival order collapses to the
        same answer, frame for frame."""
        frames = self._moving_clip()
        reference = None
        for num_threads in (2, 4, 8, 16):
            for name, order in self._arrival_orders(len(frames), num_threads).items():
                got = self._play(frames, order, observe_first=True)
                if reference is None:
                    reference = got
                self.assertEqual(got, reference,
                                 f"{name} @{num_threads} threads diverged")
        # And genuinely smooth, not merely consistently wrong.
        self.assertLessEqual(self._flips(reference), 8)

    def test_concurrent_access_is_safe(self):
        router = NonFrontalRouter()
        rng = np.random.default_rng(3)
        frames = [(t, project_kps(0, 30) + rng.normal(0, 1.0, (5, 2)).astype(np.float32))
                  for t in range(400)]
        errors = []

        def worker(offset):
            try:
                for t, k in frames[offset::4]:
                    router.verdict(k, 0.0, t)
            except Exception as exc:       # noqa: BLE001 - surfaced below
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        self.assertEqual(errors, [])
        self.assertLessEqual(len(router._tracks), 2)


class TestMultipleFaces(unittest.TestCase):
    def test_each_face_keeps_its_own_latch(self):
        router = NonFrontalRouter()
        frontal = project_kps(0, 0)
        profile = project_kps(90, 0) + np.array([600.0, 0.0], dtype=np.float32)
        for t in range(20):
            self.assertFalse(router.verdict(frontal, 0.0, t))
            self.assertTrue(router.verdict(profile, 0.0, t))
        self.assertEqual(len(router._tracks), 2)

    def test_the_track_list_stays_bounded(self):
        """A long clip with faces moving through it must not grow state without
        limit. Recency is counted in arrivals, not frames, because frames
        arrive out of order."""
        router = NonFrontalRouter(max_tracks=8)
        for t in range(300):
            offset = np.array([(t * 137) % 4000, (t * 61) % 2000], dtype=np.float32)
            router.verdict(project_kps(45, 0) + offset, 0.0, t)
        self.assertLessEqual(len(router._tracks), 8)

    def test_the_event_log_stays_bounded(self):
        """A long clip must not accumulate one entry per decisive frame."""
        router = NonFrontalRouter(history=32)
        for t in range(600):
            router.verdict(project_kps(90, 0), 0.0, t)
        self.assertLessEqual(len(router._tracks[0]['idxs']), 32)

    def test_evicted_history_survives_as_the_baseline(self):
        """Dropping old events must not lose what they decided — a face that
        went profile at frame 0 and stayed in-band since is still profile."""
        router = NonFrontalRouter(history=4)
        router.verdict(project_kps(90, 0), 0.0, 0)          # decisive: True
        for t in range(1, 40):                               # push it out
            router.verdict(project_kps(90, 0), 0.0, t)
        self.assertTrue(router._tracks[0]['baseline'])

    def test_reset_clears_everything(self):
        router = NonFrontalRouter()
        router.verdict(project_kps(90, 0), 0.0, 0)
        router.reset()
        self.assertEqual(router._tracks, [])


class TestWiring(unittest.TestCase):
    def test_process_mask_uses_the_router(self):
        import inspect
        from roop.procmgr_masking import MaskingMixin
        src = inspect.getsource(MaskingMixin.process_mask)
        self.assertIn("nonfrontal_score", src)
        self.assertIn("_nonfrontal_router", src)
        self.assertIn("frame_idx", src,
                      "the latch needs a frame index or it silently no-ops")

    def test_processmgr_builds_and_resets_it(self):
        import inspect
        from roop.ProcessMgr import ProcessMgr
        self.assertIn("NonFrontalRouter", inspect.getsource(ProcessMgr.__init__))
        self.assertIn("_nonfrontal_router.reset",
                      inspect.getsource(ProcessMgr.run_batch_inmem),
                      "state would leak between clips")

    def test_process_face_publishes_early(self):
        """The whole point of observe() is the GAP between publishing and
        querying. If the call migrates into process_mask it buys nothing."""
        import inspect
        from roop.ProcessMgr import ProcessMgr
        from roop.procmgr_masking import MaskingMixin
        self.assertIn("_nonfrontal_router.observe",
                      inspect.getsource(ProcessMgr.process_face),
                      "nothing publishes the score early any more")
        self.assertNotIn("_nonfrontal_router.observe",
                         inspect.getsource(MaskingMixin.process_mask))

    def test_the_old_inline_heuristics_are_gone(self):
        """They lived in process_mask as four hand-rolled booleans. Leaving a
        copy behind is exactly how the inline rule and the scored one would
        drift apart, so pin the constants to nonfrontal.py."""
        import inspect
        from roop.procmgr_masking import MaskingMixin
        # Code only — the comment there legitimately explains what the score
        # covers, and matching prose would make this test unwritable.
        code = "\n".join(line.split("#", 1)[0]
                         for line in inspect.getsource(MaskingMixin.process_mask).splitlines())
        for orphan in ("0.55", "0.32", "0.70", "asymmetry", "0.25"):
            self.assertNotIn(orphan, code,
                             f"{orphan!r} is still hard-coded in process_mask")


if __name__ == "__main__":
    unittest.main()
