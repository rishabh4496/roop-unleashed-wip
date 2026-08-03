"""Is this face angled enough to need the unwarped-crop masking path?

Two pieces:

`nonfrontal_score` collapses the four heuristics the router used to OR together
into ONE continuous number, normalised so that 1.0 means "exactly at the
threshold". `score > 1.0` is the same verdict the OR gave — verified exact over
34k poses, see tests — but unlike a bundle of booleans it can be reasoned about
temporally, which is what the latch below needs.

`NonFrontalRouter` is that latch. A binary routing decision driven by a noisy
per-frame score FLICKERS: the two mask paths derive the mask differently, so a
face parked near the threshold alternates between them frame to frame and the
mask boundary visibly changes on a head that is not moving. Measured on a still
head under 1 px of keypoint noise, the bare threshold flips the verdict up to
123 times in 400 frames (yaw 0 / pitch +30, i.e. an ordinary head tilted up).
Hysteresis takes that to 0.
"""

import threading

import numpy as np

from roop.face_util import kps_pose_ratios, offaxis_deg, solve_pose_5pt


# Half-width of the hysteresis band, in score units (1.0 = the threshold).
# Enter non-frontal above 1 + MARGIN, leave below 1 - MARGIN, hold in between.
#
# Sized from the score's actual noise rather than picked: on a still head with
# 1 px of keypoint jitter the 1st-99th percentile spread of the score near the
# threshold is ~0.16-0.19, so a band narrower than that does not latch. Measured
# flips per 400 frames on the worst still poses:
#
#   margin      0.00   0.05   0.10   0.15   0.20
#   yaw 0/+30    117      7      1      0      0
#   yaw 5/+30    110      6      0      0      0
#   yaw 10/+30    90      4      0      0      0
#
# 0.15 is the first value that zeroes every hot spot. Going wider only delays
# genuine transitions. A real crossing still registers: swept through two full
# nod cycles, all 8 true boundary crossings came through at this margin.
NONFRONTAL_MARGIN = 0.15

# Threshold constants, kept next to the score that normalises by them so the two
# cannot drift apart.
_ASYM_MAX = 0.25          # nose-vs-eyes asymmetry above this = turned
_YAW_RATIO_MIN = 0.55     # eye separation / eye-to-mouth below this = turned
_PITCH_LO, _PITCH_HI = 0.32, 0.70   # neutral band for the pitch proxy
_OFFAXIS_MAX = 50.0       # solved off-axis angle, deg
_INVERT_TOL = 5.0         # px of eyes-below-mouth before calling it upside down
_TGT_PITCH_MAX = 30.0     # deg, from landmark_3d_68 when that model is loaded

_PITCH_MID = (_PITCH_LO + _PITCH_HI) / 2.0
_PITCH_HALF = (_PITCH_HI - _PITCH_LO) / 2.0


def nonfrontal_score(kps, tgt_pitch_deg=0.0):
    """How non-frontal this face is, as a multiple of the routing threshold.

    Each sub-test is `metric vs threshold`, so dividing by its own threshold puts
    them all in the same units and `max` reproduces the OR exactly: OR(m_i > 1)
    is (max m_i) > 1. Returns 0.0 for keypoints it cannot read, which routes
    them down the frontal path exactly as the old code's `is_non_frontal = False`
    default did.
    """
    margins = [0.0]
    try:
        if kps is not None and len(kps) == 5:
            left_eye_x, right_eye_x, nose_x = kps[0][0], kps[1][0], kps[2][0]
            d_left = abs(nose_x - left_eye_x)
            d_right = abs(nose_x - right_eye_x)
            if d_left + d_right > 1e-5:
                margins.append(
                    (abs(d_left - d_right) / (d_left + d_right)) / _ASYM_MAX)

            yaw_ratio, pitch_ratio = kps_pose_ratios(kps)
            if yaw_ratio is not None:
                # Inverted test (fires BELOW the threshold), so the margin is the
                # reciprocal. Clamped because yaw_ratio goes to 0 at a true
                # profile and would otherwise divide by ~nothing.
                margins.append(_YAW_RATIO_MIN / max(float(yaw_ratio), 1e-3))
            if pitch_ratio is not None:
                # Two-sided band -> distance from its centre, in half-widths.
                margins.append(abs(float(pitch_ratio) - _PITCH_MID) / _PITCH_HALF)

            pose = solve_pose_5pt(kps)
            if pose is not None:
                margins.append(offaxis_deg(pose[0], pose[1]) / _OFFAXIS_MAX)

            eye_y = (kps[0][1] + kps[1][1]) / 2.0
            mouth_y = (kps[3][1] + kps[4][1]) / 2.0
            if eye_y > mouth_y:
                margins.append(float(eye_y - mouth_y) / _INVERT_TOL)

        margins.append(abs(float(tgt_pitch_deg)) / _TGT_PITCH_MAX)
        value = max(margins)
        return float(value) if np.isfinite(value) else 0.0
    except Exception:
        return 0.0


class NonFrontalRouter:
    """Per-face latch over `nonfrontal_score`, so the routing decision changes
    when the head does and not when the detector twitches.

    Faces are matched between frames by nearest keypoint centroid, the same way
    the keypoint stabilisers in one_euro.py do it, so a multi-face scene keeps
    one latch per face.

    THREADING. This is the part that decides the whole design, so it is worth
    being explicit about. The video path hands frames to workers round-robin
    (`frame_index % num_threads`), so worker i sees frames i, i+N, i+2N... —
    each worker walks the WHOLE timeline at stride N, in order, and adjacent
    frames belong to different workers.

    A single shared latch is therefore wrong, and measurably so. Eight workers
    walking the same trajectory drive one latch through every boundary crossing
    eight times over, interleaved: on a head nodding across the threshold the
    shared latch changed state 64 times where the correct answer is 8, and the
    output flickered WORSE than having no latch at all.

    So the latch is per-thread — each worker's stride-N subsequence is properly
    ordered, so its latch evolves correctly — while the track IDENTITY and a
    seed verdict are shared under a lock. The shared seed is what stops the
    other failure mode: with each worker seeding its own latch from its own
    first frame, a head parked mid-band would have workers disagreeing forever,
    which is a period-N flicker. Seeding every worker from one shared value
    means an in-band score is a no-op for all of them and they cannot diverge.

    The seed tracks the last STRONGLY determined verdict, so a worker that
    starts late on a clip seeds from something current rather than from
    whatever the face was doing in frame 0.

    Note the frame index is deliberately NOT used to match or expire tracks. An
    earlier attempt did that, and a late-arriving frame then looked like a face
    that had been missing for 190 frames, so it forked a fresh unlatched track
    and the chatter came straight back. Recency is counted in arrivals instead
    and the track list is a bounded LRU; the frame index survives only as the
    "is there a temporal dimension at all" flag.
    """

    # Faces per frame is small, so this holds several frames of history while
    # staying O(1). Tracks are evicted least-recently-used.
    MAX_TRACKS = 32

    def __init__(self, margin=NONFRONTAL_MARGIN, max_tracks=MAX_TRACKS,
                 match_scale=0.6):
        self.margin = float(margin)
        self.max_tracks = int(max_tracks)
        self.match_scale = float(match_scale)
        self._tracks = []        # [{id, centroid, strong, seq}] — shared
        self._seq = 0
        self._next_id = 0
        self._generation = 0
        self._lock = threading.Lock()
        self._tls = threading.local()

    def reset(self):
        # Bumping the generation invalidates every worker's thread-local latch
        # without needing to reach into other threads to clear them.
        with self._lock:
            self._tracks = []
            self._seq = 0
            self._next_id = 0
            self._generation += 1

    def _local_latches(self, generation):
        store = getattr(self._tls, 'latches', None)
        if store is None or store[0] != generation:
            store = (generation, {})
            self._tls.latches = store
        latches = store[1]
        # Track ids only ever increase, so over a long clip with faces coming and
        # going this dict would accumulate an entry per retired track. Keep the
        # newest ids — the shared side already caps itself at max_tracks, so
        # anything older than that cannot be matched again anyway.
        cap = self.max_tracks * 4
        if len(latches) > cap:
            for dead in sorted(latches)[:len(latches) - self.max_tracks]:
                del latches[dead]
        return latches

    def _shared_track(self, centroid, size, score, bare):
        """Match or create this face's shared track. Returns (id, seed, gen)."""
        with self._lock:
            best, best_d = None, float('inf')
            for tr in self._tracks:
                d = float(np.linalg.norm(tr['centroid'] - centroid))
                if d < best_d:
                    best_d, best = d, tr

            if best is not None and best_d <= self.match_scale * size:
                tr = best
            else:
                tr = {'id': self._next_id, 'centroid': centroid,
                      'strong': bare, 'seq': 0}
                self._next_id += 1
                self._tracks.append(tr)

            # Only a decisive score updates the shared seed, so in-band noise
            # cannot move it and late-starting workers still seed from
            # something recent.
            if score > 1.0 + self.margin:
                tr['strong'] = True
            elif score < 1.0 - self.margin:
                tr['strong'] = False

            self._seq += 1
            tr['centroid'] = centroid
            tr['seq'] = self._seq
            if len(self._tracks) > self.max_tracks:
                self._tracks.sort(key=lambda x: x['seq'])
                del self._tracks[:len(self._tracks) - self.max_tracks]
            return tr['id'], tr['strong'], self._generation

    def verdict(self, kps, tgt_pitch_deg=0.0, t=None):
        """Latched non-frontal verdict for this face.

        `t=None` means there is no temporal dimension to exploit — a still
        image, or a caller that does not know its frame index — so it falls
        straight through to the bare threshold and behaves exactly as before.
        """
        score = nonfrontal_score(kps, tgt_pitch_deg)
        bare = score > 1.0
        if t is None or self.margin <= 0.0:
            return bare
        try:
            pts = np.asarray(kps, dtype=np.float64)
            centroid = pts.mean(axis=0)
            size = max(float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1])), 1.0)
            if not np.isfinite(centroid).all():
                return bare
        except Exception:
            return bare

        track_id, seed, generation = self._shared_track(centroid, size, score, bare)

        # The latch itself, thread-local so it advances along THIS worker's
        # ordered subsequence. Applying the same score twice is a no-op, which
        # is what makes it safe for process_mask to be called more than once for
        # the same face in a frame (once per mask processor) — a "flip after N
        # consecutive frames" debounce would have counted those repeats as
        # evidence.
        latches = self._local_latches(generation)
        latched = latches.get(track_id, seed)
        if score > 1.0 + self.margin:
            latched = True
        elif score < 1.0 - self.margin:
            latched = False
        latches[track_id] = latched
        return bool(latched)
