"""One Euro filter for temporal smoothing of face keypoints across video frames.

The One Euro filter (Casiez et al., 2012) is an adaptive low-pass filter: it
smooths heavily when the signal is slow/still (kills detector jitter) and lightly
when the signal moves fast (avoids lag), so it beats a fixed-alpha EMA's single
jitter-vs-lag tradeoff.

Time `t` here is a monotonically increasing per-frame index (dt = 1 frame), so
`min_cutoff` / `beta` are expressed in per-frame units.
"""
import math
import numpy as np


def _alpha(t_e, cutoff):
    r = 2.0 * math.pi * cutoff * t_e
    return r / (r + 1.0)


class OneEuroFilter:
    """Adaptive low-pass filter over an arbitrary-shaped numpy signal (elementwise)."""

    def __init__(self, min_cutoff=0.05, beta=0.02, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def __call__(self, x, t):
        x = np.asarray(x, dtype=np.float64)
        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = np.zeros_like(x)
            self.t_prev = t
            return x
        t_e = t - self.t_prev
        if t_e <= 0:
            t_e = 1.0
        a_d = _alpha(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = _alpha(t_e, cutoff)          # per-element adaptive smoothing factor
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat


class KpsStabilizer:
    """Smooths face 5-point keypoints across frames, with nearest-centroid
    tracking so each face in a multi-face scene keeps its own filter."""

    def __init__(self, min_cutoff=0.05, beta=0.02, max_missing=8, match_scale=0.6):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.max_missing = int(max_missing)   # drop a track unseen for this many frames
        self.match_scale = float(match_scale)  # match radius as a fraction of face size
        self.tracks = []                       # [{filter, centroid, last_t}]

    def reset(self):
        self.tracks = []

    def apply(self, kps, t):
        """Return temporally-smoothed (5,2) keypoints for the face at frame `t`."""
        kps = np.asarray(kps, dtype=np.float64)
        if kps.shape != (5, 2):
            return kps.astype(np.float32)
        centroid = kps.mean(axis=0)
        size = max(float(np.ptp(kps[:, 0])), float(np.ptp(kps[:, 1])), 1.0)

        best, best_d = None, float('inf')
        for tr in self.tracks:
            d = float(np.linalg.norm(tr['centroid'] - centroid))
            if d < best_d:
                best_d, best = d, tr

        if best is not None and best_d <= self.match_scale * size and (t - best['last_t']) <= self.max_missing:
            tr = best
        else:
            tr = {'filter': OneEuroFilter(self.min_cutoff, self.beta), 'centroid': centroid, 'last_t': t}
            self.tracks.append(tr)

        smoothed = tr['filter'](kps, t)
        tr['centroid'] = smoothed.mean(axis=0)
        tr['last_t'] = t
        # prune stale tracks
        self.tracks = [x for x in self.tracks if (t - x['last_t']) <= self.max_missing]
        return smoothed.astype(np.float32)


class EnhancerStabilizer:
    """Reduces per-frame enhancer texture flicker (GFPGAN/GPEN/Codeformer shimmer)
    by temporally blending the *aligned* enhanced crop with a motion-adaptive
    blend factor (One Euro logic, but the speed is a single scalar — head motion —
    applied uniformly to the whole crop, NOT per-pixel: a per-pixel derivative
    would treat the flicker itself as "motion" and refuse to smooth it).

    Blend hard when the face is still → kills flicker; pass the current frame
    through on fast motion → avoids ghosting. Per-face tracking by kps centroid.
    """

    def __init__(self, strength=0.5, max_missing=8, match_scale=0.6, motion_beta=8.0):
        self.strength = float(min(max(strength, 0.0), 1.0))
        # strength 0 → light (base_cutoff 0.42), strength 1 → heavy (0.02)
        self.base_cutoff = 0.4 * (1.0 - self.strength) + 0.02
        self.motion_beta = float(motion_beta)
        self.max_missing = int(max_missing)
        self.match_scale = float(match_scale)
        self.tracks = []   # [{prev(float32 crop), centroid, last_t}]

    def reset(self):
        self.tracks = []

    def apply(self, crop, kps, t):
        if crop is None:
            return crop
        kps = np.asarray(kps, dtype=np.float32)
        if kps.shape != (5, 2):
            return crop
        centroid = kps.mean(axis=0)
        size = max(float(np.ptp(kps[:, 0])), float(np.ptp(kps[:, 1])), 1.0)

        best, best_d = None, float('inf')
        for tr in self.tracks:
            d = float(np.linalg.norm(tr['centroid'] - centroid))
            if d < best_d:
                best_d, best = d, tr

        matched = (best is not None and best_d <= self.match_scale * size
                   and (t - best['last_t']) <= self.max_missing
                   and best['prev'].shape == crop.shape)
        if matched:
            tr = best
            t_e = max(t - tr['last_t'], 1)
            motion = (best_d / t_e) / size            # head motion as a fraction of face size
            cutoff = self.base_cutoff + self.motion_beta * motion
            a = _alpha(t_e, cutoff)                   # → 1 (current) when fast, small when still
            out = a * crop.astype(np.float32) + (1.0 - a) * tr['prev']
            tr['prev'] = out
            tr['centroid'] = centroid
            tr['last_t'] = t
            self.tracks = [x for x in self.tracks if (t - x['last_t']) <= self.max_missing]
            return np.clip(out, 0, 255).astype(np.uint8)
        # new / unmatched track — pass through and seed.
        self.tracks.append({'prev': crop.astype(np.float32), 'centroid': centroid, 'last_t': t})
        return crop
