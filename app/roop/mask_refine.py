"""Post-inference mask edge refinement: sharp boundaries, no extra network.

The problem this solves
-----------------------
``Mask_XSeg.Run`` infers at 256x256 and returns a float probability map.
``MaskingMixin._composite_mask`` then does::

    img_mask = cv2.resize(img_mask, (target.shape[1], target.shape[0]))

with no interpolation flag, i.e. INTER_LINEAR, stretching a 256 probability map
up to the 512 crop. Two things follow, and together they are the whole "loose,
blurry boundary" report:

* every boundary is at least 2 crop-pixels wide before anything else happens,
  and bilinear makes it a linear ramp across those pixels;
* the ramp's POSITION is wherever the 256 grid put it, which on a 512 crop is
  +/-1 output pixel of quantisation, and its shape carries no information about
  where the actual image edge is.

The occluder family gets a partial remedy already — ``process_mask``
thresholds ``mask_occluder`` / ``mask_xseg3`` at 0.35 and re-blurs with a
deliberate kernel — but XSeg, the default engine, gets neither, so its raw
soft ramp is used as alpha directly.

The fix, in the order it matters
--------------------------------
1. Upsample with INTER_CUBIC rather than bilinear. Free.
2. Re-align the boundary to the image's own edges with a guided filter, using
   the crop's luma as the guide. This is the step that buys precision the
   network never had at 256: ears, jaw edge and hair strands are all places
   where the IMAGE has a strong gradient and the 256 mask has a smeared one,
   and a guided filter transfers the former's structure onto the latter. It is
   a local linear fit, not a segmentation, so it cannot invent a boundary where
   the image has none — it can only snap an approximately-right one into place.
3. Optionally steepen the remaining transition. A hard threshold would
   staircase along the boundary at 256-grid resolution; a monotone contrast
   curve narrows the ramp while keeping it sub-pixel smooth.

Cost. The guided filter is O(1) in its radius (box filters are integral-image
sums, not convolutions) and the FAST variant solves for the linear
coefficients at 1/s scale, so the per-face cost is a handful of box filters
over a 128x128 buffer plus two upsamples. Measured in ``bench_refine``. No GPU
round trip, no second session, nothing to warm up.

``cv2.ximgproc.guidedFilter`` is NOT used: this project ships plain
``opencv-python`` (4.9.0.80 in requirements.txt), not ``opencv-contrib-python``,
so ximgproc does not exist here. The implementation below is the same
algorithm (He, Sun & Tang 2013; fast variant 2015) on stock box filters.
"""

from __future__ import annotations

import os
from typing import Tuple

import cv2
import numpy as np

Mask = np.ndarray       # float32, (H, W), 0..1
Frame = np.ndarray      # uint8, (H, W, 3), BGR


# -- Tunables ---------------------------------------------------------------

# Guided-filter window, as a fraction of the crop's width.
#
# This is the one parameter that must NOT be set by intuition, and the reason is
# worth stating because the intuitive value is wrong. A narrow window looks like
# the careful choice — "only average locally" — but the window is what lets the
# local fit SEE the image edge it is supposed to snap to. If the mask boundary
# is 11px away from the real edge and the radius is 10px, no window contains
# both, and the filter sharpens the boundary exactly where it already was.
#
# Measured on a synthetic jaw edge with the mask boundary off by 11px, using
# this project's real 256 -> 512 path (truth = 256):
#
#   radius   6  -> edge 267   (no movement at all)
#   radius  10  -> edge 267
#   radius  16  -> edge 262
#   radius  24  -> edge 257
#   radius  26  -> edge 256   <- default
#   radius  32+ -> edge 256   (no further gain, wider ramp to re-steepen)
#
# So the radius has to EXCEED the error you are correcting. It is free to do so:
# a box filter is an integral-image sum, O(1) in its radius. 5% of a 512 crop
# is 26px, which covers the worst 256-grid misregistration with margin.
REFINE_RADIUS_FRAC = float(os.environ.get('ROOP_REFINE_RADIUS', '0.05') or 0.05)

# Regularisation. Small eps follows the guide's edges hard (and its noise with
# them); large eps degenerates toward a plain box blur. 1e-4 on 0..1 luma keeps
# skin texture out of the mask while still snapping to a jaw or hair edge.
REFINE_EPS = float(os.environ.get('ROOP_REFINE_EPS', '1e-4') or 1e-4)

# Coefficient-solve subsampling for the fast variant. `a` and `b` are box-
# filtered means, i.e. smooth fields by construction even where the mask is
# not, so they survive aggressive subsampling: 8 measured indistinguishable
# from 1 on the relocation test above while cutting the box-filter work 64x.
REFINE_SUBSAMPLE = int(os.environ.get('ROOP_REFINE_SUBSAMPLE', '8') or 8)

# Transition steepening. The guided filter relocates the boundary but WIDENS it
# (radius 26 takes the ramp from 32px to ~55px), so this is not optional — it
# is the second half of the same operation. Measured end to end, ramp width at
# the correct position: 0.0 -> 55px, 0.6 -> 10px, 1.0 -> 2px.
#
# Do not raise it past ~1.0. At 1.5 with a wide radius the curve collapses the
# ramp to nothing and the boundary snaps to the window edge instead of the image
# edge (measured: edge jumps from 256 to 0). 0.6 keeps a 10px anti-aliased
# transition, which is what stops the seam looking cut out with scissors.
REFINE_STEEPEN = float(os.environ.get('ROOP_REFINE_STEEPEN', '0.6') or 0.6)


# -- Guided filter ----------------------------------------------------------

def _box(src: np.ndarray, radius: int) -> np.ndarray:
    """Normalised box filter. O(1) in `radius` — an integral-image sum."""
    k = 2 * int(radius) + 1
    return cv2.boxFilter(src, -1, (k, k), normalize=True,
                         borderType=cv2.BORDER_REFLECT_101)


def fast_guided_filter(guide: np.ndarray,
                       src: np.ndarray,
                       radius: int,
                       eps: float = REFINE_EPS,
                       subsample: int = REFINE_SUBSAMPLE) -> np.ndarray:
    """Edge-preserving filter of `src` under the structure of `guide`.

    Both must be single-channel float32 in 0..1 and the same shape. Returns the
    filtered `src`, same shape and dtype.

    The output is a per-pixel linear function of the guide, ``q = a * I + b``,
    with ``a`` and ``b`` fitted over local windows. That is what makes it safe
    to run on an alpha mask: it is a *linear* re-fit, so it cannot introduce a
    boundary the guide does not have, and it cannot overshoot outside the range
    the local window spans — unlike an unsharp mask or a Lanczos resample,
    either of which will ring and put a bright or dark halo along the seam.

    The fast variant solves for ``a`` and ``b`` at ``1/subsample`` scale and
    upsamples them. Both are smooth fields (they are box-filtered means), so
    subsampling them costs almost nothing visually while cutting the box-filter
    work by ``subsample**2``.
    """
    if guide.shape != src.shape:
        guide = cv2.resize(guide, (src.shape[1], src.shape[0]),
                           interpolation=cv2.INTER_LINEAR)

    s = max(1, int(subsample))
    r = max(1, int(radius))
    if s > 1:
        h, w = guide.shape[:2]
        sw, sh = max(8, w // s), max(8, h // s)
        g_sub = cv2.resize(guide, (sw, sh), interpolation=cv2.INTER_AREA)
        p_sub = cv2.resize(src, (sw, sh), interpolation=cv2.INTER_AREA)
        r_sub = max(1, r // s)
    else:
        g_sub, p_sub, r_sub = guide, src, r

    mean_g = _box(g_sub, r_sub)
    mean_p = _box(p_sub, r_sub)
    # var(I) and cov(I, p) via the identity E[xy] - E[x]E[y]. Clamped at 0
    # because that identity is not guaranteed non-negative in float32 when the
    # window is nearly constant, and a negative variance makes `a` explode.
    var_g = np.maximum(_box(g_sub * g_sub, r_sub) - mean_g * mean_g, 0.0)
    cov_gp = _box(g_sub * p_sub, r_sub) - mean_g * mean_p

    a = cov_gp / (var_g + float(eps))
    b = mean_p - a * mean_g

    mean_a = _box(a, r_sub)
    mean_b = _box(b, r_sub)
    if s > 1:
        size = (guide.shape[1], guide.shape[0])
        mean_a = cv2.resize(mean_a, size, interpolation=cv2.INTER_LINEAR)
        mean_b = cv2.resize(mean_b, size, interpolation=cv2.INTER_LINEAR)

    return mean_a * guide + mean_b


# -- Transition shaping -----------------------------------------------------

_LUT_CACHE: dict = {}


def _steepen_lut(strength: float) -> np.ndarray:
    """256-entry smoothstep table, cached per strength."""
    key = round(float(strength), 4)
    lut = _LUT_CACHE.get(key)
    if lut is None:
        x = np.arange(256, dtype=np.float32) / 255.0
        gain = 1.0 + 3.0 * float(strength)
        y = np.clip((x - 0.5) * gain + 0.5, 0.0, 1.0)
        y = y * y * (3.0 - 2.0 * y)          # smoothstep
        lut = np.clip(y * 255.0 + 0.5, 0, 255).astype(np.uint8)
        _LUT_CACHE[key] = lut
    return lut


def steepen(mask: Mask, strength: float = REFINE_STEEPEN) -> Mask:
    """Narrow the alpha ramp without binarising it.

    A hard threshold would put the boundary on the 256-grid staircase the mask
    was inferred on, which reads as a jagged seam once it lands on a 512 or
    1024 crop. This is a monotone smoothstep about 0.5 instead: the ramp gets
    narrower, its position does not move (0.5 maps to 0.5), the ordering of
    alpha values is preserved, and the transition stays continuous — so the
    seam keeps its sub-pixel anti-aliasing.

    Applied through a 256-entry LUT rather than evaluated per pixel. The
    arithmetic is a 6-op polynomial over a 512x512 float buffer, which measured
    1.66 ms — comparable to the entire rest of this module — against 0.15 ms for
    a ``convertScaleAbs`` + ``cv2.LUT`` pair. The 8-bit quantisation it implies
    costs nothing real: ``paste_upscale`` already carries this matte as uint8,
    and 8-bit alpha is what the composite consumes either way.
    """
    s = float(strength)
    if s <= 1e-6:
        return mask
    m = np.asarray(mask, dtype=np.float32)
    # convertScaleAbs saturates into uint8 in one pass, which also does the
    # 0..1 clamp the caller would otherwise pay for separately.
    m8 = cv2.convertScaleAbs(m, alpha=255.0)
    out = cv2.LUT(m8, _steepen_lut(min(1.0, s)))
    return out.astype(np.float32) * (1.0 / 255.0)


# -- The entry point --------------------------------------------------------

def refine_mask(mask: Mask,
                crop: Frame,
                radius_frac: float = REFINE_RADIUS_FRAC,
                eps: float = REFINE_EPS,
                subsample: int = REFINE_SUBSAMPLE,
                steepen_strength: float = REFINE_STEEPEN) -> Mask:
    """Upsample `mask` to `crop`'s size and snap its boundary to `crop`'s edges.

    `mask` is any single-channel 0..1 map at any resolution (typically the
    engine's native 256); `crop` is the aligned face crop the mask belongs to,
    used only as the structural guide. Polarity does not matter — the filter is
    linear — so this works unchanged on this project's restore-polarity masks.

    Returns float32 0..1 at ``crop.shape[:2]``.
    """
    m = np.asarray(mask, dtype=np.float32)
    if m.ndim == 3 and m.shape[-1] == 1:
        m = m[..., 0]
    if m.ndim != 2:
        return np.asarray(mask, dtype=np.float32)

    h, w = crop.shape[:2]
    if m.shape != (h, w):
        # INTER_CUBIC, not INTER_LINEAR: on a 2x upsample cubic reconstructs the
        # ramp from four taps instead of two, which is a visibly tighter
        # boundary before the guided filter has done anything. Cubic overshoots
        # slightly, hence the clip — that overshoot is exactly the ringing that
        # makes LANCZOS4 the wrong choice for an alpha channel.
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_CUBIC)
        np.clip(m, 0.0, 1.0, out=m)

    guide = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    guide *= (1.0 / 255.0)
    radius = max(2, int(round(float(radius_frac) * w)))
    out = fast_guided_filter(guide, m, radius, eps=eps, subsample=subsample)
    if steepen_strength > 1e-6:
        # steepen's convertScaleAbs does the 0..1 clamp, so skip a full-buffer
        # np.clip here — it measured 0.2 ms at 512 for a result immediately
        # saturated again on the next line.
        return steepen(out, steepen_strength)
    return np.clip(out, 0.0, 1.0, out=out)


# -- Morphological alternative ---------------------------------------------

def snap_mask_to_edges(mask: Mask,
                       crop: Frame,
                       band_px: int = 6,
                       edge_percentile: float = 75.0) -> Mask:
    """Cheaper, blunter alternative to :func:`refine_mask`.

    Restricts itself to a thin band either side of the mask's 0.5 contour (the
    morphological gradient of the thresholded mask) and, inside that band only,
    pushes alpha toward 0 or 1 according to which side of the image's strongest
    local gradient each pixel falls on. Everything outside the band is returned
    untouched, so the worst case is that the boundary does not move.

    Use it when the guide is unreliable — heavy grain, heavy compression
    blocking, or a near-monochrome crop — where a guided filter's local linear
    fit locks onto noise instead of anatomy. Otherwise prefer
    :func:`refine_mask`, which is both better and about the same cost.
    """
    m = np.asarray(mask, dtype=np.float32)
    if m.ndim == 3 and m.shape[-1] == 1:
        m = m[..., 0]
    h, w = crop.shape[:2]
    if m.shape != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_CUBIC)
        np.clip(m, 0.0, 1.0, out=m)

    hard = (m > 0.5).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (2 * int(band_px) + 1,) * 2)
    band = cv2.morphologyEx(hard, cv2.MORPH_GRADIENT, k).astype(bool)
    if not band.any():
        return m

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    grad = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
                         cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    thr = float(np.percentile(grad[band], edge_percentile))
    strong = band & (grad >= thr)
    if not strong.any():
        return m

    out = m.copy()
    out[strong] = np.where(m[strong] > 0.5, 1.0, 0.0)
    # Two-pixel blur over the band only, so the snapped pixels do not become a
    # hard staircase against their unsnapped neighbours.
    soft = cv2.GaussianBlur(out, (5, 5), 0)
    out[band] = soft[band]
    return np.clip(out, 0.0, 1.0)


# -- Bench ------------------------------------------------------------------

def bench_refine(size: int = 512, mask_size: int = 256,
                 iterations: int = 200) -> Tuple[float, float]:
    """``(refine_ms, snap_ms)`` per call, for the latency claim above."""
    import time

    rng = np.random.default_rng(0)
    crop = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    mask = np.zeros((mask_size, mask_size), np.float32)
    cv2.circle(mask, (mask_size // 2, mask_size // 2), mask_size // 3, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (21, 21), 0)

    out = []
    for fn in (refine_mask, snap_mask_to_edges):
        fn(mask, crop)                      # warm caches
        t0 = time.perf_counter()
        for _ in range(iterations):
            fn(mask, crop)
        out.append((time.perf_counter() - t0) / iterations * 1000.0)
    return out[0], out[1]
