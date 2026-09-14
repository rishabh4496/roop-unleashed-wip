"""Dual-stream frequency splitting and Reinhard colour transfer.

Shared by `Enhance_UltraMax`'s dual-stream engine. Kept out of the processor
so the operators can be graded independently, without a GPU or model load.

WHY A SPLIT AT ALL. Two restorers are good at different halves of the problem
(measured data from Enhance_GPENRealistic, high-frequency std at the paste):

    swap input 2.67 | GPEN-256 2.82 | CodeFormer-512 4.11 | GPEN-512 5.14

GPEN-512 synthesises ~25% more high-frequency detail than CodeFormer. But
CodeFormer's discrete codebook is the better STRUCTURE prior -- it draws a
clean iris rim, correct tooth boundaries and symmetric lids -- and GPEN is the
model with the colour cast.  Taking the low band from CodeFormer and the high
band from GPEN is therefore not a stylistic choice; it is each network's
measured strength.

THE FILTER HAS TO BE EDGE-AWARE. A Gaussian split leaks structure across every
strong boundary, so the "high" band at a jawline carries the jaw itself, and
re-adding it at gain > 1 is the halo this project already removed from
UltraMax once. A guided filter's band is bounded by its own guide edges, so the
high band carries pores and lashes and NOT the silhouette.

`cv2.ximgproc.guidedFilter` is contrib-only and is NOT present in the
opencv-python build this app ships, so the filter is implemented here on
`cv2.boxFilter`. That is He et al. exactly, not an approximation.
"""

import cv2
import numpy as np

# Reinhard std ratios are clamped to this band. `procmgr_color` bounds its own
# LAB transfer to exactly the same [0.80, 1.20] and for the same reason: an
# unbounded ratio multiplies a colour cast instead of removing one, and on a
# low-variance crop the ratio runs away entirely.
_STD_LO, _STD_HI = 0.80, 1.20

# Guard for the variance divisions below. Small enough not to bias `a` on real
# image variance, large enough that a perfectly flat patch cannot divide by 0.
_EPS = 1e-6


def guided_filter(image, radius=8, eps=0.04, guide=None):
    """Edge-preserving smoother (He et al. 2010), O(1) box-filter form.

    `image` float32 in [0, 1], HxW or HxWxC. `guide` defaults to `image`
    (self-guided), which is what the frequency split uses: each stream is
    smoothed by its own edges.

    `eps` is in the SQUARED units of the guide, so it only means "0.04" for a
    [0, 1] input, and getting that wrong fails SILENTLY IN BOTH DIRECTIONS.
    Feed it [0, 255] data and the variance term dominates everywhere, `a` -> 1,
    and the filter becomes the IDENTITY: `frequency_split` then computes
    `d - d == 0` and injects no detail at all while looking like it ran.
    Callers here always normalise to [0, 1] first.
    """
    I = np.ascontiguousarray(image, dtype=np.float32)
    G = I if guide is None else np.ascontiguousarray(guide, dtype=np.float32)
    if G.shape[:2] != I.shape[:2]:
        raise ValueError("guide and image must share spatial dimensions")

    # ksize must be odd; radius is the half-width, matching the reference.
    k = (int(radius) * 2 + 1, int(radius) * 2 + 1)

    def _box(x):
        return cv2.boxFilter(x, -1, k, normalize=True,
                             borderType=cv2.BORDER_REFLECT)

    mean_G = _box(G)
    corr_G = _box(G * G)
    var_G = np.maximum(corr_G - mean_G * mean_G, 0.0)

    if guide is None:
        mean_I = mean_G
        cov = var_G
    else:
        mean_I = _box(I)
        cov = _box(G * I) - mean_G * mean_I

    a = cov / (var_G + float(eps) + _EPS)
    b = mean_I - a * mean_G
    return _box(a) * G + _box(b)


def frequency_split(structure, detail, radius=8, eps=0.04, gain=1.25,
                    clamp=None):
    """Combine two uint8 BGR images via edge-aware frequency split.

    Returns a uint8 BGR image whose low-frequency structure comes from
    `structure` and whose high-frequency detail comes from `detail`.

    The high band is extracted from `detail` by subtracting its own
    guided-filtered smooth; the gain amplifies that band before re-adding it
    to `structure`. An optional clamp prevents halos at large step edges
    (the same failure mode that motivated the removal of UltraMax's earlier
    unsharp mask).
    """
    s = structure.astype(np.float32) / 255.0
    d = detail.astype(np.float32) / 255.0

    # Smooth each stream by its OWN edges so the band boundary never
    # introduces a boundary of its own.
    s_smooth = guided_filter(s, radius=radius, eps=eps)
    d_smooth = guided_filter(d, radius=radius, eps=eps)
    high = d - d_smooth          # zero-mean high band from detail stream

    if clamp is not None and clamp > 0:
        limit = float(clamp) / 255.0
        np.clip(high, -limit, limit, out=high)

    out = s_smooth + high * float(gain)
    np.clip(out, 0.0, 1.0, out=out)
    return (out * 255.0).astype(np.uint8)


def frequency_split_luma(structure, detail, radius=8, eps=0.04, gain=1.25,
                          clamp=None):
    """Luminance-channel frequency split, chrominance from `structure`.

    Splits only the L channel in YCrCb space so colour never carries across
    the boundary. Chrominance is taken from `structure` (which is also
    CodeFormer's output, with the colour-fix applied) rather than from GPEN's
    chroma, which has the cast this project already measures and removes.
    """
    s_ycrcb = cv2.cvtColor(structure, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    d_ycrcb = cv2.cvtColor(detail, cv2.COLOR_BGR2YCrCb).astype(np.float32)

    s_L = s_ycrcb[:, :, 0] / 255.0
    d_L = d_ycrcb[:, :, 0] / 255.0

    s_smooth = guided_filter(s_L, radius=radius, eps=eps)
    d_smooth = guided_filter(d_L, radius=radius, eps=eps)
    high = d_L - d_smooth

    if clamp is not None and clamp > 0:
        limit = float(clamp) / 255.0
        np.clip(high, -limit, limit, out=high)

    out_L = s_smooth + high * float(gain)
    np.clip(out_L, 0.0, 1.0, out=out_L)

    result = s_ycrcb.copy()
    result[:, :, 0] = out_L * 255.0
    return cv2.cvtColor(np.clip(result, 0, 255).astype(np.uint8),
                        cv2.COLOR_YCrCb2BGR)


def reinhard_lab(source, reference, l_weight=0.5):
    """Reinhard (2001) colour transfer from `reference` to `source`, in LAB.

    Matches mean and standard deviation per channel. The L channel is matched
    at `l_weight` (0 = keep source L, 1 = full match) to avoid dragging
    contrast back toward the degraded input. Std ratios are clamped to
    [_STD_LO, _STD_HI] to prevent a low-variance crop from amplifying its
    own cast.
    """
    src = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)

    out = src.copy()
    for ch in range(3):
        s_mean, s_std = cv2.meanStdDev(src[:, :, ch])
        r_mean, r_std = cv2.meanStdDev(ref[:, :, ch])
        s_mean = float(s_mean)
        s_std = max(float(s_std), _EPS)
        r_mean = float(r_mean)
        r_std = float(r_std)

        ratio = np.clip(r_std / s_std, _STD_LO, _STD_HI)
        shifted = (src[:, :, ch] - s_mean) * ratio + r_mean
        if ch == 0:
            # Partial L match: blend between original and transferred
            shifted = src[:, :, ch] * (1.0 - l_weight) + shifted * l_weight

        out[:, :, ch] = shifted

    np.clip(out, 0, 255, out=out)
    return cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_LAB2BGR)
