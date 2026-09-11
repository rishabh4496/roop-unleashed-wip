"""Scale-aware resampling for the composite step.

Why this is not just "use a better interpolator"
------------------------------------------------
The received advice for a soft composite is to replace bilinear with
``INTER_CUBIC`` or ``INTER_LANCZOS4``. That is right for an UPSCALE and wrong
for a downscale, and the paste in ``paste_upscale`` is usually a downscale.

``IM`` maps the 512 enhanced crop onto the face's footprint in the frame.
Measured for realistic on-screen face sizes in 720p footage:

    on-screen face ~120px  ->  warp scale 0.34   downscale
    on-screen face ~200px  ->  warp scale 0.56   downscale
    on-screen face ~320px  ->  warp scale 0.90   downscale
    on-screen face ~480px  ->  warp scale 1.35   upscale
    on-screen face ~700px  ->  warp scale 1.97   upscale

For the common cases the 512 crop is being shrunk by up to 3x, and every
interpolator OpenCV offers to ``warpAffine`` is a point sampler: bilinear reads
a 2x2 neighbourhood, cubic 4x4, Lanczos 8x8. None of them is a low-pass
filter, so at scale 0.34 they all skip source pixels regardless of tap count —
that is aliasing, and no interpolator choice fixes it. It shows up as shimmer
that crawls frame to frame on a restored face (which is exactly the detail an
enhancer just added), and it is worst on the frames where the face is smallest.

Lanczos is actively the wrong answer here twice over: it does not fix the
aliasing, and its negative lobes overshoot, which puts a bright/dark ring along
the paste boundary — a visible halo where the swap meets the plate.

What actually works is to band-limit before sampling. ``INTER_AREA`` is a true
box low-pass and is the only OpenCV mode that averages every source pixel, but
``warpAffine`` does not accept it. So: pre-shrink the crop with ``INTER_AREA``
to approximately its final on-screen size, then warp the already-band-limited
image with a cheap interpolator. The warp then has little scaling left to do,
so bilinear is sufficient and no detail is skipped.
"""

from __future__ import annotations

import os
from typing import Tuple

import cv2
import numpy as np

Frame = np.ndarray

# Below this warp scale, pre-filter with INTER_AREA before warping. 0.9 rather
# than 1.0 so a warp that is barely a downscale does not pay for an extra
# resize it cannot benefit from.
PREFILTER_BELOW = float(os.environ.get('ROOP_PREFILTER_BELOW', '0.9') or 0.9)

# Leave the pre-shrink slightly larger than the final footprint, so the warp
# still has real samples to interpolate between rather than landing on a 1:1
# grid where any sub-pixel offset re-softens it.
PREFILTER_HEADROOM = float(os.environ.get('ROOP_PREFILTER_HEADROOM', '1.25') or 1.25)


def warp_scale(M: np.ndarray) -> float:
    """Uniform scale factor of a 2x3 affine. >1 enlarges, <1 shrinks.

    ``sqrt(|det|)`` is the linear scale for a similarity, which every matrix in
    this pipeline is (``estimate_norm`` forces one), and remains the correct
    area-equivalent scale for a general affine.
    """
    return float(np.sqrt(abs(np.linalg.det(np.asarray(M, dtype=np.float64)[:, :2]))))


def warp_face(src: Frame,
              IM: np.ndarray,
              size: Tuple[int, int],
              border_mode: int = cv2.BORDER_REPLICATE) -> Frame:
    """Warp a face crop into frame space without aliasing or ringing.

    Drop-in for::

        cv2.warpAffine(src, IM, size, borderMode=cv2.BORDER_REPLICATE)

    which is what ``paste_upscale`` does today — and which, with no ``flags``,
    is ``INTER_LINEAR`` at every scale.

    Three regimes:

    * **Heavy downscale** (< ``PREFILTER_BELOW``): band-limit with
      ``INTER_AREA`` to ``PREFILTER_HEADROOM`` x the target footprint, fold the
      pre-shrink into ``IM``, then warp bilinearly. This is the only path that
      removes aliasing rather than re-interpolating it.
    * **Near 1:1**: bilinear, unchanged. There is nothing to fix.
    * **Upscale**: ``INTER_CUBIC``. More taps genuinely help when the
      destination grid is denser than the source; cubic's mild overshoot is
      bounded and, unlike Lanczos, does not produce a visible edge halo.
    """
    s = warp_scale(IM)

    if s >= PREFILTER_BELOW:
        flags = cv2.INTER_CUBIC if s > 1.0 else cv2.INTER_LINEAR
        return cv2.warpAffine(src, IM, size, flags=flags, borderMode=border_mode)

    h, w = src.shape[:2]
    # Fraction of the source size to band-limit down to. Bounded above by 1
    # (never enlarge here) and below by a floor so an extreme warp cannot ask
    # for a degenerate buffer.
    target = min(1.0, max(0.02, s * PREFILTER_HEADROOM))
    sw, sh = max(8, int(round(w * target))), max(8, int(round(h * target)))
    if sw >= w or sh >= h:
        return cv2.warpAffine(src, IM, size, flags=cv2.INTER_LINEAR,
                              borderMode=border_mode)

    small = cv2.resize(src, (sw, sh), interpolation=cv2.INTER_AREA)

    # The pre-shrink is a scale about the origin, so composing it with IM is a
    # column-wise multiply of the linear part only — the translation column
    # already refers to destination coordinates and must not be scaled.
    fx, fy = w / float(sw), h / float(sh)
    IM2 = np.asarray(IM, dtype=np.float64).copy()
    IM2[:, 0] *= fx
    IM2[:, 1] *= fy
    return cv2.warpAffine(small, IM2, size, flags=cv2.INTER_LINEAR,
                          borderMode=border_mode)


def resize_mask(mask: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize an alpha mask, choosing by direction and clamping the result.

    Alpha is not an image and must not be resampled like one. Cubic and Lanczos
    both overshoot outside the source range; on a 0..1 matte that means values
    above 1 (a region that composites MORE than fully swapped, brightening the
    seam) and below 0 (which then clips to a hard line). Upsampling uses cubic
    for the tighter ramp but clamps immediately; downsampling uses INTER_AREA,
    which is both alias-free and guaranteed to stay within the source range
    because every output is a convex average of its inputs.
    """
    m = np.asarray(mask, dtype=np.float32)
    if m.ndim == 3 and m.shape[-1] == 1:
        m = m[..., 0]
    w, h = int(size[0]), int(size[1])
    if m.shape[:2] == (h, w):
        return m
    if w * h < m.shape[0] * m.shape[1]:
        return cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA)
    out = cv2.resize(m, (w, h), interpolation=cv2.INTER_CUBIC)
    return np.clip(out, 0.0, 1.0, out=out)
