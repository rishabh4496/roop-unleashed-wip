"""Adaptive lower-face boundary: stop the swap where the swapper stopped.

The distortion this module addresses
------------------------------------
The swap crop is aligned by a 5-point SimilarityTransform (``estimate_norm`` -
two eyes, nose, two mouth corners). A similarity has four degrees of freedom:
rotation, uniform scale, and translation. It cannot shear or squash, which is
why the aligned crop is geometrically faithful - but it also means the fit is
determined ENTIRELY by the eye/nose/mouth constellation. The jaw and chin are
not in the fit at all.

So the lower third of the output is whatever the net drew, positioned by the
source person's own lower-face proportion, scaled to the target's interocular
distance. Two people with the same eye spacing and different chin lengths
produce a chin that lands in a different place in the frame - and nothing in
the alignment notices, because every landmark the alignment can see still
matches perfectly.

Then the paste matte is built from the TARGET's landmarks
(``procmgr_masking.landmark_hull`` -> ``create_landmark_mask``), and that is
where a geometric mismatch becomes a visible artefact. The matte says "the face
reaches down to the target's chin"; the pixels say "the chin is 15 px higher, or
lower, than that". Three consequences, all reported as lower-face distortion:

* **Overshoot.** Source chin longer than target's: source chin pixels are
  pasted below the target's real jaw silhouette, onto neck and background.
  Reads as elongation.
* **Shortfall.** Source chin shorter: the band between the drawn chin and the
  matte edge is the net's own extrapolation of the target plate, blended over
  the target's real chin. Reads as a doubled or smeared jawline.
* **Convexity.** ``landmark_hull`` closes the polygon with ``cv2.convexHull``,
  and the region under a jaw is CONCAVE. The hull bridges straight from jaw
  corner to jaw corner across the chin, adding matte over the neck on every
  face regardless of identity - then the isotropic feather in ``blur_area``
  spreads the swap a further ``mask_size * blend/200`` px past it (45 px on a
  300 px face at the shipped ``face_mask_blend`` of 30).

What this module does
---------------------
1. :func:`lower_face_geometry` measures the source/target lower-face mismatch
   in a shared canonical space, so pose, scale and position cancel and what is
   left is pure shape difference.
2. :func:`adaptive_lower_boundary_mask` builds the matte from a concave jaw
   chain instead of a convex hull, and clamps its lower edge to the shallower
   of the two chins when they disagree - never paste below where the swapper
   actually drew a face.
3. :func:`anisotropic_feather` narrows the feather across the jaw silhouette
   while leaving it wide everywhere else. A jaw edge is a depth discontinuity
   with neck behind it; a forehead edge is skin against hair. One feather width
   cannot be right for both.
4. :func:`multiband_blend` / :func:`poisson_blend` for the final composite.

Uses only the 106-point landmarks the detector already produces
(``face.landmark_2d_106``, contour at indices 0..32, chin at 0, brows at
33..52) - no MediaPipe dependency, and no second landmark model to keep in
sync with the one the alignment uses.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

Frame = np.ndarray
Mask = np.ndarray

# 106-point layout, as used by roop.ProcessMgr._JAW_CONTOUR_IDX and
# procmgr_masking.landmark_hull.
CONTOUR_IDX = np.arange(0, 33)
BROW_SLICE = slice(33, 53)
CHIN_IDX = 0
FOREHEAD_IDX = 72

# Chin depth difference, as a fraction of the eye-to-mouth distance, above
# which the two faces are treated as geometrically incompatible in the lower
# third and the boundary gets clamped. Below it the mismatch is inside the
# feather's own width and clamping would cost coverage for nothing.
DEVIATION_THRESHOLD = float(os.environ.get('ROOP_JAW_DEVIATION', '0.08') or 0.08)

# Feather across the jaw silhouette, as a fraction of the wide feather. The jaw
# is the one boundary with a depth discontinuity behind it.
JAW_FEATHER_RATIO = float(os.environ.get('ROOP_JAW_FEATHER_RATIO', '0.35') or 0.35)


# -- Geometry ---------------------------------------------------------------

class LowerFaceGeometry:
    """Source/target lower-face mismatch, measured in canonical space."""

    __slots__ = ('chin_delta', 'jaw_width_ratio', 'eye_mouth_px',
                 'target_chin_depth', 'source_chin_depth', 'valid')

    def __init__(self, chin_delta: float, jaw_width_ratio: float,
                 eye_mouth_px: float, target_chin_depth: float,
                 source_chin_depth: float, valid: bool) -> None:
        self.chin_delta = chin_delta
        self.jaw_width_ratio = jaw_width_ratio
        self.eye_mouth_px = eye_mouth_px
        self.target_chin_depth = target_chin_depth
        self.source_chin_depth = source_chin_depth
        self.valid = valid

    @property
    def deviates(self) -> bool:
        """Whether the mismatch is large enough to need a clamped boundary."""
        return self.valid and abs(self.chin_delta) > DEVIATION_THRESHOLD

    @property
    def source_chin_is_shorter(self) -> bool:
        return self.chin_delta > 0.0

    def __repr__(self) -> str:
        return (f"LowerFaceGeometry(chin_delta={self.chin_delta:+.3f} "
                f"jaw_width_ratio={self.jaw_width_ratio:.3f} "
                f"deviates={self.deviates} valid={self.valid})")


def _affine(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ M[:, :2].T + M[:, 2]


def _head_axes(kps: Optional[Sequence[float]]) -> Tuple[np.ndarray, np.ndarray]:
    """``(up, across)`` unit vectors from the 5 keypoints.

    Image y grows downward, so an upright head yields ``up == (0, -1)`` and
    every projection below collapses to plain y/x arithmetic - matching
    ``procmgr_masking.landmark_hull``, which must agree with this module about
    where a face is.
    """
    up = np.array([0.0, -1.0], dtype=np.float64)
    if kps is not None:
        try:
            k = np.asarray(kps, dtype=np.float64).reshape(-1, 2)
            if k.shape[0] == 5:
                axis = ((k[0] + k[1]) * 0.5) - ((k[3] + k[4]) * 0.5)
                n = float(np.linalg.norm(axis))
                if n > 1e-6:
                    up = axis / n
        except Exception:
            pass
    return up, np.array([-up[1], up[0]], dtype=np.float64)


def _similarity_scale(M: np.ndarray) -> float:
    """Canonical units per frame unit, for a similarity transform."""
    a = np.asarray(M, dtype=np.float64)[:, :2]
    return float(np.sqrt(abs(np.linalg.det(a))))


def lower_face_geometry(tgt106: Optional[np.ndarray],
                        src106: Optional[np.ndarray],
                        tgt_kps: Optional[np.ndarray],
                        src_kps: Optional[np.ndarray],
                        canonical_size: int = 256) -> LowerFaceGeometry:
    """Measure the lower-face shape difference, pose and scale removed.

    Both faces are mapped through their OWN 5-point arcface transform into the
    shared canonical frame. That is the same transform the swapper aligns with,
    so in canonical space the eyes, nose and mouth of both faces coincide by
    construction and every remaining difference in the contour is shape.

    ``chin_delta`` is ``(target chin depth - source chin depth)`` as a fraction
    of the canonical eye-to-mouth distance, signed: positive means the target's
    chin sits lower than the source's, i.e. the swapper will draw a chin
    ABOVE where the target's matte reaches.
    """
    invalid = LowerFaceGeometry(0.0, 1.0, 0.0, 0.0, 0.0, False)
    if tgt106 is None or src106 is None or tgt_kps is None or src_kps is None:
        return invalid
    try:
        from roop.face_util import estimate_norm

        t106 = np.asarray(tgt106, dtype=np.float32)
        s106 = np.asarray(src106, dtype=np.float32)
        if t106.shape[0] < 33 or s106.shape[0] < 33:
            return invalid
        tk = np.asarray(tgt_kps, dtype=np.float32).reshape(-1, 2)
        sk = np.asarray(src_kps, dtype=np.float32).reshape(-1, 2)
        if tk.shape[0] != 5 or sk.shape[0] != 5:
            return invalid

        Mt = estimate_norm(tk, canonical_size)
        Ms = estimate_norm(sk, canonical_size)
        t_c = _affine(Mt, t106)
        s_c = _affine(Ms, s106)
        tk_c = _affine(Mt, tk)

        # In canonical space the head is upright by construction, so depth is
        # plain y and no axis projection is needed here.
        eye_mid_y = float((tk_c[0][1] + tk_c[1][1]) * 0.5)
        mouth_mid_y = float((tk_c[3][1] + tk_c[4][1]) * 0.5)
        eye_mouth = abs(mouth_mid_y - eye_mid_y)
        if eye_mouth <= 1e-3:
            return invalid

        t_chin = float(t_c[CONTOUR_IDX][:, 1].max()) - eye_mid_y
        s_chin = float(s_c[CONTOUR_IDX][:, 1].max()) - eye_mid_y

        t_width = float(np.ptp(t_c[CONTOUR_IDX][:, 0]))
        s_width = float(np.ptp(s_c[CONTOUR_IDX][:, 0]))
        width_ratio = (s_width / t_width) if t_width > 1e-3 else 1.0

        return LowerFaceGeometry(
            chin_delta=(t_chin - s_chin) / eye_mouth,
            jaw_width_ratio=width_ratio,
            eye_mouth_px=eye_mouth,
            target_chin_depth=t_chin,
            source_chin_depth=s_chin,
            valid=True)
    except Exception:
        return invalid


# -- Boundary construction --------------------------------------------------

def _jaw_chain(pts: np.ndarray, up: np.ndarray, across: np.ndarray,
               eye_line_along: float) -> np.ndarray:
    """Contour points below the eye line, ordered along the across-axis.

    Sorting by the across-coordinate rather than trusting the landmark index
    order gives a monotone chain that traces the jaw as it actually is -
    including its concavity under the chin - and does so without assuming
    anything about how the 106-point model orders its contour indices.
    """
    along = pts @ up
    lower = pts[along < eye_line_along]
    if lower.shape[0] < 3:
        lower = pts
    order = np.argsort(lower @ across)
    return lower[order]


def adaptive_lower_boundary_mask(frame_shape: Tuple[int, ...],
                                 tgt106: np.ndarray,
                                 tgt_kps: Optional[np.ndarray] = None,
                                 geometry: Optional[LowerFaceGeometry] = None,
                                 target_M: Optional[np.ndarray] = None,
                                 chin_margin: float = 0.04,
                                 forehead_frac: float = 0.6,
                                 canonical_size: int = 256) -> Mask:
    """Swap-polarity matte (255 = swap here) with an adaptive lower edge.

    Differs from ``procmgr_masking.create_landmark_mask`` in exactly two ways,
    both confined to the lower third:

    * the polygon's lower edge is the concave jaw chain, not a convex hull, so
      the matte does not bridge across the neck under the chin;
    * when ``geometry`` reports a real mismatch AND ``target_M`` is supplied,
      the chain is clamped so the matte never reaches below the SHALLOWER of
      the two chins. If the source chin is shorter, the matte stops where the
      swapper's chin actually is, and the target's own chin is left untouched
      instead of being blended against the net's extrapolation. If it is
      longer, the matte still stops at the target's jaw, so the overshoot is
      cropped rather than pasted onto the neck.

    ``target_M`` is the frame->canonical transform used to convert the canonical
    chin depth back into frame pixels. Leave it None and it is derived from
    ``tgt_kps`` with the same ``estimate_norm`` and the same canonical size
    :func:`lower_face_geometry` used - which is what a caller wants, because a
    transform built at a DIFFERENT canonical size would rescale the depths and
    silently move the clamp. Pass one explicitly only to reuse a matrix you know
    was built at ``canonical_size``.
    """
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    pts = np.asarray(tgt106, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 33:
        return mask

    chain, up, across, face_h = _clamped_chain(
        pts, tgt_kps, geometry, target_M, chin_margin, canonical_size)
    contour = pts[CONTOUR_IDX]
    along = pts @ up
    across_all = pts @ across
    brow_along = float(np.max(along[BROW_SLICE])) if pts.shape[0] > 52 else float(np.max(along))
    chin_along = float(np.min(along[CONTOUR_IDX]))
    eye_line_along = chin_along + 0.55 * face_h

    # -- Upper boundary: brow band + forehead extension ---------------------
    forehead_along = brow_along + int(face_h * forehead_frac)
    top_zone = across_all[along > brow_along - int(face_h * 0.15)]
    if len(top_zone) >= 2:
        left_t, right_t = float(np.min(top_zone)), float(np.max(top_zone))
    else:
        left_t, right_t = float(np.min(across_all)), float(np.max(across_all))

    upper = np.array([
        forehead_along * up + t * across
        for t in (left_t, (left_t + right_t) * 0.5, right_t)
    ], dtype=np.float64)
    upper_hull = cv2.convexHull(
        np.vstack([contour[(contour @ up) >= eye_line_along], upper]
                  ).astype(np.int32)).reshape(-1, 2).astype(np.float64)

    # Close the polygon: the upper hull ordered by the across-axis, then the
    # jaw chain back the other way. fillPoly handles the concavity that
    # fillConvexPoly could not represent.
    upper_sorted = upper_hull[np.argsort(upper_hull @ across)]
    polygon = np.vstack([upper_sorted, chain[::-1]]).astype(np.int32)
    cv2.fillPoly(mask, [polygon], 255)
    return mask


def _clamped_chain(pts: np.ndarray,
                   tgt_kps: Optional[np.ndarray],
                   geometry: Optional[LowerFaceGeometry],
                   target_M: Optional[np.ndarray],
                   chin_margin: float,
                   canonical_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """``(chain, up, across, face_h)`` - the jaw chain, clamped if it must be.

    Shared by :func:`adaptive_lower_boundary_mask` (which closes a whole
    polygon around it) and :func:`lower_face_keep_region` (which only wants the
    half-plane above it), so the two can never disagree about where the jaw is.
    """
    up, across = _head_axes(tgt_kps)
    contour = pts[CONTOUR_IDX]
    along = pts @ up

    brow_along = (float(np.max(along[BROW_SLICE])) if pts.shape[0] > 52
                  else float(np.max(along)))
    chin_along = float(np.min(along[CONTOUR_IDX]))
    face_h = max(1.0, brow_along - chin_along)
    eye_line_along = chin_along + 0.55 * face_h
    chain = _jaw_chain(contour, up, across, eye_line_along)

    if geometry is None or not geometry.deviates:
        return chain, up, across, face_h

    if target_M is None:
        # Same transform and same canonical size lower_face_geometry used, or
        # the depths it measured would be laid off at the wrong scale.
        try:
            from roop.face_util import estimate_norm

            k = np.asarray(tgt_kps, dtype=np.float32).reshape(-1, 2)
            if k.shape[0] == 5:
                target_M = estimate_norm(k, canonical_size)
        except Exception:
            target_M = None
    if target_M is None:
        return chain, up, across, face_h

    scale = _similarity_scale(target_M)          # canonical px per frame px
    if scale <= 1e-6:
        return chain, up, across, face_h

    eye_mid_along = eye_line_along
    if tgt_kps is not None:
        try:
            k = np.asarray(tgt_kps, dtype=np.float64).reshape(-1, 2)
            if k.shape[0] == 5:
                eye_mid_along = float(((k[0] + k[1]) * 0.5) @ up)
        except Exception:
            pass

    # `geometry`'s depths are canonical pixels measured downward from the eye
    # midline; `along` grows toward the crown, so a depth below the eyes is a
    # subtraction. The shallower chin wins: never keep swap below where BOTH
    # faces still have face. The margin holds the edge inside skin rather than
    # exactly on the silhouette, where a one-pixel landmark error puts it on
    # the neck.
    src_chin_along = eye_mid_along - geometry.source_chin_depth / scale
    tgt_chin_along = eye_mid_along - geometry.target_chin_depth / scale
    limit = max(src_chin_along, tgt_chin_along) + chin_margin * face_h
    limit = min(limit, brow_along - 0.2 * face_h)     # never above mid-face
    chain = chain + np.maximum(0.0, limit - (chain @ up))[:, None] * up[None, :]
    return chain, up, across, face_h


def lower_face_keep_region(frame_shape: Tuple[int, ...],
                           tgt106: np.ndarray,
                           tgt_kps: Optional[np.ndarray] = None,
                           geometry: Optional[LowerFaceGeometry] = None,
                           target_M: Optional[np.ndarray] = None,
                           chin_margin: float = 0.04,
                           canonical_size: int = 256) -> Mask:
    """255 on the crown side of the jaw chain, 0 below it.

    A TRIM rather than a replacement matte, and that distinction is the whole
    point of this function existing next to
    :func:`adaptive_lower_boundary_mask`. Intersecting an existing matte with
    this can only ever remove matte below the jaw, so it is provably incapable
    of changing the forehead, temple or hairline boundary - the regions the
    lower-face artefact has nothing to do with, and which the existing hull
    already handles. Replacing the whole polygon instead moves every boundary
    at once, which is a much larger behavioural change than the bug being
    fixed asked for.

    Two things get removed: the ``cv2.convexHull`` bridge that spans the
    concavity under the chin, and (when ``geometry`` reports a real mismatch)
    everything below the shallower of the two chins.
    """
    keep = np.zeros(frame_shape[:2], dtype=np.uint8)
    pts = np.asarray(tgt106, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 33:
        return np.full(frame_shape[:2], 255, dtype=np.uint8)

    chain, up, across, _face_h = _clamped_chain(
        pts, tgt_kps, geometry, target_M, chin_margin, canonical_size)
    if chain.shape[0] < 2:
        return np.full(frame_shape[:2], 255, dtype=np.uint8)

    # The region has to be unbounded LATERALLY as well as upward, or it clips
    # the hull wherever the hull is wider than the jaw — which is everywhere
    # above the cheekbones: the temples and the forehead extension both reach
    # past the chain's own horizontal extent, and clipping them is exactly the
    # kind of change above the jaw this function exists to avoid making.
    #
    # So walk the chain, step out sideways from each end, then close over the
    # top. `chain` is sorted along the across-axis, so chain[0] and chain[-1]
    # are its two ends and the polygon stays simple however the head is rolled.
    reach = float(np.hypot(*frame_shape[:2])) * 2.0
    left = chain[0] - reach * across
    right = chain[-1] + reach * across
    polygon = np.vstack([left, chain, right,
                         right + reach * up, left + reach * up])
    cv2.fillPoly(keep, [polygon.astype(np.int32)], 255)
    return keep


def jaw_band_weight(frame_shape: Tuple[int, ...],
                    tgt106: np.ndarray,
                    tgt_kps: Optional[np.ndarray] = None,
                    band_frac: float = 0.18) -> Mask:
    """1.0 along the jaw silhouette, 0.0 elsewhere - the anisotropy map.

    Rasterises the jaw chain as a thick polyline and blurs it, so
    :func:`anisotropic_feather` has a smooth weight to interpolate its two
    feather widths with (a hard switch between two blur radii would show up as
    a visible step where the two regions meet).
    """
    band = np.zeros(frame_shape[:2], dtype=np.uint8)
    pts = np.asarray(tgt106, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 33:
        return band.astype(np.float32)

    up, across = _head_axes(tgt_kps)
    along = pts @ up
    brow_along = float(np.max(along[BROW_SLICE])) if pts.shape[0] > 52 else float(np.max(along))
    chin_along = float(np.min(along[CONTOUR_IDX]))
    face_h = max(1.0, brow_along - chin_along)
    chain = _jaw_chain(pts[CONTOUR_IDX], up, across, chin_along + 0.55 * face_h)

    thickness = max(3, int(round(band_frac * face_h)))
    cv2.polylines(band, [chain.astype(np.int32)], False, 255, thickness,
                  lineType=cv2.LINE_AA)
    k = (thickness | 1)
    soft = cv2.GaussianBlur(band.astype(np.float32) / 255.0, (k, k), 0)
    peak = float(soft.max())
    return soft / peak if peak > 1e-6 else soft


def anisotropic_feather(mask: Mask,
                        band: Mask,
                        wide_px: int,
                        tight_px: Optional[int] = None,
                        erode_px: int = 0) -> Mask:
    """Feather `mask` narrowly inside `band` and widely outside it.

    A single feather width has to serve two very different boundaries. Across
    the forehead and temples the matte edge runs through skin and hair at the
    same depth, and a wide ramp is what hides the seam. Across the jaw it runs
    along a depth discontinuity with neck and background behind it, and the
    same wide ramp is what produces a translucent second jawline.

    Implemented as a blend of two blurred copies rather than a per-pixel
    variable-radius filter: two separable Gaussians plus a lerp are a few
    hundred microseconds at 1080p, where a true spatially-varying blur is tens
    of milliseconds, and the visible difference is nil once `band` is smooth.

    ``erode_px`` shrinks the matte before feathering, which keeps the whole
    transition zone inside face skin instead of letting half of it fall outside
    the silhouette. ``MaskingMixin.blur_area`` does the same thing with
    ``blend_px // 2``, so passing that keeps this interchangeable with it.
    """
    m = np.asarray(mask, dtype=np.float32)
    if m.max() > 1.5:
        m = m / 255.0
    if erode_px > 0:
        k_er = int(erode_px) * 2 + 1
        m = cv2.erode(m, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (k_er, k_er)), iterations=1)
    wide_k = int(max(1, wide_px)) | 1
    tight = tight_px if tight_px is not None else max(
        1, int(round(wide_px * JAW_FEATHER_RATIO)))
    tight_k = int(max(1, tight)) | 1

    soft_wide = cv2.GaussianBlur(m, (wide_k, wide_k), 0) if wide_k > 1 else m
    soft_tight = cv2.GaussianBlur(m, (tight_k, tight_k), 0) if tight_k > 1 else m

    w = np.asarray(band, dtype=np.float32)
    if w.shape != m.shape:
        w = cv2.resize(w, (m.shape[1], m.shape[0]), interpolation=cv2.INTER_LINEAR)
    w = np.clip(w, 0.0, 1.0)
    return np.clip(w * soft_tight + (1.0 - w) * soft_wide, 0.0, 1.0)


# -- Blending ---------------------------------------------------------------

def soft_lower_face_keep(frame_shape: Tuple[int, ...],
                         tgt106: np.ndarray,
                         tgt_kps: Optional[np.ndarray] = None,
                         geometry: Optional[LowerFaceGeometry] = None,
                         target_M: Optional[np.ndarray] = None,
                         feather_px: int = 9,
                         chin_margin: float = 0.04,
                         canonical_size: int = 256) -> Mask:
    """:func:`lower_face_keep_region` as a float multiplier, jaw edge feathered.

    Built to be applied to a matte that has ALREADY been feathered, as a
    multiplicative trim - which is where a trim belongs in this pipeline.
    ``MaskingMixin.blur_area`` derives its feather width from the matte's
    bounding box, so shrinking the matte beforehand narrows the seam over the
    entire face (the code in ``paste_upscale`` measures 29px -> 18px for its own
    trims and applies them afterwards for exactly this reason). Multiplying
    afterwards leaves every other boundary's feather at the width it already
    had, and the jaw gets its own narrow ramp from ``feather_px`` instead of
    inheriting the wide one - which is the point, since the jaw is the one
    boundary with a depth discontinuity behind it.

    Returns 1.0 everywhere on the crown side of the jaw, so it is a no-op above
    the cheekbones by construction.
    """
    keep = lower_face_keep_region(frame_shape, tgt106, tgt_kps=tgt_kps,
                                  geometry=geometry, target_M=target_M,
                                  chin_margin=chin_margin,
                                  canonical_size=canonical_size)
    soft = keep.astype(np.float32) / 255.0
    k = int(max(1, feather_px)) | 1
    return cv2.GaussianBlur(soft, (k, k), 0) if k > 1 else soft


def multiband_blend(foreground: Frame, background: Frame, mask: Mask,
                    levels: int = 5) -> Frame:
    """Laplacian-pyramid (Burt-Adelson) blend of `foreground` over `background`.

    Preferred over a single alpha ramp whenever the two layers differ in
    lighting or colour rather than in content: the mask is blurred once per
    pyramid level, so low frequencies (the lighting difference the seam would
    otherwise reveal) cross over gradually while high frequencies (pores, hair,
    the jaw edge itself) cross over sharply. A single ramp has to pick one
    width for both and gets a soft-focus band at whatever width it picks.

    Roughly 4-6x the cost of an alpha blend over the same region, so run it on
    the matte's bounding box, not the frame.
    """
    if foreground.shape != background.shape:
        foreground = cv2.resize(foreground,
                                (background.shape[1], background.shape[0]),
                                interpolation=cv2.INTER_CUBIC)
    m = np.asarray(mask, dtype=np.float32)
    if m.max() > 1.5:
        m = m / 255.0
    if m.ndim == 2:
        m = m[..., None]
    if m.shape[:2] != background.shape[:2]:
        m = cv2.resize(m[..., 0], (background.shape[1], background.shape[0]),
                       interpolation=cv2.INTER_LINEAR)[..., None]

    # Each level halves the resolution; below ~8 px a level carries no signal
    # and cv2.pyrDown on a 1-px axis raises.
    max_levels = int(np.floor(np.log2(max(2, min(background.shape[:2]) / 8.0))))
    levels = int(max(1, min(levels, max_levels)))

    fg = foreground.astype(np.float32)
    bg = background.astype(np.float32)

    gp_f, gp_b, gp_m = [fg], [bg], [m]
    for _ in range(levels):
        gp_f.append(cv2.pyrDown(gp_f[-1]))
        gp_b.append(cv2.pyrDown(gp_b[-1]))
        down = cv2.pyrDown(gp_m[-1])
        gp_m.append(down if down.ndim == 3 else down[..., None])

    out = gp_f[-1] * gp_m[-1] + gp_b[-1] * (1.0 - gp_m[-1])
    for i in range(levels - 1, -1, -1):
        size = (gp_f[i].shape[1], gp_f[i].shape[0])
        lap_f = gp_f[i] - cv2.pyrUp(gp_f[i + 1], dstsize=size)
        lap_b = gp_b[i] - cv2.pyrUp(gp_b[i + 1], dstsize=size)
        band = lap_f * gp_m[i] + lap_b * (1.0 - gp_m[i])
        out = cv2.pyrUp(out, dstsize=size) + band

    return np.clip(out, 0, 255).astype(np.uint8)


def poisson_blend(foreground: Frame, background: Frame, mask: Mask,
                  flags: int = cv2.NORMAL_CLONE) -> Frame:
    """``cv2.seamlessClone`` with the guards it needs to be usable in a loop.

    Poisson blending matches the gradient field rather than the pixel values,
    which is exactly right for a lighting mismatch across the jaw - and exactly
    wrong when the mask touches the frame border or is empty, where OpenCV
    either throws or returns a black plate. It also solves over the whole
    bounding box each call, so it is the most expensive option here (order 10
    ms for a 300 px face) and belongs on stills or on a final pass, not on
    every frame of a 30 fps render. Returns `background` unchanged rather than
    raising if the geometry is degenerate.

    Note that seamlessClone will also drag the swap's overall colour toward the
    plate. That is desirable for a seam and undesirable for identity, so run it
    AFTER any colour transfer, never instead of one.
    """
    m = np.asarray(mask, dtype=np.float32)
    if m.max() <= 1.5:
        m = m * 255.0
    m8 = np.clip(m, 0, 255).astype(np.uint8)
    m8 = (m8 > 127).astype(np.uint8) * 255

    # seamlessClone needs a margin: a mask touching the border has no boundary
    # condition to solve against.
    m8[:2, :] = m8[-2:, :] = m8[:, :2] = m8[:, -2:] = 0
    x, y, w, h = cv2.boundingRect(m8)
    if w < 4 or h < 4:
        return background
    center = (x + w // 2, y + h // 2)
    try:
        return cv2.seamlessClone(foreground, background, m8, center, flags)
    except cv2.error:
        return background


# -- Convenience ------------------------------------------------------------

def blend_lower_face(swapped: Frame,
                     original: Frame,
                     tgt106: np.ndarray,
                     src106: Optional[np.ndarray] = None,
                     tgt_kps: Optional[np.ndarray] = None,
                     src_kps: Optional[np.ndarray] = None,
                     target_M: Optional[np.ndarray] = None,
                     face_mask_blend: float = 30.0,
                     use_multiband: bool = True) -> Tuple[Frame, LowerFaceGeometry]:
    """End-to-end: measure, build the adaptive matte, feather it, composite.

    Operates in frame space on an already-pasted ``swapped`` frame, so it can
    be dropped in beside ``ProcessMgr.reshape_jaw_frame`` without touching the
    swap or the alignment. Returns the blended frame and the geometry that was
    measured, so a caller can log or gate on it.
    """
    geom = lower_face_geometry(tgt106, src106, tgt_kps, src_kps)
    matte = adaptive_lower_boundary_mask(
        original.shape, tgt106, tgt_kps=tgt_kps,
        geometry=geom, target_M=target_M)

    x, y, w, h = cv2.boundingRect(matte)
    if w < 8 or h < 8:
        return original, geom

    # Feather width from the matte's own extent, matching blur_area's scale so
    # the two are interchangeable.
    mask_size = int(np.sqrt(max(1, (w - 1) * (h - 1))))
    wide_px = max(1, int(mask_size * float(face_mask_blend) / 200.0)) * 2 + 1
    band = jaw_band_weight(original.shape, tgt106, tgt_kps=tgt_kps)
    alpha = anisotropic_feather(matte, band, wide_px)

    pad = wide_px
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1 = min(original.shape[1], x + w + pad)
    y1 = min(original.shape[0], y + h + pad)
    box = (slice(y0, y1), slice(x0, x1))

    out = original.copy()
    if use_multiband:
        out[box] = multiband_blend(swapped[box], original[box], alpha[box])
    else:
        a = alpha[box][..., None]
        out[box] = np.clip(a * swapped[box].astype(np.float32)
                           + (1.0 - a) * original[box].astype(np.float32),
                           0, 255).astype(np.uint8)
    return out, geom
