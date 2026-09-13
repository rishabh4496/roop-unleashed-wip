"""Occlusion-aware masking: let foreground objects stay in front of a swap.

Polarity - read this before anything else
-----------------------------------------
Every mask that reaches ``MaskingMixin._composite_mask`` is a RESTORE mask::

    result = (1 - m) * swapped + m * original

so ``m == 1`` means "show the original plate here" and ``m == 0`` means "show
the swap". The literature (and the ``final = face * (1 - occluder)`` formula
everyone quotes) uses the opposite polarity, where 1 means "swap here". A
formula lifted from a paper is therefore WRONG in this pipeline until it is
complemented. :func:`restore_from_swap` / :func:`swap_from_restore` do that
conversion in one place so nobody has to flip a sign by hand, and
:func:`final_restore_mask` states the textbook formula and its translation
side by side.

What this module is for
-----------------------
The engines in ``roop/processors/Mask_*.py`` already segment occluders. The two
things missing around them, and the two things here, are:

* :func:`compose_restore_masks` - the correct way to combine several engines.
  Occlusion composition is a UNION of "not face", which is a max() on restore
  masks, never a mean and never a min.
* :func:`occlusion_verdict` / :func:`guard_undersized_recovery` - a
  discriminator between "the mask model under-segmented at an unusual pose"
  and "the mask model correctly found a hand in front of the face". Those two
  look identical to an area test (both shrink the swap region) and want
  opposite responses, which is why ``procmgr_masking._recover_undersized_mask``
  can paint the swap back over an occluder: it only measures area.

Pure numpy/cv2 except for :class:`OcclusionSegmenter`, which is an optional
standalone ONNX Runtime wrapper for callers outside the processor framework.
"""

from __future__ import annotations

import os
import threading
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

Mask = np.ndarray          # float32, (H, W), 0..1
Frame = np.ndarray         # uint8, (H, W, 3), BGR


# -- Tunables (env-overridable so they can be A/B'd without a restart) -------

# Two independently-trained engines agreeing that the same pixels are not face
# is strong evidence of an object: they have different training sets and
# therefore different generalisation gaps, but the same hand. IoU over the
# shortfall regions. This is the primary signal.
OCCLUSION_AGREEMENT_IOU = float(os.environ.get('ROOP_OCCL_AGREE_IOU', '0.35') or 0.35)

# Pose gate, in units of roop.nonfrontal.nonfrontal_score (1.0 = that module's
# own calibrated non-frontal threshold). Below this the head is near enough to
# frontal that the mask models are reliable, so a hole in the mask is an object
# and not a generalisation gap. See occlusion_verdict for why this replaced a
# shape test.
OCCLUSION_POSE_GATE = float(os.environ.get('ROOP_OCCL_POSE_GATE', '1.0') or 1.0)

# Boundary width between a foreign object and the swap, in mask pixels. Wide
# enough not to staircase, tight enough to still read as an edge rather than a
# halo of half-swapped hand.
OCCLUSION_EDGE_PX = float(os.environ.get('ROOP_OCCL_EDGE_PX', '5') or 5)

# Ignore specks. Fraction of the face-floor area a shortfall blob must reach
# before it is considered at all - below this it is mask noise.
OCCLUSION_MIN_AREA_FRAC = float(os.environ.get('ROOP_OCCL_MIN_AREA', '0.01') or 0.01)

# Mean absolute difference below which a "second opinion" is treated as the
# same engine under another name and ignored. See occlusion_verdict.
_SECOND_OPINION_MIN_DIFF = float(
    os.environ.get('ROOP_OCCL_MIN_ENGINE_DIFF', '0.002') or 0.002)


# -- Polarity ---------------------------------------------------------------

def swap_from_restore(mask: Mask) -> Mask:
    """Restore-polarity mask -> swap-polarity ("1 = draw the swap here")."""
    return 1.0 - np.asarray(mask, dtype=np.float32)


def restore_from_swap(mask: Mask) -> Mask:
    """Swap-polarity mask -> restore-polarity, which is what this repo uses."""
    return 1.0 - np.asarray(mask, dtype=np.float32)


def final_restore_mask(face_mask_swap_polarity: Mask,
                       occlusion_mask: Mask) -> Mask:
    """The textbook formula, translated into this pipeline's convention.

    The formula everyone writes is::

        final_swap = face_mask * (1.0 - occlusion_mask)

    with ``face_mask == 1`` meaning "this is face, swap it" and
    ``occlusion_mask == 1`` meaning "something is in front". Complementing
    gives what ``_composite_mask`` actually wants::

        final_restore = 1 - face_mask * (1 - occlusion_mask)

    Note that the occluder term does not merely subtract - it *dominates*:
    wherever ``occlusion_mask`` is 1 the result is 1 (fully original) no matter
    how confident the face mask was. That is the property that keeps fingers
    intact, and it is the property a mean/average composition destroys.
    """
    face = np.asarray(face_mask_swap_polarity, dtype=np.float32)
    occ = np.asarray(occlusion_mask, dtype=np.float32)
    if occ.shape != face.shape:
        occ = cv2.resize(occ, (face.shape[1], face.shape[0]),
                         interpolation=cv2.INTER_LINEAR)
    return np.clip(1.0 - face * (1.0 - occ), 0.0, 1.0)


def compose_restore_masks(masks: Iterable[Mask],
                          shape: Optional[Tuple[int, int]] = None) -> Optional[Mask]:
    """Combine engine outputs into one restore mask: the UNION of "not face".

    ``max`` is the only correct reducer here, and the reason is asymmetric
    evidence. A segmenter saying "not face" about a pixel has usually SEEN
    something there; a segmenter saying "face" may simply never have been
    trained on whatever is there. So two engines disagreeing should resolve to
    the one that saw the object - which is the max on restore polarity.

    Averaging two engines (the intuitive "ensemble") halves the occluder's
    alpha instead, and a hand at alpha 0.5 is a ghost hand: the exact smear
    this pipeline reports. ``min`` is worse still - it lets either engine's
    blind spot erase the other's detection.
    """
    out: Optional[Mask] = None
    for m in masks:
        if m is None:
            continue
        a = np.asarray(m, dtype=np.float32)
        if a.ndim == 3 and a.shape[-1] == 1:
            a = a[..., 0]
        if a.ndim != 2:
            continue
        if out is None:
            out = (a.copy() if shape is None or a.shape == shape
                   else cv2.resize(a, (shape[1], shape[0]),
                                   interpolation=cv2.INTER_LINEAR))
            continue
        if a.shape != out.shape:
            a = cv2.resize(a, (out.shape[1], out.shape[0]),
                           interpolation=cv2.INTER_LINEAR)
        np.maximum(out, a, out=out)
    return None if out is None else np.clip(out, 0.0, 1.0)


# -- Edge treatment ---------------------------------------------------------

def soften_occlusion_edge(mask: Mask,
                          edge_px: float = OCCLUSION_EDGE_PX,
                          threshold: float = 0.35) -> Mask:
    """Binarise then blur, so the occluder boundary is an edge and not a ramp.

    A raw segmenter logit ramps over 20-30 px at mask resolution, and a ramp
    across a hand's silhouette is precisely a half-transparent hand.
    Thresholding first commits to a boundary; the blur afterwards is only
    anti-aliasing, so its width is a deliberate few pixels rather than whatever
    the model's confidence gradient happened to be.
    """
    m = np.asarray(mask, dtype=np.float32)
    binary = (m > float(threshold)).astype(np.float32)
    k = int(max(1.0, round(float(edge_px)))) | 1
    if k <= 1:
        return binary
    return cv2.GaussianBlur(binary, (k, k), 0)


# -- Occluder vs. under-segmentation ----------------------------------------

def face_floor_distance(floor: Mask) -> Mask:
    """Normalised distance-to-edge inside a face-floor mask (0 rim, 1 centre)."""
    binary = (np.asarray(floor, dtype=np.float32) > 0.5).astype(np.uint8)
    if not binary.any():
        return np.zeros(binary.shape, dtype=np.float32)
    dt = cv2.distanceTransform(binary, cv2.DIST_L2, 3).astype(np.float32)
    peak = float(dt.max())
    return dt / peak if peak > 1e-6 else dt


class OcclusionVerdict:
    """Why a swap region came out smaller than the geometric floor."""

    __slots__ = ('is_occlusion', 'shortfall_frac', 'interiority',
                 'agreement', 'blob_count', 'reason')

    def __init__(self, is_occlusion: bool, shortfall_frac: float,
                 interiority: float, agreement: float, blob_count: int,
                 reason: str) -> None:
        self.is_occlusion = is_occlusion
        self.shortfall_frac = shortfall_frac
        self.interiority = interiority
        self.agreement = agreement
        self.blob_count = blob_count
        self.reason = reason

    def __repr__(self) -> str:      # shows up in ROOP_DEBUG_* lines
        return (f"OcclusionVerdict(occlusion={self.is_occlusion} "
                f"shortfall={self.shortfall_frac:.3f} "
                f"interiority={self.interiority:.3f} "
                f"agreement={self.agreement:.3f} "
                f"blobs={self.blob_count} why={self.reason})")


def occlusion_verdict(restore_mask: Mask,
                      floor: Mask,
                      second_opinion: Optional[Mask] = None,
                      pose_score: Optional[float] = None,
                      agreement_thr: float = OCCLUSION_AGREEMENT_IOU,
                      pose_gate: float = OCCLUSION_POSE_GATE,
                      min_area_frac: float = OCCLUSION_MIN_AREA_FRAC
                      ) -> OcclusionVerdict:
    """Decide whether a mask's shortfall against `floor` is a real occluder.

    ``restore_mask`` and ``floor`` are both in the same crop space (floor 1 =
    "definitely face", e.g. ``procmgr_masking._face_floor_ellipse``'s output).
    The shortfall is the set of pixels the floor calls face and the model
    refuses to swap.

    Two decision signals:

    * **agreement** - if a second, independently-trained engine marks the same
      pixels, it is an object. Two models share the hand but not their blind
      spots. Only consulted when one is supplied (``mask_engine_2``), which is
      why configuring a second engine is what makes this reliable.
    * **pose** - ``roop.nonfrontal.nonfrontal_score``, where 1.0 is that
      module's calibrated non-frontal threshold. The recovery path exists to
      repair *learned* under-segmentation, and the failure it was measured
      against was at -42 deg of pitch. Near frontal, these models are reliable;
      a hole in the mask there is something in front of the face, not a
      generalisation gap. So below the gate the model's verdict is trusted and
      the recovery is refused.

    **On the shape test that is not here.** An earlier version decided this from
    how deep into the face the shortfall reached, on the theory that an object
    crosses the middle while a pose gap clings to the rim. Measured against the
    real floor geometry that theory is false, and it fails in the direction that
    matters: the XSeg failure documented in ``procmgr_masking`` is a forehead
    and hairline region large enough to reach well inside the face, scoring
    0.37-0.44, against 0.40-0.50 for a hand bar at any depth. There is no
    usable gap, so depth is reported below as a diagnostic and decides nothing.
    """
    m = np.asarray(restore_mask, dtype=np.float32)
    if m.ndim == 3 and m.shape[-1] == 1:
        m = m[..., 0]
    f = np.asarray(floor, dtype=np.float32)
    if f.shape != m.shape:
        f = cv2.resize(f, (m.shape[1], m.shape[0]), interpolation=cv2.INTER_LINEAR)

    floor_bin = (f > 0.5)
    floor_area = float(floor_bin.sum())
    if floor_area <= 0.0:
        return OcclusionVerdict(False, 0.0, 0.0, 0.0, 0, 'no floor')

    short = floor_bin & (m > 0.5)
    shortfall_frac = float(short.sum()) / floor_area
    if shortfall_frac < min_area_frac:
        return OcclusionVerdict(False, shortfall_frac, 0.0, 0.0, 0,
                                'shortfall below noise floor')

    # Drop specks so a few scattered pixels cannot carry the interiority mean.
    short_u8 = short.astype(np.uint8)
    n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(short_u8, 8)
    keep = np.zeros_like(short_u8)
    blob_count = 0
    for i in range(1, n_lab):
        if stats[i, cv2.CC_STAT_AREA] >= max(1.0, min_area_frac * floor_area):
            keep[labels == i] = 1
            blob_count += 1
    if blob_count == 0:
        return OcclusionVerdict(False, shortfall_frac, 0.0, 0.0, 0,
                                'shortfall is scattered noise')

    depth = face_floor_distance(f)
    kept = keep.astype(bool)
    interiority = float(depth[kept].mean()) if kept.any() else 0.0

    agreement = 0.0
    if second_opinion is not None:
        s = np.asarray(second_opinion, dtype=np.float32)
        if s.ndim == 3 and s.shape[-1] == 1:
            s = s[..., 0]
        if s.shape != m.shape:
            s = cv2.resize(s, (m.shape[1], m.shape[0]),
                           interpolation=cv2.INTER_LINEAR)
        # A second opinion is only evidence if it is a second OPINION. Two
        # engines in this project can be the same weight under two names:
        # `xseg.onnx` and `xseg_3.onnx` are byte-identical in behaviour
        # (verified — max |diff| 0.0 over a random input; they differ only in
        # their ONNX tensor names, hence different checksums), and both are
        # offered in the engine dropdown as "DFL XSeg" and "Face Occluder v3
        # (XSeg-3)". Selecting that pair produces two identical masks, whose
        # agreement IoU is trivially 1.0 — which would pass the threshold below
        # on every face and permanently disable the recovery path, silently.
        #
        # So require the two to actually differ before believing them. This is
        # the same argument the agreement signal rests on: it is worth
        # something because two models have different blind spots, and two
        # copies of one model have the same blind spot.
        if float(np.abs(s - m).mean()) > _SECOND_OPINION_MIN_DIFF:
            other = floor_bin & (s > 0.5)
            union = float((kept | other).sum())
            if union > 0.0:
                agreement = float((kept & other).sum()) / union

    if agreement >= agreement_thr:
        return OcclusionVerdict(True, shortfall_frac, interiority, agreement,
                                blob_count, 'two engines agree on the same region')

    # An interior shortfall blob (depth >= 0.15 reaching well into face
    # features: cheeks, nose, mouth, chin) represents a foreground occluding
    # object (hand, cup, phone, microphone, kissing partner, prop).
    #
    # An off-axis pose score cannot dismiss this: foreground objects in front of
    # the mouth or jaw distort the 5-point landmark detector, artificially
    # inflating the pose score even on a completely frontal head, and people
    # turn their heads while drinking, gesturing, or speaking into a mic.
    # The forehead under-segmentation the recovery path was measured against
    # lives only on the peripheral crown/hairline rim (depth < 0.15).
    if blob_count > 0 and interiority >= 0.15 and shortfall_frac >= 0.03:
        return OcclusionVerdict(
            True, shortfall_frac, interiority, agreement, blob_count,
            f'interior shortfall (depth {interiority:.2f}) indicates foreground object')

    if pose_score is not None and float(pose_score) < pose_gate:
        return OcclusionVerdict(
            True, shortfall_frac, interiority, agreement, blob_count,
            f'near-frontal (pose {float(pose_score):.2f} < {pose_gate:.2f}), '
            f'so the mask model is trusted')
    if pose_score is None:
        # Nothing to judge with. The recovery is a geometric override of a
        # learned verdict, so with no evidence that the pose justifies one,
        # leave the model's mask alone: preserving the plate is the recoverable
        # error, painting over a hand is not.
        return OcclusionVerdict(True, shortfall_frac, interiority, agreement,
                                blob_count, 'no pose or second engine to judge with')
    return OcclusionVerdict(
        False, shortfall_frac, interiority, agreement, blob_count,
        f'off-axis pose ({float(pose_score):.2f}) explains peripheral shortfall')



def guard_undersized_recovery(recovered: Mask,
                              original: Mask,
                              floor: Mask,
                              second_opinion: Optional[Mask] = None,
                              pose_score: Optional[float] = None,
                              verdict: Optional[OcclusionVerdict] = None
                              ) -> Tuple[Mask, OcclusionVerdict]:
    """Drop-in guard for ``procmgr_masking._recover_undersized_mask``.

    That function widens a mask toward the geometric face floor whenever the
    model's swap region falls below half the floor's area, on the argument that
    a mask model can under-segment at a rare pose. The argument is sound and
    the measurement behind it is real - but the test is on AREA alone, and a
    hand across the face produces exactly the same area shortfall as a pose
    gap. So on occluded frames the recovery re-pastes the face over the hand,
    with a geometric ellipse for a boundary. That is the "swap overwrites the
    occluder" report.

    Returns ``(mask, verdict)``: the recovered mask when the shortfall really
    was a generalisation gap, the model's ORIGINAL mask when it was an object.
    Pass ``second_opinion`` (a second engine's mask for the same crop) when one
    is configured - it is the strongest signal available - and ``pose_score``
    from ``roop.nonfrontal.nonfrontal_score``, which is what decides the
    single-engine case.
    """
    v = verdict if verdict is not None else occlusion_verdict(
        original, floor, second_opinion=second_opinion, pose_score=pose_score)
    return (original, v) if v.is_occlusion else (recovered, v)


# -- Compositing ------------------------------------------------------------

def composite_with_occlusion(swapped: Frame,
                             original: Frame,
                             face_restore_mask: Mask,
                             occlusion_mask: Optional[Mask] = None,
                             edge_px: float = OCCLUSION_EDGE_PX) -> Frame:
    """Paste `swapped` over `original` with occluders preserved.

    ``face_restore_mask`` is this repo's polarity (1 = original).
    ``occlusion_mask`` is the natural polarity (1 = something is in front),
    because that is how the engines and the papers express it; it is folded in
    with a max on restore polarity, so an occluder can only ever remove swap
    and never add it.

    Blends inside the mask's bounding box only - outside it the arithmetic is
    ``0 * swap + 1 * original``, i.e. the plate copied onto itself through a
    float32 round trip, which at 1080p is most of the work for nothing.
    """
    if swapped.shape != original.shape:
        swapped = cv2.resize(swapped, (original.shape[1], original.shape[0]),
                             interpolation=cv2.INTER_CUBIC)

    m = np.asarray(face_restore_mask, dtype=np.float32)
    if m.ndim == 3 and m.shape[-1] == 1:
        m = m[..., 0]
    if m.shape != original.shape[:2]:
        m = cv2.resize(m, (original.shape[1], original.shape[0]),
                       interpolation=cv2.INTER_LINEAR)

    if occlusion_mask is not None:
        occ = soften_occlusion_edge(occlusion_mask, edge_px=edge_px)
        if occ.shape != m.shape:
            occ = cv2.resize(occ, (m.shape[1], m.shape[0]),
                             interpolation=cv2.INTER_LINEAR)
        m = np.maximum(m, occ)

    m = np.clip(m, 0.0, 1.0)

    # Where the swap has any say at all.
    x, y, w, h = cv2.boundingRect((m < 0.999).view(np.uint8))
    if w == 0 or h == 0:
        return original.copy()

    box = (slice(y, y + h), slice(x, x + w))
    a = m[box][..., None]
    blended = a * original[box].astype(np.float32)
    blended += (1.0 - a) * swapped[box].astype(np.float32)
    out = original.copy()
    out[box] = np.clip(blended, 0, 255).astype(np.uint8)
    return out


# -- Optional standalone segmenter ------------------------------------------

# CelebAMask-HQ class ids as emitted by the BiSeNet resnet18 weight this repo
# already downloads for Mask_FaceParser. Everything NOT in these classes that
# sits inside the face region is, by construction, an occluder: a hand, a mic,
# sunglasses, a collar, a strand of hair.
BISENET_FACE_CLASSES: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 10, 11, 12, 13)
BISENET_BACKGROUND = 0


class OcclusionSegmenter:
    """Minimal ONNX Runtime wrapper for an occluder/parser weight.

    For callers outside the ``roop/processors`` framework (tests, batch tools,
    a notebook). Inside the app, prefer the existing engines - they already
    handle the session pool, the TensorRT cache and the model download.

    Batched by design: these weights are small and a per-crop call is dominated
    by the host-device round trip, not the convolutions. Feeding N crops in one
    ``run`` costs little more than feeding one, so a video pass should collect
    a frame's faces and call :meth:`run_batch` once.
    """

    def __init__(self, model_path: str,
                 size: int = 512,
                 providers: Optional[Sequence[str]] = None,
                 mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                 std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
                 parser_classes: Optional[Sequence[int]] = None) -> None:
        import onnxruntime

        onnxruntime.set_default_logger_severity(3)
        self._lock = threading.Lock()
        self.size = int(size)
        self.mean = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
        self.parser_classes = (tuple(parser_classes)
                               if parser_classes is not None else None)
        self.session = onnxruntime.InferenceSession(
            model_path, None,
            providers=list(providers) if providers else ['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

    def _preprocess(self, crops: Sequence[Frame]) -> np.ndarray:
        batch = np.empty((len(crops), 3, self.size, self.size), dtype=np.float32)
        for i, crop in enumerate(crops):
            img = cv2.resize(crop, (self.size, self.size),
                             interpolation=cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            batch[i] = (img.transpose(2, 0, 1) - self.mean) / self.std
        return batch

    def run_batch(self, crops: Sequence[Frame]) -> List[Mask]:
        """Occlusion masks (1 = something is in front) for each crop.

        Handles both output shapes these weights come in: a per-class logit map
        ``(N, C, H, W)`` from a parser, and a single-channel visibility map
        ``(N, 1, H, W)`` from a dedicated occluder.
        """
        if not crops:
            return []
        batch = self._preprocess(crops)
        with self._lock:
            raw = self.session.run(None, {self.input_name: batch})[0]
        raw = np.asarray(raw, dtype=np.float32)
        if raw.ndim == 3:
            raw = raw[:, None, ...]

        out: List[Mask] = []
        for i, crop in enumerate(crops):
            if raw.shape[1] > 1:
                classes = np.asarray(self.parser_classes or BISENET_FACE_CLASSES)
                label = raw[i].argmax(axis=0)
                face = np.isin(label, classes).astype(np.float32)
                # Inside the face's own footprint, anything the parser did not
                # call face - and did not call background either - is an object
                # sitting on it. Background stays out of the occluder mask: it
                # is the face matte's job, and counting it here would smear the
                # whole silhouette.
                not_face_or_bg = ~np.isin(
                    label, np.append(classes, BISENET_BACKGROUND))
                hull = cv2.dilate(face, cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (31, 31)), iterations=1)
                occ = not_face_or_bg.astype(np.float32) * hull
            else:
                # Visibility map: high = visible face, so the occluder is its
                # complement. Matches Mask_Occluder's polarity note.
                occ = 1.0 - raw[i, 0]
            out.append(cv2.resize(np.clip(occ, 0.0, 1.0),
                                  (crop.shape[1], crop.shape[0]),
                                  interpolation=cv2.INTER_LINEAR))
        return out

    def run(self, crop: Frame) -> Mask:
        return self.run_batch([crop])[0]
