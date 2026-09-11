"""Identity gating: only swap the person who was actually selected.

Three filters, applied in cost order, because the cheap ones remove most of
what the expensive one would have to adjudicate:

1. :class:`GeometryFilter` - detector score, box area, aspect ratio, and
   interocular distance. Free (it reads numbers the detector already produced)
   and it removes the background extras that no recognition model can judge
   anyway: at 20 px interocular an ArcFace embedding is noise, and noise lands
   wherever the threshold happens to be.
2. :class:`IdentityGate` - cosine similarity of the 512-d ArcFace embedding
   against the selected reference(s). One dot product per (face, reference).
3. :class:`FaceTracker` - temporal hysteresis over IoU/centroid-matched
   tracks. A single frame is never enough evidence to start swapping a new
   face, and one bad frame is never enough to stop.

Why the third one is not optional
---------------------------------
A per-frame threshold has no memory, so at 30 fps a bystander whose similarity
crosses the line for two frames produces a two-frame phantom swap - and the
target crossing the line downward for two frames produces a two-frame dropout.
Both read as flicker and both are invisible to any amount of threshold tuning,
because the threshold is not the problem: the absence of state is. Confirming
over ``confirm_frames`` and holding over ``grace_frames`` fixes both at once,
and it is the same mechanism, run in the two directions.

Scale note
----------
``roop.utilities.compute_cosine_distance`` is ``scipy.spatial.distance.cosine``,
which is ``1 - cos_sim``. The app's ``max_face_distance`` setting is therefore a
DISTANCE and the familiar "similarity > 0.65" is ``distance < 0.35``.
:func:`similarity_to_distance` and :func:`distance_to_similarity` convert, and
:data:`DEFAULT_MIN_SIMILARITY` is stated as a similarity so it can be read
without the mental complement.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Embedding = np.ndarray     # float32, (512,)
BBox = Sequence[float]     # (x1, y1, x2, y2)


# -- Defaults (see the table in the accompanying notes) ---------------------

# ArcFace/buffalo_l same-identity pairs sit well above this even across pose
# and lighting; different identities cluster near 0.0-0.25. 0.65 is comfortably
# inside the gap, which is why it is the usual recommendation.
DEFAULT_MIN_SIMILARITY = float(os.environ.get('ROOP_MIN_SIMILARITY', '0.65') or 0.65)

# Detector confidence. Distinct from identity: this is "is there a face here at
# all". Low values are what put background extras in front of the identity
# gate in the first place.
DEFAULT_MIN_DET_SCORE = float(os.environ.get('ROOP_MIN_DET_SCORE', '0.60') or 0.60)

# Below roughly this interocular distance the embedding is not informative
# enough to gate on, whatever the threshold.
DEFAULT_MIN_INTEROCULAR_PX = float(os.environ.get('ROOP_MIN_INTEROCULAR', '24') or 24)

DEFAULT_MIN_BOX_PX = float(os.environ.get('ROOP_MIN_BOX_PX', '48') or 48)
DEFAULT_MIN_BOX_FRAC = float(os.environ.get('ROOP_MIN_BOX_FRAC', '0.0008') or 0.0008)

# A face box is roughly 0.7-1.0 in w/h. Anything far outside that is a torso, a
# reflection, or a detector artefact on a patterned background.
DEFAULT_ASPECT_RANGE: Tuple[float, float] = (0.55, 1.60)

DEFAULT_CONFIRM_FRAMES = int(os.environ.get('ROOP_CONFIRM_FRAMES', '3') or 3)
DEFAULT_GRACE_FRAMES = int(os.environ.get('ROOP_GRACE_FRAMES', '8') or 8)


# -- Metric helpers ---------------------------------------------------------

def l2_normalize(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32).ravel()
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-8 else v


def cosine_similarity(a: Embedding, b: Embedding) -> float:
    """Cosine similarity in [-1, 1]. 1 = same direction = same identity."""
    return float(np.dot(l2_normalize(a), l2_normalize(b)))


def similarity_to_distance(sim: float) -> float:
    """Similarity -> the app's ``max_face_distance`` scale."""
    return 1.0 - float(sim)


def distance_to_similarity(dist: float) -> float:
    """The app's ``max_face_distance`` scale -> similarity."""
    return 1.0 - float(dist)


def best_similarity(embedding: Embedding,
                    references: Sequence[Embedding]) -> Tuple[float, int]:
    """Highest similarity against a bank of reference embeddings.

    Returns ``(similarity, index)``, or ``(-1.0, -1)`` for an empty bank. A
    per-reference max rather than a similarity against the MEAN embedding: the
    mean of several poses of one person is a vector that resembles none of them
    especially well, so averaging first systematically depresses the score of
    the very angles the bank was collected to cover. ``FaceSet.AverageEmbeddings``
    exists for callers that want the other behaviour.
    """
    if not len(references):
        return -1.0, -1
    q = l2_normalize(embedding)
    bank = np.stack([l2_normalize(r) for r in references])
    sims = bank @ q
    idx = int(np.argmax(sims))
    return float(sims[idx]), idx


def bbox_iou(a: BBox, b: BBox) -> float:
    ax0, ay0, ax1, ay1 = (float(v) for v in a[:4])
    bx0, by0, bx1, by1 = (float(v) for v in b[:4])
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0.0 else 0.0


def _centroid(box: BBox) -> Tuple[float, float]:
    return ((float(box[0]) + float(box[2])) * 0.5,
            (float(box[1]) + float(box[3])) * 0.5)


def _interocular(face: object) -> Optional[float]:
    kps = getattr(face, 'kps', None)
    if kps is None:
        return None
    try:
        k = np.asarray(kps, dtype=np.float32).reshape(-1, 2)
        if k.shape[0] < 2:
            return None
        return float(np.hypot(*(k[1] - k[0])))
    except Exception:
        return None


# -- Geometry ---------------------------------------------------------------

class GeometryFilter:
    """Cheap structural rejection, before any embedding is compared.

    Every threshold here answers "could this detection possibly be the subject
    of a deliberate swap", not "who is it". Running it first means the identity
    gate only ever adjudicates candidates whose embeddings are worth something,
    which is what makes a strict similarity threshold safe to set: most of the
    scores that used to land near the boundary were small, blurry, or
    non-face detections whose embeddings were effectively random.
    """

    def __init__(self,
                 min_det_score: float = DEFAULT_MIN_DET_SCORE,
                 min_box_px: float = DEFAULT_MIN_BOX_PX,
                 min_box_frac: float = DEFAULT_MIN_BOX_FRAC,
                 aspect_range: Tuple[float, float] = DEFAULT_ASPECT_RANGE,
                 min_interocular_px: float = DEFAULT_MIN_INTEROCULAR_PX) -> None:
        self.min_det_score = float(min_det_score)
        self.min_box_px = float(min_box_px)
        self.min_box_frac = float(min_box_frac)
        self.aspect_range = (float(aspect_range[0]), float(aspect_range[1]))
        self.min_interocular_px = float(min_interocular_px)

    def reject_reason(self, face: object,
                      frame_shape: Optional[Tuple[int, ...]] = None
                      ) -> Optional[str]:
        """``None`` when the face is worth judging, else a short reason."""
        box = getattr(face, 'bbox', None)
        if box is None:
            return 'no bbox'
        x0, y0, x1, y1 = (float(v) for v in np.asarray(box).ravel()[:4])
        w, h = x1 - x0, y1 - y0
        if w <= 0.0 or h <= 0.0:
            return 'degenerate bbox'

        score = getattr(face, 'det_score', None)
        if score is not None and float(score) < self.min_det_score:
            return f'det_score {float(score):.2f} < {self.min_det_score:.2f}'

        if min(w, h) < self.min_box_px:
            return f'box {w:.0f}x{h:.0f} under {self.min_box_px:.0f}px'

        if frame_shape is not None and self.min_box_frac > 0.0:
            frame_area = float(frame_shape[0]) * float(frame_shape[1])
            if frame_area > 0.0 and (w * h) / frame_area < self.min_box_frac:
                return f'box is {(w * h) / frame_area * 100:.3f}% of frame'

        aspect = w / h
        lo, hi = self.aspect_range
        if not (lo <= aspect <= hi):
            return f'aspect {aspect:.2f} outside [{lo:.2f}, {hi:.2f}]'

        iod = _interocular(face)
        if iod is not None and iod < self.min_interocular_px:
            return f'interocular {iod:.0f}px < {self.min_interocular_px:.0f}px'

        return None

    def accepts(self, face: object,
                frame_shape: Optional[Tuple[int, ...]] = None) -> bool:
        return self.reject_reason(face, frame_shape) is None


# -- Identity ---------------------------------------------------------------

class IdentityDecision:
    """One face's verdict, with the numbers that produced it."""

    __slots__ = ('swap', 'similarity', 'reference_index', 'track_id', 'reason')

    def __init__(self, swap: bool, similarity: float, reference_index: int,
                 track_id: Optional[int], reason: str) -> None:
        self.swap = swap
        self.similarity = similarity
        self.reference_index = reference_index
        self.track_id = track_id
        self.reason = reason

    @property
    def distance(self) -> float:
        """Same verdict on the app's ``max_face_distance`` scale."""
        return similarity_to_distance(self.similarity)

    def __repr__(self) -> str:
        return (f"IdentityDecision(swap={self.swap} sim={self.similarity:.3f} "
                f"ref={self.reference_index} track={self.track_id} "
                f"why={self.reason})")


class IdentityGate:
    """Strict embedding verification against the selected reference face(s).

    ``references`` is the embedding bank for ONE person - typically several
    captured angles. For multiple target people, hold one gate per person and
    resolve contention 1:1 (the highest-similarity pairing wins, each face
    swapped by at most one person and each person claiming at most one face);
    ``ProcessMgr``'s ``swap_mode == "selected"`` branch already implements that
    assignment, so this class deliberately does not duplicate it.
    """

    def __init__(self,
                 references: Sequence[Embedding],
                 min_similarity: float = DEFAULT_MIN_SIMILARITY,
                 geometry: Optional[GeometryFilter] = None) -> None:
        self.references = [l2_normalize(r) for r in references]
        self.min_similarity = float(min_similarity)
        self.geometry = geometry if geometry is not None else GeometryFilter()

    @classmethod
    def from_max_distance(cls, references: Sequence[Embedding],
                          max_distance: float, **kw) -> 'IdentityGate':
        """Build from the app's ``max_face_distance`` setting."""
        return cls(references,
                   min_similarity=distance_to_similarity(max_distance), **kw)

    def judge(self, face: object,
              frame_shape: Optional[Tuple[int, ...]] = None
              ) -> IdentityDecision:
        """Geometry + embedding verdict for one face. No temporal state."""
        why = self.geometry.reject_reason(face, frame_shape)
        if why is not None:
            return IdentityDecision(False, -1.0, -1, None, f'geometry: {why}')

        emb = getattr(face, 'normed_embedding', None)
        if emb is None:
            emb = getattr(face, 'embedding', None)
        if emb is None:
            return IdentityDecision(False, -1.0, -1, None, 'no embedding')
        if not self.references:
            return IdentityDecision(False, -1.0, -1, None, 'no reference face')

        sim, idx = best_similarity(emb, self.references)
        if sim < self.min_similarity:
            return IdentityDecision(
                False, sim, idx, None,
                f'similarity {sim:.3f} < {self.min_similarity:.3f}')
        return IdentityDecision(True, sim, idx, None, 'identity confirmed')


# -- Temporal -------------------------------------------------------------

class _Track:
    __slots__ = ('id', 'bbox', 'centroid', 'similarity', 'hits', 'misses',
                 'confirmed', 'last_seen', 'reference_index')

    def __init__(self, track_id: int, bbox: BBox, similarity: float,
                 reference_index: int, frame_idx: int) -> None:
        self.id = track_id
        self.bbox = tuple(float(v) for v in np.asarray(bbox).ravel()[:4])
        self.centroid = _centroid(self.bbox)
        self.similarity = float(similarity)
        self.hits = 0
        self.misses = 0
        self.confirmed = False
        self.last_seen = int(frame_idx)
        self.reference_index = int(reference_index)


class FaceTracker:
    """IoU-then-centroid association plus swap/no-swap hysteresis.

    Association is IoU first and centroid distance as the fallback, because IoU
    alone goes to zero the moment a face moves further than its own width in
    one frame - a fast pan, a cut-in, or simply a small face - and a track that
    breaks re-enters as a new one, which resets exactly the evidence the
    hysteresis was accumulating.

    Hysteresis is asymmetric on purpose:

    * a track must clear the similarity threshold on ``confirm_frames``
      consecutive frames before its faces are swapped at all, which is what
      kills single-frame phantom swaps on a bystander;
    * a confirmed track keeps being swapped for up to ``grace_frames`` frames
      of failure, which is what keeps the target's swap alive through a blink,
      a motion blur, a shout, or a hand passing over the face - all of which
      move the embedding without changing who the person is.

    The similarity carried per track is an EMA, so one outlier frame moves the
    decision far less than it moves the raw score. ``smoothing`` is the weight
    on the new observation.
    """

    def __init__(self,
                 gate: IdentityGate,
                 iou_threshold: float = 0.30,
                 centroid_frac: float = 0.75,
                 confirm_frames: int = DEFAULT_CONFIRM_FRAMES,
                 grace_frames: int = DEFAULT_GRACE_FRAMES,
                 smoothing: float = 0.4,
                 max_track_age: int = 30) -> None:
        self.gate = gate
        self.iou_threshold = float(iou_threshold)
        self.centroid_frac = float(centroid_frac)
        self.confirm_frames = int(confirm_frames)
        self.grace_frames = int(grace_frames)
        self.smoothing = float(smoothing)
        self.max_track_age = int(max_track_age)
        self._tracks: Dict[int, _Track] = {}
        self._next_id = 1

    def reset(self) -> None:
        """Call between clips - tracks must not survive a cut or a new video."""
        self._tracks.clear()
        self._next_id = 1

    def _match(self, bbox: BBox, taken: set) -> Optional[_Track]:
        box = tuple(float(v) for v in np.asarray(bbox).ravel()[:4])
        best, best_iou = None, 0.0
        for t in self._tracks.values():
            if t.id in taken:
                continue
            iou = bbox_iou(box, t.bbox)
            if iou > best_iou:
                best, best_iou = t, iou
        if best is not None and best_iou >= self.iou_threshold:
            return best

        # Fallback: nearest centroid within a fraction of the face's own size,
        # so the tolerance scales with the subject instead of being an absolute
        # pixel count that is generous at 1080p and meaningless at 4K.
        cx, cy = _centroid(box)
        reach = self.centroid_frac * max(box[2] - box[0], box[3] - box[1])
        best, best_d = None, reach
        for t in self._tracks.values():
            if t.id in taken:
                continue
            d = float(np.hypot(cx - t.centroid[0], cy - t.centroid[1]))
            if d < best_d:
                best, best_d = t, d
        return best

    def update(self, faces: Sequence[object], frame_idx: int,
               frame_shape: Optional[Tuple[int, ...]] = None
               ) -> List[IdentityDecision]:
        """Decisions for this frame's faces, in the order they were given."""
        decisions: List[IdentityDecision] = []
        taken: set = set()

        for face in faces:
            raw = self.gate.judge(face, frame_shape)
            box = getattr(face, 'bbox', None)
            if box is None:
                decisions.append(raw)
                continue

            track = self._match(box, taken)
            if track is None:
                track = _Track(self._next_id, box, max(raw.similarity, 0.0),
                               raw.reference_index, frame_idx)
                self._tracks[track.id] = track
                self._next_id += 1
            else:
                track.bbox = tuple(float(v) for v in np.asarray(box).ravel()[:4])
                track.centroid = _centroid(track.bbox)
                if raw.similarity >= 0.0:
                    a = self.smoothing
                    track.similarity = (a * raw.similarity
                                        + (1.0 - a) * track.similarity)
                if raw.reference_index >= 0:
                    track.reference_index = raw.reference_index
            taken.add(track.id)
            track.last_seen = int(frame_idx)

            if raw.swap:
                track.hits += 1
                track.misses = 0
                if track.hits >= self.confirm_frames:
                    track.confirmed = True
            else:
                track.misses += 1
                track.hits = 0

            if track.confirmed and track.misses == 0:
                decisions.append(IdentityDecision(
                    True, track.similarity, track.reference_index, track.id,
                    'identity confirmed'))
            elif track.confirmed and track.misses <= self.grace_frames:
                decisions.append(IdentityDecision(
                    True, track.similarity, track.reference_index, track.id,
                    f'held through {track.misses} bad frame(s): {raw.reason}'))
            elif track.confirmed:
                track.confirmed = False
                decisions.append(IdentityDecision(
                    False, track.similarity, track.reference_index, track.id,
                    f'dropped after {track.misses} bad frames: {raw.reason}'))
            elif raw.swap:
                decisions.append(IdentityDecision(
                    False, track.similarity, track.reference_index, track.id,
                    f'awaiting confirmation ({track.hits}/{self.confirm_frames})'))
            else:
                decisions.append(IdentityDecision(
                    False, track.similarity, track.reference_index, track.id,
                    raw.reason))

        # Retire tracks nobody matched for a while, or a busy scene accumulates
        # stale boxes that the centroid fallback can still match against.
        for tid in [t.id for t in self._tracks.values()
                    if frame_idx - t.last_seen > self.max_track_age]:
            self._tracks.pop(tid, None)

        return decisions
