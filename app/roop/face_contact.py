"""What the DETECTOR and the RECOGNISER do when two faces touch.

Two separate failures, both visible only while two people are close enough to
interact, both measured on `G:/pinokio/roop-keep/sample1.mp4` (two heads that
meet in profile and kiss). `roop/face_overlap.py` is the third member of the
family and handles the one that comes after — who owns a pixel once both faces
have been swapped.

──────────────────────────────────────────────────────────────────────────────
1. THE JUNCTION IS DETECTED AS A THIRD FACE  (`merged_indices`)

Where two faces meet, the detector regularly fires a third box straddling the
join: the left person's mouth and chin plus the right person's nose and mouth,
which really does look like one frontal face. On the measured clip it appears
on 346 of the 1074 frames where two faces are visible, and at det_score up to
0.99 — it is not a weak detection that a confidence floor would remove.

NMS cannot: `roop/nms.py` deliberately protects boxes that overlap but are not
concentric, because that is exactly what two touching faces look like, and the
merged box is offset from both of its parents.

Nothing downstream survives it either. The phantom gets a track of its own, its
embedding is a chimera of two people so it lands wherever it likes among the
identity gates, it competes for a source (one source per frame, so it can take
the real face's), it competes for pixel ownership in face_overlap, and it comes
and goes frame to frame — which is a flicker on BOTH real faces.

What it IS, geometrically, is a box that lies BETWEEN two other detections and
is almost entirely covered by them. A real face in a row of faces is not: its
neighbours sit to one side of it and leave most of it uncovered. That is the
whole rule, plus a size floor so a small face genuinely standing between two
nearer ones is not mistaken for a junction.

──────────────────────────────────────────────────────────────────────────────
2. THE RECOGNITION CROP CONTAINS THE OTHER PERSON  (`crop_contamination`)

The embedding is not read off the detection box. ArcFace fits the 5 keypoints
to a frontal template, and for a near-profile face the two eyes are only a few
pixels apart, so that fit resolves to a wide, rotated crop reaching well past
the nose — about 1.7 box widths on the measured clip. While the faces are apart
that extra area is background and harmless. Once they touch it is the other
person, and the two crops converge onto very nearly the SAME picture.

The identities then converge with them. Measured, distance from each face to
its own reference against the fraction of its crop covered by the other face:

    crop covered   n     d(own)   d(other)   read as the wrong person
      0.0 - 0.2    27     0.05      1.06      0 %
      0.2 - 0.3    33     0.43      1.02      0 %
      0.3 - 0.4    43     0.40      0.99      2 %
      0.4 - 0.6    83     0.63      0.92     14 %

Past ~0.4 the mean distance to the RIGHT person is over the 0.6 gate the track
assignment uses, and one face in seven is closer to the other person than to
itself. That single number explains all three reported symptoms: the swap
refused (over threshold), the swap applied from the wrong faceset (the other
person is nearer), and the track broken into fragments and left with no source
at all (association is embedding-gated, so contamination splits a person into
two tracks and the fragment's mean is built from the frames that split it).

Masking the neighbour out before recognition was tried and is NOT the fix: a
flat fill inside the crop destroys the embedding outright (d(own) 0.18 -> 0.74
on frames where the plain crop was still correct). The pixels cannot be removed
because they are inside the face's own aligned crop.

So the embedding is not repaired, it is DISBELIEVED. `crop_contamination`
returns the fraction per face; callers that decide identity from it — track
association, the track's mean, the per-frame matcher — skip a face whose crop
is mostly somebody else and fall back to position, which is the evidence that
survives contact. This is the same reasoning `_stitch_tracks` already uses when
appearance collapses on a turned head.
"""

import os

import cv2
import numpy as np


def _env_float(name, default):
    try:
        v = float(os.environ.get(name, ''))
    except ValueError:
        return default
    return v


# ── 1. merged ("junction") detections ────────────────────────────────────────

# Master switch: ROOP_FACE_MERGE=0 keeps every detection, as before.
MERGE_SUPPRESS = os.environ.get('ROOP_FACE_MERGE', '1') != '0'

# Fraction of the candidate's own area that its two neighbours must cover
# between them. The merged box is by construction made OF the two faces, so it
# measures high; a real face has at least the far side of its own head showing.
MERGE_COVER = _env_float('ROOP_FACE_MERGE_COVER', 0.80)

# Minimum overlap with EACH neighbour, as a fraction of the candidate's area.
# Without it a box merely enclosed by one large neighbour would qualify.
MERGE_EACH = _env_float('ROOP_FACE_MERGE_EACH', 0.15)

# Size floor against the smaller neighbour. A junction box spans two faces and
# is never much smaller than either; a distant head framed between two nearer
# ones is, and this is what keeps it.
MERGE_SIZE = _env_float('ROOP_FACE_MERGE_SIZE', 0.60)

# The two neighbours must be two different faces rather than one face detected
# twice. Same measurement, and deliberately the same value, as
# face_overlap._DUP_CONCENTRIC: centre separation over the smaller radius.
MERGE_PAIR_SEP = _env_float('ROOP_FACE_MERGE_PAIR_SEP', 0.35)

# How far along the a->b axis the candidate's centre must sit to count as
# "between" them. 0.15..0.85 excludes a box sitting on top of either end.
MERGE_BETWEEN = _env_float('ROOP_FACE_MERGE_BETWEEN', 0.15)


def _centre(b):
    return ((b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5)


def _radius(b):
    return 0.5 * max(b[2] - b[0], b[3] - b[1])


def _inter_area(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return float(w * h) if (w > 0 and h > 0) else 0.0


def _area(b):
    return max(1e-6, float((b[2] - b[0]) * (b[3] - b[1])))


def _union_cover(m, a, b):
    """Fraction of box `m` covered by `a` union `b`.

    Inclusion-exclusion rather than a raster: the union of two rectangles
    clipped to a third is exact in closed form, and the raster version of this
    was the expensive part of the first cut.
    """
    ia, ib = _inter_area(m, a), _inter_area(m, b)
    if ia <= 0.0 and ib <= 0.0:
        return 0.0
    # Overlap of a and b, itself clipped to m.
    ab = (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))
    iab = _inter_area(m, ab) if (ab[2] > ab[0] and ab[3] > ab[1]) else 0.0
    return (ia + ib - iab) / _area(m)


def _between(m, a, b):
    """Does m's centre lie between a's and b's, along the a->b axis?

    The geometric content of "straddles the junction". Two faces adjacent to a
    third both sit on the SAME side of it and fail this; the parents of a merged
    box sit one on each side by construction.
    """
    ca, cb, cm = _centre(a), _centre(b), _centre(m)
    vx, vy = cb[0] - ca[0], cb[1] - ca[1]
    n2 = vx * vx + vy * vy
    if n2 < 1e-9:
        return False
    t = ((cm[0] - ca[0]) * vx + (cm[1] - ca[1]) * vy) / n2
    return MERGE_BETWEEN < t < (1.0 - MERGE_BETWEEN)


def _min_side(b):
    return min(b[2] - b[0], b[3] - b[1])


def merged_indices(boxes):
    """Indices of `boxes` that are the junction of two of the others.

    `boxes` is any sequence of (x0, y0, x1, y1). Pure geometry — no models, no
    frame — so the rule can be pinned in a unit test against hand-built
    layouts, which is the only way to check the cases the sample clip does not
    contain (three faces in a row, a small face between two large ones).
    """
    n = len(boxes)
    if n < 3 or not MERGE_SUPPRESS:
        return []
    boxes = [tuple(float(v) for v in b) for b in boxes]
    out = []
    for k in range(n):
        m = boxes[k]
        ma = _area(m)
        hit = False
        for i in range(n):
            if i == k or hit:
                continue
            for j in range(i + 1, n):
                if j == k:
                    continue
                a, b = boxes[i], boxes[j]
                # Two different faces, not one face detected twice.
                ca, cb = _centre(a), _centre(b)
                sep = float(np.hypot(ca[0] - cb[0], ca[1] - cb[1]))
                if sep <= MERGE_PAIR_SEP * min(_radius(a), _radius(b)):
                    continue
                if not _between(m, a, b):
                    continue
                if _inter_area(m, a) / ma < MERGE_EACH:
                    continue
                if _inter_area(m, b) / ma < MERGE_EACH:
                    continue
                if _min_side(m) < MERGE_SIZE * min(_min_side(a), _min_side(b)):
                    continue
                if _union_cover(m, a, b) < MERGE_COVER:
                    continue
                out.append(k)
                hit = True
                break
    return out


def suppress_merged(faces):
    """Drop detections that are the junction of two other faces.

    Returns the surviving list (the same objects, order preserved) and the
    number removed. A face without a usable bbox is never dropped.
    """
    if not MERGE_SUPPRESS or len(faces) < 3:
        return faces, 0
    try:
        boxes = [tuple(float(v) for v in f.bbox) for f in faces]
    except Exception:
        return faces, 0
    drop = set(merged_indices(boxes))
    if not drop:
        return faces, 0
    return [f for i, f in enumerate(faces) if i not in drop], len(drop)


# ── 2. contaminated recognition crops ────────────────────────────────────────

# The ArcFace destination template, verbatim from insightface's face_align, so
# the quad computed here is the crop the recogniser actually took rather than an
# approximation of it. Copied rather than imported: this module must stay
# importable (and testable) without the model stack.
_ARCFACE_DST = np.array([[38.2946, 51.6963], [73.5318, 51.5014],
                         [56.0252, 71.7366], [41.5493, 92.3655],
                         [70.7299, 92.2041]], dtype=np.float64)

# Fraction of a face's recognition crop that another face may cover before the
# embedding stops being used to decide identity. The knee in the measured table
# in the module docstring is at 0.4 — that is where mean d(own) crosses the 0.6
# track-assignment gate — and 0.35 leaves a little room below it.
# ROOP_EMB_CONTAM=0 disables the whole mechanism.
CONTAM_MAX = _env_float('ROOP_EMB_CONTAM', 0.35)


def _crop_quad(kps, size=112.0):
    """The four corners of the ArcFace crop, in frame coordinates.

    A least-squares SIMILARITY fit (rotation, uniform scale, translation) of the
    5 keypoints onto the template — the same estimator skimage's
    SimilarityTransform uses, written out here so this module does not need
    skimage and so the maths is visible: the whole point of the function is that
    for a profile face this transform resolves to something much larger than the
    detection box, and that is worth being able to read.
    """
    src = np.asarray(kps, dtype=np.float64).reshape(5, 2)
    dst = _ARCFACE_DST * (size / 112.0)
    sm, dm = src.mean(axis=0), dst.mean(axis=0)
    sc, dc = src - sm, dst - dm
    var = float((sc ** 2).sum())
    if var < 1e-9:
        return None
    # Umeyama for the 2-D similarity case: the optimal rotation comes from the
    # cross-covariance, the scale from the ratio of variances.
    cov = dc.T @ sc / 5.0
    u, s, vt = np.linalg.svd(cov)
    d = np.array([1.0, 1.0])
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[1] = -1.0
    r = u @ np.diag(d) @ vt
    scale = float(s @ d) / (var / 5.0)
    if not np.isfinite(scale) or scale < 1e-9:
        return None
    # Forward transform maps frame -> crop; we want the crop's corners in the
    # frame, so invert it.
    a = scale * r
    t = dm - a @ sm
    inv = np.linalg.inv(a)
    corners = np.array([[0.0, 0.0], [size, 0.0], [size, size], [0.0, size]])
    return ((corners - t) @ inv.T).astype(np.float32)


def _quad_box_overlap(quad, box):
    """Fraction of `quad`'s area covered by axis-aligned `box`."""
    rect = np.array([[box[0], box[1]], [box[2], box[1]],
                     [box[2], box[3]], [box[0], box[3]]], dtype=np.float32)
    try:
        inter, _ = cv2.intersectConvexConvex(quad, rect, True)
    except Exception:
        return 0.0
    area = float(cv2.contourArea(quad))
    return float(inter) / area if area > 1e-6 else 0.0


def crop_contamination(faces):
    """Per face, the largest fraction of its recognition crop owned by another.

    Returns a list of floats, one per face, 0.0 when nothing overlaps or the
    geometry could not be built. Cheap: a 2x2 solve and one convex-polygon
    intersection per pair, only for pairs whose boxes are anywhere near.
    """
    n = len(faces)
    if n < 2:
        return [0.0] * n
    quads, boxes = [], []
    for f in faces:
        try:
            boxes.append(tuple(float(v) for v in f.bbox))
        except Exception:
            boxes.append(None)
        kps = getattr(f, 'kps', None)
        quads.append(_crop_quad(kps) if kps is not None else None)
    out = [0.0] * n
    for i in range(n):
        if quads[i] is None:
            continue
        for j in range(n):
            if i == j or boxes[j] is None:
                continue
            out[i] = max(out[i], _quad_box_overlap(quads[i], boxes[j]))
    return out


def annotate(faces):
    """Suppress junction detections, then tag what is left with how much of each
    recognition crop belongs to somebody else.

    Called once per detection, so every consumer downstream — the tracking scan,
    the identity-lock matcher, the per-frame matcher — reads the same verdict
    instead of each re-deriving it from a different subset of the faces.
    """
    faces, dropped = suppress_merged(faces)
    if faces and CONTAM_MAX > 0:
        for f, c in zip(faces, crop_contamination(faces)):
            try:
                f['_emb_contam'] = float(c)
            except Exception:
                pass
    return faces, dropped


def unreliable(face):
    """Is this face's embedding too contaminated to decide identity with?"""
    if CONTAM_MAX <= 0:
        return False
    try:
        return float(face.get('_emb_contam', 0.0)) >= CONTAM_MAX
    except Exception:
        return False
