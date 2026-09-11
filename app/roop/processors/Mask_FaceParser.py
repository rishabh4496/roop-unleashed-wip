import os
import numpy as np
import cv2
import onnxruntime
import roop.globals

from roop.typing import Frame
from roop.utilities import resolve_relative_path, conditional_download
from roop import session_pool


# BiSeNet (yakhyo/face-parsing, resnet18) trained on CelebAMask-HQ — 19 classes:
#  0 background  1 skin     2 l_brow  3 r_brow  4 l_eye   5 r_eye   6 eye_g(glasses)
#  7 l_ear       8 r_ear    9 ear_r   10 nose   11 mouth  12 u_lip  13 l_lip
#  14 neck       15 neck_l  16 cloth  17 hair   18 hat
#
# The swap region is the inner face: skin + brows + eyes + nose + lips/mouth.
# Everything else — hair, hat, glasses, ears, neck, cloth, background — is kept
# from the original, so part boundaries (hair fringe over the forehead, glasses,
# ears) are excluded precisely. This is sharper than XSeg's coarse whole-face
# blob at those boundaries; XSeg, however, is trained for arbitrary occluders
# (hands) which this model has no class for — hence offered as a separate engine.
_FACE_CLASSES = np.array([1, 2, 3, 4, 5, 10, 11, 12, 13], dtype=np.int64)

# The 19 classes, grouped the way a person thinks about a face rather than the
# way CelebAMask-HQ numbers them: nobody wants to decide about `l_brow` and
# `r_brow` separately, and a mask that included one brow and not the other
# would be a bug in every case. Each group is include/exclude plus a GROW in
# pixels — grow is what makes the control useful rather than merely present,
# because the model's boundaries are tight and a swap that stops exactly at
# the parsed hairline still shows a seam.
#
# The default set is exactly _FACE_CLASSES, so an untouched install produces
# the same mask it always did, down to the bit (see _region_mask below).
PARSER_REGIONS = {
    'skin':   [1],
    'brows':  [2, 3],
    'eyes':   [4, 5],
    'nose':   [10],
    'mouth':  [11, 12, 13],
    'glasses': [6],
    'ears':   [7, 8, 9],
    'neck':   [14, 15],
    'hair':   [17],
    'hat':    [18],
    'cloth':  [16],
}
PARSER_DEFAULT_ON = ('skin', 'brows', 'eyes', 'nose', 'mouth')
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_MODEL_URL = 'https://github.com/yakhyo/face-parsing/releases/download/weights/resnet18.onnx'
_MODEL_FILE = 'resnet18.onnx'

# Edge-aware feather instead of the blanket Gaussian (see Run).
# ROOP_PARSER_EDGE=0 restores the original isotropic blur exactly.
#
# Every default here is measured, over 36 real aligned crops from facesets/,
# against two numbers: `band` is the fraction of the crop carrying partial
# alpha (lower = a crisper boundary, which is the actual complaint) and
# `edge_ratio` is the mean image gradient ON the mask's 0.5 contour divided by
# the crop's own mean gradient (higher = the boundary sits on real anatomy
# rather than mid-skin).
#
#   XSeg legacy, for reference             band 3.67%   edge_ratio 2.706
#   blanket Gaussian sigma=3 (was here)    band 4.57%   edge_ratio 2.801
#   edge-aware r=.012 sigma=1.5 steep=0    band 8.59%   edge_ratio 3.080
#   edge-aware r=.012 sigma=1.0 steep=.15  band 3.51%   edge_ratio 3.128
#   edge-aware r=.012 sigma=1.0 steep=.25  band 2.97%   edge_ratio 3.132  <- default
#   edge-aware r=.012 sigma=0.8 steep=.40  band 2.31%   edge_ratio 3.141
#
# Two things that table settles. First, the guided filter WITHOUT steepening is
# worse than what it replaces: it aligns the boundary better but spreads it to
# 8.59%, because with nothing to relocate an edge-aware filter is still a blur.
# The steepening is not a refinement of this feature, it is half of it.
#
# Second, radius barely matters (0.012 and 0.02 measured identical) and that is
# the expected result, not a surprise: this mask is inferred at 512 on a 512
# crop, so it is already registered and the filter is snapping a boundary
# rather than moving one. Contrast mask_refine's own default of 0.05, which is
# sized for XSeg's 256 mask landing ~11px off on the same crop. Reusing that
# value here would drag a correct boundary around.
#
# Steepen is capped well below the point where an argmax mask collapses to a
# binary staircase. 0.4 measures tighter still and is left available, but 0.25
# keeps a genuinely anti-aliased transition, which is what stops the seam
# reading as cut out with scissors.
_EDGE_AWARE = os.environ.get('ROOP_PARSER_EDGE', '1').strip().lower() not in ('0', 'off', 'false')
_EDGE_RADIUS = float(os.environ.get('ROOP_PARSER_EDGE_RADIUS', '0.012') or 0.012)
_EDGE_SIGMA = float(os.environ.get('ROOP_PARSER_EDGE_SIGMA', '1.0') or 1.0)
_EDGE_STEEPEN = float(os.environ.get('ROOP_PARSER_EDGE_STEEPEN', '0.25') or 0.25)


def parser_region_settings():
    """(included group names, {group: grow px}) from roop.globals.

    Read per call rather than cached: the settings are pushed onto globals by
    the request that is running, and a preview and a render can disagree
    within one process.
    """
    on = getattr(roop.globals, 'parser_regions', None)
    if not isinstance(on, (list, tuple, set)) or not on:
        on = PARSER_DEFAULT_ON
    on = tuple(g for g in on if g in PARSER_REGIONS)
    grow = getattr(roop.globals, 'parser_region_grow', None)
    grow = grow if isinstance(grow, dict) else {}
    return (on or PARSER_DEFAULT_ON), grow


def _region_mask(labels):
    """Binary swap region from the (512,512) class-id map.

    The fast path is the one that matters: with the default groups and no
    grow, this is `np.isin(labels, _FACE_CLASSES)` — the exact expression this
    used to be — so nobody who never opens the region panel pays for it or
    gets a different mask than they did before.

    Growing is per GROUP, dilated separately before the union. Dilating the
    union instead would be cheaper and wrong: it would push the outer boundary
    of the whole face outward whichever group you meant to grow, so asking for
    a little more mouth would also swallow a ring of background.

    A NEGATIVE grow erodes instead. The two directions are not symmetric in
    what they are for: growing covers a seam the model's tight boundary leaves
    visible, shrinking pulls the swap back off a part the model claimed too
    much of — most usefully `skin`, which is the group that borders hair and
    absorbs a stray fringe pixel when the parting moves. Eroding a group to
    nothing is allowed and simply drops it, which is what asking for -50px on a
    5px-wide part means.
    """
    on, grow = parser_region_settings()
    ids = sorted({c for g in on for c in PARSER_REGIONS[g]})
    active_grow = {g: int(grow.get(g) or 0) for g in on}
    if not any(v != 0 for v in active_grow.values()):
        return np.isin(labels, np.asarray(ids, dtype=np.int64)).astype(np.float32)

    out = np.zeros(labels.shape, dtype=np.uint8)
    for g in on:
        part = np.isin(labels, np.asarray(PARSER_REGIONS[g], dtype=np.int64))
        px = active_grow[g]
        if px != 0:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (abs(px) * 2 + 1,) * 2)
            op = cv2.dilate if px > 0 else cv2.erode
            part = op(part.astype(np.uint8), k, iterations=1)
        out |= part.astype(np.uint8)
    return out.astype(np.float32)


class Mask_FaceParser():
    plugin_options: dict = None

    model = None

    processorname = 'mask_faceparser'
    type = 'mask'

    def __init__(self):
        # Opt-in SessionPool (ROOP_DETMASK_POOL) of independent TensorRT sessions
        # so the mask runs concurrently across worker threads. None → single shared
        # session serialised by the global lock (original safe default).
        self.pool = None

    def Initialize(self, plugin_options: dict):
        if self.plugin_options is not None:
            if self.plugin_options["devicename"] != plugin_options["devicename"]:
                self.Release()

        self.plugin_options = plugin_options
        if self.model is None:
            model_dir = resolve_relative_path('../models')
            conditional_download(model_dir, [_MODEL_URL])
            model_path = os.path.join(model_dir, _MODEL_FILE)
            onnxruntime.set_default_logger_severity(3)

            def _build(_i=0):
                return onnxruntime.InferenceSession(
                    model_path, None, providers=roop.globals.execution_providers)

            self.model = _build()
            self.input_name = self.model.get_inputs()[0].name
            self.output_name = self.model.get_outputs()[0].name

            # replace Mac mps with cpu for the moment
            self.devicename = self.plugin_options["devicename"].replace('mps', 'cpu')

            # Optional multi-session pool: up to N threads run the mask concurrently,
            # each on its own TensorRT context.
            if session_pool.detmask_pooling_enabled():
                n = session_pool.detmask_pool_size()
                extras = [_build(i) for i in range(n - 1)]
                self.pool = session_pool.SessionPool(
                    lambda i, _e=([self.model] + extras): _e[i], n)

    def RunLabels(self, img1):
        """Raw (512,512) per-pixel class-id map, before any region grouping,
        blur or inversion -- split out from Run() so a caller (RealityUX) that
        needs the raw semantic classes, not just the default face/not-face
        mask, doesn't have to re-run inference itself."""
        resized = cv2.resize(img1, (512, 512), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - _MEAN) / _STD
        blob = np.transpose(rgb, (2, 0, 1))[None, ...].astype(np.float32)

        if self.pool is not None:
            with self.pool.lease() as sess:
                ort_outs = sess.run([self.output_name], {self.input_name: blob})
        else:
            ort_outs = self.model.run([self.output_name], {self.input_name: blob})
        return ort_outs[0][0].argmax(0)                             # (512,512) class ids

    def Run(self, img1, keywords: str) -> Frame:
        # img1 is the aligned face crop (BGR uint8). Returned mask matches the
        # other engines' convention: 1.0 = keep ORIGINAL (exclude from swap),
        # 0.0 = use the swapped pixels. process_mask resizes it to the crop.
        labels = self.RunLabels(img1)
        face = _region_mask(labels)                                # 1 inside swap region
        if _EDGE_AWARE:
            # Feather along the IMAGE's own edges instead of isotropically.
            # The blanket Gaussian below is not wrong so much as blind: it
            # spreads the boundary by the same 3 sigma whether it is running
            # along a real hairline (where the feather should be tight, because
            # the two sides are different objects) or across mid-cheek (where a
            # wide ramp is what hides the seam). Measured over 36 real faces,
            # the blanket version widens the partial-alpha band from XSeg's
            # 3.67% of the crop to 4.57% — this engine's boundaries are
            # sharper than XSeg's and the post-processing was giving that back.
            #
            # Note the parameters differ from mask_refine's own defaults, and
            # deliberately: those are tuned for XSeg, whose 256 mask lands ~11px
            # off on a 512 crop and needs a wide radius to RELOCATE. This mask
            # is inferred at 512 on a 512 crop, so it is already registered and
            # only needs snapping — a wide radius would drag a correct boundary
            # around, and steepening an argmax mask that is already hard
            # collapses it to a binary staircase (measured).
            from roop import mask_refine

            face = mask_refine.refine_mask(
                face, img1,
                radius_frac=_EDGE_RADIUS, steepen_strength=_EDGE_STEEPEN)
            face = cv2.GaussianBlur(face, (0, 0), sigmaX=_EDGE_SIGMA)
        else:
            # Soften edges so the blend matches the smooth XSeg output.
            face = cv2.GaussianBlur(face, (0, 0), sigmaX=3)
        face = np.clip(face, 0.0, 1.0)
        # invert: keep original everywhere outside the face region
        return (1.0 - face).astype(np.float32)

    def Release(self):
        if self.pool is not None:
            self.pool.release()
            self.pool = None
        del self.model
        self.model = None
