"""RetinaFace (retinaface_10g / retinaface_resnet50) — alternate face detector engines.

Same hybrid pattern as YOLOFace (roop/yoloface.py): this model only detects
(bbox + 5 keypoints); face_util pairs it with buffalo_l's aux models for
identity + landmarks, so the rest of the pipeline sees the exact same Face
objects. RetinaFace has higher recall than SCRFD on hard poses/lighting,
which reduces detection dropouts (swap blink) at the source.

Two distinct model families are supported, each with its own output/anchor
convention:
  - '10g': FaceFusion's retinaface_10g.onnx. Same anchor-free, distance-
    regression head as insightface's SCRFD, so it's routed straight through
    insightface's own ModelRouter/RetinaFace class, which already decodes it
    correctly.
  - 'r50': a biubug6/Pytorch_Retinaface ResNet50 export. This is a classic
    *anchor-based* detector (predefined box sizes per stride, decoded via
    prior boxes + variance), a completely different convention from
    '10g'/SCRFD, so it needs its own PriorBox + decode implementation
    (RetinaFace3Output below) and its own preprocessing (mean-subtract only,
    BGR, no std scaling — not the 127.5/128 SCRFD convention).
"""

import os
import threading
import cv2
import numpy as np
import onnxruntime
import roop.globals
from roop.utilities import resolve_relative_path, conditional_download

def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis]
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]
    return e_x / div


# biubug6/Pytorch_Retinaface PriorBox config (ResNet50 variant).
_R50_MIN_SIZES = ((16, 32), (64, 128), (256, 512))
_R50_STEPS = (8, 16, 32)
_R50_VARIANCE = (0.1, 0.2)

_prior_cache = {}


def _generate_priors(image_size, min_sizes_cfg=_R50_MIN_SIZES, steps=_R50_STEPS):
    """Anchor-based PriorBox generation, matching biubug6/Pytorch_Retinaface's
    box_utils.PriorBox exactly (grid + anchor order must match how the network
    was trained). Result is normalized (cx, cy, w, h) in [0, 1]; cached per
    image_size since it's static for a fixed input resolution."""
    cached = _prior_cache.get(image_size)
    if cached is not None:
        return cached

    h, w = image_size
    anchors = []
    for k, step in enumerate(steps):
        fh = int(np.ceil(h / step))
        fw = int(np.ceil(w / step))
        min_sizes = min_sizes_cfg[k]
        num_anchors = len(min_sizes)

        cy = (np.arange(fh, dtype=np.float32) + 0.5) * step / h
        cx = (np.arange(fw, dtype=np.float32) + 0.5) * step / w
        grid_cy, grid_cx = np.meshgrid(cy, cx, indexing='ij')
        grid_cy = grid_cy.reshape(-1)
        grid_cx = grid_cx.reshape(-1)

        sizes = np.array(min_sizes, dtype=np.float32)
        cx_rep = np.repeat(grid_cx, num_anchors)
        cy_rep = np.repeat(grid_cy, num_anchors)
        skx = np.tile(sizes / w, grid_cx.shape[0])
        sky = np.tile(sizes / h, grid_cy.shape[0])
        anchors.append(np.stack([cx_rep, cy_rep, skx, sky], axis=1))

    priors = np.concatenate(anchors, axis=0).astype(np.float32)
    _prior_cache[image_size] = priors
    return priors


def _decode_boxes(loc, priors, variances=_R50_VARIANCE):
    """box_utils.decode from biubug6/Pytorch_Retinaface. Returns normalized
    (x1, y1, x2, y2) in [0, 1]."""
    boxes = np.concatenate((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])
    ), axis=1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def _decode_landmarks(pre, priors, variances=_R50_VARIANCE):
    """box_utils.decode_landm counterpart — 5 (x, y) points, normalized."""
    return np.concatenate([
        priors[:, :2] + pre[:, 2 * i:2 * i + 2] * variances[0] * priors[:, 2:]
        for i in range(5)
    ], axis=1)


class RetinaFace3Output:
    """biubug6/Pytorch_Retinaface ResNet50 decoder — anchor-based, NOT the
    insightface/SCRFD anchor-free convention. See module docstring."""

    def __init__(self, model_file=None, session=None):
        self.model_file = model_file
        self.session = session
        self.taskname = 'detection'
        self.nms_thresh = 0.4
        self.det_thresh = 0.5
        # biubug6 preprocessing: BGR, mean-subtract only, no std scaling.
        self.input_mean = (104.0, 117.0, 123.0)

        if self.session is None:
            self.session = onnxruntime.InferenceSession(self.model_file, providers=['CPUExecutionProvider'])

        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        if isinstance(input_shape[2], str):
            self.input_size = None
        else:
            self.input_size = tuple(input_shape[2:4][::-1])
        self.input_name = input_cfg.name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def prepare(self, ctx_id, **kwargs):
        if ctx_id < 0:
            self.session.set_providers(['CPUExecutionProvider'])
        if 'nms_thresh' in kwargs:
            self.nms_thresh = kwargs['nms_thresh']
        if 'det_thresh' in kwargs:
            self.det_thresh = kwargs['det_thresh']
        if 'input_size' in kwargs:
            self.input_size = kwargs['input_size']

    def detect(self, img, input_size=None, max_num=0, metric='default'):
        input_size = self.input_size if input_size is None else input_size
        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img

        blob = cv2.dnn.blobFromImage(det_img, 1.0, input_size, self.input_mean, swapRB=False)
        net_outs = self.session.run(self.output_names, {self.input_name : blob})

        loc = net_outs[0][0]
        conf = net_outs[1][0]
        landms = net_outs[2][0]

        # The nakamura196 r50 export already applies softmax inside the graph:
        # conf rows are probabilities summing to 1 (face score in column 1),
        # empirically peaking at ~0.999 on a clear face. Do NOT re-softmax it —
        # that squashes a true 0.999 down to ~0.73, which silently falls below
        # det_thresh (e.g. 0.8) and drops every detection. Guard defensively in
        # case a future export emits raw logits instead (rows not ~1).
        if abs(float(conf[0].sum()) - 1.0) < 1e-3:
            scores = conf[:, 1]
        else:
            scores = softmax(conf)[:, 1]
        priors = _generate_priors((input_size[1], input_size[0]))

        boxes = _decode_boxes(loc, priors)
        boxes[:, 0::2] *= input_size[0]
        boxes[:, 1::2] *= input_size[1]
        boxes /= det_scale

        kpss = _decode_landmarks(landms, priors)
        kpss[:, 0::2] *= input_size[0]
        kpss[:, 1::2] *= input_size[1]
        kpss /= det_scale
        kpss = kpss.reshape((kpss.shape[0], -1, 2))

        pos_inds = np.where(scores >= self.det_thresh)[0]
        pos_scores = scores[pos_inds]
        pos_boxes = boxes[pos_inds]
        pos_kpss = kpss[pos_inds]

        order = pos_scores.argsort()[::-1]
        pos_scores = pos_scores[order]
        pos_boxes = pos_boxes[order]
        pos_kpss = pos_kpss[order]

        pre_det = np.hstack((pos_boxes, pos_scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = self.nms(pre_det)

        det = pre_det[keep, :]
        kpss = pos_kpss[keep, :, :]

        return det, kpss

    def nms(self, dets):
        thresh = self.nms_thresh
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep


_MODEL_10G_URL = "https://huggingface.co/facefusion/models-3.0.0/resolve/main/retinaface_10g.onnx"
_MODEL_R50_URL = "https://huggingface.co/nakamura196/retinaface-r50-onnx/resolve/main/retinaface_r50.onnx"

_detector_10g = None
_detector_r50 = None
_detector_lock = threading.Lock()
# One shared session; insightface detector objects are not thread-safe (they
# mutate det_thresh/center cache), so detect calls are serialised. Detection
# is cheap relative to the swap, so this is acceptable — same as YOLOFace.
_detect_lock = threading.Lock()


def get_detector(model_type='10g'):
    """Lazily build the shared RetinaFace detector, honouring force_cpu and the
    current execution providers."""
    global _detector_10g, _detector_r50
    if model_type == 'r50':
        if _detector_r50 is not None:
            return _detector_r50
        url = _MODEL_R50_URL
        file = "retinaface_r50.onnx"
    else:
        if _detector_10g is not None:
            return _detector_10g
        url = _MODEL_10G_URL
        file = "retinaface_10g.onnx"

    with _detector_lock:
        if model_type == 'r50':
            if _detector_r50 is not None:
                return _detector_r50
        else:
            if _detector_10g is not None:
                return _detector_10g

        model_dir = resolve_relative_path('../models')
        conditional_download(model_dir, [url])
        model_path = os.path.join(model_dir, file)

        if roop.globals.CFG is not None and roop.globals.CFG.force_cpu:
            providers = ["CPUExecutionProvider"]
        else:
            providers = roop.globals.execution_providers

        if model_type == 'r50':
            session = onnxruntime.InferenceSession(model_path, providers=providers)
            det = RetinaFace3Output(model_path, session=session)
            _detector_r50 = det
        else:
            from insightface.model_zoo import get_model
            det = get_model(model_path, providers=providers)
            if det is None or not hasattr(det, 'detect'):
                raise RuntimeError(f'insightface could not route {file} to a detector')
            _detector_10g = det
            
    return det


def detect(frame, det_size=640, det_thresh=0.5, model_type='10g'):
    """Run detection; returns (bboxes (N,5) incl. score, kpss (N,5,2)) in
    original frame coordinates — the same contract as yoloface.detect."""
    det = get_detector(model_type)
    nms_thresh = getattr(roop.globals, 'face_detector_nms', 0.40)
    with _detect_lock:
        det.det_thresh = det_thresh
        if hasattr(det, 'nms_thresh'):
            det.nms_thresh = nms_thresh
        return det.detect(frame, input_size=(int(det_size), int(det_size)), max_num=0)


def release_detector():
    global _detector_10g, _detector_r50
    with _detector_lock:
        _detector_10g = None
        _detector_r50 = None
