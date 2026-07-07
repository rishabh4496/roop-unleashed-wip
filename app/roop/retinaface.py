"""RetinaFace (retinaface_10g / retinaface_resnet50) — alternate face detector engines.

Same hybrid pattern as YOLOFace (roop/yoloface.py): this model only detects
(bbox + 5 keypoints); face_util pairs it with buffalo_l's aux models for
identity + landmarks, so the rest of the pipeline sees the exact same Face
objects. RetinaFace has higher recall than SCRFD on hard poses/lighting,
which reduces detection dropouts (swap blink) at the source.

The ONNX is FaceFusion's retinaface export. Its output layout is the
standard insightface distance-regression head, and insightface's own
ModelRouter recognises it (>= 5 outputs -> RetinaFace class), so we reuse
insightface's proven anchor decode instead of reimplementing it.
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

def distance2bbox(points, distance, max_shape=None):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)

def distance2kps(points, distance, max_shape=None):
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i%2] + distance[:, i]
        py = points[:, i%2+1] + distance[:, i+1]
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)

class RetinaFace3Output:
    def __init__(self, model_file=None, session=None):
        self.model_file = model_file
        self.session = session
        self.taskname = 'detection'
        self.center_cache = {}
        self.nms_thresh = 0.4
        self.det_thresh = 0.5
        self.input_mean = 127.5
        self.input_std = 128.0
        
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

        blob = cv2.dnn.blobFromImage(det_img, 1.0/self.input_std, input_size, (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
        net_outs = self.session.run(self.output_names, {self.input_name : blob})
        
        loc = net_outs[0][0]
        conf = net_outs[1][0]
        landms = net_outs[2][0]
        
        scores = softmax(conf)[:, 1]
        
        input_height = blob.shape[2]
        input_width = blob.shape[3]
        feat_stride_fpn = [8, 16, 32]
        anchor_centers_list = []
        
        for stride in feat_stride_fpn:
            height = input_height // stride
            width = input_width // stride
            anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
            anchor_centers = (anchor_centers * stride).reshape((-1, 2))
            anchor_centers = np.stack([anchor_centers]*2, axis=1).reshape((-1, 2))
            anchor_centers_list.append(anchor_centers)
            
        anchor_centers = np.vstack(anchor_centers_list)
        
        loc_scaled = np.zeros_like(loc)
        landms_scaled = np.zeros_like(landms)
        idx = 0
        for stride in feat_stride_fpn:
            height = input_height // stride
            width = input_width // stride
            count = height * width * 2
            loc_scaled[idx:idx+count] = loc[idx:idx+count] * stride
            landms_scaled[idx:idx+count] = landms[idx:idx+count] * stride
            idx += count
            
        bboxes = distance2bbox(anchor_centers, loc_scaled) / det_scale
        kpss = distance2kps(anchor_centers, landms_scaled) / det_scale
        kpss = kpss.reshape((kpss.shape[0], -1, 2))
        
        pos_inds = np.where(scores >= self.det_thresh)[0]
        pos_scores = scores[pos_inds]
        pos_bboxes = bboxes[pos_inds]
        pos_kpss = kpss[pos_inds]
        
        order = pos_scores.argsort()[::-1]
        pos_scores = pos_scores[order]
        pos_bboxes = pos_bboxes[order]
        pos_kpss = pos_kpss[order]
        
        pre_det = np.hstack((pos_bboxes, pos_scores[:, np.newaxis])).astype(np.float32, copy=False)
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
