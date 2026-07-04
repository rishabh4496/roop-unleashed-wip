"""YOLOFace (yoloface_8n) detector — an alternate face detector engine.

insightface's default SCRFD detector is fast and accurate on frontal faces but
weaker on steep profiles / partially-occluded faces. YOLOFace tends to catch
those better. It only produces bbox + 5 keypoints (not identity/landmarks), so
face_util pairs it with the existing buffalo_l aux models (recognition + 106/68
landmarks) to build full Face objects — see face_util._hybrid_yolo_faces.

Output decoding mirrors FaceFusion's yoloface path: the model emits (1, 20, N)
= 4 bbox (cx,cy,w,h) + 1 score + 15 (5 kps × x,y,score) in a 640-canvas the
downscaled frame is pasted into top-left of; canvas coords scale back to the
original by the resize ratio.
"""

import os
import threading

import cv2
import numpy as np
import onnxruntime

import roop.globals
from roop.utilities import resolve_relative_path, conditional_download

_YOLO_URL = "https://huggingface.co/facefusion/models-3.0.0/resolve/main/yoloface_8n.onnx"

_detector = None
_detector_lock = threading.Lock()


def _nms(boxes, scores, iou_thresh=0.4):
    """Plain IoU non-max suppression. boxes: (N,4) x1y1x2y2."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thresh]
    return keep


class YoloFaceDetector:
    def __init__(self, providers):
        model_dir = resolve_relative_path('../models')
        conditional_download(model_dir, [_YOLO_URL])
        model_path = os.path.join(model_dir, "yoloface_8n.onnx")
        so = onnxruntime.SessionOptions()
        self.session = onnxruntime.InferenceSession(model_path, so, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        # A single session shared across worker threads → guard run() so
        # concurrent detect calls don't corrupt the context. Detection is the
        # cheap stage for yoloface_8n, so serialising it is acceptable.
        self._lock = threading.Lock()

    def detect(self, frame, det_size=640, det_thresh=0.5):
        h0, w0 = frame.shape[:2]
        scale = min(det_size / w0, det_size / h0)
        rw, rh = int(round(w0 * scale)), int(round(h0 * scale))
        resized = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((det_size, det_size, 3), dtype=frame.dtype)
        canvas[:rh, :rw, :] = resized
        blob = ((canvas.astype(np.float32) - 127.5) / 128.0).transpose(2, 0, 1)[np.newaxis]

        with self._lock:
            out = self.session.run(None, {self.input_name: blob})[0]

        det = np.squeeze(out).T                     # (N, 20)
        if det.ndim != 2 or det.shape[1] < 20:
            return np.zeros((0, 5), np.float32), np.zeros((0, 5, 2), np.float32)
        bbox_raw = det[:, 0:4]
        score_raw = det[:, 4]
        kps_raw = det[:, 5:20]
        keep = score_raw > det_thresh
        bbox_raw, score_raw, kps_raw = bbox_raw[keep], score_raw[keep], kps_raw[keep]
        if bbox_raw.shape[0] == 0:
            return np.zeros((0, 5), np.float32), np.zeros((0, 5, 2), np.float32)

        # cx,cy,w,h (canvas coords) → x1,y1,x2,y2 (original coords)
        inv = 1.0 / scale
        cx, cy, w, h = bbox_raw[:, 0], bbox_raw[:, 1], bbox_raw[:, 2], bbox_raw[:, 3]
        x1 = (cx - w / 2) * inv
        y1 = (cy - h / 2) * inv
        x2 = (cx + w / 2) * inv
        y2 = (cy + h / 2) * inv
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        kps = kps_raw.reshape(-1, 5, 3)[:, :, :2] * inv

        idx = _nms(boxes, score_raw, iou_thresh=0.4)
        boxes, score_raw, kps = boxes[idx], score_raw[idx], kps[idx]

        # Clamp to frame bounds.
        boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, w0 - 1)
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, h0 - 1)

        bboxes = np.concatenate([boxes, score_raw[:, None]], axis=1).astype(np.float32)
        return bboxes, kps.astype(np.float32)


def get_detector():
    """Lazily build the shared YOLOFace detector, honouring force_cpu and the
    current execution providers."""
    global _detector
    if _detector is not None:
        return _detector
    with _detector_lock:
        if _detector is None:
            if roop.globals.CFG.force_cpu:
                providers = ["CPUExecutionProvider"]
            else:
                providers = roop.globals.execution_providers
            _detector = YoloFaceDetector(providers)
    return _detector


def release_detector():
    global _detector
    with _detector_lock:
        _detector = None
