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

import roop.globals
from roop.utilities import resolve_relative_path, conditional_download

_MODEL_10G_URL = "https://huggingface.co/facefusion/models-3.0.0/resolve/main/retinaface_10g.onnx"
_MODEL_R50_URL = "https://github.com/facefusion/models/releases/download/v1/retinaface_resnet50.onnx"

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
        file = "retinaface_resnet50.onnx"
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

        from insightface.model_zoo import get_model
        model_dir = resolve_relative_path('../models')
        conditional_download(model_dir, [url])
        model_path = os.path.join(model_dir, file)
        if roop.globals.CFG is not None and roop.globals.CFG.force_cpu:
            providers = ["CPUExecutionProvider"]
        else:
            providers = roop.globals.execution_providers
        det = get_model(model_path, providers=providers)
        if det is None or not hasattr(det, 'detect'):
            raise RuntimeError(f'insightface could not route {file} to a detector')
        
        if model_type == 'r50':
            _detector_r50 = det
        else:
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
