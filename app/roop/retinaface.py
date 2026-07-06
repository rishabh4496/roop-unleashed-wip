"""RetinaFace (retinaface_10g) — alternate face detector engine.

Same hybrid pattern as YOLOFace (roop/yoloface.py): this model only detects
(bbox + 5 keypoints); face_util pairs it with buffalo_l's aux models for
identity + landmarks, so the rest of the pipeline sees the exact same Face
objects. RetinaFace-10G has higher recall than SCRFD on hard poses/lighting,
which reduces detection dropouts (swap blink) at the source.

The ONNX is FaceFusion's retinaface_10g export. Its output layout is the
standard insightface distance-regression head, and insightface's own
ModelRouter recognises it (>= 5 outputs -> RetinaFace class), so we reuse
insightface's proven anchor decode instead of reimplementing it.
"""

import os
import threading

import roop.globals
from roop.utilities import resolve_relative_path, conditional_download

_MODEL_URL = "https://huggingface.co/facefusion/models-3.0.0/resolve/main/retinaface_10g.onnx"
_MODEL_FILE = "retinaface_10g.onnx"

_detector = None
_detector_lock = threading.Lock()
# One shared session; insightface detector objects are not thread-safe (they
# mutate det_thresh/center cache), so detect calls are serialised. Detection
# is cheap relative to the swap, so this is acceptable — same as YOLOFace.
_detect_lock = threading.Lock()


def get_detector():
    """Lazily build the shared RetinaFace detector, honouring force_cpu and the
    current execution providers."""
    global _detector
    if _detector is not None:
        return _detector
    with _detector_lock:
        if _detector is None:
            from insightface.model_zoo import get_model
            model_dir = resolve_relative_path('../models')
            conditional_download(model_dir, [_MODEL_URL])
            model_path = os.path.join(model_dir, _MODEL_FILE)
            if roop.globals.CFG is not None and roop.globals.CFG.force_cpu:
                providers = ["CPUExecutionProvider"]
            else:
                providers = roop.globals.execution_providers
            det = get_model(model_path, providers=providers)
            if det is None or not hasattr(det, 'detect'):
                raise RuntimeError('insightface could not route retinaface_10g.onnx to a detector')
            _detector = det
    return _detector


def detect(frame, det_size=640, det_thresh=0.5):
    """Run detection; returns (bboxes (N,5) incl. score, kpss (N,5,2)) in
    original frame coordinates — the same contract as yoloface.detect."""
    det = get_detector()
    with _detect_lock:
        det.det_thresh = det_thresh
        return det.detect(frame, input_size=(int(det_size), int(det_size)), max_num=0)


def release_detector():
    global _detector
    with _detector_lock:
        _detector = None
