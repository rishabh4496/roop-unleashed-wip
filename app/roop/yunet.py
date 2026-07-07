import os
import cv2
import numpy as np
import threading
import roop.globals
from roop.utilities import resolve_relative_path, conditional_download

_MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
_MODEL_FILE = "face_detection_yunet_2023mar.onnx"

_detector = None
_detector_lock = threading.Lock()
_detect_lock = threading.Lock()


def get_detector():
    """Lazily load OpenCV cv2.FaceDetectorYN detector, downloading weights if necessary."""
    global _detector
    if _detector is not None:
        return _detector
    with _detector_lock:
        if _detector is None:
            model_dir = resolve_relative_path('../models')
            conditional_download(model_dir, [_MODEL_URL])
            model_path = os.path.join(model_dir, _MODEL_FILE)
            # Create FaceDetectorYN instance. Default size (320, 320) will be dynamically adjusted during detect.
            det = cv2.FaceDetectorYN.create(
                model=model_path,
                config="",
                inputSize=(320, 320),
                scoreThreshold=0.6,
                nmsThreshold=0.3,
                topK=5000
            )
            _detector = det
    return _detector


def detect(frame, det_size=640, det_thresh=0.5):
    """Run detection; returns (bboxes (N,5) incl. score, kpss (N,5,2)) in
    original frame coordinates."""
    det = get_detector()
    h, w = frame.shape[:2]
    
    # Scale frame dynamically so the longest side matches det_size
    scale = float(det_size) / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    nms_thresh = getattr(roop.globals, 'face_detector_nms', 0.40)
    with _detect_lock:
        det.setInputSize((new_w, new_h))
        det.setScoreThreshold(det_thresh)
        det.setNmsThreshold(nms_thresh)
        _, faces = det.detect(resized)
        
    if faces is None or len(faces) == 0:
        return np.zeros((0, 5), dtype=np.float32), np.zeros((0, 5, 2), dtype=np.float32)
        
    bboxes = []
    kpss = []
    for face in faces:
        # face shape: bbox [x, y, w, h] (0:4), landmarks [5, 2] (4:14), score (14)
        x1, y1, width, height = face[0:4]
        score = face[14]
        
        # Scale back to original coordinates
        ox1 = x1 / scale
        oy1 = y1 / scale
        ox2 = (x1 + width) / scale
        oy2 = (y1 + height) / scale
        
        bboxes.append([ox1, oy1, ox2, oy2, score])
        
        # Convert landmarks to shape (5, 2)
        lm = face[4:14].reshape((5, 2)) / scale
        kpss.append(lm)
        
    return np.array(bboxes, dtype=np.float32), np.array(kpss, dtype=np.float32)


def release_detector():
    global _detector
    with _detector_lock:
        _detector = None
