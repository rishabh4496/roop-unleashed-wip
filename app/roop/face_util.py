import threading
import contextlib
from queue import Queue
from typing import Any
import insightface

import roop.globals
from roop.typing import Frame, Face
from roop import session_pool

import cv2
import numpy as np
from skimage import transform as trans
from roop.capturer import get_video_frame
from roop.utilities import resolve_relative_path, conditional_download

# Pool of independent insightface FaceAnalysis instances (opt-in, ROOP_DETMASK_POOL).
#
# Detection is ~43% of video time and runs single-threaded behind the global GPU
# lock. A single FaceAnalysis instance is NOT safe to call concurrently (its models
# share buffers/caches), so to detect on N threads at once each worker leases its
# OWN instance from this pool — the same pattern the swapper uses. Instances keep
# the normal providers (TensorRT) so per-call speed is unchanged (CUDA FP32 was
# benchmarked slower); only the serialisation is removed. Without the env var the
# pool is size 1 and detection falls back to the single shared instance serialised
# by the global lock — the original, known-safe behaviour.
FACE_ANALYSER = None              # primary instance (== FACE_ANALYSER_POOL[0])
FACE_ANALYSER_POOL = []           # all instances (length 1 when not pooling)
_ANALYSER_Q = None                # lease queue (only used when pooling)
_ANALYSER_DET_SIZE = None         # det_size the pool was built with (rebuild on change)
_ANALYSER_DET_THRESH = None       # det_thresh the pool was built with (rebuild on change)
_ANALYSER_ENGINE = None           # detector engine the pool was built with (rebuild on change)
THREAD_LOCK_ANALYSER = threading.Lock()
THREAD_LOCK_SWAPPER = threading.Lock()
FACE_SWAPPER = None


def _desired_det_size():
    val = getattr(roop.globals, 'face_detector_size', '640')
    if isinstance(val, bool):
        return (640, 640) if val else (320, 320)
    try:
        sz = int(val)
        return (sz, sz)
    except Exception:
        return (640, 640)


# Engines that bring their OWN detector and only borrow buffalo_l's aux models
# via _hybrid_detector_faces (which skips taskname == 'detection').
_HYBRID_ENGINES = ('yoloface', 'retinaface', 'retinaface_r50', 'yunet')


def _current_engine():
    return getattr(roop.globals, 'detector_engine', 'scrfd')


def _hybrid_engine_active() -> bool:
    return _current_engine() in _HYBRID_ENGINES


def analysis_pooled() -> bool:
    """True when >1 independent FaceAnalysis instance exists, i.e. detection can
    run lock-free concurrently (each worker leases its own instance)."""
    return len(FACE_ANALYSER_POOL) > 1


def _build_face_analyser():
    model_path = resolve_relative_path('..')
    allowed_modules = roop.globals.g_desired_face_analysis
    if roop.globals.CFG.force_cpu:
        providers = ["CPUExecutionProvider"]
    else:
        providers = roop.globals.execution_providers
    fa = insightface.app.FaceAnalysis(
        name="buffalo_l", root=model_path, providers=providers, allowed_modules=allowed_modules)
    fa.prepare(
        ctx_id=0,
        det_size=_desired_det_size(),
        det_thresh=getattr(roop.globals, 'face_detector_threshold', 0.60),
    )
    # With a hybrid engine selected, buffalo_l's own SCRFD detector is never
    # called — _hybrid_detector_faces skips taskname == 'detection' and uses the
    # engine's detector instead. insightface asserts 'detection' is present at
    # construction and prepare() needs it, so it can only be dropped afterwards;
    # doing so frees one unused det_10g session PER pool instance. fa.det_model
    # is left None deliberately: _detect_faces_raw's det_size/det_thresh
    # overrides are all guarded on getattr(fa.det_model, ...) being non-None.
    # _ensure_face_analyser tracks the engine so switching back to SCRFD (which
    # does need fa.get()) rebuilds the pool rather than hitting the hole.
    if _hybrid_engine_active():
        fa.models.pop('detection', None)
        fa.det_model = None
    return fa


def _ensure_face_analyser():
    """(Re)build the FaceAnalysis pool when missing, when the requested module set
    changed, or when the detection resolution (face_detector_size) or threshold changed. Returns
    the primary instance."""
    global FACE_ANALYSER, FACE_ANALYSER_POOL, _ANALYSER_Q, _ANALYSER_DET_SIZE, _ANALYSER_DET_THRESH
    global _ANALYSER_ENGINE
    # Fast path (no lock): pool is built once before the run and the module set,
    # det_size, det_thresh, and engine are stable during it, so the hot per-frame detect path skips the lock.
    cur_det_thresh = getattr(roop.globals, 'face_detector_threshold', 0.60)
    cur_engine = _current_engine()
    if (FACE_ANALYSER_POOL
            and roop.globals.g_current_face_analysis == roop.globals.g_desired_face_analysis
            and _ANALYSER_DET_SIZE == _desired_det_size()
            and _ANALYSER_DET_THRESH == cur_det_thresh
            and _ANALYSER_ENGINE == cur_engine):
        return FACE_ANALYSER
    with THREAD_LOCK_ANALYSER:
        if (not FACE_ANALYSER_POOL
                or roop.globals.g_current_face_analysis != roop.globals.g_desired_face_analysis
                or _ANALYSER_DET_SIZE != _desired_det_size()
                or _ANALYSER_DET_THRESH != cur_det_thresh
                or _ANALYSER_ENGINE != cur_engine):
            roop.globals.g_current_face_analysis = roop.globals.g_desired_face_analysis
            _ANALYSER_DET_SIZE = _desired_det_size()
            _ANALYSER_DET_THRESH = cur_det_thresh
            _ANALYSER_ENGINE = cur_engine
            if roop.globals.CFG.force_cpu:
                print("Forcing CPU for Face Analysis")
            n = session_pool.detmask_pool_size() if session_pool.detmask_pooling_enabled() else 1
            FACE_ANALYSER_POOL = [_build_face_analyser() for _ in range(n)]
            FACE_ANALYSER = FACE_ANALYSER_POOL[0]
            q = Queue()
            for fa in FACE_ANALYSER_POOL:
                q.put(fa)
            _ANALYSER_Q = q
            if n > 1:
                print(f"[FaceAnalysis] pool of {n} TRT instances — detection runs {n}-way concurrent (lock-free).")
    return FACE_ANALYSER


def release_face_analyser():
    global FACE_ANALYSER, FACE_ANALYSER_POOL, _ANALYSER_Q
    with THREAD_LOCK_ANALYSER:
        FACE_ANALYSER = None
        FACE_ANALYSER_POOL = []
        _ANALYSER_Q = None
    try:
        from roop.yoloface import release_detector
        release_detector()
    except Exception:
        pass
    try:
        from roop.retinaface import release_detector as _release_retina
        _release_retina()
    except Exception:
        pass
    try:
        from roop.yunet import release_detector as _release_yunet
        _release_yunet()
    except Exception:
        pass


def get_face_analyser() -> Any:
    return _ensure_face_analyser()


@contextlib.contextmanager
def lease_face_analyser():
    """Lease one FaceAnalysis instance for a detect call. With a pool each waiting
    thread gets its OWN instance (safe concurrency); the queue blocks once all N are
    out, capping concurrency at the pool size. Without a pool it yields the single
    shared instance (caller serialises via the global lock)."""
    _ensure_face_analyser()
    if analysis_pooled():
        fa = _ANALYSER_Q.get()
        try:
            yield fa
        finally:
            _ANALYSER_Q.put(fa)
    else:
        yield FACE_ANALYSER


def _refine_kps_from_68(face) -> None:
    """Replace the detector's 5 arcface keypoints with ones derived from the
    68-point landmarks (eye centers, nose tip, mouth corners). The 68-point
    model is more stable than the 5-point regressor at angles, so this reduces
    residual alignment wobble. No-op when the 68-point landmarks are absent.

    Standard dlib/300-W ordering: left eye 36-41, right eye 42-47, nose tip 30,
    left/right mouth corners 48/54 (all image-left/right, matching the arcface
    template's kps order)."""
    lm = getattr(face, 'landmark_3d_68', None)
    if lm is None:
        return
    try:
        pts = np.asarray(lm)[:, :2].astype(np.float32)
        if pts.shape[0] < 68:
            return
        refined = np.array([
            pts[36:42].mean(axis=0),   # left eye center
            pts[42:48].mean(axis=0),   # right eye center
            pts[30],                    # nose tip
            pts[48],                    # left mouth corner
            pts[54],                    # right mouth corner
        ], dtype=np.float32)
        face.kps = refined
    except Exception:
        pass


def _scale_face_coords(face, inv_scale: float) -> None:
    """Scale a Face's spatial fields by inv_scale in place (used to map faces
    detected on an upscaled frame back to original-frame coordinates)."""
    for attr in ('bbox', 'kps', 'landmark_2d_106'):
        v = getattr(face, attr, None)
        if v is not None:
            try:
                setattr(face, attr, np.asarray(v, dtype=np.float32) * inv_scale)
            except Exception:
                pass
    lm68 = getattr(face, 'landmark_3d_68', None)
    if lm68 is not None:
        try:
            lm68 = np.asarray(lm68, dtype=np.float32).copy()
            lm68[:, :2] *= inv_scale   # only x,y are pixel coords; z is depth
            face.landmark_3d_68 = lm68
        except Exception:
            pass


def _offset_face_coords(face, ox: float, oy: float) -> None:
    """Shift a Face's spatial fields by (ox, oy) in place (used to map faces
    detected in a cropped ROI back to full-frame coordinates)."""
    bbox = getattr(face, 'bbox', None)
    if bbox is not None:
        try:
            b = np.asarray(bbox, dtype=np.float32).copy()
            b[0::2] += ox
            b[1::2] += oy
            face.bbox = b
        except Exception:
            pass
    for attr in ('kps', 'landmark_2d_106'):
        v = getattr(face, attr, None)
        if v is not None:
            try:
                a = np.asarray(v, dtype=np.float32).copy()
                a[:, 0] += ox
                a[:, 1] += oy
                setattr(face, attr, a)
            except Exception:
                pass
    lm68 = getattr(face, 'landmark_3d_68', None)
    if lm68 is not None:
        try:
            lm68 = np.asarray(lm68, dtype=np.float32).copy()
            lm68[:, 0] += ox   # only x,y are pixel coords; z is depth
            lm68[:, 1] += oy
            face.landmark_3d_68 = lm68
        except Exception:
            pass


def get_all_faces_in_roi(frame, bbox, pad_ratio=1.0, min_crop=160):
    """Detect faces within a padded crop around `bbox` (a tracked face's
    previous/predicted location) instead of the full frame. The detector's
    input canvas size is unchanged, so a small tracked face fills far more of
    it — improving recall on rotated/angled faces at no extra compute versus
    a full-frame detect. Returns faces in full-frame coordinates, or an empty
    list if none were found in the crop (caller decides whether to fall back
    to a full-frame detect on a miss)."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return []
    mx, my = bw * pad_ratio, bh * pad_ratio
    cx1, cy1, cx2, cy2 = x1 - mx, y1 - my, x2 + mx, y2 + my
    if cx2 - cx1 < min_crop:
        pad = (min_crop - (cx2 - cx1)) / 2.0
        cx1 -= pad
        cx2 += pad
    if cy2 - cy1 < min_crop:
        pad = (min_crop - (cy2 - cy1)) / 2.0
        cy1 -= pad
        cy2 += pad
    cx1, cy1 = max(0, int(round(cx1))), max(0, int(round(cy1)))
    cx2, cy2 = min(w, int(round(cx2))), min(h, int(round(cy2)))
    if cx2 <= cx1 or cy2 <= cy1:
        return []

    crop = frame[cy1:cy2, cx1:cx2]
    faces = get_all_faces(crop) or []
    for face in faces:
        _offset_face_coords(face, cx1, cy1)
    return faces


def _hybrid_detector_faces(frame, fa, bboxes, kpss):
    """Wrap raw detector output (bbox + 5 kps per face) into full Face objects
    using buffalo_l's aux models (recognition + 106/68 landmarks) — mirrors
    insightface FaceAnalysis.get but with the detector swapped out. Lets any
    alternate detector feed the exact same Face objects the pipeline expects
    (embedding, landmark_2d_106, optional 68)."""
    from insightface.app.common import Face
    if bboxes.shape[0] == 0:
        return []
    ret = []
    for i in range(bboxes.shape[0]):
        face = Face(bbox=bboxes[i, 0:4], kps=kpss[i], det_score=bboxes[i, 4])
        for taskname, model in fa.models.items():
            if taskname == 'detection':
                continue
            model.get(frame, face)
        ret.append(face)
    return ret


def _hybrid_yolo_faces(frame, fa):
    from roop.yoloface import get_detector
    det = get_detector()
    det_size = _desired_det_size()[0]
    det_thresh = getattr(roop.globals, 'face_detector_threshold', 0.60)
    bboxes, kpss = det.detect(frame, det_size=det_size, det_thresh=det_thresh)
    return _hybrid_detector_faces(frame, fa, bboxes, kpss)


def _hybrid_retinaface_faces(frame, fa, model_type='10g'):
    from roop import retinaface
    det_size = _desired_det_size()[0]
    det_thresh = getattr(roop.globals, 'face_detector_threshold', 0.60)
    bboxes, kpss = retinaface.detect(frame, det_size=det_size, det_thresh=det_thresh, model_type=model_type)
    return _hybrid_detector_faces(frame, fa, bboxes, kpss)


def _hybrid_yunet_faces(frame, fa):
    from roop import yunet
    det_size = _desired_det_size()[0]
    det_thresh = getattr(roop.globals, 'face_detector_threshold', 0.60)
    bboxes, kpss = yunet.detect(frame, det_size=det_size, det_thresh=det_thresh)
    return _hybrid_detector_faces(frame, fa, bboxes, kpss)


def _detect_faces_raw(frame, det_size=None, det_thresh=None):
    """Run the selected detector engine and return raw Face objects (unsorted) without rescues."""
    engine = getattr(roop.globals, 'detector_engine', 'scrfd')
    nms_thresh = getattr(roop.globals, 'face_detector_nms', 0.40)
    with lease_face_analyser() as fa:
        for model in fa.models.values():
            if hasattr(model, 'nms_thresh'):
                model.nms_thresh = nms_thresh
                
        # Temporarily override detection size and threshold on the underlying model if requested
        orig_input_size = getattr(fa.det_model, 'input_size', None)
        orig_det_thresh = getattr(fa.det_model, 'det_thresh', None)
        if det_size is not None and orig_input_size is not None:
            fa.det_model.input_size = (det_size, det_size)
        if det_thresh is not None and orig_det_thresh is not None:
            fa.det_model.det_thresh = det_thresh
            
        try:
            if engine == 'yoloface':
                faces = _hybrid_yolo_faces(frame, fa)
            elif engine == 'retinaface':
                faces = _hybrid_retinaface_faces(frame, fa, model_type='10g')
            elif engine == 'retinaface_r50':
                faces = _hybrid_retinaface_faces(frame, fa, model_type='r50')
            elif engine == 'yunet':
                faces = _hybrid_yunet_faces(frame, fa)
            else:
                faces = fa.get(frame)
        finally:
            if det_size is not None and orig_input_size is not None:
                fa.det_model.input_size = orig_input_size
            if det_thresh is not None and orig_det_thresh is not None:
                fa.det_model.det_thresh = orig_det_thresh
                
    return faces or []


def _rescue_upscaled(frame: Frame):
    """Retry detection on a 2x upscale of the frame, mapping any faces back to
    the original coordinate space. For frames where the face is too small for
    the current det size to catch."""
    try:
        up = cv2.resize(frame, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
        faces = _detect_faces_raw(up)
        if faces:
            for f in faces:
                _scale_face_coords(f, 0.5)
            return faces
    except Exception:
        pass
    return None


def _rescue_downscaled(frame: Frame):
    """Retry detection with a smaller input_size (320x320) directly on the detector.
    SCRFD anchors are tuned for faces in the 50-200px range; when a close-up face 
    fills most of the frame (400px+ face in a 640px det-size pass) confidence scores 
    drop sharply. By dropping the detector's input size to 320, the image is shrunk 
    more aggressively before passing through the network, putting the face back 
    within the anchor sweet-spot.

    Also lowers the detection threshold slightly for this rescue attempt since
    close-up faces that partially exit the frame may have lower confidence."""
    try:
        h, w = frame.shape[:2]
        if min(h, w) < 128:
            return None

        # Temporarily lower the detection threshold
        orig_thresh = getattr(roop.globals, 'face_detector_threshold', 0.50)
        lowered_thresh = max(0.30, orig_thresh - 0.15)
        
        # Override det_size directly on the model via _detect_faces_raw
        faces = _detect_faces_raw(frame, det_size=320, det_thresh=lowered_thresh)
        
        if faces:
            return faces
    except Exception:
        pass
    return None


def _rescue_padded(frame: Frame):
    """Pad the frame by 25% on all sides. When a face is extremely close-up or
    clipped at the boundaries, anchor-based detectors fail because critical face
    structure is cut off. Adding padding provides background padding, allowing
    the detector to lock on."""
    try:
        h, w = frame.shape[:2]
        pad = int(max(h, w) * 0.25)
        padded = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        
        faces = _detect_faces_raw(padded)
        if faces:
            for f in faces:
                f.bbox[0] -= pad
                f.bbox[1] -= pad
                f.bbox[2] -= pad
                f.bbox[3] -= pad
                if getattr(f, 'kps', None) is not None:
                    f.kps -= pad
                if getattr(f, 'landmark_2d_106', None) is not None:
                    f.landmark_2d_106 = np.asarray(f.landmark_2d_106) - pad
                if getattr(f, 'landmark_3d_68', None) is not None:
                    lm = np.asarray(f.landmark_3d_68)
                    lm[:, :2] -= pad
                    f.landmark_3d_68 = lm
            return faces
    except Exception:
        pass
    return None


def _unrotate_face_coords(face, orig_w, orig_h, angle):
    """Map face bbox, kps, and landmarks back from a rotated canvas to original frame space."""
    def unrot_pts(pts):
        pts = np.asarray(pts, dtype=np.float32)
        unrot = np.zeros_like(pts)
        if angle == "clockwise":
            unrot[..., 0] = pts[..., 1]
            unrot[..., 1] = orig_h - 1.0 - pts[..., 0]
        elif angle == "anticlockwise":
            unrot[..., 0] = orig_w - 1.0 - pts[..., 1]
            unrot[..., 1] = pts[..., 0]
        return unrot

    x1, y1, x2, y2 = face.bbox
    corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    unrot_corners = unrot_pts(corners)
    face.bbox = np.array([
        unrot_corners[:, 0].min(),
        unrot_corners[:, 1].min(),
        unrot_corners[:, 0].max(),
        unrot_corners[:, 1].max()
    ], dtype=np.float32)

    if getattr(face, 'kps', None) is not None:
        face.kps = unrot_pts(face.kps)
    if getattr(face, 'landmark_2d_106', None) is not None:
        face.landmark_2d_106 = unrot_pts(face.landmark_2d_106)
    if getattr(face, 'landmark_3d_68', None) is not None:
        lm = np.asarray(face.landmark_3d_68, dtype=np.float32)
        lm[:, :2] = unrot_pts(lm[:, :2])
        face.landmark_3d_68 = lm


def _rescue_rotated(frame: Frame):
    """Retry detection on 90° rotated frame variants when standard orientation detection finds no faces."""
    try:
        h, w = frame.shape[:2]
        rot_cw = rotate_clockwise(frame)
        faces = _detect_faces_raw(rot_cw)
        if faces:
            for f in faces:
                _unrotate_face_coords(f, w, h, "clockwise")
            return faces

        rot_acw = rotate_anticlockwise(frame)
        faces = _detect_faces_raw(rot_acw)
        if faces:
            for f in faces:
                _unrotate_face_coords(f, w, h, "anticlockwise")
            return faces
    except Exception:
        pass
    return None


def _detect_faces(frame):
    """Run the selected detector engine and return raw Face objects (unsorted).
    Applies small-face (upscale), close-up scale (downscale), clipped boundary (padded), and rotated face rescues."""
    faces = _detect_faces_raw(frame)
    if not faces:
        # 1. Small-face rescue
        if getattr(roop.globals, 'rescue_small_faces', False):
            faces = _rescue_upscaled(frame)
        # 2. Close-up rescue
        if not faces:
            faces = _rescue_downscaled(frame)
        # 3. Boundary padding rescue
        if not faces:
            faces = _rescue_padded(frame)
        # 4. Rotated face rescue
        if not faces:
            faces = _rescue_rotated(frame)

    if faces and getattr(roop.globals, 'refine_landmarks', False):
        for f in faces:
            _refine_kps_from_68(f)
    return faces or []


def get_first_face(frame: Frame) -> Any:
    try:
        faces = get_all_faces(frame)
        if faces:
            return min(faces, key=lambda x: x.bbox[0])
    except Exception:
        pass
    return None


def get_all_faces(frame: Frame) -> Any:
    try:
        faces = _detect_faces(frame)
        if not faces:
            return []
        return sorted(faces, key=lambda x: x.bbox[0])
    except Exception:
        return []


def _attach_source_crops(face, img):
    """Pre-warp and cache the aligned source-face crops the image-source swap
    models (BlendSwap / UniFace) consume. Cheap (a couple of small warps) and
    tiny to keep (112² + 256²), so we don't retain the full source image.
    `img` must be the frame `face.kps` are expressed in."""
    try:
        kps = getattr(face, 'kps', None)
        if kps is None:
            return
        crop112, _ = align_crop(img, kps, 112, mode='arcface_112_v2')
        crop256, _ = align_crop(img, kps, 256, mode='ffhq_512')
        face['_src_crop_arcface_112_v2'] = crop112
        face['_src_crop_ffhq_256'] = crop256
    except Exception:
        pass


def extract_face_images(source_filename, video_info, extra_padding=-1.0):
    face_data = []
    source_image = None

    if video_info[0]:
        frame = get_video_frame(source_filename, video_info[1])
        if frame is not None:
            source_image = frame
        else:
            return face_data
    else:
        source_image = cv2.imdecode(np.fromfile(source_filename, dtype=np.uint8), cv2.IMREAD_COLOR)

    faces = get_all_faces(source_image)
    if faces is None:
        return face_data

    i = 0
    for face in faces:
        (startX, startY, endX, endY) = face["bbox"].astype("int")
        startX, endX, startY, endY = clamp_cut_values(startX, endX, startY, endY, source_image)
        if extra_padding > 0.0:
            # ── Close-up shortcut ─────────────────────────────────────────────
            # When the face bbox covers the majority of the source image there is
            # no useful room to pad. The padded-crop loop would just reproduce the
            # same image and the re-detection would fail for the same reason the
            # initial call nearly failed. Use the original detection directly.
            img_h, img_w = source_image.shape[:2]
            img_area = img_h * img_w
            bbox_arr = face["bbox"].astype("int")
            bw = max(0, min(bbox_arr[2], img_w) - max(0, bbox_arr[0]))
            bh = max(0, min(bbox_arr[3], img_h) - max(0, bbox_arr[1]))
            is_close_up = (bw * bh) > img_area * 0.40

            if is_close_up or source_image.shape[:2] == (512, 512):
                i += 1
                _attach_source_crops(face, source_image)
                face_data.append([face, source_image])
                continue

            found = False
            for i in range(1, 3):
                (startX, startY, endX, endY) = face["bbox"].astype("int")
                startX, endX, startY, endY = clamp_cut_values(startX, endX, startY, endY, source_image)
                cutout_padding = extra_padding
                # top needs extra room for detection
                padding = int((endY - startY) * cutout_padding)
                oldY = startY
                startY -= padding

                factor = 0.25 if i == 1 else 0.5
                cutout_padding = factor
                padding = int((endY - oldY) * cutout_padding)
                endY += padding
                padding = int((endX - startX) * cutout_padding)
                startX -= padding
                endX += padding
                startX, endX, startY, endY = clamp_cut_values(
                    startX, endX, startY, endY, source_image
                )
                face_temp = source_image[startY:endY, startX:endX]
                face_temp = resize_image_keep_content(face_temp)
                testfaces = get_all_faces(face_temp)
                if testfaces is not None and len(testfaces) > 0:
                    i += 1
                    _attach_source_crops(testfaces[0], face_temp)
                    face_data.append([testfaces[0], face_temp])
                    found = True
                    break

            if not found:
                # All padded-crop re-detections failed (can happen when the face is near
                # an image edge, or the padded 512×512 crop has too much background for
                # the detector at the current threshold). Fall back to the original
                # (unpadded) detection — it is valid, just without the wider context.
                print("Re-detection in padded crop failed — falling back to original unpadded detection.")
                (startX, startY, endX, endY) = face["bbox"].astype("int")
                startX, endX, startY, endY = clamp_cut_values(startX, endX, startY, endY, source_image)
                face_temp = source_image[startY:endY, startX:endX]
                if face_temp.size > 0:
                    _attach_source_crops(face, source_image)
                    face_data.append([face, face_temp])
            continue

        face_temp = source_image[startY:endY, startX:endX]
        if face_temp.size < 1:
            continue

        # face.kps are in source_image coords (not the bbox crop), so warp the
        # source crops from source_image.
        _attach_source_crops(face, source_image)
        i += 1
        face_data.append([face, face_temp])
    return face_data


def clamp_cut_values(startX, endX, startY, endY, image):
    if startX < 0:
        startX = 0
    if endX > image.shape[1]:
        endX = image.shape[1]
    if startY < 0:
        startY = 0
    if endY > image.shape[0]:
        endY = image.shape[0]
    return startX, endX, startY, endY



def face_offset_top(face: Face, offset):
    face["bbox"][1] += offset
    face["bbox"][3] += offset
    lm106 = face.landmark_2d_106
    add = np.full_like(lm106, [0, offset])
    face["landmark_2d_106"] = lm106 + add
    return face


def resize_image_keep_content(image, new_width=512, new_height=512):
    dim = None
    (h, w) = image.shape[:2]
    if h > w:
        r = new_height / float(h)
        dim = (int(w * r), new_height)
    else:
        # Calculate the ratio of the width and construct the dimensions
        r = new_width / float(w)
        dim = (new_width, int(h * r))
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    (h, w) = image.shape[:2]
    if h == new_height and w == new_width:
        return image
    resize_img = np.zeros(shape=(new_height, new_width, 3), dtype=image.dtype)
    offs = (new_width - w) if h == new_height else (new_height - h)
    startoffs = int(offs // 2) if offs % 2 == 0 else int(offs // 2) + 1
    offs = int(offs // 2)

    if h == new_height:
        resize_img[0:new_height, startoffs : new_width - offs] = image
    else:
        resize_img[startoffs : new_height - offs, 0:new_width] = image
    return resize_img


def rotate_image_90(image, rotate=True):
    if rotate:
        return np.rot90(image)
    else:
        return np.rot90(image, 1, (1, 0))


def rotate_anticlockwise(frame):
    return rotate_image_90(frame)


def rotate_clockwise(frame):
    return rotate_image_90(frame, False)


def rotate_image_180(image):
    return np.rot90(image, 2)


# alignment code from insightface https://github.com/deepinsight/insightface/blob/master/python-package/insightface/utils/face_align.py

arcface_dst = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

# Normalized (0-1) 5-point warp templates for swap models NOT trained on the
# insightface arcface alignment. Values from FaceFusion face_helper.py
# WARP_TEMPLATE_SET; multiplied by the crop size at estimation time.
#   arcface_112_v1 — ghost_* and simswap_* training alignment
#   mtcnn_512      — hififace training alignment
#   ffhq_512       — uniface/blendswap training alignment (FFHQ-style)
WARP_TEMPLATES = {
    "arcface_112_v1": np.array(
        [
            [0.35473214, 0.45658929],
            [0.64526786, 0.45658929],
            [0.50000000, 0.61154464],
            [0.37913393, 0.77687500],
            [0.62086607, 0.77687500],
        ],
        dtype=np.float32,
    ),
    # blendswap source alignment (FaceFusion face_helper WARP_TEMPLATE_SET)
    "arcface_112_v2": np.array(
        [
            [0.34191607, 0.46157411],
            [0.65653393, 0.45983393],
            [0.50022500, 0.64050536],
            [0.37097589, 0.82469196],
            [0.63151696, 0.82325089],
        ],
        dtype=np.float32,
    ),
    "mtcnn_512": np.array(
        [
            [0.36562865, 0.46733799],
            [0.63305391, 0.46585885],
            [0.50019127, 0.61942959],
            [0.39032951, 0.77598822],
            [0.61178945, 0.77476328],
        ],
        dtype=np.float32,
    ),
    "ffhq_512": np.array(
        [
            [0.37691676, 0.46864664],
            [0.62285697, 0.46912813],
            [0.50123859, 0.61331904],
            [0.39308822, 0.72541100],
            [0.61150205, 0.72490465],
        ],
        dtype=np.float32,
    ),
}


# ── 5-keypoint pose proxies ──────────────────────────────────────────────────
# Two roll-invariant ratios derived from the 5 arcface keypoints ALONE, so they
# cost nothing extra (no landmark_3d_68 inference, which is not even loaded in
# the default config — see ProcessMgr's `modules` list) and stay valid whenever
# the detector has run.
#
#   yaw_ratio   = |eye_L - eye_R| / |eye_mid - mouth_mid|
#   pitch_ratio = where the nose tip sits along the eye_mid -> mouth_mid axis,
#                 as a fraction of that axis
#
# Both are normalised by the eye-to-mouth distance, which — unlike the eye
# SEPARATION that the alignment fit leans on — stays well-conditioned all the
# way to a 90 deg profile. Being ratios of distances (and of a projection onto
# an axis the face itself defines) they are invariant to in-plane roll, so a
# tilted head does not perturb them.
#
# Measured against the project's own 3D reference face (face_3d_recon._REF3D_68,
# orthographic projection):
#
#   yaw    0    20    40    55    65    75    85    90      <- degrees
#   ratio  .72  .68   .60   .50   .43   .19   .06   .00     <- yaw_ratio, pitch 0
#
#   pitch -40  -25   -10    0    +10   +25   +40
#   ratio  .20  .35   .45  .52   .57   .66   .77            <- pitch_ratio, yaw 0
#
# NOTE on pitch_ratio: its sensitivity falls off as yaw grows (at a true 90 deg
# profile the nose and the eye->mouth axis all lie in the sagittal plane, which
# is parallel to the image plane, so pitch barely changes the projection — the
# ratio becomes V-shaped and near-useless there). That is fine: it exists to
# catch the FRONTAL extreme-pitch case, where it is most sensitive, and high yaw
# is already covered by yaw_ratio.
def kps_pose_ratios(kps):
    """(yaw_ratio, pitch_ratio) from the 5 arcface keypoints, or (None, None)
    when they are unusable (missing, wrong count, or degenerate)."""
    try:
        if kps is None:
            return None, None
        pts = np.asarray(kps, dtype=np.float32)
        if pts.shape[0] != 5:
            return None, None
        eye_mid   = (pts[0] + pts[1]) / 2.0
        mouth_mid = (pts[3] + pts[4]) / 2.0
        axis = mouth_mid - eye_mid
        vert = float(np.linalg.norm(axis))
        if vert < 1e-6:
            return None, None
        yaw_ratio   = float(np.linalg.norm(pts[1] - pts[0])) / vert
        pitch_ratio = float(np.dot(pts[2] - eye_mid, axis / vert) / vert)
        return yaw_ratio, pitch_ratio
    except Exception:
        return None, None


# Opt-in profile alignment (see estimate_norm). Default OFF: it changes the crop
# geometry — and therefore the swap output — for high-yaw faces.
#
# Read live from roop.globals.yaw_align (seeded from the ROOP_YAW_ALIGN env var,
# overridden per run by the Face Swap toggle) rather than captured here at import
# time, so flipping the toggle takes effect on the next run without an app
# restart — the point of an opt-in visual change is being able to A/B it.
# Engage only on near-profile faces (~70 deg+). Deliberately TIGHTER than the
# 0.55 gate the masking path uses: the constrained fit disagrees with the plain
# least-squares fit by up to ~9 deg even at yaw 55, so a looser gate here would
# visibly change mid-angle faces that already look correct.
YAW_ALIGN_RATIO = 0.40

# Three modes, selected by roop.globals.yaw_align:
#   'off'       — the plain fixed-template least-squares fit (default, unchanged)
#   'stabilize' — keep the frontal template, but take the rotation from the
#                 eye->mouth axis so pitch stops leaking into in-plane roll
#   'pose'      — replace the template itself with the reference head projected
#                 at the estimated yaw (see _pose_template)
# Booleans are accepted for backwards compatibility: True == 'stabilize'.
YAW_ALIGN_MODES = ('off', 'stabilize', 'pose')


def _yaw_align_mode():
    raw = getattr(roop.globals, 'yaw_align', False)
    if raw is True:
        return 'stabilize'
    if not raw:
        return 'off'
    mode = str(raw).strip().lower()
    return mode if mode in YAW_ALIGN_MODES else 'off'


# ── Reference head, 5-point ──────────────────────────────────────────────────
# Derived from face_3d_recon._REF3D_68 rather than duplicated, so the pose code
# and the 3-D reconstruction code can never drift apart on head shape.
def _reference_5pt():
    from roop.face_3d_recon import _REF3D_68
    ref = np.asarray(_REF3D_68, dtype=np.float64)
    return np.vstack([
        ref[36:42].mean(axis=0),   # left eye centre
        ref[42:48].mean(axis=0),   # right eye centre
        ref[30],                   # nose tip
        ref[48],                   # left mouth corner
        ref[54],                   # right mouth corner
    ])


def _project_reference(yaw_deg):
    """The reference head's 5 points at a given yaw, orthographic, image axes."""
    y = np.radians(yaw_deg)
    rot = np.array([[np.cos(y), 0.0, np.sin(y)],
                    [0.0, 1.0, 0.0],
                    [-np.sin(y), 0.0, np.cos(y)]])
    pts = _reference_5pt() @ rot.T
    return np.column_stack([pts[:, 0], -pts[:, 1]])


def _build_yaw_lookup():
    """yaw (deg) -> yaw_ratio, so a measured ratio can be inverted back to an
    angle. Monotonically decreasing, so np.interp needs both arrays reversed."""
    yaws = np.arange(0.0, 90.5, 0.5)
    ratios = []
    for y in yaws:
        q = _project_reference(y)
        vert = np.linalg.norm((q[3] + q[4]) / 2.0 - (q[0] + q[1]) / 2.0)
        ratios.append(np.linalg.norm(q[1] - q[0]) / vert)
    return yaws, np.asarray(ratios)


_LUT_YAWS, _LUT_RATIOS = _build_yaw_lookup()


def _yaw_from_ratio(ratio):
    return float(np.interp(ratio, _LUT_RATIOS[::-1], _LUT_YAWS[::-1]))


def _pose_template(yaw_deg, base_dst):
    """The destination points for a head at `yaw_deg`, placed to match the
    frontal template's centroid and scale.

    The fixed frontal template asks a 90 deg profile — whose two eyes project to
    the SAME point — to land on two well-separated template positions. No
    similarity transform can do that, so the fit settles on a compromise that
    shears and rotates the face; measured mean residual is 60 px on a 512 crop
    versus 8 px frontal. Projecting the reference head at the estimated yaw gives
    a template that is congruent to the input, so the fit becomes well posed
    (residual -60% at high yaw, and exactly 0 at zero pitch).

    Scale is fixed from the FRONTAL reference, not the posed one, so the face
    keeps a constant size in the crop as it turns — otherwise the head would
    appear to zoom during a turn. Measured placement barely moves versus the
    fixed template (nose 295.5 -> 300.0 px at 90 deg yaw), so this corrects the
    distortion without re-framing the face.
    """
    posed = _project_reference(yaw_deg)
    frontal = _project_reference(0.0)
    frontal_vert = np.linalg.norm((frontal[3] + frontal[4]) / 2.0
                                  - (frontal[0] + frontal[1]) / 2.0)
    base_vert = np.linalg.norm((base_dst[3] + base_dst[4]) / 2.0
                               - (base_dst[0] + base_dst[1]) / 2.0)
    if frontal_vert < 1e-9:
        return None
    scaled = posed * (base_vert / frontal_vert)
    return scaled - scaled.mean(axis=0) + np.asarray(base_dst, np.float64).mean(axis=0)


def _constrained_norm(lmk, dst):
    """Similarity transform whose ROTATION is taken from the eye_mid -> mouth_mid
    axis instead of from an unconstrained 5-point least-squares fit.

    Why: at high yaw the two eyes project to nearly the same point (their
    separation collapses to zero at 90 deg), so the least-squares fit is
    ill-conditioned in rotation and starts absorbing PITCH as in-plane roll.
    Measured swing of the crop rotation as the head nods from -25 to +25 deg:

        yaw    0     30     60     75     90
        LS   0.00   9.93  20.14  25.35  30.51  deg   <- current behaviour
        ours 0.00   0.49   0.49   0.28   0.00  deg   <- rotation held steady

    A 30 deg rotation swing on a nodding profile head feeds the swapper an
    off-distribution crop and shows up as rotational wobble frame to frame.

    The eye_mid -> mouth_mid axis stays well-conditioned at every yaw, so we fix
    the rotation from it and still solve scale + translation by least squares
    over all 5 points (so those keep every point's information).

    Caveat: the mean per-point fit residual gets slightly WORSE off-neutral-pitch
    (e.g. yaw 90 / pitch +25: 64.0 -> 69.9 px on a 512 crop). That is expected and
    is not a regression — the lower residual was being bought by rotating the
    face, which is precisely the pathology being removed.
    """
    def axis_angle(p):
        v = ((p[3] + p[4]) / 2.0) - ((p[0] + p[1]) / 2.0)
        return np.arctan2(v[1], v[0])

    theta = axis_angle(dst) - axis_angle(lmk)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)

    src_m, dst_m = lmk.mean(0), dst.mean(0)
    rot_sc = (lmk - src_m) @ R.T
    denom = float((rot_sc ** 2).sum())
    if denom < 1e-9:
        return None
    scale = float(((dst - dst_m) * rot_sc).sum() / denom)
    if not np.isfinite(scale) or scale <= 0:
        return None

    M = np.zeros((2, 3), dtype=np.float64)
    M[:, :2] = scale * R
    M[:, 2] = dst_m - scale * (R @ src_m)
    return M


def _maybe_constrained(lmk, dst):
    """Profile-specific alignment for near-profile faces, per the selected mode.
    Returns None to mean 'use the normal fit' — which is every frontal and
    mid-angle face, and every face at all when the mode is 'off'."""
    mode = _yaw_align_mode()
    if mode == 'off':
        return None
    yaw_ratio, _ = kps_pose_ratios(lmk)
    if yaw_ratio is None or yaw_ratio >= YAW_ALIGN_RATIO:
        return None

    lmk = np.asarray(lmk, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)

    if mode == 'pose':
        posed = _pose_template(_yaw_from_ratio(yaw_ratio), dst)
        if posed is None:
            return None
        # A pose-matched template is congruent to the input, so the ordinary
        # least-squares similarity is already well posed — no rotation
        # constraint needed on top.
        tform = trans.SimilarityTransform()
        tform.estimate(lmk, posed)
        m = tform.params[0:2, :]
        return m if np.isfinite(m).all() else None

    return _constrained_norm(lmk, dst)


def estimate_norm(lmk, image_size=112, mode="arcface"):
    assert lmk.shape == (5, 2)

    # Force SimilarityTransform (use_affine = False) for all faces.
    # AffineTransform introduces non-uniform scaling and shearing, which causes
    # severe perspective warping and facial distortion (e.g. stretched foreheads)
    # on angled/close-up faces. Using SimilarityTransform preserves the face's
    # natural aspect ratio and matches the training distribution of the models.
    use_affine = False

    if mode in WARP_TEMPLATES:
        dst = WARP_TEMPLATES[mode] * float(image_size)
        M = _maybe_constrained(lmk, dst)
        if M is not None:
            return M
        tform = trans.AffineTransform() if use_affine else trans.SimilarityTransform()
        tform.estimate(lmk, dst)
        return tform.params[0:2, :]

    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
        diff_x = 0
    elif image_size % 128 == 0:
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio
    elif image_size % 512 == 0:
        ratio = float(image_size) / 512.0
        diff_x = 32.0 * ratio
    else:
        # Generic fallback so a swap model with an unusual output size (e.g.
        # 320/384) can't hit an UnboundLocalError here — scale like the 112
        # template with no x-shift, matching insightface's base alignment.
        ratio = float(image_size) / 112.0
        diff_x = 0

    dst = arcface_dst * ratio
    dst[:, 0] += diff_x
    M = _maybe_constrained(lmk, dst)
    if M is not None:
        return M
    tform = trans.AffineTransform() if use_affine else trans.SimilarityTransform()
    tform.estimate(lmk, dst)
    M = tform.params[0:2, :]
    return M



# aligned, M = norm_crop2(f[1], face.kps, 512)
def align_crop(img, landmark, image_size=112, mode="arcface"):
    M = estimate_norm(landmark, image_size, mode)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped, M


def square_crop(im, S):
    if im.shape[0] > im.shape[1]:
        height = S
        width = int(float(im.shape[1]) / im.shape[0] * S)
        scale = float(S) / im.shape[0]
    else:
        width = S
        height = int(float(im.shape[0]) / im.shape[1] * S)
        scale = float(S) / im.shape[1]
    resized_im = cv2.resize(im, (width, height))
    det_im = np.zeros((S, S, 3), dtype=np.uint8)
    det_im[: resized_im.shape[0], : resized_im.shape[1], :] = resized_im
    return det_im, scale


def transform(data, center, output_size, scale, rotation):
    scale_ratio = scale
    rot = float(rotation) * np.pi / 180.0
    # translation = (output_size/2-center[0]*scale_ratio, output_size/2-center[1]*scale_ratio)
    t1 = trans.SimilarityTransform(scale=scale_ratio)
    cx = center[0] * scale_ratio
    cy = center[1] * scale_ratio
    t2 = trans.SimilarityTransform(translation=(-1 * cx, -1 * cy))
    t3 = trans.SimilarityTransform(rotation=rot)
    t4 = trans.SimilarityTransform(translation=(output_size / 2, output_size / 2))
    t = t1 + t2 + t3 + t4
    M = t.params[0:2]
    cropped = cv2.warpAffine(data, M, (output_size, output_size), borderValue=0.0)
    return cropped, M


def trans_points2d(pts, M):
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.0], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        # print('new_pt', new_pt.shape, new_pt)
        new_pts[i] = new_pt[0:2]

    return new_pts


def trans_points3d(pts, M):
    scale = np.sqrt(M[0][0] * M[0][0] + M[0][1] * M[0][1])
    # print(scale)
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.0], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        # print('new_pt', new_pt.shape, new_pt)
        new_pts[i][0:2] = new_pt[0:2]
        new_pts[i][2] = pts[i][2] * scale

    return new_pts


def trans_points(pts, M):
    if pts.shape[1] == 2:
        return trans_points2d(pts, M)
    else:
        return trans_points3d(pts, M)
    
def create_blank_image(width, height):
    img = np.zeros((height, width, 4), dtype=np.uint8)
    img[:] = [0,0,0,0]
    return img

