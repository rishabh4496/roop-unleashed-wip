import math
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
    # is left None deliberately: the det_size/det_thresh writes onto it in
    # _detect_faces_raw are guarded on getattr(fa.det_model, ...) being non-None,
    # and are only how SCRFD is steered — a hybrid engine is passed the effective
    # values as arguments instead, so the hole costs it nothing.
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


def _hybrid_yolo_faces(frame, fa, det_size, det_thresh):
    # Module-level detect(), not get_detector().detect(): the instance has to be
    # LEASED for the call, or concurrent workers share one ORT session again.
    from roop import yoloface
    bboxes, kpss = yoloface.detect(frame, det_size=det_size, det_thresh=det_thresh)
    return _hybrid_detector_faces(frame, fa, bboxes, kpss)


def _hybrid_retinaface_faces(frame, fa, det_size, det_thresh, model_type='10g'):
    from roop import retinaface
    bboxes, kpss = retinaface.detect(frame, det_size=det_size, det_thresh=det_thresh, model_type=model_type)
    return _hybrid_detector_faces(frame, fa, bboxes, kpss)


def _hybrid_yunet_faces(frame, fa, det_size, det_thresh):
    from roop import yunet
    bboxes, kpss = yunet.detect(frame, det_size=det_size, det_thresh=det_thresh)
    return _hybrid_detector_faces(frame, fa, bboxes, kpss)


def _detect_faces_raw(frame, det_size=None, det_thresh=None):
    """Run the selected detector engine and return raw Face objects (unsorted) without rescues.

    det_size / det_thresh override the configured detection resolution and
    confidence floor for THIS call. They must reach whichever engine is selected,
    not just SCRFD: the overrides used to be written onto `fa.det_model`, which a
    hybrid engine does not own (_build_face_analyser pops it, leaving None), and
    the hybrid branches then re-read the unchanged globals. That made
    _rescue_downscaled — the close-up rescue, and the only rescue that varies
    detector PARAMETERS rather than the image — an exact re-run of the pass that
    had just returned nothing, on four of the five engines.
    """
    engine = getattr(roop.globals, 'detector_engine', 'scrfd')
    nms_thresh = getattr(roop.globals, 'face_detector_nms', 0.40)
    eff_size = det_size if det_size is not None else _desired_det_size()[0]
    eff_thresh = (det_thresh if det_thresh is not None
                  else getattr(roop.globals, 'face_detector_threshold', 0.60))
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
                faces = _hybrid_yolo_faces(frame, fa, eff_size, eff_thresh)
            elif engine == 'retinaface':
                faces = _hybrid_retinaface_faces(frame, fa, eff_size, eff_thresh, model_type='10g')
            elif engine == 'retinaface_r50':
                faces = _hybrid_retinaface_faces(frame, fa, eff_size, eff_thresh, model_type='r50')
            elif engine == 'yunet':
                faces = _hybrid_yunet_faces(frame, fa, eff_size, eff_thresh)
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
        
        # This is the ONLY rescue that varies detector parameters instead of the
        # image, so it is worth nothing unless the override reaches the engine
        # actually in use — see the note in _detect_faces_raw.
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
        elif angle == "180":
            unrot[..., 0] = orig_w - 1.0 - pts[..., 0]
            unrot[..., 1] = orig_h - 1.0 - pts[..., 1]
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
    """Retry detection on rotated frame variants when the upright pass finds nothing.

    All THREE turns are tried, not just the quarter ones. A detector that misses
    a face on its side misses an inverted one too, and neither quarter turn
    reaches it — both leave an upside-down face lying sideways, which is the
    orientation the pass already failed on. The half turn is the only one that
    presents it upright, so leaving it out means an inverted face in an
    otherwise empty frame is simply never found.
    """
    try:
        h, w = frame.shape[:2]
        for angle, rotated in (("clockwise", rotate_clockwise),
                               ("anticlockwise", rotate_anticlockwise),
                               ("180", rotate_image_180)):
            faces = _detect_faces_raw(rotated(frame))
            if faces:
                for f in faces:
                    _unrotate_face_coords(f, w, h, angle)
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


# ── In-plane roll detection (autorotate_faces) ───────────────────────────────
#
# The decision "is this face lying on its side, and which way do I turn the
# frame to stand it up?" rests entirely on estimating the face's DOWN axis
# (eyes -> chin) in image space.  Which landmarks that axis is built from
# matters far more than it looks, because an out-of-plane turn (yaw) shears
# every candidate axis sideways and so masquerades as in-plane roll.
#
# Measured over a yaw 0-90 / pitch +-30 sweep of the project's own reference
# head (tests/facegeom.py), with the head held perfectly upright — so every
# reading below is pure error:
#
#     eye-midpoint -> NOSE TIP     up to 45.7 deg   unusable
#     eye-line perpendicular       up to 90.0 deg   unusable
#     eye-midpoint -> MOUTH-mid    up to  9.1 deg   good
#
# The nose tip protrudes along +z, so turning the head swings it sideways
# almost as much as a real 45 deg roll would.  The eye line collapses to zero
# length at full profile and its direction becomes noise.  The mouth midpoint
# sits on the head's vertical midline and barely moves — it is the only one of
# the three that stays honest, so it is what is used.
#
# A 90 deg turn of the frame lands the face within 45 deg of upright exactly
# when its roll is between 45 and 135 deg — that is the honest break-even band,
# and the two edges of it want different treatment.
#
# LOWER is pushed out from 45 to 54.5 because the failure directions are not
# symmetric there: declining to rotate a sideways face merely forgoes an
# improvement, while rotating an upright one corrupts the crop the swap is
# built from.  54.5 is not a guess — swept against the same synthetic head, it
# is the first value with ZERO false positives over the full yaw 0-90 /
# pitch +-30 grid at every roll up to 44 deg, while still firing on every face
# in the 70-110 deg band.  Below it the 9.1 deg axis error stacks onto real
# roll and upright faces start getting turned.
#
# UPPER stays at the true break-even 135 and deliberately does NOT inherit that
# margin.  Nothing near 135 deg is at risk of being an upright face, so the
# only thing a margin buys there is lost recall — and it is expensive: a face
# at 120 deg is 90 deg better off for being turned.
#
# Past UPPER a 90 deg turn no longer helps — but a 180 deg one does, and that
# is what the band above it is for.  An inverted face is exactly the case a
# quarter turn cannot reach: at 165 deg either quarter turn leaves it at 75,
# still on its side, while a half turn puts it at -15.  The two branches meet
# without a seam because at exactly 135 deg they are equivalent (both land the
# face at 45), so which side of the boundary a noisy reading falls on does not
# matter.  Nor can the half turn misfire on an upright face: reaching 135 deg
# of measured tilt would take 135 deg of error against a 9.1 deg axis.
FACE_ROLL_LOWER = 54.5          # vs the 9.1 deg worst-case axis error
FACE_ROLL_UPPER = 135.0         # above this a quarter turn stops helping and a half turn starts


def face_down_axis(face):
    """The face's chin direction as (dx, dy) in image space, or None.

    Built from the 5 detector keypoints, which every detector engine emits
    (align_crop already depends on them), so this works where the 106-point
    landmarks are unavailable.
    """
    kps = getattr(face, 'kps', None)
    if kps is None or len(kps) < 5:
        return None
    kps = np.asarray(kps, dtype=np.float32)
    eye_mid = (kps[0] + kps[1]) / 2.0
    mouth_mid = (kps[3] + kps[4]) / 2.0
    axis = mouth_mid - eye_mid
    if not np.isfinite(axis).all() or float(np.hypot(axis[0], axis[1])) < 1e-3:
        return None
    return float(axis[0]), float(axis[1])


def face_roll_tilt(face):
    """In-plane tilt of *face* in degrees, or None. 0 = upright, +-180 = upside down."""
    axis = face_down_axis(face)
    if axis is None:
        return None
    return float(np.degrees(np.arctan2(axis[0], axis[1])))


def _action_for_down_axis(dx, dy):
    """Which way to turn the FRAME so a face whose chin points (dx, dy) stands up.

    Verified against the shipped rotate helpers rather than reasoned about: the
    forward point map of np.rot90 sends a direction (dx, dy) to (dy, -dx), so a
    chin pointing image-left (-x) is stood upright by rotate_anticlockwise, and
    a chin pointing image-right (+x) by rotate_clockwise.  A chin pointing back
    UP the frame is an inverted face, which only a half turn reaches.
    """
    tilt = float(np.degrees(np.arctan2(dx, dy)))
    if abs(tilt) < FACE_ROLL_LOWER:
        return None
    if abs(tilt) > FACE_ROLL_UPPER:
        return "rotate_180"
    return "rotate_anticlockwise" if tilt < 0 else "rotate_clockwise"


def face_rotation_action(face, frame_shape=None):
    """Return 'rotate_clockwise' / 'rotate_anticlockwise' / 'rotate_180' / None.

    Single source of truth: ProcessMgr (render) and core (the Frame Editor crop
    preview) both call this, so the render and the preview can never disagree
    about orientation.
    """
    # 1. Keypoint midline — measured worst-case error 9.1 deg under any pose.
    axis = face_down_axis(face)
    if axis is not None:
        action = _action_for_down_axis(*axis)
        if action is not None:
            return action
        return None     # a trustworthy axis said "upright"; do not second-guess it

    # 2. 106-point midline, when the detector supplies it but keypoints are
    #    degenerate. forehead[72] -> chin[0] is the same midline measurement.
    lm106 = getattr(face, 'landmark_2d_106', None)
    if lm106 is not None and len(lm106) > 72:
        action = _action_for_down_axis(float(lm106[0][0]) - float(lm106[72][0]),
                                       float(lm106[0][1]) - float(lm106[72][1]))
        if action is not None:
            return action
        return None

    # 3. Last resort: a bbox wider than it is tall means a face on its side, but
    #    carries no sign, so guess from which half of the frame it sits in.
    if frame_shape is None:
        return None
    bbox_w = float(face.bbox[2]) - float(face.bbox[0])
    bbox_h = float(face.bbox[3]) - float(face.bbox[1])
    if bbox_w > 1.1 * bbox_h:
        width = frame_shape[1]
        bbox_cx = float(face.bbox[0]) + bbox_w / 2.0
        return "rotate_anticlockwise" if bbox_cx >= width / 2.0 else "rotate_clockwise"
    return None


def rotation_improves_upright(before_face, after_face):
    """True when re-detecting on the rotated frame actually stood the face up.

    The orientation call is a heuristic on noisy landmarks; this is the check
    that makes acting on it safe. A rotation that left the face no more upright
    than it started (or stood it on its head) is rejected, so a bad call costs
    an unrotated swap instead of a corrupted one.
    """
    before = face_roll_tilt(before_face)
    after = face_roll_tilt(after_face)
    if before is None or after is None:
        return True     # nothing to judge with; keep the pre-existing behaviour
    return abs(after) < abs(before) - 5.0


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
# Gate for 'stabilize' mode ONLY. Engages on near-profile faces (~70 deg+),
# deliberately TIGHTER than the 0.55 gate the masking path uses: the constrained
# fit disagrees with the plain least-squares fit by up to ~9 deg even at yaw 55,
# so a looser gate here would visibly change mid-angle faces that already look
# correct.
#
# 'pose' mode does NOT use this. It is keyed on a real yaw+pitch solve and fades
# in over an angle band instead — see POSE_ALIGN_ONSET_DEG. A yaw_ratio gate
# cannot serve it, because pitch inflates yaw_ratio: at yaw 90 / pitch -30 the
# ratio reads 0.411, above this threshold, so the most extreme pose in the range
# was being treated as a mid-angle face and left uncorrected.
YAW_ALIGN_RATIO = 0.40

# Three modes, selected by roop.globals.yaw_align:
#   'off'       — the plain fixed-template least-squares fit (default, unchanged)
#   'stabilize' — keep the frontal template, but take the rotation from the
#                 eye->mouth axis so pitch stops leaking into in-plane roll
#   'pose'      — replace the template itself with the reference head projected
#                 at the SOLVED yaw and pitch (see _pose_template), faded in
#                 over an off-axis band so the crop never jumps
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


def _project_reference(yaw_deg, pitch_deg=0.0):
    """The reference head's 5 points at a given yaw/pitch, orthographic, image
    axes. Pitch is applied after yaw (R = Rx @ Ry), matching the convention the
    pose solve below decomposes back out."""
    y, p = np.radians(yaw_deg), np.radians(pitch_deg)
    ry = np.array([[np.cos(y), 0.0, np.sin(y)],
                   [0.0, 1.0, 0.0],
                   [-np.sin(y), 0.0, np.cos(y)]])
    rx = np.array([[1.0, 0.0, 0.0],
                   [0.0, np.cos(p), -np.sin(p)],
                   [0.0, np.sin(p), np.cos(p)]])
    pts = _reference_5pt() @ (rx @ ry).T
    return np.column_stack([pts[:, 0], -pts[:, 1]])


# The reference head centred and flipped into image axes (+y DOWN), which is the
# space the observed keypoints live in. Built once: the pose solve runs per face
# per frame, and this is a constant.
def _reference_5pt_image():
    ref = np.asarray(_reference_5pt(), dtype=np.float64)
    ref = ref - ref.mean(axis=0)
    return ref * np.array([1.0, -1.0, 1.0])


_REF5_IMAGE = _reference_5pt_image()

# The reference head is CONSTANT, so the least-squares solve against it reduces
# to one precomputed pseudo-inverse and a 3x5 @ 5x2 matmul: 0.79 us against
# 6.49 us for np.linalg.lstsq, which redoes a full SVD on every call.
_REF5_PINV = np.linalg.pinv(_REF5_IMAGE)


def solve_pose_5pt(kps):
    """(yaw, pitch, roll) in degrees from the 5 arcface keypoints, or None.

    Why not the ratio lookups above: `yaw_ratio` and `pitch_ratio` are scalar
    proxies, and each is contaminated by the OTHER angle. Measured on the
    project's own reference head, pitch pushes yaw_ratio UP —

        yaw_ratio   yaw 0   yaw 45   yaw 75   yaw 90
        pitch   0    0.721    0.508    0.185    0.000
        pitch -30    0.906    0.693    0.460    0.411
        pitch -40    1.066    0.855    0.643    0.596

    — so a profile head that is ALSO tilted reads like a mid-angle head, and
    every gate keyed on yaw_ratio silently disengages on exactly the pose that
    needs it most (see `_maybe_constrained`, and the mask router's
    `is_non_frontal`). No threshold on a contaminated scalar can fix that; the
    two angles have to be separated.

    This solves for them jointly instead, by weak perspective (scaled
    orthographic): the observed points are x = s·R₂ₓ₃·X for the reference head
    X, so least-squares gives the 2x3 matrix directly, and orthonormalising its
    rows recovers R and s. Five points against six unknowns is well posed
    because the reference points are not coplanar — the nose tip stands proud of
    the eye/mouth plane, which is what carries the pitch and yaw signal.

    Accurate to <0.05 deg against synthetic projections over the whole
    yaw 0-90 x pitch +/-40 x roll +/-30 grid, including a true 90 deg profile
    where the eye separation has collapsed to zero and the ratio proxies are
    degenerate.
    """
    try:
        if kps is None:
            return None
        x = np.asarray(kps, dtype=np.float64)
        if x.shape != (5, 2) or not np.isfinite(x).all():
            return None
        x = x - x.mean(axis=0)
        if float((x ** 2).sum()) < 1e-9:
            return None            # all five points coincide

        # A maps the reference head onto the observed points; rows are s*r1, s*r2.
        A = _REF5_PINV @ x

        # The rest is 3-vector arithmetic, done in plain Python floats rather
        # than numpy. On 3-element vectors every numpy call costs far more in
        # dispatch than in arithmetic, and there are about ten of them here
        # (norm, dot, cross, clip, arcsin, two arctan2, degrees...). Writing
        # them out took this function from 42.7 us to 10.8 us — worth it because
        # it runs per face per frame in both the alignment and the mask router.
        a0x, a0y, a0z = float(A[0][0]), float(A[1][0]), float(A[2][0])
        a1x, a1y, a1z = float(A[0][1]), float(A[1][1]), float(A[2][1])

        n1 = math.sqrt(a0x * a0x + a0y * a0y + a0z * a0z)
        if not (n1 > 1e-9):
            return None
        # Nearest orthonormal pair: normalise the first row, then remove its
        # component from the second (Gram-Schmidt). r3 completes the frame.
        r1x, r1y, r1z = a0x / n1, a0y / n1, a0z / n1
        dot = a1x * r1x + a1y * r1y + a1z * r1z
        r2x, r2y, r2z = a1x - dot * r1x, a1y - dot * r1y, a1z - dot * r1z
        n2 = math.sqrt(r2x * r2x + r2y * r2y + r2z * r2z)
        if not (n2 > 1e-9):
            return None
        r2x, r2y, r2z = r2x / n2, r2y / n2, r2z / n2
        # r3 = r1 x r2, the third row of R = [r1; r2; r3].
        r3x = r1y * r2z - r1z * r2y
        r3y = r1z * r2x - r1x * r2z
        r3z = r1x * r2y - r1y * r2x

        pitch = math.degrees(math.asin(max(-1.0, min(1.0, -r3y))))
        yaw = -math.degrees(math.atan2(r3x, r3z))
        roll = math.degrees(math.atan2(r1y, r2y))
        if not (math.isfinite(yaw) and math.isfinite(pitch) and math.isfinite(roll)):
            return None
        return yaw, pitch, roll
    except Exception:
        return None


def offaxis_deg(yaw_deg, pitch_deg):
    """How far the head has turned away from the camera, combining yaw and pitch
    into the single angle between the face's forward axis and the view axis.

    Using this instead of |yaw| is what makes a turned-AND-tilted head rank as
    the extreme pose it is: yaw 75 / pitch 40 is 79 deg off-axis, not 75."""
    c = (np.cos(np.radians(yaw_deg)) * np.cos(np.radians(pitch_deg)))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


# NOTE: a yaw-ratio -> yaw lookup table used to live here, because the 'pose'
# mode read the head's yaw by inverting that scalar. It has no callers since
# solve_pose_5pt replaced it: inverting yaw_ratio could not recover pitch at all,
# and could not even recover YAW on a face that was pitched, because pitch
# inflates the ratio (see solve_pose_5pt). Removed rather than left as a second,
# worse way to ask the same question.


def _pose_template(yaw_deg, base_dst, pitch_deg=0.0):
    """The destination points for a head at `yaw_deg`/`pitch_deg`, placed to
    match the frontal template's centroid and scale.

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

    Pitch matters as much as yaw and used to be missing here. A head tilted up
    or down foreshortens the eye->mouth distance, and the fixed frontal template
    answers by stretching the crop to put the mouth back where a level head's
    would be. Measured crop scale over yaw 0-90 x pitch +/-40:

        fixed template   0.864 .. 1.199   (1.39x swing)
        this template    1.004 .. 1.005   (1.00x swing)

    A 39% swing means the pasted face changes size as the head moves — it
    breathes — which reads as both misalignment and per-frame wobble. Holding
    the crop scale flat is most of what makes an angled swap sit still.
    """
    posed = _project_reference(yaw_deg, pitch_deg)
    frontal = _project_reference(0.0, 0.0)
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


# Where the 'pose' template fades in and out, as an off-axis angle in degrees
# (see offaxis_deg). Below ONSET the result is bit-identical to the default fit;
# above FULL the template is entirely pose-matched; between, the two templates
# are crossfaded.
#
# There is a band rather than a threshold because a threshold FLICKERS. The
# crop geometry either side of a hard gate differs by a finite jump, so a head
# sitting near it — or detector noise on a head that is not moving at all —
# pops between two different transforms frame to frame. A crossfade makes the
# alignment a continuous function of pose, so no per-frame pose wobble can
# produce a discontinuous crop. Measured worst per-frame crop jump along a
# 0->90 deg turn with a nod riding on it: 1.09 deg rotation / 0.77% scale with
# the fixed template, 0.00 / 0.00% with this one.
#
# The band is also not free at the low end, which is what stops it starting at
# zero. This template is the REFERENCE head's projection, while arcface_dst is
# an empirical template, and the two differ even at zero pose — mean 7.7 px,
# max 12.2 px on a 512 crop, worth -1.28% of crop scale. That gap is a fixed
# cost paid by every face, and real faces vary in eye-separation and
# eye-to-mouth proportion, so near frontal the fixed template is the safer of
# the two: its least-squares fit spreads the discrepancy over all five points
# instead of forcing one head's proportions onto everybody.
#
# So engage where the pose error clearly exceeds that fixed cost. Pose error
# grows with off-axis angle (crop scale is ~2% off at 20 deg, ~4.5% at 30,
# ~10% at 40) while the shape mismatch stays ~1.3%. Measured crop-scale swing
# over yaw 0-90 x pitch +/-40, and per-frame jitter under 1 px keypoint noise:
#
#   band            swing    frontal   jitter @frontal / @yaw45 / @yaw90+tilt
#   fixed template  1.389x       —        0.423%   0.479%   0.907%
#   onset 20 full 50 1.080x    unchanged  0.451%   0.609%   0.587%
#   onset 15 full 40 1.071x    unchanged  0.413%   0.538%   0.582%
#
# 15/40 is the better of the two on every measure, and reaches full correction
# by 40 deg — which matters because a head tilted up or down 40 deg is exactly
# the case that used to get no correction at all.
POSE_ALIGN_ONSET_DEG = 15.0
POSE_ALIGN_FULL_DEG = 40.0


def _smoothstep(edge0, edge1, x):
    """Hermite 3t^2-2t^3 ramp. Chosen over a linear ramp because its derivative
    is zero at BOTH ends, so the template blend has no slope kink where it
    starts or finishes — a kink would show up as a small but visible change in
    how the crop tracks a turning head."""
    if edge1 <= edge0:
        return 0.0 if x < edge1 else 1.0
    t = (float(x) - edge0) / (edge1 - edge0)
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def _weight_for_pose(yaw, pitch):
    return _smoothstep(POSE_ALIGN_ONSET_DEG, POSE_ALIGN_FULL_DEG,
                       offaxis_deg(yaw, pitch))


def pose_align_weight(lmk):
    """How much pose-matched template to blend in for these keypoints, 0..1.

    Returns 0.0 for frontal faces, which is what keeps the alignment bit-exact
    where nothing needs correcting."""
    pose = solve_pose_5pt(lmk)
    if pose is None:
        return 0.0
    return _weight_for_pose(pose[0], pose[1])


def _maybe_constrained(lmk, dst):
    """Pose-specific alignment, per the selected mode. Returns None to mean
    'use the normal fit' — which is every frontal face, and every face at all
    when the mode is 'off'."""
    mode = _yaw_align_mode()
    if mode == 'off':
        return None

    lmk = np.asarray(lmk, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)

    if mode == 'pose':
        # Solve yaw and pitch properly rather than inverting a yaw-only scalar.
        # The old path read yaw from `yaw_ratio`, which pitch inflates, so it
        # both missed tilted profiles entirely and modelled every face as
        # perfectly level — leaving the up/down case uncorrected.
        # Solved once and reused: this runs per face per frame, and
        # pose_align_weight would otherwise repeat the whole solve.
        pose = solve_pose_5pt(lmk)
        if pose is None:
            return None
        yaw, pitch, _ = pose
        w = _weight_for_pose(yaw, pitch)
        if w <= 0.0:
            return None                      # frontal: leave the default fit alone
        posed = _pose_template(yaw, dst, pitch)
        if posed is None:
            return None
        # Blend the TEMPLATES, not the two fitted matrices. Interpolating target
        # landmark positions is a geometric operation that stays a valid
        # configuration at every w; averaging two affine matrices is not, and
        # can shear the result mid-blend.
        target = (1.0 - w) * dst + w * np.asarray(posed, dtype=np.float64)
        # A pose-matched template is congruent to the input, so the ordinary
        # least-squares similarity is already well posed — no rotation
        # constraint needed on top.
        tform = trans.SimilarityTransform()
        tform.estimate(lmk, target)
        m = tform.params[0:2, :]
        return m if np.isfinite(m).all() else None

    # 'stabilize' keeps its own tighter near-profile gate: unlike the pose
    # template, its rotation constraint does NOT converge to the default fit as
    # the face approaches frontal (it disagrees by ~9 deg even at yaw 55), so it
    # cannot be faded in from a low angle without visibly changing mid angles.
    yaw_ratio, _ = kps_pose_ratios(lmk)
    if yaw_ratio is None or yaw_ratio >= YAW_ALIGN_RATIO:
        return None
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
    # Replicate the frame edge instead of filling with BLACK.
    #
    # This crop is the swap model's input. When a face runs off the edge of the
    # frame — the "only part of the face is visible" case — the aligned crop
    # samples outside the image, and the default BORDER_CONSTANT/0 filled that
    # with pure black: measured 22,449 of 65,536 pixels (34% of the crop) for a
    # face at the left edge. The swap models were trained on complete crops, so
    # a third of the input being a hard black wedge is far outside their
    # distribution, and whatever they hallucinate over it gets pasted back onto
    # the frame. Replicated edge pixels are not correct either, but they are
    # continuous with the face and in-distribution, which is what the models can
    # actually cope with — and it is what the paste-back path (procmgr_masking)
    # and every other warp in the pipeline already use.
    #
    # Costs nothing anywhere else: warpAffine only consults the border mode for
    # samples that fall OUTSIDE the source image, so for a face fully inside the
    # frame this is bit-identical to the old call (verified).
    warped = cv2.warpAffine(img, M, (image_size, image_size),
                            borderMode=cv2.BORDER_REPLICATE)
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

