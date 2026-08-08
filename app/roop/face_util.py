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
from roop.nms import bind_instance_nms

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
    else:
        # The default engine decodes and suppresses inside insightface (det_10g
        # routes to its RetinaFace class, whatever our engine name says), so its
        # own nms() is the one that would delete the second of two touching
        # faces. Give this INSTANCE the shared rule the other engines use
        # (identical signature and contract; site-packages untouched) —
        # otherwise the DEFAULT engine would be the only one still dropping
        # them. See roop/nms.py.
        bind_instance_nms(fa.det_model)
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
# HISTORICAL. This was the hard gate for 'stabilize' mode. Nothing gates on it
# any more — both modes now key on a real yaw+pitch solve and fade in over an
# angle band (POSE_ALIGN_ONSET_DEG, STAB_ALIGN_ONSET_DEG) — but the constant is
# kept because the tests use it to pin down how badly the proxy misleads: pitch
# inflates yaw_ratio, so at yaw 90 / pitch -30 the ratio reads 0.411, above this
# threshold, and the most extreme pose in the range was read as a mid-angle face
# and left uncorrected. See solve_pose_5pt.
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


def _project_reference(yaw_deg, pitch_deg=0.0, jaw=0.0):
    """The reference head's 5 points at a given yaw/pitch, orthographic, image
    axes. Pitch is applied after yaw (R = Rx @ Ry), matching the convention the
    pose solve below decomposes back out.

    `jaw` slides the two mouth corners down the head's own eye->mouth axis, as a
    fraction of that distance, BEFORE the rotation — the sixth degree of freedom
    the five keypoints actually have (see solve_pose_jaw_5pt). jaw = 0 leaves
    the arithmetic untouched, so every existing caller is bit-identical.
    """
    y, p = np.radians(yaw_deg), np.radians(pitch_deg)
    ry = np.array([[np.cos(y), 0.0, np.sin(y)],
                   [0.0, 1.0, 0.0],
                   [-np.sin(y), 0.0, np.cos(y)]])
    rx = np.array([[1.0, 0.0, 0.0],
                   [0.0, np.cos(p), -np.sin(p)],
                   [0.0, np.sin(p), np.cos(p)]])
    ref = _reference_5pt()
    if jaw:
        ref = np.asarray(ref, dtype=np.float64).copy()
        drop = float(jaw) * ((ref[3] + ref[4]) / 2.0 - (ref[0] + ref[1]) / 2.0)
        ref[3] = ref[3] + drop
        ref[4] = ref[4] + drop
    pts = ref @ (rx @ ry).T
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


# ── The jaw axis, and the normal equations as polynomials in it ──────────────
# The mouth corners are two of the five keypoints and the jaw is a moving part,
# so the observed points are a SIX degree-of-freedom family, not five: yaw,
# pitch, roll, scale and translation, plus how far the mouth has opened. This is
# that sixth direction — both mouth corners sliding down the head's own
# eye->mouth axis — expressed in the same centred image-axis frame as
# _REF5_IMAGE, so a head with the jaw at `j` is _REF5_IMAGE + j * _JAW5.
def _jaw_axis_5pt():
    ref = np.asarray(_REF5_IMAGE, dtype=np.float64)
    drop = (ref[3] + ref[4]) / 2.0 - (ref[0] + ref[1]) / 2.0
    axis = np.zeros((5, 3), dtype=np.float64)
    axis[3] = drop
    axis[4] = drop
    return axis


_JAW5 = _jaw_axis_5pt()
# Centred, because a pose solve removes translation before it does anything else.
_JAW5C = _JAW5 - _JAW5.mean(axis=0)

# X(j) = _REF5_IMAGE + j*_JAW5C, so the normal-equation matrix X(j)^T X(j) is a
# QUADRATIC in j and the right-hand side X(j)^T x is LINEAR in it. Precomputing
# the coefficients turns each Newton step below into 3x3 arithmetic on constants
# instead of re-forming and re-factorising a least-squares problem.
_JQ0 = _REF5_IMAGE.T @ _REF5_IMAGE
_JQ1 = _REF5_IMAGE.T @ _JAW5C + _JAW5C.T @ _REF5_IMAGE
_JQ2 = _JAW5C.T @ _JAW5C

# Unpacked into plain floats for the hot loop, for the same reason the tail of
# solve_pose_5pt is: on 3x3 data every numpy call costs more in dispatch than in
# arithmetic.
_JQ0F = tuple(float(v) for v in (_JQ0[0, 0], _JQ0[0, 1], _JQ0[0, 2],
                                 _JQ0[1, 1], _JQ0[1, 2], _JQ0[2, 2]))
_JQ1F = tuple(float(v) for v in (_JQ1[0, 0], _JQ1[0, 1], _JQ1[0, 2],
                                 _JQ1[1, 1], _JQ1[1, 2], _JQ1[2, 2]))
_JQ2F = tuple(float(v) for v in (_JQ2[0, 0], _JQ2[0, 1], _JQ2[0, 2],
                                 _JQ2[1, 1], _JQ2[1, 2], _JQ2[2, 2]))

# How far the mouth is allowed to travel, as a fraction of the eye->mouth
# distance. Negative because the parameter also absorbs a real difference in
# eye-to-mouth PROPORTION between the reference head and the person in frame —
# which is welcome, it is exactly the "one head's proportions forced onto
# everybody" cost the pose template otherwise pays — but only within the range
# real anatomy spans. Outside it the solve has gone somewhere the model does not
# describe, and clamping keeps a bad frame merely uncorrected instead of wrong.
JAW_SOLVE_MIN = -0.35
JAW_SOLVE_MAX = 0.90


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
        out = _decompose_projection(A)
        if out is None:
            return None
        return out[0], out[1], out[2]
    except Exception:
        return None


def _decompose_projection(A):
    """(yaw, pitch, roll, scale) from a 3x2 weak-perspective matrix, or None.

    THE single place this project turns a fitted projection into angles. Both
    solvers call it, deliberately: two pose sources that disagreed about which
    way a head faces is a bug this project has already had, and it stayed
    invisible for a long time (see face_3d_recon's 180-degree offset). A second
    copy of this arithmetic would be the same trap with a shorter fuse.

    `A` is indexed [row][col] and may be a numpy array or nested tuples, so the
    jaw solve can hand over plain floats without building an array for it.

    Written out in plain Python floats rather than numpy: on 3-element vectors
    every numpy call costs far more in dispatch than in arithmetic, and there
    are about ten of them here (norm, dot, cross, clip, arcsin, two arctan2,
    degrees...). Doing it this way took solve_pose_5pt from 42.7 us to 10.8 —
    worth it because that runs per face per frame in both the alignment and the
    mask router.
    """
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
    r3x = r1y * r2z - r1z * r2y
    r3y = r1z * r2x - r1x * r2z
    r3z = r1x * r2y - r1y * r2x

    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -r3y))))
    yaw = -math.degrees(math.atan2(r3x, r3z))
    roll = math.degrees(math.atan2(r1y, r2y))
    if not (math.isfinite(yaw) and math.isfinite(pitch) and math.isfinite(roll)):
        return None
    return yaw, pitch, roll, 0.5 * (n1 + n2)


# How many Gauss-Newton steps to take on the jaw, once the right basin is known.
# Worst pose error over |yaw| 5-90 x pitch +/-40 x roll +/-90, jaw values chosen
# deliberately NOT to land on the scan grid below:
#
#   steps      2        3        4        5        6
#   worst  0.1827   0.0019   0.0000   0.0000   0.0000  deg
#
# Four converges; five is one step of margin, and the loop exits early on any
# frame that settles sooner, which is most of them.
#
# Measure this with jaw values OFF the scan grid. A grid whose jaws happen to
# coincide with the scan points reports 0.0000 at every step count, because the
# scan lands on the answer and the refinement has nothing left to do — which
# says nothing at all about whether the refinement works.
_JAW_NEWTON_STEPS = 5

# The rotation-consistency error is NOT unimodal in the jaw. Besides the true
# minimum there is a second, shallower one 0.4 to 0.8 higher, which reads as a
# head tilted ~35 deg further back with the mouth open wider. Two facts about it
# set the search below, and both were measured rather than assumed:
#
#   * Its floor (error ~1e-4) is well BELOW the true basin's shoulders (~5e-3 a
#     tenth of a jaw away from the true minimum, whose floor is ~1e-31). So
#     ranking sampled points does not work — a grid that misses the true
#     minimum by a little ranks the decoy first. Only floors may be compared
#     with floors, which means refining before choosing.
#   * Gauss-Newton settles into whichever basin it starts in, and no small fixed
#     set of starting points covers both for every pose. Starting only at 0
#     misreads every yaw-0 head tilted UP (over a 9360-pose grid, 104 misreads,
#     all at pitch +40, worst 8.6 deg); adding a second and a third fixed start
#     moved the failures around rather than removing them.
#
# So: sample the range on a grid comfortably finer than the basin spacing, take
# the best sample AND the best sample from the other basin, refine both, and
# choose on the error each one actually reaches.
_JAW_SCAN_STEP = 0.1
_JAW_BASIN_SEP = 0.25

# And when refinement finds that both basins fit EXACTLY, which happens.
#
# At yaw 0 the ambiguity is not numerical, it is real. A frontal head projects
# its eyes to one height, its mouth to another and its nose in between, and the
# nose's sideways offset — the thing that separates a tilt from a jaw once the
# head has turned — is zero. Two unknowns, two informative heights, a quadratic
# system, two roots. Measured on an exact projection of a head at pitch +10 with
# the mouth 5% closed: jaw -0.05 gives error 1.1e-31, and jaw +0.83 with pitch
# +66.7 gives 2.1e-31. Both are the truth as far as five points can tell.
#
# So the choice is a prior, and there is an obvious one. The decoy is ALWAYS the
# root with the WIDER mouth — it sits 0.4 to 0.8 above the true jaw at every
# pose sampled — and the two readings are not equally costly to get wrong:
# taking the more closed mouth on a head that really is tilted back merely
# leaves the alignment uncorrected, while taking the wider one on an ordinary
# face swings the template through 57 degrees of pitch that is not there. So
# prefer the most closed reading, and let a rival overturn it only by fitting
# distinctly better. (Ordered on the signed jaw, not its magnitude: when the
# true jaw is negative — a face whose mouth simply sits higher than the
# reference head's — the decoy is the one NEARER zero, and ranking by magnitude
# picks it.)
_JAW_TIE_RATIO = 0.5
# The floor exists so that two roots which BOTH fit exactly are not ranked on
# the ratio of two numbers that are pure rounding noise — at that point the
# comparison is meaningless and the prior should simply win.
#
# It has to sit near the true floor of an exact fit (~1e-30), NOT at some
# generously small number. 1e-8 was tried and is a bug: a candidate that has
# converged to within 0.06 of the right jaw scores ~1e-9, which is twenty-one
# orders of magnitude WORSE than the exact root sitting next to it and is in no
# sense a tie — but the floor called it perfect, stopped the comparison, and
# kept it. That cost 14 misreads of up to 5.9 deg on the pose grid, every one of
# them a case where the right answer had already been computed and was thrown
# away. On real footage the error never approaches either number (it runs 1e-2
# to 1e-1, dominated by how far a real face is from the reference head), so the
# floor is inert there and the ratio does all the work.
_JAW_TIE_FLOOR = 1e-20


def _jaw_fit_at(j, b, q, want_step=True):
    """Fit the reference head with its jaw at `j`, and say how far the result is
    from being a rigid pose.

    Returns `(A, err, step)`: the 3x2 weak-perspective matrix, the squared
    rotation-consistency error (scale-free, so it is comparable across
    candidates), and the Gauss-Newton step that would reduce it — or None if the
    normal equations are singular. `want_step=False` skips the derivative, which
    is about 40% of the work and is not needed while merely scoring candidates.

    `b` is the right-hand side split into its constant and jaw-linear parts, `q`
    the three quadratic coefficients of X(j)^T X(j). Both are precomputed by the
    caller because they do not depend on j.
    """
    (b0_00, b0_01, b0_10, b0_11, b0_20, b0_21,
     b1_00, b1_01, b1_10, b1_11, b1_20, b1_21) = b
    q0, q1, q2 = q

    jj = j * j
    g11 = q0[0] + j * q1[0] + jj * q2[0]
    g12 = q0[1] + j * q1[1] + jj * q2[1]
    g13 = q0[2] + j * q1[2] + jj * q2[2]
    g22 = q0[3] + j * q1[3] + jj * q2[3]
    g23 = q0[4] + j * q1[4] + jj * q2[4]
    g33 = q0[5] + j * q1[5] + jj * q2[5]

    # Adjugate of the symmetric 3x3, then Cramer.
    c11 = g22 * g33 - g23 * g23
    c12 = g13 * g23 - g12 * g33
    c13 = g12 * g23 - g13 * g22
    det = g11 * c11 + g12 * c12 + g13 * c13
    if not (abs(det) > 1e-12):
        return None
    c22 = g11 * g33 - g13 * g13
    c23 = g13 * g12 - g11 * g23
    c33 = g11 * g22 - g12 * g12
    inv = 1.0 / det

    # A = G^-1 B, columns a0 (image x) and a1 (image y).
    a00 = (c11 * b0_00 + c12 * b0_10 + c13 * b0_20 + j * (c11 * b1_00 + c12 * b1_10 + c13 * b1_20)) * inv
    a10 = (c12 * b0_00 + c22 * b0_10 + c23 * b0_20 + j * (c12 * b1_00 + c22 * b1_10 + c23 * b1_20)) * inv
    a20 = (c13 * b0_00 + c23 * b0_10 + c33 * b0_20 + j * (c13 * b1_00 + c23 * b1_10 + c33 * b1_20)) * inv
    a01 = (c11 * b0_01 + c12 * b0_11 + c13 * b0_21 + j * (c11 * b1_01 + c12 * b1_11 + c13 * b1_21)) * inv
    a11 = (c12 * b0_01 + c22 * b0_11 + c23 * b0_21 + j * (c12 * b1_01 + c22 * b1_11 + c23 * b1_21)) * inv
    a21 = (c13 * b0_01 + c23 * b0_11 + c33 * b0_21 + j * (c13 * b1_01 + c23 * b1_11 + c33 * b1_21)) * inv

    # The fit is a scaled rotation exactly when its two rows have equal norm AND
    # are orthogonal, so the residual is the PAIR
    #
    #     u = |a0|^2 - |a1|^2        v = 2 (a0 . a1)
    #
    # and the jaw is the j that drives both to zero. Both terms are needed: an
    # in-plane roll of theta turns (u, v) through 2*theta, so u alone is
    # identically zero at 45 degrees of roll, and a root find on it wanders off
    # to the clamp. u^2 + v^2 is roll-invariant, which a head lying on its side
    # in shot makes a requirement rather than a nicety.
    n0 = a00 * a00 + a10 * a10 + a20 * a20
    n1 = a01 * a01 + a11 * a11 + a21 * a21
    u = n0 - n1
    v = 2.0 * (a00 * a01 + a10 * a11 + a20 * a21)
    s2 = 0.5 * (n0 + n1)
    if not (s2 > 1e-12):
        return None
    # Divided by the scale so the error means the same thing on a face 40 px
    # across and one 400 px across — otherwise the scan below would compare
    # candidates by how big the face is.
    err = (u * u + v * v) / (s2 * s2)
    if not want_step:
        return ((a00, a01), (a10, a11), (a20, a21)), err, 0.0

    # dA/dj = G^-1 (B' - G' A) reuses the same inverse, so the exact derivative
    # costs one more application rather than a second solve.
    p11 = q1[0] + 2.0 * j * q2[0]
    p12 = q1[1] + 2.0 * j * q2[1]
    p13 = q1[2] + 2.0 * j * q2[2]
    p22 = q1[3] + 2.0 * j * q2[3]
    p23 = q1[4] + 2.0 * j * q2[4]
    p33 = q1[5] + 2.0 * j * q2[5]

    r00 = b1_00 - (p11 * a00 + p12 * a10 + p13 * a20)
    r10 = b1_10 - (p12 * a00 + p22 * a10 + p23 * a20)
    r20 = b1_20 - (p13 * a00 + p23 * a10 + p33 * a20)
    r01 = b1_01 - (p11 * a01 + p12 * a11 + p13 * a21)
    r11 = b1_11 - (p12 * a01 + p22 * a11 + p23 * a21)
    r21 = b1_21 - (p13 * a01 + p23 * a11 + p33 * a21)

    d00 = (c11 * r00 + c12 * r10 + c13 * r20) * inv
    d10 = (c12 * r00 + c22 * r10 + c23 * r20) * inv
    d20 = (c13 * r00 + c23 * r10 + c33 * r20) * inv
    d01 = (c11 * r01 + c12 * r11 + c13 * r21) * inv
    d11 = (c12 * r01 + c22 * r11 + c23 * r21) * inv
    d21 = (c13 * r01 + c23 * r11 + c33 * r21) * inv

    du = 2.0 * ((a00 * d00 + a10 * d10 + a20 * d20)
                - (a01 * d01 + a11 * d11 + a21 * d21))
    dv = 2.0 * ((d00 * a01 + d10 * a11 + d20 * a21)
                + (a00 * d01 + a10 * d11 + a20 * d21))
    # Gauss-Newton on (u, v) jointly — a plain Newton on either one alone
    # inherits that term's blind roll angle.
    den = du * du + dv * dv
    step = 0.0 if not (den > 1e-12) else -(u * du + v * dv) / den
    if not math.isfinite(step):
        step = 0.0
    return ((a00, a01), (a10, a11), (a20, a21)), err, step


def solve_pose_jaw_5pt(kps):
    """(yaw, pitch, roll, jaw) in degrees + jaw fraction, or None.

    Same weak-perspective model as solve_pose_5pt, with the mouth allowed to
    open. That matters because it is not a small correction:

        frontal head, jaw dropping, NOTHING else moving
        jaw drop        0%     15%     30%     45%     60%
        solve_pose_5pt  0.0   -11.0   -18.7   -24.1   -28.0 deg of "pitch"

    A talking head reports up to 28 degrees of pitch that is not there, and
    _pose_template then projects the reference head into a position the person
    is not in — which is worse than not correcting at all. Two consequences, both
    measured on REAL detected keypoints (seven faces from this project's test
    clip, mouth opened synthetically so nothing else changes):

      * The crop breathes. Opening the mouth 45% of the eye-to-mouth distance
        moves the reported pitch by up to 17 deg and swings the crop scale
        1.36x on average, 1.46x worst. With the jaw solved out, the reported
        pitch is constant to 0.1 deg and the swing is 1.03x / 1.06x.
      * The correction fires on faces that have not moved. A frontal head with
        the mouth 45% open reported 24 deg of pitch, which is most of the way
        through the 15->40 deg engagement band, so the template was being
        swapped in on the strength of an expression. It now scores 0 and the
        alignment is bit-identical to leaving the mode off.

    Note what this does NOT claim: on turned heads the correction was mostly
    engaging for real reasons. Across 18 high-angle frames sampled from that
    clip the engagement verdict does not change on any of them, and the pitch
    moves by only a few degrees — those heads really are tilted. The win is
    steadiness across an expression, not a different set of frames.

    WHY A LINEAR SOLVE CANNOT DO THIS, and this one can. After centring, the
    jaw displacement lies EXACTLY in the span of the reference head's own
    coordinates (least-squares residual 4e-16 — see tests/test_open_mouth_pose),
    so adding a jaw column to the linear system changes nothing: both eyes sit
    at one y and both mouth corners at another, and dropping the jaw is a
    reweighting of that axis, which is what pitch is. What breaks the tie is the
    ROTATION CONSTRAINT — a jaw drop leaves a residual no rigid rotation can
    produce. The unconstrained fit answers a jaw drop by stretching the y axis,
    so its two projection rows come out unequal and non-orthogonal; driving that
    inconsistency to zero is a one-dimensional problem in the jaw. Not a convex
    one, though — see _JAW_SCAN_STEP and _JAW_TIE_RATIO for the two basins and
    the genuine yaw-0 ambiguity, which is most of what the code below is doing.

    ACCURACY, and where it is worst. Exact to 0.0000 deg over |yaw| 5-90 x
    pitch +/-40 x roll +/-90 x jaw -0.21..0.74. At |yaw| BELOW about 5 it can be
    wrong by up to 36 deg, because that is where the two readings described under
    _JAW_TIE_RATIO stop being distinguishable — 28 of 9576 grid poses, every one
    of them at yaw 0, half over-reading the pitch and half under-reading it.
    That sounds bad until it is compared with the alternative, which is what
    matters here. Under 0.5 px of keypoint noise on a 180 px face:

        pitch error, mean / p95 / sd across draws
        yaw pitch   solve_pose_5pt          this
          0     0   16.5 / 28.2 / 10.0      0.5 /  1.1 / 0.6
          0    20   22.3 / 39.1 / 13.9      3.3 / 34.6 / 9.0   <- the bad case
          0    40   25.1 / 46.3 / 16.4      6.2 / 12.9 / 5.8
         15    40   24.5 / 45.3 / 16.0      1.5 /  3.9 / 1.8
         60    40   16.5 / 31.1 / 10.9      0.4 /  0.9 / 0.5

    It is better on the mean and on the spread in every condition, including the
    bad one. The spread is the part that shows on video: what reaches the render
    is the engagement weight, and on a STILL head at yaw 15 / pitch 40 (true
    weight 1.0) the jaw-blind pose delivers 0.341 +/- 0.404 — the correction
    fading in and out on a face that is not moving — against 1.000 +/- 0.001
    here. At yaw 0 / pitch 0 (true weight 0.0) it is 0.178 +/- 0.210 against
    0.000 +/- 0.000. Most of what looked like pose-mode flicker was this.

    Costs ~9x solve_pose_5pt (112 us against 13), so it is used only where it
    changes an outcome: the 'pose' and 'stabilize' alignment templates, both
    opt-in. solve_pose_5pt stays the answer everywhere else — the mask router,
    the mouth and eye restore fades, the tracking gates — unchanged and
    bit-exact. Those read pose to decide how far a head has TURNED, where a few
    degrees of jaw-borrowed pitch changes no verdict; the templates are the one
    consumer that projects a head at the angle it is handed.
    """
    try:
        if kps is None:
            return None
        x = np.asarray(kps, dtype=np.float64)
        if x.shape != (5, 2) or not np.isfinite(x).all():
            return None
        x = x - x.mean(axis=0)
        if float((x ** 2).sum()) < 1e-9:
            return None

        # Right-hand side, also linear in j: X(j)^T x = B0 + j*B1.
        B0 = _REF5_IMAGE.T @ x
        B1 = _JAW5C.T @ x
        b0_00, b0_01 = float(B0[0][0]), float(B0[0][1])
        b0_10, b0_11 = float(B0[1][0]), float(B0[1][1])
        b0_20, b0_21 = float(B0[2][0]), float(B0[2][1])
        b1_00, b1_01 = float(B1[0][0]), float(B1[0][1])
        b1_10, b1_11 = float(B1[1][0]), float(B1[1][1])
        b1_20, b1_21 = float(B1[2][0]), float(B1[2][1])

        b = (b0_00, b0_01, b0_10, b0_11, b0_20, b0_21,
             b1_00, b1_01, b1_10, b1_11, b1_20, b1_21)
        q = (_JQ0F, _JQ1F, _JQ2F)

        # 1. Score the whole range cheaply, and pick one candidate per basin.
        scored = []
        jc = JAW_SOLVE_MIN
        while jc <= JAW_SOLVE_MAX + 1e-9:
            got = _jaw_fit_at(jc, b, q, False)
            if got is not None:
                scored.append((got[1], jc))
            jc += _JAW_SCAN_STEP
        if not scored:
            return None
        scored.sort()
        starts = [scored[0][1]]
        for _err, cand in scored[1:]:
            if abs(cand - starts[0]) >= _JAW_BASIN_SEP:
                starts.append(cand)
                break

        # 2. Refine each, and let them compete on the error they reach.
        refined = []
        for start in starts:
            j = start
            for _step in range(_JAW_NEWTON_STEPS):
                got = _jaw_fit_at(j, b, q)
                if got is None:
                    break
                # A trust region, because Gauss-Newton on a near-flat error can
                # throw the jaw across the whole range in one step and not come
                # back.
                step = max(-0.4, min(0.4, got[2]))
                j_new = max(JAW_SOLVE_MIN, min(JAW_SOLVE_MAX, j + step))
                settled = abs(j_new - j) < 1e-5
                j = j_new
                if settled:
                    break
            # Re-fit at the jaw this start settled on — both to score it and so
            # the pose returned and the jaw returned describe the same head
            # rather than being one Gauss-Newton step apart.
            got = _jaw_fit_at(j, b, q)
            if got is not None:
                refined.append((got[1], j, got[0]))

        if not refined:
            return None
        # 3. The most closed mouth wins unless another fits distinctly better —
        #    see _JAW_TIE_RATIO for why that is the safe way round.
        refined.sort(key=lambda c: c[1])
        best_err, best_j, best_A = refined[0]
        for err, j, A in refined[1:]:
            if best_err <= _JAW_TIE_FLOOR:
                break
            if err < best_err * _JAW_TIE_RATIO:
                best_err, best_j, best_A = err, j, A

        out = _decompose_projection(best_A)
        if out is None:
            return None
        return out[0], out[1], out[2], best_j
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


def _pose_template(yaw_deg, base_dst, pitch_deg=0.0, jaw=0.0):
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

    The MOUTH does the same thing, for the same reason, and it moves far more
    often than the head does. The template is therefore projected with the jaw
    the solve found (see solve_pose_jaw_5pt) while the scale still comes from
    the CLOSED-mouth frontal reference — so an opening mouth moves the
    template's mouth corners down to meet it instead of shrinking the whole crop
    to drag them back up. Crop-scale swing as the mouth opens from shut to 60%
    on an otherwise still head at yaw 50 / pitch -20:

        no correction   1.532x
        this template   1.000x

    On a FRONTAL head the same sweep stays at 1.475x, and that is not a failure
    to fix it — it is this mode declining to act. A head that has not turned
    scores zero on the engagement band, so the alignment is bit-identical to
    leaving the mode off, mouth or no mouth. Correcting it there would mean
    re-framing every talking close-up away from the arcface crop the swap models
    were trained on, which is a change to argue on output, not on geometry.
    """
    posed = _project_reference(yaw_deg, pitch_deg, jaw)
    frontal = _project_reference(0.0, 0.0)
    frontal_vert = np.linalg.norm((frontal[3] + frontal[4]) / 2.0
                                  - (frontal[0] + frontal[1]) / 2.0)
    base_vert = np.linalg.norm((base_dst[3] + base_dst[4]) / 2.0
                               - (base_dst[0] + base_dst[1]) / 2.0)
    if frontal_vert < 1e-9:
        return None
    scaled = posed * (base_vert / frontal_vert)
    return scaled - scaled.mean(axis=0) + np.asarray(base_dst, np.float64).mean(axis=0)


def _pose_placement(yaw_deg, pitch_deg, base_dst, jaw=0.0):
    """The (scale, shift) that `_pose_template` places the reference head with.

    Exists so anything ELSE derived from the reference head can be put in the
    same crop coordinates as the alignment template — which is the whole basis
    for `pose_visibility_polygon` being able to say where the face is in a crop.
    Returns (None, None) when the reference is degenerate.

    `_pose_template` is deliberately NOT rewritten in terms of this: it is the
    tested alignment path and re-associating its arithmetic would perturb the
    last bits. A test instead pins the two together, so they cannot drift.
    """
    posed = _project_reference(yaw_deg, pitch_deg, jaw)
    frontal = _project_reference(0.0, 0.0)
    frontal_vert = np.linalg.norm((frontal[3] + frontal[4]) / 2.0
                                  - (frontal[0] + frontal[1]) / 2.0)
    base_vert = np.linalg.norm((base_dst[3] + base_dst[4]) / 2.0
                               - (base_dst[0] + base_dst[1]) / 2.0)
    if frontal_vert < 1e-9:
        return None, None
    scale = float(base_vert / frontal_vert)
    shift = (np.asarray(base_dst, np.float64).mean(axis=0)
             - (posed * scale).mean(axis=0))
    return scale, shift


# ── Where the face SURFACE is, at a pose ─────────────────────────────────────
# The paste matte says where to put swapped pixels. Its two terms — an ellipse in
# canonical crop space, and the convex hull of the 106 landmarks — both describe
# a FRONTAL face's footprint, and neither knows that a turned head hides half of
# its own face. The 106-pt model predicts the far-side contour whether or not it
# is behind the skull, so on a profile that hull reaches back over hair and neck,
# and the swap gets pasted there.
#
# Measured in crop space, with pose-matched alignment active, the matte covers
# this much more than the visible face surface:
#
# SIZE AND SHAPE OF THE EFFECT, honestly, on the third attempt at describing it.
#
# It was reported as "with the trim on, only the middle of the face is swapped
# and the rest keeps the original texture", with a screenshot showing a seam
# across the brow. Measured against the real matte and split by region, over
# yaw 0-75 x pitch -40..+20:
#
#   below the brow line   0.0%   at every pose tested
#   forehead              0.0 - 8.1%
#
# ALL of it was forehead. Nothing else was ever removed at any pose — so the two
# previous versions of this comment, which described it as an under-chin
# correction that fires on a tilted head, were both describing the pitch
# DEPENDENCE of a forehead cut and calling it a chin cut.
#
# The forehead is the one part of the face where neither shape has evidence. The
# 106-pt landmarks stop at the eyebrows, so `landmark_hull` extrapolates upward
# by 0.6 of the brow-chin distance and `_face_patch_rows` extrapolates by
# carrying the reference ellipsoid `brow_extend` past the brows. Two guesses at
# the same unknown, and the difference between them was being cut out of the
# face. On a head tilted down they diverge most, which is where it was reported.
#
# The trim's TOP EDGE is therefore opened up before it is applied (see
# `_visibility_matte`): the left/right terminator is kept, which is the real
# visible-surface information and the reason this layer exists, and its opinion
# about how high the face goes is discarded. On synthetic geometry that takes the
# whole layer to 0.0% at every pose — i.e. everything it was measurably doing was
# the artefact. Whatever value it has left rests on a REAL detector's 106 points
# over-claiming the far side of a real profile, which the synthetic head does not
# reproduce and which nothing here can measure. Hence: off by default, and do not
# reach for it.
#
# An earlier version of this comment claimed 10.6% at yaw 88 and framed it as a
# yaw fix. That was measured with the polygon placed against the wrong template
# (see swap_template_points) — 53 px out of register with the face, so it was
# "trimming" more only because it was trimming the wrong pixels.
#
# The reason it cannot be tighter is fundamental to the approach: the polygon
# comes from a REFERENCE head, and real heads differ from it by more than the
# over-coverage being removed. The margin needed to stop it clipping a wider jaw
# than the reference is most of the margin the trim was trying to reclaim.
#
# So this is kept as a cheap, safe, small improvement — NOT as the fix for
# extreme-angle distortion. That is angle_fade_weight below. Do not re-tune this
# expecting a large win; the ceiling is set by head-shape variation, and beating
# it needs a per-face visibility estimate rather than a reference head.
#
# CRITICALLY it must never cut real face: a matte that clips visible skin is a
# worse artefact (a hard edge across a cheek) than the soft over-coverage it is
# removing. 0.04 is the smallest margin at which zero truly-visible landmarks are
# trimmed at any pose over yaw 0-88 x pitch -40..+20.
#
# NOTE WHAT THAT PROPERTY CANNOT SEE. It is stated in landmarks, and there are no
# landmarks above the eyebrows — so the forehead cut described above passed this
# test at every pose while being the only thing the layer did. A safety property
# expressed in terms of the evidence is blind exactly where the evidence stops,
# which is exactly where extrapolation happens and where the bug was.
#
# Measured poses with at least one landmark clipped, against margin:
#
#   margin    0.010  0.015  0.020  0.025  0.030  0.040
#   poses        13     11      6      4      1      0
#   mean trim  5.0%   4.4%   3.8%   3.5%   3.1%   2.5%
#
# The test behind that table is deliberately STRICTER than the shipped behaviour:
# it checks the hard polygon, while the applied trim is feathered and weighted, so
# a landmark it counts as "clipped" would in practice only sit in the ramp. Being
# conservative on the one property that produces a visible hard edge is the right
# trade.
VIS_POLY_MARGIN = 0.04          # dilation, as a fraction of the crop size

# ...and a hard ceiling on the whole layer, because the margin above is sized
# against HEAD SHAPE variation and that is not the error that hurts.
#
# The polygon is drawn for the pose the 5-point solve reports, and that solve
# fits one reference head, so a person whose nose is more prominent than the
# reference reads 15-20 deg further round than they are (see
# tests/test_pose_shape.py — yaw 30 read as 44.6). The polygon is then the
# visible surface of a head turned 45 deg, applied to a head turned 30, and it
# cuts the far side of a face that is entirely visible. Half the head keeps the
# original texture beside the swapped half, with a line down the middle. That is
# the failure this layer actually gets reported for, and no safety margin
# expressed in head widths can prevent it, because the input is wrong by more
# than a head.
#
# So the trim is additionally bounded by how much it is allowed to REMOVE.
# Whatever the polygon says, the layer may not take more than this fraction of
# the matte; past that its weight is scaled back until it does not. A wrong pose
# then costs an over-trimmed cheek — soft, because the trim is feathered — rather
# than a split face.
#
# 0.25 sits above everything the layer legitimately does (measured max 10.2% of
# the matte, at pitch -40, mean 2.5% over the pose grid) and well below the ~50%
# a mirrored or badly over-estimated yaw asks for. It is a backstop, not a tuning
# knob: if a clip is hitting it, the pose estimate is wrong and the layer should
# be off for that clip.
VIS_TRIM_MAX_FRAC = 0.25

# Head model for the visible-surface test: an ellipsoid fitted to the project's
# own reference head, so this cannot disagree with the pose solve or the
# alignment template about head shape. AZ is the front-to-back extent of the
# FACE mass rather than of the whole skull — the patch below only covers the
# face, and an ellipsoid deep enough to hold the back of the head would put the
# terminator in the wrong place.
def _face_patch_rows(n_u=161, n_v=64, brow_extend=0.62):
    """The face surface as rows of constant height, each row a (points, normals)
    pair in reference-head coordinates.

    Rows rather than a flat point cloud because the visible span at a fixed
    height is one CONTIGUOUS interval in u, so its two ends are the left and
    right edge of the visible region. That gives an exact outline; a convex hull
    over the visible points does not — at profile the hull is still a full-width
    oval, and measured, it trimmed 0.5% where this trims 10.6%.

    `brow_extend` carries the patch above the eyebrows to the hairline, matching
    `landmark_hull`'s 0.6 forehead extension. Without it the polygon would cut
    the forehead off every face, frontal ones included.

    Returned PADDED into (rows, n_u) grids with a validity mask rather than as a
    list of ragged rows. The ragged form cost 1.1 ms per face — a Python loop over
    42 rows, against a stage budget where a whole enhancer pass is 3-24 ms — and
    none of that was arithmetic. Padded, the visibility test is one matrix-vector
    product and two argmaxes.
    """
    from roop.face_3d_recon import _REF3D_68
    ref = np.asarray(_REF3D_68, dtype=np.float64)
    ref = ref - ref.mean(axis=0)
    ax = float(np.abs(ref[:, 0]).max())
    ay = float(np.abs(ref[:, 1]).max())
    az = ax * 0.92

    brow_y = float(ref[17:27, 1].max())
    chin_y = float(ref[:17, 1].min())
    top_y = brow_y + brow_extend * (brow_y - chin_y)

    v_all = np.linspace(chin_y, top_y, n_v)
    t = np.clip(v_all / ay, -1.0, 1.0)
    half = ax * np.sqrt(np.maximum(0.0, 1.0 - t * t))
    keep_row = half > 1e-6
    v_all, half = v_all[keep_row], half[keep_row]

    # One u grid per row, spanning that row's own half-width.
    frac = np.linspace(-1.0, 1.0, n_u)[None, :]
    u = half[:, None] * frac
    v = np.repeat(v_all[:, None], n_u, axis=1)

    s = 1.0 - (u / ax) ** 2 - (v / ay) ** 2
    valid = s > 0.0
    valid &= valid.sum(axis=1, keepdims=True) >= 2      # a row needs two ends
    z = az * np.sqrt(np.maximum(s, 0.0))

    pts = np.stack([u, v, z], axis=-1)
    # True ellipsoid normal, so the terminator sits where the surface really
    # turns away instead of at a hand-picked angle.
    nrm = np.stack([u / ax ** 2, v / ay ** 2, z / az ** 2], axis=-1)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-12)
    return pts, nrm, valid


_FACE_PATCH = None


def _face_patch():
    """The patch, built once. Lazy so importing face_util does not pull in
    face_3d_recon, matching how `_reference_5pt` reaches for the same module."""
    global _FACE_PATCH
    if _FACE_PATCH is None:
        _FACE_PATCH = _face_patch_rows()
    return _FACE_PATCH


def pose_visibility_polygon(yaw_deg, pitch_deg, base_dst, jaw=0.0):
    """Outline of the face surface still facing the camera at this pose, in the
    SAME crop pixel coordinates as the alignment template.

    Returns an (N, 2) int32 polygon, or None when the pose hides the face
    entirely or the reference is degenerate. Not convex — see `_face_patch_rows`.
    """
    scale, shift = _pose_placement(yaw_deg, pitch_deg, base_dst, jaw)
    # `scale is None` is not enough: a degenerate template (all-zero, or NaN from
    # a bad detection) yields a scale of 0 or NaN, which sails through and hands
    # back a polygon collapsed onto a point — a matte trimmed to nothing, i.e. a
    # face that silently stops being swapped. Fail to None and skip the trim.
    if scale is None or not (scale > 0.0) or not np.isfinite(shift).all():
        return None

    y, p = np.radians(float(yaw_deg)), np.radians(float(pitch_deg))
    ry = np.array([[np.cos(y), 0.0, np.sin(y)],
                   [0.0, 1.0, 0.0],
                   [-np.sin(y), 0.0, np.cos(y)]])
    rx = np.array([[1.0, 0.0, 0.0],
                   [0.0, np.cos(p), -np.sin(p)],
                   [0.0, np.sin(p), np.cos(p)]])
    R = rx @ ry                       # same order as _project_reference

    pts, nrm, valid = _face_patch()

    # Front-facing test. The rotated normal's z component is just n . R[2], so
    # this is one matrix-vector product over the whole grid rather than a full
    # rotation of every normal.
    vis = (nrm @ R[2]) > 0.0
    vis &= valid
    rows_ok = vis.any(axis=1)
    if int(rows_ok.sum()) < 3:
        return None

    # First and last visible sample in each row. Valid because the visible span
    # at a fixed height is one contiguous interval (see _face_patch_rows) — these
    # two are therefore the left and right edge of the visible region, not merely
    # two points inside it.
    n_u = vis.shape[1]
    first = vis.argmax(axis=1)
    last = n_u - 1 - vis[:, ::-1].argmax(axis=1)

    ri = np.nonzero(rows_ok)[0]
    edges = np.concatenate([pts[ri, first[ri]], pts[ri, last[ri]]])
    q = edges @ R.T
    xy = np.column_stack([q[:, 0], -q[:, 1]])          # image axes, +y down

    k = len(ri)
    # Left edge bottom-to-top, then right edge back down: a closed ring.
    ring = np.vstack([xy[:k], xy[k:][::-1]])
    return (ring * scale + shift).astype(np.int32)


# ── How much to trust the swap at this pose ──────────────────────────────────
# Every swap model here is trained on near-frontal aligned crops. Past roughly
# 55 deg off-axis the crop is outside that distribution and the model stops
# reconstructing and starts inventing — which is why hyperswap, hififace and
# inswapper each fail DIFFERENTLY at different angles, and why no amount of
# alignment work fixes it: the 5-point fit's residual against the training
# template grows 44 -> 85 px on a 512 crop from frontal to profile, and it does
# so in every alignment mode, because no similarity transform can map a profile
# onto a frontal template.
#
# So this does not try to make the model right. It bounds how wrong the result
# can look, by fading the swapped crop back toward the plate as the pose leaves
# the range the models can actually serve. That is the one lever that applies
# identically to all 13 swappers, because it does not care which one produced
# the crop.
#
# Onset is where the residual reaches ~1.5x its frontal value (yaw 70 at pitch
# 0, 65 deg off-axis); full fade at 90, where a swapper has nothing left to work
# from. Faded with the same smoothstep as the alignment band, for the same
# reason: a hard gate on a noisy per-frame pose flickers.
ANGLE_FADE_ONSET_DEG = 60.0
ANGLE_FADE_FULL_DEG = 90.0


def angle_fade_weight(yaw_deg, pitch_deg, strength):
    """How much to fade the swap back toward the plate, 0..1.

    `strength` is the user's ceiling in percent — the fade reaches
    `strength/100` at 90 deg off-axis and 0 below the onset, so 0 disables this
    entirely and every face is bit-identical to leaving it off.
    """
    try:
        s = float(strength) / 100.0
    except (TypeError, ValueError):
        return 0.0
    if not (s > 0.0):
        return 0.0
    s = min(s, 1.0)
    w = _smoothstep(ANGLE_FADE_ONSET_DEG, ANGLE_FADE_FULL_DEG,
                    offaxis_deg(yaw_deg, pitch_deg))
    return s * w


def _axis_angle(p):
    """Direction of the eye_mid -> mouth_mid axis, in image radians."""
    v = ((p[3] + p[4]) / 2.0) - ((p[0] + p[1]) / 2.0)
    return np.arctan2(v[1], v[0])


def _similarity_at_angle(lmk, dst, theta):
    """Similarity transform with the ROTATION pinned to `theta`, and scale and
    translation solved by least squares over all 5 points.

    Pinning the rotation to the value the unconstrained fit would have chosen
    reproduces that fit exactly — the least-squares similarity decouples, so the
    optimal scale and translation for the optimal rotation are the optimal scale
    and translation. That is what lets the crossfade below start from the plain
    fit and converge back to it, instead of stepping.
    """
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
    return M if np.isfinite(M).all() else None


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

# Where 'stabilize' fades its rotation constraint in and out, also as an
# off-axis angle. Same reasoning as the pose band above — a threshold flickers —
# but this mode had a far worse case of it, so the numbers are worth recording.
#
# It used to engage on a hard `yaw_ratio < 0.40` gate. Two things were wrong
# with that. First, yaw_ratio is pitch-contaminated (see solve_pose_5pt), so the
# gate wandered: it fired at yaw 56 on a level head, at yaw 66 with 20 deg of
# nod, and never at all below yaw 90 once the nod reached 30 deg — the mode was
# silently dead on exactly the tilted profiles it was written for. Second, and
# much worse, the two fits differ by up to 30 deg of crop rotation at high yaw,
# so crossing the gate was a step change of that size. Measured:
#
#   still head, 1 px keypoint noise      rot sd    rot range
#     yaw 66 / pitch -20   hard gate     5.411 deg   12.61 deg
#     yaw 90 / pitch -30   hard gate     6.157 deg   21.88 deg
#     yaw 90 / pitch -30   this band     0.360 deg    2.64 deg
#     yaw 90 / pitch -30   mode 'off'    0.446 deg    2.69 deg
#
# A head that is not moving at all had its crop rotating +/- 11 deg frame to
# frame, driven by nothing but detector noise sitting on the gate. The mode sold
# as the fix for rotational wobble was the largest source of it in the pipeline.
# Worst single-frame jump along a -90..+90 turn with a 40 deg nod riding on it:
# 18.43 deg with the gate, 0.046 deg with this band, 0.104 deg with the mode off.
#
# Note this fades the ROTATION ANGLE between the two fits, not the templates.
# An earlier attempt to fade 'stabilize' in was abandoned because the rotation
# constraint does not converge to the default fit near frontal — it disagrees by
# ~9 deg even at yaw 55 — so a template blend would have changed mid angles. That
# objection does not apply here: interpolating from the angle the plain fit
# ALREADY chose is exactly the plain fit at w=0 (see _similarity_at_angle), so
# the low end of the band converges by construction. Verified: yaw 30 and 40 are
# bit-identical to 'off', and the deviation reaches only 0.85 deg at yaw 55.
#
# Onset 40 rather than 55 (which would have left every currently-untouched angle
# untouched) because the nod-coupled rotation swing this mode exists to remove is
# large well below the old gate, and was going uncorrected:
#
#   nod-coupled rotation swing, pitch -25..+25    yaw 45   yaw 60   yaw 75
#     mode 'off'                                  14.99    20.14    25.35 deg
#     hard gate (shipped)                         14.99    20.14     0.92 deg
#     this band                                   10.85     2.84     0.92 deg
STAB_ALIGN_ONSET_DEG = 40.0
STAB_ALIGN_FULL_DEG = 70.0


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


def pose_weight_for(yaw, pitch):
    """The pose-align crossfade weight for an ALREADY SOLVED yaw/pitch.

    `pose_align_weight` below is the same thing from raw keypoints. Callers that
    already hold a solve want this one — otherwise they pay for a second solve
    and, worse, can end up keying on a pose that differs from the one the
    alignment template used."""
    return _weight_for_pose(yaw, pitch)


def yaw_align_mode():
    """The active alignment mode. Public alias for `_yaw_align_mode`, for the
    pipeline stages that must only act while pose-matched alignment is on."""
    return _yaw_align_mode()


def pose_align_weight(lmk):
    """How much pose-matched template to blend in for these keypoints, 0..1.

    Returns 0.0 for frontal faces, which is what keeps the alignment bit-exact
    where nothing needs correcting.

    Keyed on the jaw-aware solve, so an open mouth no longer engages the
    correction on a head that has not turned. Before, a frontal face with the
    mouth 45% open reported 24 deg of pitch, which is most of the way through
    the 15->40 deg band — the template was being swapped in on the strength of
    an expression."""
    pose = solve_pose_jaw_5pt(lmk)
    if pose is None:
        return 0.0
    return _weight_for_pose(pose[0], pose[1])


def _stabilized_norm(lmk, dst):
    """'stabilize' mode: take the crop ROTATION from the eye_mid -> mouth_mid
    axis instead of from the unconstrained 5-point least-squares fit, faded in
    over STAB_ALIGN_ONSET_DEG..STAB_ALIGN_FULL_DEG of off-axis angle.

    Why constrain the rotation at all: at high yaw the two eyes project to
    nearly the same point (their separation collapses to zero at 90 deg), so the
    least-squares fit is ill-conditioned in rotation and starts absorbing PITCH
    as in-plane roll. Measured swing of the crop rotation as the head nods from
    -25 to +25 deg:

        yaw    0     30     60     75     90
        LS   0.00   9.93  20.14  25.35  30.51  deg   <- unconstrained
        ours 0.00   0.49   0.49   0.28   0.00  deg   <- rotation held steady

    A 30 deg rotation swing on a nodding profile head feeds the swapper an
    off-distribution crop and shows up as rotational wobble frame to frame. The
    eye_mid -> mouth_mid axis stays well-conditioned at every yaw, so the
    rotation comes from it while scale and translation are still solved by least
    squares over all 5 points (so those keep every point's information).

    Why fade rather than switch: see the band constants above — the two fits are
    up to 30 deg apart, so any threshold between them is a step change that
    detector noise alone will straddle.

    Returns None below the onset, which is where the plain fit already holds
    rotation steady and nothing needs constraining.

    Caveat: the mean per-point fit residual gets slightly WORSE off-neutral-pitch
    (e.g. yaw 90 / pitch +25: 64.0 -> 69.9 px on a 512 crop). That is expected and
    is not a regression — the lower residual was being bought by rotating the
    face, which is precisely the pathology being removed.

    SCALE. Pinning the rotation is not free: the least-squares scale is solved
    AT the pinned angle, and rotating away from the angle the unconstrained fit
    chose shortens the projection of the target points onto the rotated source,
    so the scale falls with it — by roughly cos(delta), and delta reaches 30 deg
    at high yaw. The mode therefore used to fix the rotation wobble by
    introducing a scale one, and the scale one is larger. Measured crop-scale
    swing over yaw 0-90 x pitch +/-40:

        mode 'off'                     1.389x
        rotation pinned, scale free    1.575x   <- worse than doing nothing
        rotation pinned, scale held    1.071x

    So the scale is held to the value a POSE-MATCHED template implies, which is
    the same quantity 'pose' mode holds flat and is the reason that mode does
    not breathe (see _pose_template). Both the angle and the scale are faded by
    the same w, so at w = 0 this is still the plain fit exactly — the scale
    correction is a ratio of 1.0 there by construction, not merely nearly one.
    """
    pose = solve_pose_jaw_5pt(lmk)
    if pose is None:
        return None
    w = _smoothstep(STAB_ALIGN_ONSET_DEG, STAB_ALIGN_FULL_DEG,
                    offaxis_deg(pose[0], pose[1]))
    if w <= 0.0:
        return None

    tform = trans.SimilarityTransform()
    tform.estimate(lmk, dst)
    m_plain = tform.params[0:2, :]
    if not np.isfinite(m_plain).all():
        return None

    # Blend the ANGLE, taking the short way round the circle so a pair straddling
    # +/-pi interpolates through 0 rather than sweeping the long way.
    theta_plain = float(np.arctan2(m_plain[1, 0], m_plain[0, 0]))
    theta_axis = float(_axis_angle(dst) - _axis_angle(lmk))
    delta = (theta_axis - theta_plain + np.pi) % (2.0 * np.pi) - np.pi
    M = _similarity_at_angle(lmk, dst, theta_plain + w * delta)
    if M is None:
        return None
    return _hold_scale(M, lmk, dst, pose, w, m_plain)


def _mat_scale(M):
    """Uniform scale of a 2x3 similarity."""
    return float(np.hypot(M[0, 0], M[1, 0]))


def _rescale_about(M, lmk, dst, k):
    """Multiply a similarity's scale by `k`, keeping the source centroid landing
    on the destination centroid.

    Scaling the linear part alone would also move the crop, because the
    translation column is expressed for the old scale; re-deriving it from the
    centroids keeps the face framed exactly where it was and changes only how
    much of it fits in the box.
    """
    A = M[:, :2] * float(k)
    out = np.zeros((2, 3), dtype=np.float64)
    out[:, :2] = A
    out[:, 2] = dst.mean(0) - A @ lmk.mean(0)
    return out if np.isfinite(out).all() else M


def _hold_scale(M, lmk, dst, pose, w, m_plain):
    """Hold the crop scale to the pose-matched template's, faded in by `w`.

    The target is the scale the ordinary least-squares fit would choose against
    a template projected at this head's own yaw and pitch — a template congruent
    to the input, so its scale is the one that keeps the face the same size on
    the crop however the head is turned. Blended from the plain fit's scale so
    w = 0 is a no-op.
    """
    posed = _pose_template(pose[0], dst, pose[1], pose[3])
    if posed is None:
        return M
    t = trans.SimilarityTransform()
    t.estimate(lmk, np.asarray(posed, dtype=np.float64))
    mp = t.params[0:2, :]
    if not np.isfinite(mp).all():
        return M
    s_cur = _mat_scale(M)
    if not (s_cur > 1e-9):
        return M
    s_target = (1.0 - w) * _mat_scale(m_plain) + w * _mat_scale(mp)
    if not (s_target > 1e-9):
        return M
    return _rescale_about(M, lmk, dst, s_target / s_cur)


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
        #
        # Jaw-aware, because two of the five keypoints are the mouth corners:
        # a jaw-blind solve reports a talking head as a tilted one and the
        # template then corrects toward a pose the person is not in. See
        # solve_pose_jaw_5pt.
        pose = solve_pose_jaw_5pt(lmk)
        if pose is None:
            return None
        yaw, pitch, _, jaw = pose
        w = _weight_for_pose(yaw, pitch)
        if w <= 0.0:
            return None                      # frontal: leave the default fit alone
        posed = _pose_template(yaw, dst, pitch, jaw)
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

    return _stabilized_norm(lmk, dst)


def swap_template_points(image_size, mode="arcface"):
    """The 5 destination points `estimate_norm` fits the keypoints to, in crop
    pixels — i.e. WHERE THE FACE LANDS in the crop, for this model's template and
    this crop size.

    Split out of estimate_norm because anything else that needs to know where the
    face sits in the crop has to ask the same question, and guessing it is a trap:
    every crop size this app actually uses (128/256/512/1024) falls through the
    `% 112` test and lands on the `% 128` branch, which scales by size/128 AND
    shifts x. Reconstructing it as `arcface_dst * size/112` — the obvious guess —
    is wrong by 13 px at 128 and 53 px at 512, and wrong in a way that still looks
    plausible on screen. `pose_visibility_polygon`'s placement was built on exactly
    that guess.
    """
    if mode in WARP_TEMPLATES:
        return WARP_TEMPLATES[mode] * float(image_size)

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
    return dst


def estimate_norm(lmk, image_size=112, mode="arcface"):
    assert lmk.shape == (5, 2)

    # Force SimilarityTransform (use_affine = False) for all faces.
    # AffineTransform introduces non-uniform scaling and shearing, which causes
    # severe perspective warping and facial distortion (e.g. stretched foreheads)
    # on angled/close-up faces. Using SimilarityTransform preserves the face's
    # natural aspect ratio and matches the training distribution of the models.
    use_affine = False

    dst = swap_template_points(image_size, mode)
    M = _maybe_constrained(lmk, dst)
    if M is not None:
        return M
    tform = trans.AffineTransform() if use_affine else trans.SimilarityTransform()
    tform.estimate(lmk, dst)
    return tform.params[0:2, :]



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

