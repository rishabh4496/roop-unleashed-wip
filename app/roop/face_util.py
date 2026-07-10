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
    return fa


def _ensure_face_analyser():
    """(Re)build the FaceAnalysis pool when missing, when the requested module set
    changed, or when the detection resolution (face_detector_size) or threshold changed. Returns
    the primary instance."""
    global FACE_ANALYSER, FACE_ANALYSER_POOL, _ANALYSER_Q, _ANALYSER_DET_SIZE, _ANALYSER_DET_THRESH
    # Fast path (no lock): pool is built once before the run and the module set,
    # det_size, and det_thresh are stable during it, so the hot per-frame detect path skips the lock.
    cur_det_thresh = getattr(roop.globals, 'face_detector_threshold', 0.60)
    if (FACE_ANALYSER_POOL
            and roop.globals.g_current_face_analysis == roop.globals.g_desired_face_analysis
            and _ANALYSER_DET_SIZE == _desired_det_size()
            and _ANALYSER_DET_THRESH == cur_det_thresh):
        return FACE_ANALYSER
    with THREAD_LOCK_ANALYSER:
        if (not FACE_ANALYSER_POOL
                or roop.globals.g_current_face_analysis != roop.globals.g_desired_face_analysis
                or _ANALYSER_DET_SIZE != _desired_det_size()
                or _ANALYSER_DET_THRESH != cur_det_thresh):
            roop.globals.g_current_face_analysis = roop.globals.g_desired_face_analysis
            _ANALYSER_DET_SIZE = _desired_det_size()
            _ANALYSER_DET_THRESH = cur_det_thresh
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


def _detect_faces(frame):
    """Run the selected detector engine and return raw Face objects (unsorted).
    Applies small-face (upscale), close-up scale (downscale), and clipped boundary (padded) rescues."""
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

