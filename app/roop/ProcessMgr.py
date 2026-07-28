import os
import cv2
import time
import numpy as np
import psutil

from roop.ProcessOptions import ProcessOptions

from roop.face_util import get_first_face, get_all_faces, rotate_anticlockwise, rotate_clockwise, analysis_pooled
from roop.utilities import compute_cosine_distance, get_device, str_to_class
import roop.vr_util as vr

from typing import Any, List, Callable
from roop.typing import Frame, Face
from roop.procmgr_masking import MaskingMixin
from roop.procmgr_color import ColorTransferMixin
from roop.procmgr_tiling import PixelBoostMixin
from roop.procmgr_tracking import TrackingMixin
from roop import recognizer_adaface as _ada
from roop import live_preview as _live_preview
from roop.procmgr_runtime import _PROFILE, _TRACK_VETO_DIST, _TRACK_VETO_MARGIN, _TRACK_VETO_SINGLE, _TRACK_EMB_MAX, _DEBUG_MATCH, COLOR_RESET, COLOR_CYAN, COLOR_YELLOW, _prof, _prof_report, _gpu_guard, PROGRESS_BAR_FORMAT, wait_while_paused
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Thread, Lock, local

# Guards the one-time build of the expression restorer (see _expression_restorer).
_EXPR_BUILD_LOCK = Lock()
from queue import Queue, Full as _QueueFull, Empty as _QueueEmpty


# Per-frame diagnostic pose logging. Computing source yaw/pitch (estimate_pose)
# every frame purely to print a line is wasted CPU in the hot loop and starves
# the GPU. Off by default; flip to True only when debugging pose correction.
_DEBUG_POSE_LOG = False

# Per-face angle/mask-routing diagnostic (ROOP_DEBUG_ANGLE=1). Prints the yaw and
# pitch proxies, the non-frontal verdict, which masking path was taken, and how
# much of the canonical crop the unwarped box actually covers. Use on a single
# preview frame — it prints per face per processor, so it is noisy on a video.

# ── Optional per-stage timing probe (enable with env ROOP_PROFILE=1) ─────────
# Sums wall-clock per pipeline stage across all worker threads. "share" is each
# stage's slice of total CPU work; "ms/call" is the real per-frame / per-face
# cost. Zero overhead when disabled, so it never affects normal runs. A report
# is printed once per video at the end of run_batch_inmem.
# Opt-in batched swap: run the pixel-boost tiles through one inference call.
_BATCH_SWAP = os.environ.get('ROOP_BATCH_SWAP', '0') == '1'



# ── Jaw / chin reshape warp ──────────────────────────────────────────────────
# inswapper-family swappers transfer identity but keep the TARGET's face
# geometry — the swapped face is bounded by the target's jaw/chin outline. This
# post-composite pass warps the lower-face silhouette toward the SOURCE person's
# jaw/chin shape using a smooth thin-plate-spline displacement field applied with
# cv2.remap (liquify style): neighbouring pixels are dragged continuously so no
# inpainting / gap-filling is needed. Off unless the user enables it.
#
# The 106-pt face-silhouette landmarks (indices 0..32) of the source and target
# are aligned into a shared canonical space via their 5-pt arcface transforms so
# the pure shape difference is isolated from pose / scale / position, then mapped
# back to the target frame. Central features (5 kps) and the ROI border are
# pinned (zero displacement) so only the jaw region moves and the periphery is
# untouched, giving a seamless paste-back.
_JAW_CONTOUR_IDX = np.arange(0, 33)


def _jaw_tps_solve(P, V):
    """Thin-plate-spline weights for control points P (K,2), scalar values V (K,)."""
    K = P.shape[0]
    r = np.sqrt(np.sum((P[:, None, :] - P[None, :, :]) ** 2, axis=2))
    with np.errstate(divide='ignore', invalid='ignore'):
        U = (r ** 2) * np.log(r + 1e-6)
    U[~np.isfinite(U)] = 0.0
    Ph = np.hstack([np.ones((K, 1)), P])
    A = np.zeros((K + 3, K + 3), dtype=np.float64)
    A[:K, :K] = U
    A[:K, K:] = Ph
    A[K:, :K] = Ph.T
    b = np.zeros((K + 3,), dtype=np.float64)
    b[:K] = V
    sol = np.linalg.solve(A, b)
    return sol[:K], sol[K:]


def _jaw_tps_eval(P, w, a, G):
    """Evaluate a TPS field (from _jaw_tps_solve) at grid points G (M,2)."""
    r = np.sqrt(np.sum((G[:, None, :] - P[None, :, :]) ** 2, axis=2))
    with np.errstate(divide='ignore', invalid='ignore'):
        U = (r ** 2) * np.log(r + 1e-6)
    U[~np.isfinite(U)] = 0.0
    return U @ w + a[0] + a[1] * G[:, 0] + a[2] * G[:, 1]


def reshape_jaw_frame(result, tgt106, src106, tgt_kps, src_kps, strength,
                      grid=48, roi_margin=0.5):
    """Warp `result` so the target face shape moves toward the source: the jaw
    silhouette plus the cheeks and lower face (interior expansion below the eye
    line). Central features (eyes/nose/mouth) are pinned so identity/expression
    are preserved.

    Returns the warped frame (or the original on any degenerate input). Pure
    numpy/cv2 — no model inference — so it is cheap and thread-safe.
    """
    try:
        from roop.face_util import estimate_norm

        s = float(strength)
        if s <= 0.0:
            return result
        if tgt106 is None or src106 is None or tgt_kps is None or src_kps is None:
            return result
        tgt106 = np.asarray(tgt106, dtype=np.float32)
        src106 = np.asarray(src106, dtype=np.float32)
        if tgt106.shape[0] < 33 or src106.shape[0] < 33:
            return result
        tgt_kps = np.asarray(tgt_kps, dtype=np.float32).reshape(-1, 2)
        src_kps = np.asarray(src_kps, dtype=np.float32).reshape(-1, 2)
        if tgt_kps.shape[0] != 5 or src_kps.shape[0] != 5:
            return result

        S = 256
        Mt = estimate_norm(tgt_kps, S)   # frame(target) → canonical
        Ms = estimate_norm(src_kps, S)   # source-image → canonical

        def _aff(M, pts):
            return pts @ M[:, :2].T + M[:, 2]

        tgt_c = _aff(Mt, tgt106)
        src_c = _aff(Ms, src106)

        # Silhouette (jaw + cheeks + chin) displacement toward the source shape,
        # computed in the shared canonical space.
        cont_c = tgt_c[_JAW_CONTOUR_IDX]
        disp_c = s * (src_c[_JAW_CONTOUR_IDX] - cont_c)

        IMt = cv2.invertAffineTransform(Mt)
        A = IMt[:, :2]                     # linear part only (transforms displacements)
        tgt_f = _aff(IMt, tgt_c)

        H, W = result.shape[:2]
        x0, y0 = tgt_f.min(axis=0)
        x1, y1 = tgt_f.max(axis=0)
        bw, bh = x1 - x0, y1 - y0
        mx, my = bw * roi_margin, bh * roi_margin
        rx0 = int(max(0, np.floor(x0 - mx)))
        ry0 = int(max(0, np.floor(y0 - my)))
        rx1 = int(min(W, np.ceil(x1 + mx)))
        ry1 = int(min(H, np.ceil(y1 + my)))
        if rx1 - rx0 < 8 or ry1 - ry0 < 8:
            return result

        # ── Cheek / lower-face expansion ──────────────────────────────────────
        # The silhouette reshape alone only tugs the outer edge. Carry it inward
        # across the cheeks and lower face with interior control points that
        # follow each below-the-eye-line contour point toward the face centre,
        # fading to zero so the pinned central features (eyes/nose/mouth) stay
        # put. Points landing near a pinned kp are dropped (TPS conditioning).
        canon_kps = _aff(Mt, tgt_kps)      # == the arcface template positions
        eye_y = 0.5 * (canon_kps[0, 1] + canon_kps[1, 1])
        center = canon_kps[2]              # nose tip — inward anchor
        lower = cont_c[:, 1] > eye_y       # cheeks + jaw + chin (exclude temples)

        pts_c = [cont_c]
        dsp_c = [disp_c]
        if np.any(lower):
            int_c, int_d = [], []
            for t in (0.35, 0.6):
                int_c.append(cont_c[lower] + t * (center - cont_c[lower]))
                int_d.append((1.0 - t) * disp_c[lower])
            int_c = np.vstack(int_c)
            int_d = np.vstack(int_d)
            int_f = _aff(IMt, int_c)
            dmin = np.min(np.linalg.norm(int_f[:, None, :] - tgt_kps[None, :, :], axis=2), axis=1)
            keep = dmin > 0.05 * max(bw, 1.0)
            if np.any(keep):
                pts_c.append(int_c[keep])
                dsp_c.append(int_d[keep])

        moved_c = np.vstack(pts_c)
        moved_dsp = np.vstack(dsp_c)

        # Map the moved points and their displacement back to the target frame.
        tgt_move = _aff(IMt, moved_c)
        delta_move = moved_dsp @ A.T
        dst_move = tgt_move + delta_move
        if np.max(np.abs(delta_move)) < 0.5:
            return result  # shapes already match — nothing to do

        ring = np.array([
            [rx0, ry0], [(rx0 + rx1) / 2, ry0], [rx1, ry0],
            [rx1, (ry0 + ry1) / 2], [rx1, ry1], [(rx0 + rx1) / 2, ry1],
            [rx0, ry1], [rx0, (ry0 + ry1) / 2],
        ], dtype=np.float32)

        P = np.vstack([dst_move, tgt_kps, ring]).astype(np.float64)
        n_anchor = len(tgt_kps) + len(ring)
        Vx = np.concatenate([delta_move[:, 0], np.zeros(n_anchor)])
        Vy = np.concatenate([delta_move[:, 1], np.zeros(n_anchor)])

        P_local = P - np.array([rx0, ry0])
        wx, ax = _jaw_tps_solve(P_local, Vx)
        wy, ay = _jaw_tps_solve(P_local, Vy)

        roi_h, roi_w = ry1 - ry0, rx1 - rx0
        gxs = np.linspace(0, roi_w - 1, grid)
        gys = np.linspace(0, roi_h - 1, grid)
        GX, GY = np.meshgrid(gxs, gys)
        G = np.stack([GX.ravel(), GY.ravel()], axis=1)
        ux = _jaw_tps_eval(P_local, wx, ax, G).reshape(grid, grid).astype(np.float32)
        uy = _jaw_tps_eval(P_local, wy, ay, G).reshape(grid, grid).astype(np.float32)

        ux_full = cv2.resize(ux, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)
        uy_full = cv2.resize(uy, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)

        xx, yy = np.meshgrid(np.arange(roi_w, dtype=np.float32),
                             np.arange(roi_h, dtype=np.float32))
        map_x = (xx - ux_full).astype(np.float32)
        map_y = (yy - uy_full).astype(np.float32)

        roi = result[ry0:ry1, rx0:rx1]
        warped = cv2.remap(roi, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
        result[ry0:ry1, rx0:rx1] = warped
        return result
    except Exception as e:
        print(f"[ProcessMgr] jaw reshape failed: {e}")
        return result






from tqdm import tqdm
from roop.ffmpeg_writer import FFMPEG_VideoWriter
from roop.StreamWriter import StreamWriter
from roop import swap_batcher
import roop.globals





# Poor man's enum to be able to compare to int
class eNoFaceAction():
    USE_ORIGINAL_FRAME = 0
    RETRY_ROTATED = 1
    SKIP_FRAME = 2
    # NB: no trailing comma — `3,` makes this a tuple and every `==` against the
    # int no_face_action silently fails ("Skip Frame if no similar face" never fired).
    SKIP_FRAME_IF_DISSIMILAR = 3
    USE_LAST_SWAPPED = 4





def create_queue(temp_frame_paths: List[str]) -> Queue[str]:
    queue: Queue[str] = Queue()
    for frame_path in temp_frame_paths:
        queue.put(frame_path)
    return queue


def pick_queue(queue: Queue[str], queue_per_future: int) -> List[str]:
    queues = []
    for _ in range(queue_per_future):
        if not queue.empty():
            queues.append(queue.get())
    return queues


def _detect_face_in_roi(frame: np.ndarray, last_bbox: np.ndarray):
    """When full-frame detection misses, crop last-known face region and retry.

    Remaps the detected face's 2-D coordinates back to full-frame space so the
    result can be used in process_face() without any special casing.
    Returns a Face object in full-frame coords, or None.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = last_bbox.astype(int)
    face_w = max(1, x2 - x1)
    face_h = max(1, y2 - y1)
    pad = int(max(face_w, face_h) * 0.5)          # 50 % padding on each side

    rx1 = max(0, x1 - pad)
    ry1 = max(0, y1 - pad)
    rx2 = min(w, x2 + pad)
    ry2 = min(h, y2 + pad)

    crop = frame[ry1:ry2, rx1:rx2]
    if crop.size == 0 or crop.shape[0] < 32 or crop.shape[1] < 32:
        return None

    # Upscale small crops so the detector can resolve them
    crop_size = max(rx2 - rx1, ry2 - ry1)
    scale = max(1.0, 320.0 / crop_size)
    if scale > 1.1:
        new_w = int((rx2 - rx1) * scale)
        new_h = int((ry2 - ry1) * scale)
        crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        scale = 1.0

    try:
        from roop.face_util import lease_face_analyser
        with lease_face_analyser() as fa:
            faces = fa.get(crop)
        if not faces:
            return None
        face = min(faces, key=lambda f: f.bbox[0])
    except Exception:
        return None

    # Remap all 2-D coordinates from (scaled) crop space to full-frame space
    def _remap2d(pts):
        if pts is None:
            return None
        pts = pts.copy().astype(np.float32)
        pts[:, 0] = pts[:, 0] / scale + rx1
        pts[:, 1] = pts[:, 1] / scale + ry1
        return pts

    face.bbox = face.bbox.copy().astype(np.float32)
    face.bbox[0] = face.bbox[0] / scale + rx1
    face.bbox[1] = face.bbox[1] / scale + ry1
    face.bbox[2] = face.bbox[2] / scale + rx1
    face.bbox[3] = face.bbox[3] / scale + ry1

    if face.kps is not None:
        face.kps = _remap2d(face.kps)
    if getattr(face, 'landmark_2d_106', None) is not None:
        face.landmark_2d_106 = _remap2d(face.landmark_2d_106)
    if getattr(face, 'landmark_3d_68', None) is not None:
        lm3d = face.landmark_3d_68.copy().astype(np.float32)
        lm3d[:, 0] = lm3d[:, 0] / scale + rx1
        lm3d[:, 1] = lm3d[:, 1] / scale + ry1
        lm3d[:, 2] = lm3d[:, 2] / scale          # depth scales with image scale
        face.landmark_3d_68 = lm3d

    return face


class ProcessMgr(MaskingMixin, ColorTransferMixin, PixelBoostMixin, TrackingMixin):
    plugins = {
        'faceswap'          : 'FaceSwapInsightFace',
        'mask_clip2seg'     : 'Mask_Clip2Seg',
        'mask_xseg'         : 'Mask_XSeg',
        'mask_xseg3'        : 'Mask_XSeg3',
        'mask_occluder'     : 'Mask_Occluder',
        'mask_faceparser'   : 'Mask_FaceParser',
        'mask_mobilesam'    : 'Mask_MobileSAM',
        'mask_fastsam'      : 'Mask_FastSAM',
        'mask_sam2'         : 'Mask_SAM2',
        'codeformer'        : 'Enhance_CodeFormer',
        'gfpgan'            : 'Enhance_GFPGAN',
        'dmdnet'            : 'Enhance_DMDNet',
        'gpen'              : 'Enhance_GPEN',
        'restoreformer++'   : 'Enhance_RestoreFormerPPlus',
        'keep'              : 'Enhance_KEEP',
        'colorizer'         : 'Frame_Colorizer',
        'filter_generic'    : 'Frame_Filter',
        'removebg'          : 'Frame_Masking',
        'upscale'           : 'Frame_Upscale'
    }

    def __init__(self, progress):
        # FIX: All mutable state as instance attributes (previously class-level,
        # which caused processor/model references to persist across ProcessMgr instances
        # and prevented VRAM from being released between runs).
        self.input_face_datas = []
        self.target_face_datas = []
        self.target_face_groups = []   # parallel to target_face_datas: person id per face
        self.imagemask = None
        self.processors = []
        self.options = None
        self.num_threads = 1
        self.current_index = 0
        self.processing_threads = 1
        self.buffer_wait_time = 0.1
        self.lock = Lock()
        self.frames_queue = None
        self.processed_queue = None
        self.videowriter = None
        self.streamwriter = None
        self.progress_gradio = None
        self.total_frames = 0
        # Cumulative count of target faces handed to process_face — used by the
        # learned runtime estimator to derive average faces/frame (density).
        self.total_swaps = 0
        # Set by core.live_swap on the shared preview ProcessMgr: preview
        # renders must not be published to the batch live-view frame.
        self.is_preview = False
        self._psutil_proc = None       # cached psutil.Process for the progress bar
        self.num_frames_no_face = 0
        self.last_swapped_frame = None
        self.output_to_file = None
        self.output_to_cam = None
        # One Euro stabilizers (video only); active flag is set per-run.
        self.kps_stabilizer = None    # smooths face keypoints (anti-wobble)
        self.enh_stabilizer = None    # smooths enhancer output (anti-flicker)
        self._stab_active = False
        self._stab_t = 0
        # 2-pass parallel stabilization: when only kps stabilization is on, pass 1
        # precomputes the (order-dependent) smoothed kps sequentially, then pass 2
        # swaps in parallel using a per-frame lookup (no temporal dependency), so
        # stabilized video keeps multi-thread speed.
        # {frame_idx: [(raw_centroid(2,), smoothed_kps(5,2)), ...]}
        self._precomputed_kps = None
        self._precomputed_mode = False
        # Cross-frame swap batcher (Phase 2): coalesces concurrent swap calls
        # from worker threads into one batched inference. Set per video run.
        self._swap_batcher = None
        # Parallel stabilization (opt-in): per-thread stabilizer instances via
        # thread-local storage so the (order-dependent) enhancer/kps stabilizers
        # can run multi-threaded on contiguous frame blocks instead of forcing
        # single-thread. Factories rebuild fresh instances per worker block.
        self._parallel_stab = False
        self._tls = local()
        self._kps_stab_factory = None
        self._enh_stab_factory = None
        # Temporal detection (anti-flicker): when active, swap_faces consumes
        # the pre-pass faces per frame instead of re-detecting.
        # _temporal_faces: {frame_idx (0-based within trim): [Face, ...]}
        # _temporal_covered: how many frames the pre-pass actually scanned — the
        # cache is only authoritative below this index.
        self._temporal_mode = False
        self._temporal_faces = None
        self._temporal_covered = 0
        self._track_scanned = 0
        # Per-faceset canvas masks: {faceset_idx (int): {'exclude_mask': arr, 'include_mask': arr,
        #                                                  'ref_kps': arr, 'is_canonical': bool}}
        self.face_masks = {}

        if progress is not None:
            self.progress_gradio = progress

    def reuseOldProcessor(self, name:str):
        for p in self.processors:
            if p.processorname == name:
                return p
        return None


    def initialize(self, input_faces, target_faces, options):
        self.input_face_datas = input_faces
        self.target_face_datas = target_faces
        # Decide ONCE per run whether AdaFace drives identity matching, and warm
        # every captured target face. All-or-nothing: a run must not compare some
        # pairs on one metric and some on another against a single threshold.
        # No-op unless ROOP_ADAFACE is set. Swap identity is never affected —
        # w600k still feeds the swapper.
        try:
            _ada.begin_run(target_faces)
        except Exception as e:
            print(f'[AdaFace] init skipped ({e})')
        # Multi-angle target groups: person id per target face (default = each its
        # own person). Multiple angles of one person share an id; matching uses
        # the min distance across a person's angles → robust to pose (anti-flicker).
        self.target_face_groups = list(roop.globals.TARGET_FACE_GROUP)
        if len(self.target_face_groups) != len(target_faces):
            self.target_face_groups = list(range(len(target_faces)))
        self.num_frames_no_face = 0
        self.last_swapped_frame = None
        self.last_found_bboxes = None
        self.options = options
        devicename = get_device()

        # Build the One Euro stabilizers when requested. They only take effect in
        # the sequential video path (run_batch_inmem sets _stab_active).
        # Factories rebuild fresh, independent stabilizer instances (used to give
        # each worker its own state in the parallel-stabilization path).
        if getattr(options, 'stabilize_face', False):
            method = getattr(options, 'stabilize_method', 'one_euro')
            if method == 'ema':
                from roop.one_euro import EmaKpsStabilizer
                self._kps_stab_factory = lambda: EmaKpsStabilizer(alpha=0.3)
            else:
                from roop.one_euro import KpsStabilizer
                _mc = getattr(options, 'stabilize_min_cutoff', 0.05)
                _bt = getattr(options, 'stabilize_beta', 0.02)
                self._kps_stab_factory = lambda: KpsStabilizer(min_cutoff=_mc, beta=_bt)
            self.kps_stabilizer = self._kps_stab_factory()
        else:
            self.kps_stabilizer = None
            self._kps_stab_factory = None
        if getattr(options, 'stabilize_enhancer', False):
            from roop.one_euro import EnhancerStabilizer
            _st = getattr(options, 'stabilize_enhancer_strength', 0.5)
            self._enh_stab_factory = lambda: EnhancerStabilizer(strength=_st)
            self.enh_stabilizer = self._enh_stab_factory()
        else:
            self.enh_stabilizer = None
            self._enh_stab_factory = None
        self._stab_active = False
        self._stab_t = 0

        # Only request the analysis sub-models actually needed → faster detection.
        # landmark_3d_68 (the 1k3d68 model, run per face on every frame) is consumed
        # ONLY by the optional pose features (3D recon / source bank / frontalization),
        # all of which are off by default and guard their use with hasattr(). Skipping
        # it when they're all off removes one per-face model inference every frame with
        # no quality change. It is re-added automatically when any of those is enabled.
        modules = ["landmark_2d_106", "detection", "recognition"]
        if (getattr(options, 'use_3d_recon', False)
                or getattr(options, 'use_source_bank', False)
                or getattr(options, 'use_frontalization', False)
                or getattr(roop.globals, 'refine_landmarks', False)):
            modules.insert(0, "landmark_3d_68")
        roop.globals.g_desired_face_analysis = modules
        if options.swap_mode == "all_female" or options.swap_mode == "all_male":
            roop.globals.g_desired_face_analysis.append("genderage")

        for p in self.processors:
            newp = next((x for x in options.processors.keys() if x == p.processorname), None)
            if newp is None:
                p.Release()
                del p

        newprocessors = []
        for key, extoption in options.processors.items():
            p = self.reuseOldProcessor(key)
            if p is None:
                classname = self.plugins[key]
                module = 'roop.processors.' + classname
                p = str_to_class(module, classname)
            if p is not None:
                extoption.update({"devicename": devicename})
                p.Initialize(extoption)
                newprocessors.append(p)
            else:
                print(f"Not using {module}")
        self.processors = newprocessors

        # ── Parse manual mask JSON (written by the canvas masking modal) ──────
        # New format: {"0": {"exclude": "data:...", "canonical": true}, "1": {...}}
        # Old format: {"exclude": "data:...", "canonical": true}  (treated as faceset 0)
        # face_masks: {faceset_idx (int): {'exclude_mask': arr, 'include_mask': arr,
        #                                   'ref_kps': arr, 'is_canonical': bool}}
        self.face_masks = {}
        mask_src = self.options.imagemask
        if isinstance(mask_src, str) and mask_src.strip().startswith('{'):
            try:
                import json as _json, base64 as _b64
                raw = _json.loads(mask_src)
                blend_amount = 20.0
                if self.input_face_datas and len(self.input_face_datas[0].faces) > 0:
                    blend_amount = self.input_face_datas[0].faces[0].mask_offsets[4]

                def _decode_mask(data_url):
                    if not data_url:
                        return None
                    try:
                        _, b64 = data_url.split(',', 1)
                        arr = np.frombuffer(_b64.b64decode(b64), dtype=np.uint8)
                        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
                        if img is None or not np.any(img):
                            return None
                        img = self.blur_area(img, blend_amount)
                        return img.astype(np.float32) / 255.0
                    except Exception:
                        return None

                def _parse_one_faceset_entry(mask_data):
                    """Decode one faceset's mask dict → face_masks entry, or None."""
                    exclude_mask = _decode_mask(mask_data.get('exclude'))
                    include_mask = _decode_mask(mask_data.get('include'))
                    if exclude_mask is None and include_mask is None:
                        return None
                    ref_kps = None
                    raw_kps = mask_data.get('ref_kps')
                    if raw_kps:
                        try:
                            ref_kps = np.array(raw_kps, dtype=np.float32)
                        except Exception:
                            pass
                    return {
                        'exclude_mask': exclude_mask,
                        'include_mask': include_mask,
                        'ref_kps': ref_kps,
                        'is_canonical': bool(mask_data.get('canonical', False)),
                    }

                # Detect format: new = all top-level keys are digit strings.
                top_keys = list(raw.keys())
                is_new_format = bool(top_keys) and all(k.isdigit() for k in top_keys)
                if is_new_format:
                    for k, v in raw.items():
                        if isinstance(v, dict):
                            entry = _parse_one_faceset_entry(v)
                            if entry is not None:
                                self.face_masks[int(k)] = entry
                else:
                    # Old flat format → treat as faceset 0
                    entry = _parse_one_faceset_entry(raw)
                    if entry is not None:
                        self.face_masks[0] = entry

            except Exception as e:
                print(f"[ProcessMgr] Failed to parse mask JSON: {e}")
                self.face_masks = {}
        # Clear legacy imagemask — we only use face_masks now
        self.options.imagemask = None

        self.options.frame_processing = False
        for p in self.processors:
            if p.type.startswith("frame_"):
                self.options.frame_processing = True

        # ── Pose-aware source crop warping ───────────────────────────────────
        # Cache a 512-px align_crop of each source face for use each frame.
        # No network inference at this stage — the crop is stored as face_3d.
        if getattr(self.options, 'use_3d_recon', False):
            try:
                from roop.face_util import align_crop, get_first_face
                for fs in self.input_face_datas:
                    if fs.face_3d_bank is not None:
                        continue   # already cached from a previous run
                    # Cache a 512-px align_crop for EVERY source face (parallel to
                    # fs.faces / fs.ref_images), so the per-frame 3D recon can warp
                    # whichever face the source bank selects — not just face[0].
                    bank = []
                    for idx in range(len(fs.faces)):
                        entry = None
                        try:
                            src_img = fs.ref_images[idx] if idx < len(fs.ref_images) else None
                            if src_img is not None:
                                src_face = get_first_face(src_img)
                                if src_face is not None and getattr(src_face, 'kps', None) is not None:
                                    src_crop, src_M = align_crop(src_img, src_face.kps, 512)
                                    # Store the crop, the source → crop affine M, and the 3D landmarks
                                    src_lm68 = None
                                    if getattr(src_face, 'landmark_3d_68', None) is not None:
                                        src_lm68 = src_face.landmark_3d_68[:, :2].astype(np.float32)
                                    entry = {'src_crop': src_crop, 'src_M': src_M, 'src_lm68': src_lm68}
                        except Exception as e:
                            print(f"[ProcessMgr] Pose-aware source cache (idx {idx}) failed: {e}")
                        bank.append(entry)
                    fs.face_3d_bank = bank
                    # Back-compat: face_3d points at the first valid cached crop.
                    fs.face_3d = next((e for e in bank if e is not None), None)
            except Exception as e:
                print(f"[ProcessMgr] Pose-aware source cache failed: {e}")

        # ── Multi-angle source bank: precompute per-face poses ────────────────
        # For each face in every FaceSet, estimate its head yaw/pitch from
        # landmark_3d_68 so process_face() can select the closest-angle source.
        if getattr(self.options, 'use_source_bank', False):
            try:
                import math as _math
                from roop.face_3d_recon import estimate_pose, decompose_yaw_pitch
                for fs in self.input_face_datas:
                    if fs.face_poses is not None:
                        continue   # already computed from a previous initialize() call
                    if len(fs.faces) < 2:
                        # Single-face facesets don't need pose selection
                        fs.face_poses = None
                        continue
                    poses = []
                    for idx, face in enumerate(fs.faces):
                        yaw_d = pitch_d = None
                        curr_face = face
                        if not hasattr(curr_face, 'landmark_3d_68') or curr_face.landmark_3d_68 is None:
                            if idx < len(fs.ref_images) and fs.ref_images[idx] is not None:
                                from roop.face_util import get_first_face
                                redetected = get_first_face(fs.ref_images[idx])
                                if redetected is not None:
                                    curr_face = redetected
                                    if hasattr(redetected, 'landmark_3d_68'):
                                        face.landmark_3d_68 = redetected.landmark_3d_68
                                        face['landmark_3d_68'] = redetected.landmark_3d_68
                                        if hasattr(redetected, 'embedding'):
                                            face.embedding = redetected.embedding
                                            face['embedding'] = redetected.embedding

                        if (hasattr(curr_face, 'landmark_3d_68')
                                and curr_face.landmark_3d_68 is not None):
                            try:
                                lm = curr_face.landmark_3d_68[:, :2].astype(np.float32)
                                rvec, _ = estimate_pose(lm, 512)
                                y, p = decompose_yaw_pitch(rvec)
                                yaw_d = _math.degrees(y)
                                pitch_d = _math.degrees(p)
                            except Exception:
                                pass
                        poses.append((yaw_d, pitch_d))
                    fs.face_poses = poses
                    valid = [(y, p) for (y, p) in poses if y is not None]
                    print(f"[SourceBank] FaceSet with {len(fs.faces)} faces — "
                          f"poses: {[(f'{y:.1f}°', f'{p:.1f}°') for y,p in valid]}")
            except Exception as e:
                print(f"[ProcessMgr] Source bank pose precomputation failed: {e}")


    def run_batch(self, source_files, target_files, threads:int = 1):
        progress_bar_format = PROGRESS_BAR_FORMAT
        self.total_frames = len(source_files)
        self.num_threads = threads
        with tqdm(total=self.total_frames, desc='Processing', unit='frame', dynamic_ncols=True, bar_format=progress_bar_format) as progress:
            with ThreadPoolExecutor(max_workers=threads) as executor:
                futures = []
                queue = create_queue(source_files)
                queue_per_future = max(len(source_files) // threads, 1)
                while not queue.empty():
                    future = executor.submit(self.process_frames, source_files, target_files, pick_queue(queue, queue_per_future), lambda: self.update_progress(progress))
                    futures.append(future)
                for future in as_completed(futures):
                    future.result()


    def process_frames(self, source_files: List[str], target_files: List[str], current_files, update: Callable[[], None]) -> None:
        for f in current_files:
            wait_while_paused()
            if not roop.globals.processing:
                return
            temp_frame = cv2.imdecode(np.fromfile(f, dtype=np.uint8), cv2.IMREAD_COLOR)
            if temp_frame is not None:
                try:
                    if self.options.frame_processing:
                        with _gpu_guard():
                            frame = temp_frame
                            for p in self.processors:
                                frame = p.Run(frame)
                            resimg = frame
                    else:
                        # process_frame serialises only its GPU primitives (under
                        # TensorRT); CPU work overlaps across threads.
                        resimg = self.process_frame(temp_frame)
                except RuntimeError as exc:
                    # Catch per-frame GPU failures (CUDA error 999, OOM, etc.) so a
                    # single bad frame does not abort the entire batch.  Write the
                    # unprocessed original frame instead so the output is continuous.
                    err_str = str(exc)
                    if 'CUDA' in err_str or 'cuda' in err_str or 'onnxruntime' in err_str.lower():
                        print(f'[ProcessMgr] GPU error on {f} — writing original frame: {err_str[:200]}')
                        resimg = temp_frame   # fall back to unmodified frame
                    else:
                        raise   # non-GPU errors propagate normally
                if resimg is not None:
                    i = source_files.index(f)
                    cv2.imwrite(target_files[i], resimg)
            if update:
                update()


    def read_frames_thread(self, cap, frame_start, frame_end, num_threads):
        num_frame = 0
        total_num = frame_end - frame_start
        if frame_start > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

        while True and roop.globals.processing:
            # Pause the reader; consumers drain the queue then block on get(),
            # so the whole pipeline pauses and resumes from the same frame.
            wait_while_paused()
            if not roop.globals.processing:
                break
            with _prof('decode'):
                ret, frame = cap.read()
            if not ret:
                break
            _thr = num_frame % num_threads
            while True:
                try:
                    self.frames_queue[_thr].put((num_frame, frame), timeout=0.1)
                    break
                except _QueueFull:
                    if not roop.globals.processing:
                        break
            num_frame += 1
            if num_frame == total_num:
                break

        for i in range(num_threads):
            self.frames_queue[i].put(None)


    def read_frames_webp_thread(self, bgr_frames, frame_start, frame_end, num_threads):
        """Feed pre-decoded BGR frames (from animated webp via PIL) into the processing queue."""
        subset = bgr_frames[frame_start:frame_end] if frame_end > frame_start else bgr_frames[frame_start:]
        for num_frame, frame in enumerate(subset):
            wait_while_paused()
            if not roop.globals.processing:
                break
            _thr = num_frame % num_threads
            while True:
                try:
                    self.frames_queue[_thr].put((num_frame, frame), timeout=0.1)
                    break
                except _QueueFull:
                    if not roop.globals.processing:
                        break
        for i in range(num_threads):
            self.frames_queue[i].put(None)


    def process_videoframes(self, threadindex, progress) -> None:
        while True:
            item = self.frames_queue[threadindex].get()
            if item is None:
                self.processing_threads -= 1
                self.processed_queue[threadindex].put((False, None))
                return
            else:
                frame_idx, frame = item
                try:
                    with _prof('frame_total'):
                        if self.options.frame_processing:
                            with _gpu_guard():
                                out = frame
                                for p in self.processors:
                                    out = p.Run(out)
                                resimg = out
                        else:
                            # process_frame serialises only its GPU primitives (under
                            # TensorRT), so CPU work overlaps across threads.
                            resimg = self.process_frame(frame, frame_idx=frame_idx)
                except RuntimeError as exc:
                    err_str = str(exc)
                    if 'CUDA' in err_str or 'cuda' in err_str or 'onnxruntime' in err_str.lower():
                        print(f'[ProcessMgr] GPU error on video frame {threadindex} — writing original: {err_str[:200]}')
                        resimg = frame  # fall back to unmodified frame
                    else:
                        # Fatal non-GPU RuntimeError: drain our input queue and post
                        # sentinel so write_frames_thread doesn't hang forever.
                        try:
                            while True:
                                self.frames_queue[threadindex].get_nowait()
                        except Exception:
                            pass
                        self.processing_threads -= 1
                        self.processed_queue[threadindex].put((False, None))
                        roop.globals.processing = False
                        raise
                except Exception:
                    # Any other exception (cv2.error, MemoryError, etc.) — same
                    # drain-and-signal so write_frames_thread unblocks.
                    try:
                        while True:
                            self.frames_queue[threadindex].get_nowait()
                    except Exception:
                        pass
                    self.processing_threads -= 1
                    self.processed_queue[threadindex].put((False, None))
                    roop.globals.processing = False
                    raise
                self.processed_queue[threadindex].put((True, resimg))
                del frame
                progress()


    def write_frames_thread(self):
        nextindex = 0
        num_producers = self.num_threads
        
        while True:
            process, frame = self.processed_queue[nextindex % self.num_threads].get()
            nextindex += 1
            if frame is not None:
                with _prof('encode'):
                    if self.output_to_file:
                        self.videowriter.write_frame(frame)
                    if self.output_to_cam:
                        self.streamwriter.WriteToStream(frame)
                del frame
            elif process == False:
                num_producers -= 1
                if num_producers < 1:
                    return


    def run_batch_inmem(self, output_method, source_video, target_video, frame_start, frame_end, fps, threads:int = 1, skip_audio=False):
        # Stabilization scheduling (temporal smoothing needs frames in order; the
        # multithreaded reader strides them out-of-order):
        #  - kps-only stabilization → 2-pass: precompute smoothed kps sequentially
        #    (pass 1, below once the source is set up) then swap in parallel
        #    (pass 2), keeping multi-thread speed.
        #  - enhancer stabilization smooths the enhanced OUTPUT, which only exists
        #    during the swap, so it can't be precomputed → fall back to the
        #    original single-thread sequential path.
        #  - parallel stabilization (opt-in ROOP_STAB_PARALLEL): process contiguous
        #    frame blocks per thread, each with its own stabilizer + warm-up, so
        #    BOTH kps and enhancer stabilization run multi-threaded.
        self._precomputed_mode = False
        self._precomputed_kps = None
        self._stab_active = False
        self._parallel_stab = False
        # Temporal detection (anti-flicker): its pre-pass gap-fills detection
        # misses AND applies the kps/lm106 smoothing itself, so the per-frame
        # kps stabilizer and the kps-only 2-pass become redundant — disable
        # them here so nothing double-smooths. (Enhancer flicker smoothing is
        # output-based and unaffected.)
        self._temporal_mode = bool(getattr(roop.globals, 'temporal_detection', False))
        self._temporal_faces = None
        self._temporal_covered = 0
        if self._temporal_mode:
            self.kps_stabilizer = None
            self._kps_stab_factory = None
        _want_kps_stab = self.kps_stabilizer is not None
        _want_enh_stab = self.enh_stabilizer is not None
        _parallel_ok = os.environ.get('ROOP_STAB_PARALLEL', '0') == '1'
        use_parallel_stab = (_want_kps_stab or _want_enh_stab) and threads > 1 and _parallel_ok
        _two_pass_ok = os.environ.get('ROOP_STAB_2PASS', '1') != '0'
        use_2pass = (not use_parallel_stab) and _want_kps_stab and not _want_enh_stab and threads > 1 and _two_pass_ok
        if (_want_kps_stab or _want_enh_stab) and not use_2pass and not use_parallel_stab:
            if threads != 1:
                print("[Stabilize] Forcing single thread for temporal smoothing.")
            threads = 1
            self._stab_active = True
            self._stab_t = 0
            if self.kps_stabilizer is not None:
                self.kps_stabilizer.reset()
            if self.enh_stabilizer is not None:
                self.enh_stabilizer.reset()

        # Animated WebP: OpenCV cannot decode it — use PIL-based reader instead
        is_awebp = source_video.lower().endswith('.webp')
        cap = None
        awebp_frames = None

        if is_awebp:
            from roop.capturer import _load_animated_webp
            import roop.capturer as _capturer_mod
            _load_animated_webp(source_video)
            awebp_frames = _capturer_mod._awebp_frames or []
            if awebp_frames:
                height, width = awebp_frames[0].shape[:2]
            else:
                width, height = 0, 0
            frame_count = len(awebp_frames[frame_start:frame_end]) if frame_end > frame_start else len(awebp_frames[frame_start:])
        else:
            cap = cv2.VideoCapture(source_video)
            frame_count = (frame_end - frame_start) + 1
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # NVDEC: swap the cv2 reader for a GPU-decode ffmpeg pipe when the
            # file probes OK (no-op otherwise; ROOP_NVDEC=0 disables). Must use
            # the SOURCE dims, before any processed_resolution override.
            from roop.nvdec_reader import wrap_capture
            cap = wrap_capture(cap, source_video, width, height, fps, tag='swap decode')

        processed_resolution = None
        for p in self.processors:
            if hasattr(p, 'getProcessedResolution'):
                processed_resolution = p.getProcessedResolution(width, height)
                print(f"Processed resolution: {processed_resolution}")
        if processed_resolution is not None:
            width = processed_resolution[0]
            height = processed_resolution[1]

        self.output_to_file = output_method != "Virtual Camera"
        self.output_to_cam = output_method == "Virtual Camera" or output_method == "Both"

        # Writer creation happens HERE (before the pre-passes) because resume
        # detection may shift frame_start forward — the temporal/SAM2/track
        # pre-passes and the 2-pass stabilizer must then only scan the frames
        # that still need encoding.
        if self.output_to_file:
            use_resume = (not is_awebp) and os.environ.get('ROOP_RESUME', '1') == '1'
            if use_resume:
                from roop.segment_writer import SegmentedVideoWriter
                self.videowriter = SegmentedVideoWriter(
                    target_video, (width, height), fps,
                    codec=roop.globals.video_encoder, crf=roop.globals.video_quality,
                    source_video=source_video, frame_start=frame_start, frame_end=frame_end,
                    signature=str(getattr(roop.globals, '_run_signature', '') or ''))
                skip = self.videowriter.resume_frames
                if skip >= frame_count > 0:
                    # Everything was already encoded by the interrupted run —
                    # just finalize (concat) and return; the caller's audio
                    # restore / renaming flow proceeds as if freshly rendered.
                    print(f'[Resume] all {frame_count} frames were already encoded '
                          f'by a previous run — finalizing without re-rendering.')
                    self.videowriter.close()
                    self.videowriter = None
                    if cap is not None:
                        cap.release()
                    return
                if skip > 0:
                    print(f'[Resume] found {skip} already-encoded frames from an '
                          f'interrupted run — resuming at frame {frame_start + skip}. '
                          f'(Delete {os.path.basename(target_video)}.resume.json to force a fresh render.)')
                    frame_start += skip
                    frame_count -= skip
            else:
                self.videowriter = FFMPEG_VideoWriter(target_video, (width, height), fps, codec=roop.globals.video_encoder, crf=roop.globals.video_quality, audiofile=None)
        if self.output_to_cam:
            self.streamwriter = StreamWriter((width, height), int(fps))

        # 2-pass stabilization, pass 1: precompute smoothed kps sequentially so
        # pass 2 (the swap) can run multi-threaded. Done before auto-tuning so the
        # tuner calibrates the real pass-2 workload.
        if use_2pass:
            self._precomputed_kps = self._precompute_stabilized_kps(
                source_video, awebp_frames, frame_start, frame_end, frame_count)
            self._precomputed_mode = True
            print(f"[Stabilize] 2-pass: precomputed smoothed kps for "
                  f"{len(self._precomputed_kps)} frames; pass 2 runs multi-threaded.")

        # Worker thread count is exactly the user's "Max. Number of Threads"
        # setting (no auto-tuning).
        self.total_frames = frame_count
        self.num_threads = threads
        self.processing_threads = self.num_threads
        self.frames_queue = []
        self.processed_queue = []
        # A little buffering per thread smooths variable per-frame times so the
        # reader/writer don't stall worker threads (matters now that CUDA runs
        # workers concurrently instead of serialised behind one GPU lock).
        qdepth = 1 if threads <= 1 else 3
        for _ in range(threads):
            self.frames_queue.append(Queue(qdepth))
            self.processed_queue.append(Queue(qdepth))

        # SAM2 temporal-mask pre-pass: track the faces across the trimmed clip and
        # cache a full-frame mask per frame, so the (still parallel) swap below can
        # look them up. Opt-in — only runs when the SAM2 engine is selected.
        sam2_p = next((p for p in self.processors
                       if getattr(p, 'processorname', None) == 'mask_sam2'), None)
        if sam2_p is not None and not is_awebp:
            try:
                self._precompute_sam2(sam2_p, source_video, frame_start, frame_end, frame_count)
            except Exception as e:
                print(f'[SAM2] pre-pass failed ({e}); falling back to unmasked swap')
                sam2_p.precomputed = {}

        # Identity-lock pre-pass: track each person across the clip and assign a
        # single source per track, so the per-frame embedding match can't flip
        # identities mid-video. Opt-in, only for "selected" mode on real video.
        self._track_mode = False
        self._track_assignments = {}

        # Temporal detection pre-pass (anti-flicker): detect + track every frame
        # once, gap-fill short detection misses and smooth kps/lm106/bbox per
        # track; the (still parallel) swap pass then consumes the cached faces
        # per frame instead of re-detecting, so the swap can't blink out on a
        # missed detection. The same scan yields the identity-lock assignments,
        # so no separate tracking pass is needed when both are enabled.
        if self._temporal_mode:
            try:
                self._precompute_temporal(source_video, awebp_frames, frame_start, frame_end, frame_count)
                # `and self._temporal_mode`: the pre-pass turns itself off when it
                # found nothing usable, and then its identity assignments are empty
                # too — so fall through to the standalone track pass below rather
                # than locking identities off an empty scan.
                self._track_mode = (self._temporal_mode
                                    and roop.globals.track_identities
                                    and self.options.swap_mode == "selected"
                                    and len(self.target_face_datas) > 0)
            except Exception as e:
                print(f'[Temporal] detection pre-pass failed ({e}); using per-frame detection')
                self._temporal_mode = False
                self._temporal_faces = None
                self._temporal_covered = 0

        if (not self._temporal_mode and roop.globals.track_identities and not is_awebp
                and self.options.swap_mode == "selected"
                and len(self.target_face_datas) > 0):
            try:
                self._precompute_tracks(source_video, frame_start, frame_end, frame_count)
                self._track_mode = True
            except Exception as e:
                print(f'[Track] identity pre-pass failed ({e}); using per-frame matching')
                self._track_mode = False

        progress_bar_format = PROGRESS_BAR_FORMAT
        try:
            if use_parallel_stab:
                print(f"[Stabilize] parallel stabilization ON (threads={threads}, warm-up overlap) — "
                      f"both kps and enhancer smoothing run multi-threaded.")
                with tqdm(total=self.total_frames, desc='Processing', unit='frames', dynamic_ncols=True, bar_format=progress_bar_format) as progress:
                    self._run_stab_parallel(source_video, awebp_frames, frame_start, frame_end,
                                            frame_count, threads, lambda: self.update_progress(progress))
            else:
                if is_awebp:
                    readthread = Thread(target=self.read_frames_webp_thread, args=(awebp_frames, frame_start, frame_end, threads))
                else:
                    readthread = Thread(target=self.read_frames_thread, args=(cap, frame_start, frame_end, threads))
                readthread.start()

                writethread = Thread(target=self.write_frames_thread)
                writethread.start()

                # Cross-frame swap batcher (opt-in): coalesce concurrent swap calls from
                # the worker threads into one batched inference. Needs >1 thread and the
                # batch-dynamic swap session (ROOP_BATCH_SWAP). Off → unchanged behavior.
                self._swap_batcher = self._make_swap_batcher(threads)

                try:
                    with tqdm(total=self.total_frames, desc='Processing', unit='frames', dynamic_ncols=True, bar_format=progress_bar_format) as progress:
                        with ThreadPoolExecutor(thread_name_prefix='swap_proc', max_workers=self.num_threads) as executor:
                            futures = []
                            for threadindex in range(threads):
                                future = executor.submit(self.process_videoframes, threadindex, lambda: self.update_progress(progress))
                                futures.append(future)
                            for future in as_completed(futures):
                                future.result()
                finally:
                    if self._swap_batcher is not None:
                        self._swap_batcher.stop()
                        self._swap_batcher.report()
                        self._swap_batcher = None
                    # Join with timeouts so an exception path never leaves background
                    # threads running (or holding the videowriter open). Timeouts are a
                    # safety net: normally both threads exit quickly because workers have
                    # already set roop.globals.processing=False and sent their sentinels.
                    readthread.join(timeout=5)
                    writethread.join(timeout=10)
        finally:
            # Always release the capture and close writers regardless of which path ran
            # and whether it raised.  The write thread MUST be joined (above) before we
            # close videowriter, otherwise the pipe stdin close races with an in-flight
            # write_frame() call and corrupts the temp file.
            if cap is not None:
                cap.release()
            if self.output_to_file and self.videowriter is not None:
                self.videowriter.close()
                self.videowriter = None
            if self.output_to_cam and self.streamwriter is not None:
                self.streamwriter.Close()
                self.streamwriter = None
            self.frames_queue.clear()
            self.processed_queue.clear()
            self._precomputed_mode = False
            self._precomputed_kps = None
            self._temporal_mode = False
            self._temporal_faces = None
            self._temporal_covered = 0
        _prof_report()


    def _make_swap_batcher(self, threads):
        """Build the cross-frame swap batcher when opted in. Requires >1 thread,
        a swapper exposing RunBatchMulti, and the batch-dynamic session
        (ROOP_BATCH_SWAP). Returns None otherwise (→ normal per-call swap)."""
        if not swap_batcher.xframe_enabled() or threads <= 1:
            return None
        if not _BATCH_SWAP:
            print("[BatchSwap] ROOP_BATCH_SWAP_XFRAME needs ROOP_BATCH_SWAP=1 "
                  "(batch-dynamic session) — skipping cross-frame batching.")
            return None
        swap_p = next((p for p in self.processors if getattr(p, 'type', None) == 'swap'), None)
        if swap_p is None or not hasattr(swap_p, 'RunBatchMulti'):
            return None
        pooled = getattr(swap_p, 'pool', None) is not None
        try:
            max_b = int(os.environ.get('ROOP_BATCH_SWAP_MAX', str(threads)))
        except ValueError:
            max_b = threads
        max_b = max(2, min(max_b, threads))
        b = swap_batcher.SwapBatcher(
            swap_p.RunBatchMulti, lambda: _gpu_guard(pooled=pooled),
            max_batch=max_b, max_wait_ms=6.0)
        print(f"[BatchSwap] cross-frame batching ON (max_batch={max_b}, threads={threads}).")
        return b


    def _run_stab_parallel(self, source_video, awebp_frames, frame_start, frame_end,
                           frame_count, threads, progress_cb):
        """Parallel stabilization (opt-in). Decodes the clip in chunks sequentially
        into memory (no seek → HEVC-safe), splits each chunk into contiguous
        per-thread sub-blocks, and runs each sub-block IN ORDER with its own
        stabilizer instances. A warm-up overlap primes each block's filter from
        the frames just before it, so block boundaries are seam-free. Frames are
        written in order through the already-open videowriter. Deadlock-free
        fork-join (plain thread start/join, no queues)."""
        WU = max(0, int(os.environ.get('ROOP_STAB_WARMUP', '4') or '4'))
        try:
            CHUNK = int(os.environ.get('ROOP_STAB_CHUNK', '0') or '0') or max(threads * 24, 192)
        except ValueError:
            CHUNK = max(threads * 24, 192)

        self._parallel_stab = True
        self._stab_active = True
        cap = None

        def _gen_frames():
            nonlocal cap
            if awebp_frames is not None:
                subset = awebp_frames[frame_start:frame_end] if frame_end > frame_start else awebp_frames[frame_start:]
                for fr in subset:
                    yield fr
            else:
                cap = cv2.VideoCapture(source_video)
                from roop.nvdec_reader import wrap_capture
                cap = wrap_capture(cap, source_video,
                                   int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                   int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                                   cap.get(cv2.CAP_PROP_FPS), tag='stabilized decode')
                if frame_start > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
                produced = 0
                while produced < frame_count:
                    ret, fr = cap.read()
                    if not ret or fr is None:
                        break
                    yield fr
                    produced += 1

        gen = _gen_frames()
        carry = []
        chunk_start = 0
        # Queue(2): reader can pre-fill one extra chunk so GPU never waits.
        # Using 2 (not 1) also avoids deadlock on cancel — reader won't block
        # on put() if the consumer has stopped and we drain below before join().
        prefetch_q = Queue(2)

        def _reader():
            while roop.globals.processing:
                # Pause: stop reading new frames while paused
                while getattr(roop.globals, 'pause', False) and roop.globals.processing:
                    time.sleep(0.05)
                if not roop.globals.processing:
                    break
                chunk = []
                for fr in gen:
                    chunk.append(fr)
                    if len(chunk) >= CHUNK:
                        break
                if not chunk:
                    break
                prefetch_q.put(chunk)
                if not roop.globals.processing:
                    break
            prefetch_q.put(None)  # sentinel: always sent, even on cancel

        rt = Thread(target=_reader, name='stab_reader', daemon=True)
        rt.start()

        # Background write thread: chunk N results are queued here immediately
        # after workers join, so the sequential FFMPEG write overlaps with workers
        # processing chunk N+1 instead of stalling between chunks.
        # Queue(1): one chunk can be in-flight; main blocks only when write is
        # slower than processing (correct back-pressure; prevents unbounded RAM).
        _write_q = Queue(1)

        _writer_exc = [None]  # propagate write errors back to the main thread

        def _writer():
            try:
                while True:
                    item = _write_q.get()
                    if item is None:
                        break
                    cs, res, clen = item
                    for gi in range(cs, cs + clen):
                        if not roop.globals.processing:
                            break
                        fr = res.pop(gi, None)  # pop frees the frame ref immediately after write
                        if fr is None:
                            continue
                        if self.output_to_file:
                            self.videowriter.write_frame(fr)
                        if self.output_to_cam:
                            self.streamwriter.WriteToStream(fr)
            except Exception as exc:
                _writer_exc[0] = exc

        _wt = Thread(target=_writer, name='stab_writer', daemon=True)
        _wt.start()

        _chunk_no = 0
        try:
            while True:
                _t_get0 = time.perf_counter()
                chunk = prefetch_q.get()
                _t_get = time.perf_counter() - _t_get0   # read-starvation wait
                if chunk is None:
                    break
                if not roop.globals.processing:
                    break

                combined = carry + chunk
                base = len(carry)
                base_global = chunk_start - base
                n = max(1, min(threads, len(chunk)))
                results = {}
                _block_times = {}   # per WORKER wall time → imbalance (see dispatch note below)

                # ── Closure-capture fix ──────────────────────────────────────
                # _process_block is re-defined each loop iteration. Without default-arg
                # capture, every thread closure shares the same `combined`,
                # `base`, `base_global` and `results` variables — dangerous when
                # Python advances the outer loop before threads finish reading them.
                # Binding them as default args freezes the values per chunk.
                def _process_block(a, b,
                                    _combined=combined, _base=base,
                                    _base_global=base_global, _results=results):
                    self._tls.kps = self._kps_stab_factory() if self._kps_stab_factory else None
                    self._tls.enh = self._enh_stab_factory() if self._enh_stab_factory else None
                    ca = _base + a
                    for ci in range(max(0, ca - WU), ca):   # warm-up: prime filter, discard
                        if not roop.globals.processing:
                            return
                        self._tls.t = _base_global + ci
                        try:
                            # Pass the real frame index so the temporal-detection /
                            # SAM2 / identity-track caches stay usable in this path.
                            self.process_frame(_combined[ci], frame_idx=_base_global + ci)
                        except Exception:
                            pass
                    for ci in range(ca, _base + b):
                        if not roop.globals.processing:
                            return
                        gi = _base_global + ci
                        self._tls.t = gi
                        try:
                            out = self.process_frame(_combined[ci], frame_idx=gi)
                        except Exception:
                            out = _combined[ci]
                        _results[gi] = out if out is not None else _combined[ci]
                        progress_cb()

                # ── Work-stealing block dispatch ──────────────────────────────
                # Per-frame cost varies a lot (face count/size, masking, close-up
                # rescue paths) and isn't temporally uniform, so statically handing
                # each of the n threads one fixed contiguous range let one "unlucky"
                # thread's range dominate the chunk's wall time while the other
                # threads sat idle after finishing early (visible as the printed
                # "imbalance" stat, sometimes >50% of the chunk's proc time). Splitting
                # into more, smaller blocks and handing them out through a shared
                # queue to a fixed pool of n workers lets idle workers pick up the
                # next block instead of idling. Each block still gets its own
                # WU-frame warm-up (unchanged, seam-free), so this costs more
                # redundant warm-up compute as granularity increases — tune via
                # ROOP_STAB_BLOCKS_PER_THREAD (default 1 = one block per thread,
                # identical boundaries/output to the old static split).
                try:
                    _bpt = max(1, int(os.environ.get('ROOP_STAB_BLOCKS_PER_THREAD', '1') or '1'))
                except ValueError:
                    _bpt = 1
                n_blocks = max(1, min(n * _bpt, len(chunk)))
                bstep = len(chunk) / n_blocks
                block_q = Queue()
                for bi in range(n_blocks):
                    a = int(round(bi * bstep))
                    b = len(chunk) if bi == n_blocks - 1 else int(round((bi + 1) * bstep))
                    if b > a:
                        block_q.put((a, b))

                def _runner(_wi, _bt=_block_times, _q=block_q, _pb=_process_block):
                    _w0 = time.perf_counter()
                    while roop.globals.processing:
                        try:
                            a, b = _q.get_nowait()
                        except _QueueEmpty:
                            break
                        _pb(a, b)
                    _bt[_wi] = time.perf_counter() - _w0

                workers = [Thread(target=_runner, args=(i,), name=f'stab_proc{i}') for i in range(n)]
                _t_proc0 = time.perf_counter()
                for w in workers:
                    w.start()
                for w in workers:
                    w.join()
                _t_proc = time.perf_counter() - _t_proc0   # compute time (slowest worker gates this)

                # Queue the chunk for the background write thread.
                # Blocks only when FFMPEG is slower than frame processing
                # (correct back-pressure; prevents unbounded memory growth).
                _t_put0 = time.perf_counter()
                _write_q.put((chunk_start, results, len(chunk)))
                _t_put = time.perf_counter() - _t_put0     # write back-pressure stall

                if _PROFILE:
                    _bts = list(_block_times.values())
                    _imbal = (max(_bts) - min(_bts)) if _bts else 0.0
                    _fps = len(chunk) / _t_proc if _t_proc > 0 else 0.0
                    
                    reset = "\033[0m"
                    bold = "\033[1m"
                    cyan = "\033[96m"
                    green = "\033[92m"
                    yellow = "\033[93m"
                    magenta = "\033[95m"
                    
                    fps_color = green if _fps >= 30 else (yellow if _fps >= 15 else "\033[91m")
                    stall_color = yellow if (_t_put * 1000) > 100 else cyan
                    wait_color = yellow if (_t_get * 1000) > 100 else cyan
                    
                    print(f"{magenta}{bold}[STAB CHUNK {_chunk_no:3d}]{reset} "
                          f"frames={bold}{green}{len(chunk):4d}{reset} | "
                          f"read_wait={wait_color}{_t_get*1000:6.0f}ms{reset} | "
                          f"proc={bold}{fps_color}{_t_proc:6.2f}s ({_fps:5.1f} FPS){reset} | "
                          f"write_stall={stall_color}{_t_put*1000:6.0f}ms{reset} | "
                          f"imbalance={yellow}{_imbal:5.2f}s{reset} "
                          f"(slow={max(_bts) if _bts else 0:.2f}s / fast={min(_bts) if _bts else 0:.2f}s)",
                          flush=True)
                _chunk_no += 1

                carry = combined[-WU:] if WU > 0 else []
                chunk_start += len(chunk)
        finally:
            # Drain _write_q before sending the sentinel when:
            #  - cancel: discard queued frames so the writer exits promptly.
            #  - writer died (IOError / broken pipe): the queue may still hold an
            #    item that nobody will ever consume; put(None) would block forever
            #    on the full Queue(1) without this drain.
            if not roop.globals.processing or not _wt.is_alive():
                try:
                    while True:
                        _write_q.get_nowait()
                except Exception:
                    pass
            _write_q.put(None)   # sentinel — always signals writer to stop
            _wt.join(timeout=15)

            # Drain the prefetch queue so _reader's put() can't block
            try:
                while not prefetch_q.empty():
                    prefetch_q.get_nowait()
            except Exception:
                pass
            rt.join(timeout=5)
            self._parallel_stab = False
            for _a in ('kps', 'enh', 't'):
                if hasattr(self._tls, _a):
                    delattr(self._tls, _a)
            if cap is not None:
                cap.release()
            # Re-raise write error AFTER all cleanup so cap/prefetch are always freed
            if _writer_exc[0] is not None:
                raise _writer_exc[0]


    def update_progress(self, progress: Any = None) -> None:
        # Reuse one Process handle instead of rebuilding it every frame.
        process = self._psutil_proc
        if process is None:
            process = self._psutil_proc = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / 1024 / 1024 / 1024
        mem_str = f"{COLOR_CYAN}{memory_usage:.2f}GB{COLOR_RESET}"
        thread_str = f"{COLOR_YELLOW}{self.num_threads}{COLOR_RESET}"
        progress.set_postfix({
            'memory_usage': mem_str,
            'execution_threads': thread_str
        })
        progress.update(1)
        if self.progress_gradio is not None:
            n = progress.n
            total = self.total_frames
            rate = progress.format_dict.get('rate', 0.0) if hasattr(progress, 'format_dict') else 0.0
            fps_str = f" ({rate:.1f} FPS)" if rate and rate > 0 else ""
            desc = f"Processing frame {n} / {total}{fps_str}"
            self.progress_gradio((n, total), desc=desc, total=total, unit='frames')


    def _publish_live(self, frame):
        # Feed the processing box's live view. This used to keep a full-frame
        # COPY per frame for the UI, which is why it was turned off; the module
        # now throttles to ~2 frames a second and downscales + encodes once, so
        # the cost no longer scales with frame rate. See roop/live_preview.py.
        _live_preview.publish(frame)

    def process_frame(self, frame:Frame, frame_idx=None):
        # ── Pause support ────────────────────────────────────────────────────
        # Spin-wait while the user has paused. Checks every 50 ms so the UI
        # stays responsive. Exits immediately if processing is cancelled.
        while getattr(roop.globals, 'pause', False) and roop.globals.processing:
            time.sleep(0.05)

        if len(self.input_face_datas) < 1 and not self.options.show_face_masking:
            return frame
        temp_frame = frame.copy()
        num_swapped, temp_frame = self.swap_faces(frame, temp_frame, stabilize=True, frame_idx=frame_idx)
        if num_swapped > 0:
            if roop.globals.no_face_action == eNoFaceAction.SKIP_FRAME_IF_DISSIMILAR:
                if len(self.input_face_datas) > num_swapped:
                    return None
            self.num_frames_no_face = 0
            self.last_swapped_frame = temp_frame.copy()
            self._publish_live(temp_frame)
            return temp_frame
        if roop.globals.no_face_action == eNoFaceAction.USE_LAST_SWAPPED:
            if self.last_swapped_frame is not None and self.num_frames_no_face < self.options.max_num_reuse_frame:
                self.num_frames_no_face += 1
                ret = self.last_swapped_frame.copy()
                self._publish_live(ret)
                return ret
            self._publish_live(frame)
            return frame
        elif roop.globals.no_face_action == eNoFaceAction.USE_ORIGINAL_FRAME:
            self._publish_live(frame)
            return frame
        if roop.globals.no_face_action == eNoFaceAction.SKIP_FRAME:
            return None
        else:
            ret = self.retry_rotated(frame)
            self._publish_live(ret)
            return ret

    def retry_rotated(self, frame):
        copyframe = frame.copy()
        copyframe = rotate_clockwise(copyframe)
        temp_frame = copyframe.copy()
        num_swapped, temp_frame = self.swap_faces(copyframe, temp_frame)
        if num_swapped > 0:
            return rotate_anticlockwise(temp_frame)
        
        copyframe = frame.copy()
        copyframe = rotate_anticlockwise(copyframe)
        temp_frame = copyframe.copy()
        num_swapped, temp_frame = self.swap_faces(copyframe, temp_frame)
        if num_swapped > 0:
            return rotate_clockwise(temp_frame)
        del copyframe
        return frame

















    # ── Stabilizer accessors: in the parallel path each worker has its own
    #    instances + frame-time via thread-local storage; otherwise the shared
    #    single-thread instances are used. ─────────────────────────────────────
    def _expression_restorer(self):
        """Lazily built and shared across worker threads.

        The models are ~537 MB and stateless per call, so one instance is right
        unless ROOP_EXPR_POOL asks for more; the restorer handles its own
        exclusion (a lock when unpooled, a slot lease when pooled).

        Construction is locked because this is now called OUTSIDE the GPU guard —
        the guard has to know whether the restorer is pooled before it can decide
        whether to serialise, so it can no longer be what protects the build.
        """
        r = getattr(self, '_expr_restorer', None)
        if r is not None:
            return r
        with _EXPR_BUILD_LOCK:
            r = getattr(self, '_expr_restorer', None)
            if r is None:
                from roop.processors.Expression_LivePortrait import Expression_LivePortrait
                from roop.utilities import get_device
                r = Expression_LivePortrait()
                r.Initialize({"devicename": get_device()})
                self._expr_restorer = r
        return r

    def _cur_kps_stab(self):
        return getattr(self._tls, 'kps', None) if self._parallel_stab else self.kps_stabilizer

    def _cur_enh_stab(self):
        return getattr(self._tls, 'enh', None) if self._parallel_stab else self.enh_stabilizer

    def _cur_stab_t(self):
        return getattr(self._tls, 't', 0) if self._parallel_stab else self._stab_t

    def _apply_stab(self, face):
        """Replace a face's 5-point kps with the temporally smoothed version."""
        ks = self._cur_kps_stab()
        if ks is None:
            return
        try:
            if face is not None and getattr(face, 'kps', None) is not None:
                face.kps = ks.apply(face.kps, self._cur_stab_t())
        except Exception:
            pass

    def swap_faces(self, frame, temp_frame, stabilize=False, frame_idx=None):
        num_faces_found = 0

        # Stash the current frame index per-thread so the SAM2 mask engine can look
        # up its precomputed full-frame mask for this frame from inside process_mask
        # (set here because every worker thread enters swap_faces once per frame).
        self._tls.frame_idx = frame_idx

        # Tick once per frame if any stabilizer is active. Parallel path uses a
        # per-thread frame index (TLS) instead of this shared counter.
        if (stabilize and self._stab_active and not self._parallel_stab
                and (self.kps_stabilizer is not None or self.enh_stabilizer is not None)):
            self._stab_t += 1
        do_kps_stab = stabilize and self._stab_active and self._cur_kps_stab() is not None
        # 2-pass parallel stabilization: replace kps with the value precomputed
        # for this frame in pass 1 instead of running the (stateful) stabilizer.
        precomp = stabilize and self._precomputed_mode and frame_idx is not None

        # Temporal detection: consume the pre-pass faces for this frame instead
        # of re-detecting (already gap-filled + smoothed; keeps workers parallel
        # and detection out of the hot loop). None → normal per-frame detection.
        _tfaces = None
        if self._temporal_mode and frame_idx is not None and self._temporal_faces is not None:
            # Only trust the cache for frames the pre-pass actually scanned. Inside
            # that range a missing entry legitimately means "nobody was there";
            # past it the frame was never looked at, so detect it normally instead
            # of silently leaving it un-swapped (see the coverage guard in
            # _precompute_temporal).
            if frame_idx < self._temporal_covered:
                _tfaces = self._temporal_faces.get(frame_idx) or []

        if self.options.swap_mode == "first":
            if _tfaces is not None:
                face = min(_tfaces, key=lambda f: f.bbox[0]) if _tfaces else None
            else:
                with _prof('detect'), _gpu_guard(pooled=analysis_pooled()):  # detect: lock-free when pooled
                    face = get_first_face(frame)
                    if face is None and self.last_found_bboxes is not None:
                        face = _detect_face_in_roi(frame, self.last_found_bboxes[0])
            if face is None:
                return num_faces_found, frame
            self.last_found_bboxes = np.array([face.bbox])   # cache for next frame
            if precomp:
                face.kps = self._lookup_precomputed_kps(frame_idx, face)
            elif do_kps_stab:
                self._apply_stab(face)
            num_faces_found += 1
            temp_frame = self.process_face(self.options.selected_index, face, temp_frame)
            del face

        else:
            if _tfaces is not None:
                faces = list(_tfaces)   # copy — faces.clear() below must not wipe the cache
            else:
                with _prof('detect'), _gpu_guard(pooled=analysis_pooled()):  # detect: lock-free when pooled
                    faces = get_all_faces(frame)
                    if not faces and self.last_found_bboxes is not None:
                        recovered = []
                        for bbox in self.last_found_bboxes:
                            f = _detect_face_in_roi(frame, bbox)
                            if f is not None:
                                recovered.append(f)
                        if recovered:
                            faces = recovered
            if not faces:
                return num_faces_found, frame
            self.last_found_bboxes = np.array([f.bbox for f in faces])   # cache for next frame
            if precomp:
                for f in faces:
                    f.kps = self._lookup_precomputed_kps(frame_idx, f)
            elif do_kps_stab:
                for f in faces:
                    self._apply_stab(f)

            if self.options.swap_mode == "all":
                for face in faces:
                    num_faces_found += 1
                    temp_frame = self.process_face(self.options.selected_index, face, temp_frame)

            elif self.options.swap_mode == "all_input":
                for i, face in enumerate(faces):
                    num_faces_found += 1
                    if i < len(self.input_face_datas):
                        temp_frame = self.process_face(i, face, temp_frame)
                    else:
                        break

            elif self.options.swap_mode == "selected" and getattr(self, '_track_mode', False) and frame_idx is not None:
                # Identity-lock: use the source assigned to this person's TRACK in
                # the pre-pass (matched by combination of spatial distance and embedding cosine similarity),
                # so the source can't flip frame-to-frame as it can with per-frame embedding matching.
                # Falls through to per-frame matching if untracked.
                entries = self._track_assignments.get(frame_idx)
                if not entries:
                    # Fallback lookup to nearest tracked frame within a 5-frame window
                    for offset in range(1, 6):
                        entries = self._track_assignments.get(frame_idx - offset)
                        if entries:
                            break
                        entries = self._track_assignments.get(frame_idx + offset)
                        if entries:
                            break
                entries = entries or []
                claimed = set()
                claimed_sources_in_frame = set()
                
                groups = self.target_face_groups
                uniq = sorted(set(groups)) if groups else []
                rank = {g: r for r, g in enumerate(uniq)}
                single_person = len(uniq) <= 1
                threshold = self.options.face_distance_threshold
                # When AdaFace drives identity, distances are on ITS scale —
                # comparing them against max_face_distance would be meaningless.
                id_threshold = _ada.active_threshold(threshold)

                # person group id -> its captured target-face (angle) indices.
                # Hoisted out of the per-face fallback below because the source
                # veto (next block) needs it for every face, not just the ones
                # that fall through.
                persons = {}
                for i, g in enumerate(groups[:len(self.target_face_datas)]):
                    persons.setdefault(g, []).append(i)
                # source index -> the captured angles of the person that source
                # belongs to, so a track's source can be checked against the face
                # actually in front of us.
                rank_to_tis = {}
                for g, tis in persons.items():
                    r = self.options.selected_index if single_person else rank[g]
                    rank_to_tis.setdefault(r, []).extend(tis)

                def _dist_to_source(face, src):
                    """Cosine distance from *face* to the closest captured angle of
                    the person that source index *src* belongs to (None if unknown)."""
                    tis = rank_to_tis.get(src)
                    if not tis or getattr(face, 'embedding', None) is None:
                        return None
                    ds = [_ada.identity_distance(self.target_face_datas[ti], face, frame)
                          for ti in tis
                          if getattr(self.target_face_datas[ti], 'embedding', None) is not None]
                    ds = [d for d in ds if d is not None]
                    return min(ds) if ds else None

                def _dist_to_any_person(face):
                    """Closest captured angle of ANY selected person (inf if none)."""
                    ds = []
                    for tis in persons.values():
                        for ti in tis:
                            d = _ada.identity_distance(self.target_face_datas[ti], face, frame)
                            if d is not None:
                                ds.append(d)
                    return min(ds) if ds else float('inf')

                # ── Claim order ──────────────────────────────────────────────
                # A source can be applied at most ONCE per frame (below), so
                # whichever face reaches it first wins and every other face in
                # the frame is refused. The faces arrive in x order, which has
                # nothing to do with who they are: a false detection on the
                # background could take the source and leave the real face
                # unswapped for that frame. Nothing downstream could recover
                # from it — the real face hits `src in claimed_sources_in_frame`
                # and is skipped.
                #
                # So let the best claim first. Real detections outrank
                # gap-filled ones (an interpolated face carries the TRACK MEAN
                # as its embedding, so it would otherwise always score better
                # than the real face it is competing with), then closest first.
                # Ordering only — no face is added or dropped here.
                faces = sorted(faces, key=lambda f: (
                    bool(f.get('_interpolated')) if isinstance(f, dict) else False,
                    _dist_to_any_person(f)))

                for face in faces:
                    best_j, best_cost = -1, float('inf')
                    if entries:
                        bb = face.bbox
                        c = np.array([(bb[0] + bb[2]) * 0.5, (bb[1] + bb[3]) * 0.5], np.float32)
                        for j, entry in enumerate(entries):
                            if j in claimed:
                                continue
                            cent, src_index = entry[0], entry[1]
                            d_spatial = float(np.hypot(cent[0] - c[0], cent[1] - c[1]))
                            if len(entry) > 2 and entry[2] is not None:
                                d_cosine = float(compute_cosine_distance(entry[2], face.embedding))
                            else:
                                d_cosine = 0.0

                            # Appearance GATE, matching the one the tracking scan
                            # already applies when it builds these tracks. Without
                            # it the multiplicative cost below can always be won by
                            # a small spatial distance no matter how wrong the face
                            # looks, which is how a track hands its source to
                            # whoever is standing closest. Refusing the association
                            # outright is the standard tracking-by-detection rule.
                            if _TRACK_EMB_MAX > 0 and d_cosine > _TRACK_EMB_MAX:
                                continue

                            # Combine spatial distance penalized by cosine similarity distance
                            cost = d_spatial * (1.0 + 2.5 * d_cosine)
                            if cost < best_cost:
                                best_cost, best_j = cost, j
                                
                    src_index = None
                    veto = None
                    if best_j >= 0:
                        cand = entries[best_j][1]
                        if cand is not None and cand < len(self.input_face_datas):
                            src_index = cand
                        # ── Source veto ──────────────────────────────────────
                        # The entry above is chosen by POSITION (cost is spatial,
                        # only nudged by cosine), so on its own it will happily
                        # hand person A's faceset to person B when they cross, or
                        # swap an unselected bystander who happens to stand where
                        # a track was. Identity locking is meant to stabilise WHICH
                        # source a person gets, never to override WHO the person is.
                        # So the track's source is accepted only if this face really
                        # is that person:
                        #   - reject if another selected person explains the face
                        #     distinctly better (relative test — catches crossings
                        #     and look-alikes without needing an absolute cutoff);
                        #   - reject if the face matches the assigned person no
                        #     better than VETO (absolute test — catches bystanders
                        #     and hard ID switches);
                        #   - reject if that source was already used in this frame
                        #     (one source per face, one face per source).
                        # A rejected face falls through to per-frame matching below,
                        # which is threshold-gated and strictly 1:1 — so it either
                        # gets its OWN person's source or stays unswapped.
                        # The gate is deliberately looser than the match threshold:
                        # it is a veto on clear mismatches, not a re-selection, so a
                        # blurred/turned frame of the right person still swaps
                        # (that anti-flicker property is the point of tracking).
                        if src_index is not None and _TRACK_VETO_DIST > 0:
                            d_own = _dist_to_source(face, src_index)
                            d_other = None
                            if not single_person:
                                others = [d for r2 in rank_to_tis if r2 != src_index
                                          for d in [_dist_to_source(face, r2)] if d is not None]
                                d_other = min(others) if others else None
                            # The ABSOLUTE test is only meaningful when there is
                            # someone else's faceset to protect. With a single
                            # selected person it can only ever take swaps away:
                            # d_own is computed from the CURRENT frame's embedding,
                            # and a profile / motion-blurred / steeply pitched frame
                            # of the right person routinely exceeds 0.85 in scipy
                            # cosine distance (0..2). Vetoing there drops the face
                            # to per-frame matching at the TIGHTER match threshold,
                            # which also fails — so the swap blinks off for exactly
                            # the hard frames tracking exists to carry, and blinks
                            # back on when the head comes round. That on/off is the
                            # shaking. There is no bystander risk to trade against
                            # when only one person is selected, so skip it.
                            multi_person = len(rank_to_tis) > 1
                            if src_index in claimed_sources_in_frame:
                                veto = 'source already used this frame'
                            elif (multi_person and d_own is not None
                                    and d_own > _ada.scale(_TRACK_VETO_DIST, threshold)):
                                veto = f'face is {d_own:.2f} from its assigned person (> {_TRACK_VETO_DIST})'
                            elif (not multi_person and _TRACK_VETO_SINGLE > 0
                                    and d_own is not None
                                    and d_own > _ada.scale(_TRACK_VETO_SINGLE, threshold)):
                                # Opt-in catch for a tracker identity switch when
                                # only one person is selected — see the constant.
                                veto = (f'face is {d_own:.2f} from the selected person '
                                        f'(> {_TRACK_VETO_SINGLE}, single-person veto)')
                            elif (d_own is not None and d_other is not None
                                    and d_other + _ada.scale(_TRACK_VETO_MARGIN, threshold) < d_own):
                                veto = f'another person fits better ({d_other:.2f} vs {d_own:.2f})'
                            if veto:
                                src_index = None
                        # Claim the entry even when its source is unresolved, so a
                        # second face this frame can't re-match the same track —
                        # but NOT when we vetoed on identity, because then this
                        # entry most likely belongs to one of the other faces.
                        if veto is None:
                            claimed.add(best_j)

                    if _DEBUG_MATCH:
                        print(f"[TRACKMATCH] f={frame_idx} faces={len(faces)} "
                              f"entry_srcs={[e[1] for e in entries]} best_j={best_j} "
                              f"src={src_index} claimed={sorted(claimed)}"
                              + (f" VETO: {veto}" if veto else ""))

                    if src_index is not None:
                        claimed_sources_in_frame.add(src_index)
                        temp_frame = self.process_face(src_index, face, temp_frame)
                        num_faces_found += 1
                    else:
                        # Fall back to per-frame multi-angle matching — the same
                        # logic live preview uses — so these frames still swap
                        # instead of being silently skipped. Threshold-gated and
                        # 1:1 (a source already used this frame is skipped), so a
                        # face here can only ever get its OWN person's source.
                        best_g, best_d = None, id_threshold
                        for g, tis in persons.items():
                            r_src = self.options.selected_index if single_person else rank.get(g, 0)
                            if r_src in claimed_sources_in_frame:
                                continue
                            embs = [getattr(self.target_face_datas[ti], 'embedding', None) for ti in tis]
                            embs = [e for e in embs if e is not None]
                            if not embs or getattr(face, 'embedding', None) is None:
                                continue
                            _ds = [_ada.identity_distance(self.target_face_datas[ti], face, frame)
                                   for ti in tis]
                            _ds = [x for x in _ds if x is not None]
                            if not _ds:
                                continue
                            d = min(_ds)
                            if d <= best_d:
                                best_d, best_g = d, g

                        if _DEBUG_MATCH:
                            _dd = {}
                            for g, tis in persons.items():
                                try:
                                    _dd[g] = round(min(compute_cosine_distance(
                                        self.target_face_datas[ti].embedding, face.embedding)
                                        for ti in tis), 3)
                                except Exception:
                                    _dd[g] = None
                            print(f"[TRACKFALL] f={frame_idx} best_g={best_g} best_d={best_d:.3f} "
                                  f"claimed_src={sorted(claimed_sources_in_frame)} thr={threshold} d={_dd}")

                        if best_g is not None:
                            src_index = self.options.selected_index if single_person else rank[best_g]
                            if src_index < len(self.input_face_datas):
                                claimed_sources_in_frame.add(src_index)
                                temp_frame = self.process_face(src_index, face, temp_frame)
                                num_faces_found += 1

            elif self.options.swap_mode == "selected":
                # Multi-angle matching: assign each captured target PERSON their
                # single closest detected face (min distance across that person's
                # stored angles), within the distance threshold. A turned head
                # still matches via a side/back angle, so the swap doesn't drop
                # out frame-to-frame (no flicker).
                #
                # Crucially this is a 1:1 assignment: a selected person maps to
                # AT MOST ONE face per frame, and a face is swapped by AT MOST
                # ONE person. This stops a look-alike or hard-pose bystander from
                # also being swapped just for landing under the distance cutoff
                # (which a per-face "swap everything under threshold" loop did).
                groups = self.target_face_groups
                uniq = sorted(set(groups)) if groups else []
                rank = {g: r for r, g in enumerate(uniq)}
                single_person = len(uniq) <= 1
                threshold = self.options.face_distance_threshold
                # When AdaFace drives identity, distances are on ITS scale —
                # comparing them against max_face_distance would be meaningless.
                id_threshold = _ada.active_threshold(threshold)

                # person group id -> list of its target-face (angle) indices
                persons = {}
                for i, g in enumerate(groups[:len(self.target_face_datas)]):
                    persons.setdefault(g, []).append(i)

                # (distance, person_g, face_idx) for every pair within threshold,
                # using each person's closest angle to that face.
                candidates = []
                for fidx, face in enumerate(faces):
                    for g, tis in persons.items():
                        _ds = [_ada.identity_distance(self.target_face_datas[ti], face, frame)
                               for ti in tis]
                        _ds = [x for x in _ds if x is not None]
                        if not _ds:
                            continue
                        d = min(_ds)
                        if d <= id_threshold:
                            candidates.append((d, g, fidx))
                candidates.sort(key=lambda c: c[0])   # greedily assign closest pairs first

                # ── Diagnostic (preview / ROOP_DEBUG_MATCH) ──────────────────
                # Multi-person "person 2 never swaps" bugs are almost always one
                # of: (a) the backend only has ONE captured person group so
                # single_person collapses two faces onto one source, (b) a person
                # sits just over the distance threshold this frame, or (c) fewer
                # source facesets than persons. Surface all three at a glance.
                if _DEBUG_MATCH:
                    try:
                        dists = {fidx: {g: round(min(compute_cosine_distance(
                                    self.target_face_datas[ti].embedding, faces[fidx].embedding)
                                    for ti in tis), 3) for g, tis in persons.items()}
                                 for fidx in range(len(faces))}
                        print(f"[MATCH] persons={len(persons)} single_person={single_person} "
                              f"faces={len(faces)} sources={len(self.input_face_datas)} "
                              f"thr={threshold} dist(face->person)={dists}")
                    except Exception as _e:
                        print(f"[MATCH] diag failed: {_e}")

                claimed_faces, claimed_persons = set(), set()
                for d, g, fidx in candidates:
                    if fidx in claimed_faces or g in claimed_persons:
                        continue
                    claimed_faces.add(fidx)
                    claimed_persons.add(g)
                    src_index = self.options.selected_index if single_person else rank[g]
                    if src_index < len(self.input_face_datas):
                        temp_frame = self.process_face(src_index, faces[fidx], temp_frame)
                        num_faces_found += 1
                    elif _DEBUG_MATCH:
                        print(f"[MATCH] person g={g} matched face {fidx} but src_index="
                              f"{src_index} >= sources({len(self.input_face_datas)}) — NOT swapped")

            elif self.options.swap_mode == "all_female" or self.options.swap_mode == "all_male":
                gender = 'F' if self.options.swap_mode == "all_female" else 'M'
                for face in faces:
                    if face.sex == gender:
                        num_faces_found += 1
                        temp_frame = self.process_face(self.options.selected_index, face, temp_frame)

            for face in faces:
                del face
            faces.clear()

        if roop.globals.vr_mode and num_faces_found % 2 > 0:
            num_faces_found = 0
            return num_faces_found, frame
        if num_faces_found == 0:
            return num_faces_found, frame

        # ── Apply manual include / exclude masks ────────────────────────────
        # Canonical masks and ref_kps warp masks are applied inside process_face.
        # This fallback full-frame blend only runs for genuinely legacy masks
        # (no ref_kps, not canonical) saved before face-crop tracking was added.
        # Uses faceset-0 entry as the representative mask for the whole frame.
        _legacy_fm = self.face_masks.get(0)
        if (_legacy_fm is not None
                and _legacy_fm.get('ref_kps') is None
                and not _legacy_fm.get('is_canonical', False)):
            h, w = frame.shape[:2]
            combined = np.zeros((h, w), dtype=np.float32)

            exc = _legacy_fm.get('exclude_mask')
            if exc is not None:
                if exc.shape[:2] != (h, w):
                    exc = cv2.resize(exc, (w, h), interpolation=cv2.INTER_LINEAR)
                combined = np.maximum(combined, exc)

            inc = _legacy_fm.get('include_mask')
            if inc is not None:
                if inc.shape[:2] != (h, w):
                    inc = cv2.resize(inc, (w, h), interpolation=cv2.INTER_LINEAR)
                combined = combined * (1.0 - inc)

            temp_frame = self.simple_blend_with_mask(temp_frame, frame, combined)

        return num_faces_found, temp_frame


    def rotation_action(self, original_face:Face, frame:Frame):
        (height, width) = frame.shape[:2]

        bounding_box_width = original_face.bbox[2] - original_face.bbox[0]
        bounding_box_height = original_face.bbox[3] - original_face.bbox[1]
        horizontal_face = bounding_box_width > bounding_box_height

        center_x = width // 2.0
        start_x = original_face.bbox[0]
        end_x = original_face.bbox[2]
        bbox_center_x = start_x + (bounding_box_width // 2.0)

        forehead_x = original_face.landmark_2d_106[72][0]
        chin_x = original_face.landmark_2d_106[0][0]

        if horizontal_face:
            if chin_x < forehead_x:
                return "rotate_anticlockwise"
            elif forehead_x < chin_x:
                return "rotate_clockwise"
            if bbox_center_x >= center_x:
                return "rotate_anticlockwise"
            if bbox_center_x < center_x:
                return "rotate_clockwise"

        return None


    def auto_rotate_frame(self, original_face, frame:Frame):
        target_face = original_face
        original_frame = frame
        rotation_action = self.rotation_action(original_face, frame)
        if rotation_action == "rotate_anticlockwise":
            frame = rotate_anticlockwise(frame)
        elif rotation_action == "rotate_clockwise":
            frame = rotate_clockwise(frame)
        return target_face, frame, rotation_action


    def auto_unrotate_frame(self, frame:Frame, rotation_action):
        if rotation_action == "rotate_anticlockwise":
            return rotate_clockwise(frame)
        elif rotation_action == "rotate_clockwise":
            return rotate_anticlockwise(frame)
        return frame


    def process_face(self, face_index, target_face:Face, frame:Frame):
        from roop.face_util import align_crop

        # Count each target face processed (density = total_swaps / frames). A
        # lost increment under thread races is harmless for a coarse average.
        self.total_swaps += 1

        # Capture full-frame dimensions before any rotation rebind.
        # 'frame' may be reassigned to a smaller rotcutframe below when
        # autorotate_faces is active.  mask_ref_kps are always in the
        # original full-frame coordinate space, so the warp path needs these.
        orig_fh, orig_fw = frame.shape[:2]

        enhanced_frame = None
        # inputface is assigned after pose computation below (supports source bank)
        inputface = None

        rotation_action = None
        if roop.globals.autorotate_faces:
            rotation_action = self.rotation_action(target_face, frame)
            if rotation_action is not None:
                (startX, startY, endX, endY) = target_face["bbox"].astype("int")
                width = endX - startX
                height = endY - startY
                offs = int(max(width, height) * 0.25)
                rotcutframe, startX, startY, endX, endY = self.cutout(frame, startX - offs, startY - offs, endX + offs, endY + offs)
                if rotation_action == "rotate_anticlockwise":
                    rotcutframe = rotate_anticlockwise(rotcutframe)
                elif rotation_action == "rotate_clockwise":
                    rotcutframe = rotate_clockwise(rotcutframe)
                rotface = get_first_face(rotcutframe)
                if rotface is None:
                    rotation_action = None
                else:
                    saved_frame = frame.copy()
                    frame = rotcutframe
                    target_face = rotface

        # ── Model output size (inswapper uses 128 × 128) ─────────────────────
        swap_p = next((p for p in self.processors if p.type == 'swap'), None)
        model_output_size = getattr(swap_p, 'model_output_size', 128)

        subsample_size = self.options.subsample_size
        # Ensure subsample_size is an integer multiple of model_output_size
        if subsample_size < model_output_size:
            subsample_size = model_output_size
        subsample_total = subsample_size // model_output_size

        # Align with the swap model's training template (ghost/simswap use
        # arcface_112_v1, hififace uses mtcnn_512; inswapper family = arcface).
        # M is carried through masks/paste/mouth-restore generically, so the
        # rest of the pipeline is template-agnostic.
        swap_template = getattr(swap_p, 'model_template', 'arcface')
        aligned_img, M = align_crop(frame, target_face.kps, subsample_size, mode=swap_template)
        fake_frame = aligned_img
        target_face.matrix = M
        # Stash the crop affine per-thread so the SAM2 mask engine can warp its
        # precomputed full-frame mask into this exact crop space (see process_mask).
        self._tls.cur_M = M

        # ── Shared landmark / pose computation ────────────────────────────────
        # Computed once and reused by source-bank selection, 3D recon, and
        # frontalization.  Guards against missing landmark_3d_68 gracefully.
        import math as _math
        tgt_lm68_crop = None
        tgt_yaw_deg   = 0.0
        tgt_pitch_deg = 0.0
        try:
            if (hasattr(target_face, 'landmark_3d_68')
                    and target_face.landmark_3d_68 is not None):
                from roop.face_3d_recon import (
                    landmarks_to_crop_space, estimate_pose, decompose_yaw_pitch,
                )
                tgt_lm68_crop = landmarks_to_crop_space(target_face.landmark_3d_68, M)
                rvec, _ = estimate_pose(tgt_lm68_crop, subsample_size)
                ty, tp  = decompose_yaw_pitch(rvec)
                tgt_yaw_deg   = _math.degrees(ty)
                tgt_pitch_deg = _math.degrees(tp)
        except Exception:
            pass   # landmarks unavailable — features that need pose will no-op

        # ── Option 1: Multi-angle source bank ────────────────────────────────
        # Select the source face whose pose best matches this target frame.
        # Falls back to faces[0] when the feature is off or poses are absent.
        # selected_src_idx is carried into 3D recon below so the two features
        # compose (recon warps the bank-selected face, not always face[0]).
        selected_src_idx = 0
        if 0 <= face_index < len(self.input_face_datas):
            fs = self.input_face_datas[face_index]
            if len(fs.faces) > 0:
                inputface = fs.faces[0]   # default
            if (getattr(self.options, 'use_source_bank', False)
                    and len(fs.faces) > 1
                    and fs.face_poses is not None):
                best_idx  = 0
                best_dist = float('inf')
                for i, (yaw_d, pitch_d) in enumerate(fs.face_poses):
                    if yaw_d is None:
                        continue
                    dist = (tgt_yaw_deg - yaw_d) ** 2 + (tgt_pitch_deg - pitch_d) ** 2
                    if dist < best_dist:
                        best_dist = dist
                        best_idx  = i
                selected_src_idx = best_idx
                inputface = fs.faces[best_idx]

        if inputface is None:
            return frame

        # ── 3D source pose matching ───────────────────────────────────────────
        # Warp the crop of the source-bank-SELECTED face (selected_src_idx) toward
        # the target pose. GATED to IMAGE-source swappers only (BlendSwap/UniFace,
        # embedding_mode == "image"): those consume a source CROP, so a pose-warped
        # crop can genuinely help. Embedding-based swappers (inswapper/ghost/
        # hyperswap/simswap) take a near pose-invariant identity vector — feeding
        # them an embedding re-extracted from a sheared/flipped crop only DEGRADES
        # identity, so recon is a deliberate no-op for them. Falls back to the first
        # cached crop (face_3d) when the selected index has no bank entry.
        _swap_is_image = (getattr(swap_p, 'embedding_mode', 'normed_emap') == 'image')
        _recon_face_data = None
        if _swap_is_image and 0 <= face_index < len(self.input_face_datas):
            _recon_fs = self.input_face_datas[face_index]
            _bank = getattr(_recon_fs, 'face_3d_bank', None)
            if _bank is not None and 0 <= selected_src_idx < len(_bank):
                _recon_face_data = _bank[selected_src_idx]
            if _recon_face_data is None:
                _recon_face_data = _recon_fs.face_3d
        if (getattr(self.options, 'use_3d_recon', False)
                and _swap_is_image
                and inputface is not None
                and _recon_face_data is not None):
            try:
                from roop.face_3d_recon import Face3DRecon, landmarks_to_crop_space
                from roop.face_util import get_first_face as _gff

                face_data = _recon_face_data
                src_crop_512 = face_data.get('src_crop')
                src_lm68_img = face_data.get('src_lm68')

                if src_crop_512 is not None and tgt_lm68_crop is not None:
                    src_M_512 = face_data.get('src_M')
                    if src_lm68_img is not None and src_M_512 is not None:
                        lm_512 = landmarks_to_crop_space(src_lm68_img, src_M_512)
                        src_lm68_crop = lm_512 * (subsample_size / 512.0)
                    else:
                        src_lm68_crop = np.full((68, 2), subsample_size / 2.0,
                                                dtype=np.float32)

                    recon = Face3DRecon.instance()
                    src_crop_ss = cv2.resize(src_crop_512, (subsample_size, subsample_size))
                    posed_crop = recon.get_posed_source_crop(
                        src_crop_ss, src_lm68_crop, tgt_lm68_crop,
                        img_size=subsample_size,
                    )
                    if _DEBUG_POSE_LOG:
                        try:
                            from roop.face_3d_recon import estimate_pose, decompose_yaw_pitch
                            sv, _ = estimate_pose(src_lm68_crop, subsample_size)
                            sy, sp = decompose_yaw_pitch(sv)
                            dy = tgt_yaw_deg - _math.degrees(sy)
                            dp = tgt_pitch_deg - _math.degrees(sp)
                            if abs(dy) > 15 or abs(dp) > 15:
                                print(f"[3DRecon] pose correction: Δyaw={dy:+.1f}° Δpitch={dp:+.1f}°")
                        except Exception:
                            pass

                    with _gpu_guard(pooled=analysis_pooled()):  # re-detection on posed crop: lock-free when pooled
                        posed_face = _gff(posed_crop)
                    if (posed_face is not None
                            and getattr(posed_face, 'kps', None) is not None):
                        # Image-source swapper: feed it the POSE-MATCHED source
                        # crop. Re-derive the aligned source crops from posed_crop
                        # using the face re-detected ON that crop (so kps and image
                        # are in the same space), then drop the cached source blob
                        # so the swapper rebuilds from the new crop.
                        # NB: copy.copy() crashes on an insightface Face (its
                        # __getattr__ returns None for missing dunders → pickle/copy
                        # calls None); shallow-copy via the dict constructor.
                        from roop.face_util import _attach_source_crops
                        posed_input = type(inputface)(inputface)
                        posed_input['kps'] = posed_face.kps
                        _attach_source_crops(posed_input, posed_crop)
                        _blob_key = f"_srcblob_{getattr(swap_p, 'loaded_model_key', '')}"
                        try:
                            if _blob_key in posed_input:
                                del posed_input[_blob_key]
                        except Exception:
                            pass
                        inputface = posed_input
            except Exception as e:
                print(f"[ProcessMgr] Pose-aware embedding failed: {e}")

        # ── Option 2: Target frontalization ──────────────────────────────────
        # Warp the aligned crop toward frontal before the swap, then apply
        # the inverse affine after the swap to restore the original pose.
        M_frontal = None
        aligned_for_swap = aligned_img   # may be replaced by frontalized version

        if (getattr(self.options, 'use_frontalization', False)
                and tgt_lm68_crop is not None
                and inputface is not None):
            try:
                ft_threshold = getattr(self.options, 'frontalization_threshold', 25.0)
                if abs(tgt_yaw_deg) > ft_threshold or abs(tgt_pitch_deg) > ft_threshold:
                    from roop.face_frontalize import frontalize_crop
                    # frontal_lm68=None → auto-computed via solvePnP re-projection
                    frontalized, M_frontal = frontalize_crop(
                        aligned_img, tgt_lm68_crop,
                    )
                    if M_frontal is not None:
                        aligned_for_swap = frontalized
                        print(f"[Frontalize] Δyaw={tgt_yaw_deg:+.1f}° Δpitch={tgt_pitch_deg:+.1f}°"
                              f" — frontalization applied")
            except Exception as e:
                print(f"[ProcessMgr] Frontalization failed: {e}")

        fake_frame = aligned_for_swap

        for p in self.processors:
            if p.type == 'swap':
              with _prof('swap'):
                subsample_frames = self.implode_pixel_boost(aligned_for_swap, model_output_size, subsample_total)
                # Only skip the global GPU lock when THIS processor owns a real
                # SessionPool (per-thread TRT contexts). Without one, concurrent
                # threads would share a single non-thread-safe TensorRT context
                # and corrupt/hang the CUDA context.
                _pooled = getattr(p, 'pool', None) is not None
                if self._swap_batcher is not None and hasattr(p, 'RunBatchMulti'):
                    # Cross-frame batching: submit every tile to the batcher (which
                    # coalesces them with crops from other worker threads), then
                    # collect. The batcher thread holds the GPU guard.
                    tiles = list(subsample_frames)
                    for _ in range(0, self.options.num_swap_steps):
                        handles = [self._swap_batcher.submit(inputface, target_face,
                                   self.prepare_crop_frame(t, p)) for t in tiles]
                        tiles = [self.normalize_swap_frame(self._swap_batcher.wait(h), p)
                                 for h in handles]
                    swap_result_frames = tiles
                elif _BATCH_SWAP and len(subsample_frames) > 1 and hasattr(p, 'RunBatch'):
                    # Batch the pixel-boost tiles through one inference call.
                    tiles = list(subsample_frames)
                    for _ in range(0, self.options.num_swap_steps):
                        prepared = [self.prepare_crop_frame(t, p) for t in tiles]   # CPU
                        with _gpu_guard(pooled=_pooled):
                            outs = p.RunBatch(inputface, target_face, prepared)
                        tiles = [self.normalize_swap_frame(o, p) for o in outs]      # CPU
                    swap_result_frames = tiles
                else:
                    swap_result_frames = []
                    for sliced_frame in subsample_frames:
                        for _ in range(0, self.options.num_swap_steps):
                            sliced_frame = self.prepare_crop_frame(sliced_frame, p)   # CPU
                            with _gpu_guard(pooled=_pooled):
                                sliced_frame = p.Run(inputface, target_face, sliced_frame)
                            sliced_frame = self.normalize_swap_frame(sliced_frame, p)  # CPU
                        swap_result_frames.append(sliced_frame)
                fake_frame = self.explode_pixel_boost(swap_result_frames, model_output_size, subsample_total, subsample_size)
                fake_frame = fake_frame.astype(np.uint8)
                
                # Dynamic color tone correction: transfer target crop's skin tone
                # and lighting highlights/shadows to the swapped face.
                try:
                    fake_frame = self.apply_color_transfer(fake_frame, aligned_img)
                except Exception as e:
                    print(f"[ProcessMgr] Face color transfer failed: {e}")
                
                scale_factor = 0.0
                # ── Defrontalize after swap (Option 2) ────────────────────────
                if M_frontal is not None:
                    try:
                        from roop.face_frontalize import defrontalize_crop
                        fake_frame = defrontalize_crop(fake_frame, M_frontal)
                    except Exception as e:
                        print(f"[ProcessMgr] Defrontalization failed: {e}")
            elif p.type == 'mask':
                with _prof('mask'), _gpu_guard(pooled=getattr(p, 'pool', None) is not None):  # mask: lock-free when pooled
                    fake_frame = self.process_mask(p, aligned_img, fake_frame, orig_frame=frame, target_face=target_face, M=M, tgt_pitch_deg=tgt_pitch_deg)
                    if enhanced_frame is not None:
                        enhanced_frame = self.process_mask(p, aligned_img, enhanced_frame, orig_frame=frame, target_face=target_face, M=M, tgt_pitch_deg=tgt_pitch_deg)
            else:
                # Pooled (no global lock) ONLY when this enhancer built its own
                # SessionPool (e.g. RestoreFormer++). Enhancers without a pool
                # (GFPGAN/GPEN/CodeFormer/DMDNet) must take the global lock, or
                # concurrent threads corrupt/hang their single TensorRT context.
                with _prof('enhance'), _gpu_guard(pooled=getattr(p, 'pool', None) is not None):
                    enhanced_frame, scale_factor = p.Run(self.input_face_datas[face_index], target_face, fake_frame)

        # ── Anti-flicker: temporally smooth the enhanced aligned crop ─────────
        # enhanced_frame is registered to the canonical face template, so a
        # motion-adaptive blend across frames removes enhancer texture shimmer
        # without ghosting on movement. Skipped when autorotate rebound the face
        # (rotated crop space is inconsistent frame-to-frame).
        _es = self._cur_enh_stab()
        if (self._stab_active and _es is not None
                and enhanced_frame is not None and rotation_action is None):
            enhanced_frame = _es.apply(enhanced_frame, target_face.kps, self._cur_stab_t())

        # ── Expression restore ────────────────────────────────────────────────
        # Put the target's own expression back on the swapped face. Runs AFTER
        # the enhancer and its flicker stabiliser deliberately: the stabiliser
        # damps per-frame change to kill enhancer shimmer, and restoring first
        # would let it smooth the expression straight back out. aligned_img is
        # the untouched target crop in the same face-template space, so it is
        # the driving face — no extra detection or alignment needed.
        _ex = float(getattr(roop.globals, 'expression_restore_strength', 0.0) or 0.0)
        if _ex > 0.0:
            try:
                # Built before the guard so its pooled-ness can decide the guard:
                # with a pool of independent TensorRT contexts this stage does not
                # need the global lock, and holding it would keep the one stage
                # that runs single-file while swap/mask/detect all run N-wide.
                restorer = self._expression_restorer()
                with _prof('expression'), _gpu_guard(pooled=restorer.pooled):
                    _region = getattr(roop.globals, 'expression_restore_region', 'all')
                    if enhanced_frame is not None:
                        enhanced_frame = restorer.Run(enhanced_frame, aligned_img, _ex, _region)
                    else:
                        fake_frame = restorer.Run(fake_frame, aligned_img, _ex, _region)
            except Exception as e:
                print(f"[ProcessMgr] expression restore failed: {e}")

        # ── Skin detail transfer ──────────────────────────────────────────────
        # The generator produces smooth 256px skin and the enhancer hallucinates
        # flickery pores; inject the REAL high-frequency layer (pores, fine
        # texture, grain) from the actual target footage instead. aligned_img and
        # the swapped/enhanced crop share the same face-template space, so the
        # detail lands registered. High-pass is zero-mean → color/lighting are
        # untouched. strength<=0 is a bit-identical no-op.
        _dt = float(getattr(roop.globals, 'detail_transfer_strength', 0.0) or 0.0)
        if _dt > 0.0:
            try:
                if enhanced_frame is not None:
                    enhanced_frame = self.apply_detail_transfer(enhanced_frame, aligned_img, _dt)
                else:
                    fake_frame = self.apply_detail_transfer(fake_frame, aligned_img, _dt)
            except Exception as e:
                print(f"[ProcessMgr] Detail transfer failed: {e}")

        # ── Apply manual mask in canonical face-crop space ────────────────────
        # combined=1 → keep original pixels (aligned_img)   [exclude / red paint]
        # combined=0 → keep swapped pixels  (fake_frame)
        #
        # Two modes:
        #   canonical=True  — mask painted directly on face crop; resize to subsample_size.
        #   ref_kps         — legacy mask in full-frame coords; warp via M_ref.
        #                     Uses orig_fh/orig_fw so autorotate_faces doesn't corrupt dims.
        #
        # face_index selects the per-faceset mask; falls back to faceset 0 when only
        # one mask was painted (single-face / "first found" scenarios).
        _fm = self.face_masks.get(face_index)
        if _fm is None and self.face_masks:
            _fm = self.face_masks.get(0)
        if _fm is not None:
            _exc_mask  = _fm.get('exclude_mask')
            _inc_mask  = _fm.get('include_mask')
            _ref_kps   = _fm.get('ref_kps')
            _canonical = _fm.get('is_canonical', False)
            if _canonical:
                try:
                    def _resize_to_ss(mask):
                        if mask is None:
                            return None
                        mh, mw = mask.shape[:2]
                        if (mh, mw) == (subsample_size, subsample_size):
                            return mask
                        m8 = cv2.resize((mask * 255.0).clip(0, 255).astype(np.uint8),
                                        (subsample_size, subsample_size),
                                        interpolation=cv2.INTER_LINEAR)
                        return m8.astype(np.float32) / 255.0

                    exc_can = _resize_to_ss(_exc_mask)
                    inc_can = _resize_to_ss(_inc_mask)

                    combined_can = np.zeros((subsample_size, subsample_size), dtype=np.float32)
                    if exc_can is not None:
                        combined_can = np.maximum(combined_can, exc_can)
                    if inc_can is not None:
                        combined_can = combined_can * (1.0 - inc_can)

                    if np.any(combined_can > 0):
                        c3 = combined_can[:, :, np.newaxis]
                        fake_frame = (fake_frame.astype(np.float32) * (1.0 - c3) +
                                      aligned_img.astype(np.float32) * c3).astype(np.uint8)
                        if enhanced_frame is not None:
                            eh, ew = enhanced_frame.shape[:2]
                            if (eh, ew) != (subsample_size, subsample_size):
                                c3e = cv2.resize(combined_can, (ew, eh),
                                                 interpolation=cv2.INTER_LINEAR)[:, :, np.newaxis]
                                orig_enh = cv2.resize(aligned_img, (ew, eh),
                                                      interpolation=cv2.INTER_CUBIC)
                            else:
                                c3e, orig_enh = c3, aligned_img
                            enhanced_frame = (enhanced_frame.astype(np.float32) * (1.0 - c3e) +
                                              orig_enh.astype(np.float32) * c3e).astype(np.uint8)
                except Exception as e:
                    print(f"[ProcessMgr] Canonical mask application failed: {e}")

            elif _ref_kps is not None:
                try:
                    from roop.face_util import estimate_norm
                    # Use original (pre-rotation-rebind) frame dimensions so that
                    # ref_kps — which are always in full-frame coords — map correctly.
                    fh, fw = orig_fh, orig_fw
                    M_ref = estimate_norm(_ref_kps, subsample_size)

                    def _to_canonical(mask):
                        if mask is None:
                            return None
                        mh, mw = mask.shape[:2]
                        m8 = (
                            cv2.resize((mask * 255.0).clip(0, 255).astype(np.uint8),
                                       (fw, fh), interpolation=cv2.INTER_LINEAR)
                            if (mh, mw) != (fh, fw)
                            else (mask * 255.0).clip(0, 255).astype(np.uint8)
                        )
                        c = cv2.warpAffine(m8, M_ref, (subsample_size, subsample_size),
                                           flags=cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                        return c.astype(np.float32) / 255.0

                    exc_can = _to_canonical(_exc_mask)
                    inc_can = _to_canonical(_inc_mask)

                    combined_can = np.zeros((subsample_size, subsample_size), dtype=np.float32)
                    if exc_can is not None:
                        combined_can = np.maximum(combined_can, exc_can)
                    if inc_can is not None:
                        combined_can = combined_can * (1.0 - inc_can)

                    if np.any(combined_can > 0):
                        c3 = combined_can[:, :, np.newaxis]
                        fake_frame = (fake_frame.astype(np.float32) * (1.0 - c3) +
                                      aligned_img.astype(np.float32) * c3).astype(np.uint8)
                        if enhanced_frame is not None:
                            eh, ew = enhanced_frame.shape[:2]
                            if (eh, ew) != (subsample_size, subsample_size):
                                c3e = cv2.resize(combined_can, (ew, eh),
                                                 interpolation=cv2.INTER_LINEAR)[:, :, np.newaxis]
                                orig_enh = cv2.resize(aligned_img, (ew, eh),
                                                      interpolation=cv2.INTER_CUBIC)
                            else:
                                c3e, orig_enh = c3, aligned_img
                            enhanced_frame = (enhanced_frame.astype(np.float32) * (1.0 - c3e) +
                                              orig_enh.astype(np.float32) * c3e).astype(np.uint8)
                except Exception as e:
                    print(f"[ProcessMgr] Warp-based mask application failed: {e}")

        upscale = 512
        orig_width = fake_frame.shape[1]
        if orig_width != upscale:
            # interpolation must be a keyword — positionally it lands in `dst`
            # and is silently ignored (resize falls back to INTER_LINEAR).
            fake_frame = cv2.resize(fake_frame, (upscale, upscale), interpolation=cv2.INTER_CUBIC)
        mask_offsets = [0, 0, 0, 0, 20.0, 10.0] if inputface is None else inputface.mask_offsets

        face_lm = target_face.landmark_2d_106 if hasattr(target_face, 'landmark_2d_106') and target_face.landmark_2d_106 is not None else None
        # kps gives create_landmark_mask the head's up-axis, so the forehead
        # extension follows a tilted head instead of image-up.
        face_kps = getattr(target_face, 'kps', None)
        if enhanced_frame is None:
            scale_factor = int(upscale / orig_width)
            result = self.paste_upscale(fake_frame, fake_frame, target_face.matrix, frame, scale_factor, mask_offsets, face_landmarks=face_lm, face_kps=face_kps)
        else:
            result = self.paste_upscale(fake_frame, enhanced_frame, target_face.matrix, frame, scale_factor, mask_offsets, face_landmarks=face_lm, face_kps=face_kps)

        if self.options.restore_original_mouth:
            mouth_cutout, mouth_bb, mouth_polygon = self.create_mouth_mask(target_face, frame, mask_offsets)
            result = self.apply_mouth_area(result, mouth_cutout, mouth_bb, mouth_polygon, mask_offsets[5], yaw=tgt_yaw_deg, pitch=tgt_pitch_deg)

        # ── Face-shape reshape (post-composite) ───────────────────────────────
        # Warp the target's jaw/chin/cheek silhouette + lower face toward the
        # source person's shape. Applies to ANY swapper: the identity swappers
        # (inswapper/reswapper/simswap/…) all keep the TARGET's jaw & chin bone
        # structure, so this geometric liquify is the only thing that moves the
        # lower silhouette toward the source — which is exactly what the UI
        # toggle promises. It is a pure numpy/cv2 warp of the composited result
        # (no re-swap), strength-controlled and opt-in, so leaving it available
        # for every model just gives the user the lever. Skipped under autorotate
        # (result lives in rotated-crop space, so frame-space landmarks would be
        # inconsistent).
        if (getattr(roop.globals, 'jaw_reshape', False)
                and rotation_action is None
                and inputface is not None):
            result = reshape_jaw_frame(
                result,
                getattr(target_face, 'landmark_2d_106', None),
                getattr(inputface, 'landmark_2d_106', None),
                getattr(target_face, 'kps', None),
                getattr(inputface, 'kps', None),
                getattr(roop.globals, 'jaw_reshape_strength', 0.5),
            )

        if rotation_action is not None:
            fake_frame = self.auto_unrotate_frame(result, rotation_action)
            result = self.paste_simple(fake_frame, saved_frame, startX, startY)
        
        return result





























    def unload_models():
        pass


    def release_resources(self):
        for p in self.processors:
            p.Release()
        self.processors.clear()
        # FIX: Null out writer references after closing so GC can collect them
        if self.videowriter is not None:
            self.videowriter.close()
            self.videowriter = None
        if self.streamwriter is not None:
            self.streamwriter.Close()
            self.streamwriter = None
        # FIX: Clear face data and cached frame references so nothing holds VRAM-backed data
        self.input_face_datas = []
        self.target_face_datas = []
        self.target_face_groups = []
        self.last_swapped_frame = None