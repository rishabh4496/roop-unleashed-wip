"""Expression restorer — put the TARGET's expression back onto the swapped face.

Why this exists
---------------
Identity swappers take a crop that carries the expression and an identity vector,
and the generator regresses toward the mean expression of its training set. The
further an expression is from that mean — a wide laugh, crying, a grimace — the
harder it is compressed. Face restorers then push further toward a clean,
neutral face prior. Nothing in that chain is trying to preserve *how open the
mouth is*, so swapped faces come out stiff.

LivePortrait's implicit-keypoint representation separates a face into a canonical
shape, a head pose, and an *expression deformation*. That last term is exactly
what the swap flattened, and it can be read off the original target frame and
re-applied.

Expression-only transfer
------------------------
The usual LivePortrait use is animating a still from a different driving video,
which needs pose transfer, relative-motion handling and a first-frame reference.
None of that applies here: the "source" (swapped crop) and the "driving"
(original target crop) are the SAME frame in the SAME aligned crop space, so the
head pose, scale and translation already match. Only the expression differs.

So instead of replacing the pose, keep it and substitute just the expression:

    x_s     = scale_s * (kp_s @ R_s + exp_s)   + t_s
    x_drive = scale_s * (kp_s @ R_s + exp_mix) + t_s
    exp_mix = exp_s + strength * (exp_d - exp_s)

which reduces to

    x_drive = x_s + scale_s * strength * (exp_d - exp_s)

The rotation cancels out of the *difference*, so the transfer cannot introduce
pose drift even if the head-pose estimate is noisy — the only thing moving is the
expression delta. strength = 0 returns x_s exactly, i.e. a no-op.

Models (~537 MB, downloaded on first use) are the FasterLivePortrait ONNX
exports, which suit this project's onnxruntime/TensorRT stack.
"""

import contextlib
import os
import threading

import cv2
import numpy as np
import onnxruntime

import roop.globals
from roop.gridsample5d import ensure_patched_model
from roop.typing import Frame
from roop.utilities import resolve_relative_path, conditional_download

_BASE = "https://huggingface.co/warmshao/FasterLivePortrait/resolve/main/liveportrait_onnx/"

MODELS = {
    "appearance": {"file": "appearance_feature_extractor.onnx", "url": _BASE + "appearance_feature_extractor.onnx"},
    "motion":     {"file": "motion_extractor.onnx",             "url": _BASE + "motion_extractor.onnx"},
    "warping":    {"file": "warping_spade.onnx",                "url": _BASE + "warping_spade.onnx"},
    "stitching":  {"file": "stitching.onnx",                    "url": _BASE + "stitching.onnx"},
}

# LivePortrait's implicit keypoints are not semantic, but these index groups are
# the ones its own eye/lip retargeting modules drive, so they are what "eyes
# only" / "lips only" restrict the expression delta to.
LIP_INDICES = (6, 12, 14, 17, 19, 20)
EYE_INDICES = (11, 13, 15, 16, 18)

INPUT_SIZE = 256          # LivePortrait's native crop size
NUM_BINS = 66             # head-pose regression bins


def _provider_names(providers):
    return [str(p[0] if isinstance(p, (tuple, list)) else p).lower() for p in providers]


def _has_gpu_provider(providers):
    """True if anything in the list can execute the rewritten graph on a GPU."""
    return any('tensorrt' in n or 'cuda' in n for n in _provider_names(providers))


def warping_providers(providers):
    """Providers that can execute the STOCK warping module.

    This is the fallback path. Normally roop.gridsample5d rewrites the two 5-D
    GridSample nodes into ops every backend can build, and the session just uses
    the standard provider list; this function is what remains for when that
    rewrite is disabled or fails.

    warping_spade warps a 5-D feature volume (1,32,16,64,64) with GridSample.
    onnxruntime's **CUDA** GridSample kernel is 4-D only, so with CUDA in the
    list the node is assigned to it and the run dies with "Only 4-D tensor is
    supported" — measured, not assumed. TensorRT executes it fine, and the CPU
    kernel handles 5-D but takes ~1.9 s per face, which is unusable for video and
    only tolerable for a single preview frame.

    So: keep TensorRT if present, otherwise fall back to CPU, and never leave
    CUDA to claim the node. The other three models are ordinary 4-D networks and
    keep the normal provider list.

    (FasterLivePortrait also ships warping_spade-fix.onnx, which replaces the
    node with a custom `GridSample3D` op. That needs a plugin library this
    project does not carry — the model fails to even load — so it is not used.)
    """
    kept = [p for p in providers
            if 'cuda' not in str(p[0] if isinstance(p, (tuple, list)) else p).lower()]
    if not any('tensorrt' in str(p[0] if isinstance(p, (tuple, list)) else p).lower()
               for p in kept):
        return ['CPUExecutionProvider']
    if not any('cpu' in str(p[0] if isinstance(p, (tuple, list)) else p).lower()
               for p in kept):
        kept.append('CPUExecutionProvider')
    return kept


@contextlib.contextmanager
def _quiet_trt_parser():
    """Silence the TensorRT parser errors this model provokes on purpose.

    TensorRT cannot build the two GridSample nodes either — its builder asserts
    `addGridSample: input.getDimensions().nbDims == 4` and the parser reports
    INVALID_NODE for /dense_motion_network/GridSample and /GridSample. That looks
    exactly like a crash in the terminal but is NOT a failure: onnxruntime
    partitions those two nodes to the CPU provider and runs everything else on
    TensorRT. Measured here: 0.25 s warm for the mixed partition versus 1.93 s on
    CPU alone, so the GPU still does nearly all the work.

    SessionOptions.log_severity_level does NOT suppress them — they come from the
    TensorRT logger bridged through onnxruntime's GLOBAL logger, not the
    session's — so the global severity has to be raised instead. The window is
    kept as narrow as possible: session construction only, which is where the
    parse happens, and it is restored afterwards even on failure. onnxruntime
    exposes no getter for the current severity, so it is restored to ORT's own
    default of 2 (warning).
    """
    onnxruntime.set_default_logger_severity(4)      # fatal only
    try:
        yield
    finally:
        onnxruntime.set_default_logger_severity(2)  # ORT's default: warning


# ── pure geometry (no models, unit-tested) ───────────────────────────────────

def headpose_pred_to_degree(pred):
    """The pose heads emit a 66-bin distribution, not an angle: expected bin
    index, rescaled to degrees. Matches LivePortrait's own conversion."""
    pred = np.asarray(pred, dtype=np.float32).reshape(-1, NUM_BINS)
    pred = pred - pred.max(axis=1, keepdims=True)          # stabilised softmax
    e = np.exp(pred)
    prob = e / e.sum(axis=1, keepdims=True)
    idx = np.arange(NUM_BINS, dtype=np.float32)
    return (prob * idx).sum(axis=1) * 3.0 - 97.5


def get_rotation_matrix(pitch_deg, yaw_deg, roll_deg):
    """Row-vector rotation matrix, so keypoints compose as `kp @ R`."""
    p, y, r = (np.radians(np.asarray(a, dtype=np.float32).reshape(-1))
               for a in (pitch_deg, yaw_deg, roll_deg))
    n = p.shape[0]
    zeros, ones = np.zeros(n, np.float32), np.ones(n, np.float32)
    rot_x = np.stack([ones, zeros, zeros,
                      zeros, np.cos(p), -np.sin(p),
                      zeros, np.sin(p), np.cos(p)], 1).reshape(n, 3, 3)
    rot_y = np.stack([np.cos(y), zeros, np.sin(y),
                      zeros, ones, zeros,
                      -np.sin(y), zeros, np.cos(y)], 1).reshape(n, 3, 3)
    rot_z = np.stack([np.cos(r), -np.sin(r), zeros,
                      np.sin(r), np.cos(r), zeros,
                      zeros, zeros, ones], 1).reshape(n, 3, 3)
    return np.transpose(rot_z @ rot_y @ rot_x, (0, 2, 1))


def transform_keypoint(kp, exp, scale, t, rot):
    """kp/exp: (B,N,3); scale: (B,1); t: (B,3); rot: (B,3,3) -> (B,N,3).
    The z translation is dropped, as in LivePortrait."""
    kp = np.asarray(kp, np.float32).reshape(kp.shape[0], -1, 3)
    exp = np.asarray(exp, np.float32).reshape(kp.shape[0], -1, 3)
    t = np.asarray(t, np.float32).reshape(kp.shape[0], 3).copy()
    t[:, 2] = 0.0
    out = (kp @ rot) + exp
    out = out * np.asarray(scale, np.float32).reshape(-1, 1, 1)
    return out + t[:, None, :]


def blend_expression(exp_source, exp_driving, strength, region="all"):
    """Move the source expression toward the driving one by `strength`.

    strength 0 returns the source expression unchanged (bit-exact no-op), 1.0
    adopts the target's expression fully, and values above 1 exaggerate past it
    — the equivalent of a micro-expression boost, for expressions the swap
    compressed rather than removed."""
    src = np.asarray(exp_source, np.float32)
    dst = np.asarray(exp_driving, np.float32)
    if strength == 0.0:
        return src.copy()
    delta = (dst - src) * float(strength)
    if region != "all":
        idx = LIP_INDICES if region == "lips" else EYE_INDICES
        gate = np.zeros_like(delta)
        flat = delta.reshape(delta.shape[0], -1, 3)
        gflat = gate.reshape(gate.shape[0], -1, 3)
        for i in idx:
            if i < flat.shape[1]:
                gflat[:, i, :] = flat[:, i, :]
        delta = gate
    return src + delta


def driving_keypoints(x_source, scale_source, exp_source, exp_driving,
                      strength, region="all"):
    """x_s + scale * (exp_mix - exp_s), the reduced form derived in the module
    docstring. Rotation and translation cancel, so this cannot move the head."""
    exp_mix = blend_expression(exp_source, exp_driving, strength, region)
    delta = (exp_mix - np.asarray(exp_source, np.float32))
    delta = delta.reshape(x_source.shape)
    return x_source + delta * np.asarray(scale_source, np.float32).reshape(-1, 1, 1)


def concat_feat(kp_source, kp_driving):
    """Stitching input: both keypoint sets flattened and concatenated."""
    b = kp_source.shape[0]
    return np.concatenate([kp_source.reshape(b, -1), kp_driving.reshape(b, -1)],
                          axis=1).astype(np.float32)


def _to_input(bgr):
    rgb = cv2.cvtColor(cv2.resize(bgr, (INPUT_SIZE, INPUT_SIZE),
                                  interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) / 255.0
    return np.transpose(x, (2, 0, 1))[None]


class Expression_LivePortrait:
    plugin_options: dict = None
    processorname = 'expression_liveportrait'
    type = 'expression'

    def __init__(self):
        self.sessions = {}
        self.devicename = 'cpu'
        self._lock = threading.Lock()

    # ── lifecycle ────────────────────────────────────────────────────────────

    def Initialize(self, plugin_options: dict):
        if self.plugin_options is not None and \
                self.plugin_options.get("devicename") != plugin_options.get("devicename"):
            self.Release()
        self.plugin_options = plugin_options
        self.devicename = str(plugin_options.get("devicename", "cpu")).replace('mps', 'cpu')
        if self.sessions:
            return
        model_dir = resolve_relative_path('../models/liveportrait')
        os.makedirs(model_dir, exist_ok=True)
        conditional_download(model_dir, [m["url"] for m in MODELS.values()])
        providers = roop.globals.execution_providers

        # The stock warping module cannot run entirely on the GPU: its two 5-D
        # GridSample nodes are rejected by TensorRT and by onnxruntime's CUDA
        # kernel alike, so they land on the CPU and split the rest into nine
        # separate TensorRT subgraphs. Rewriting them into TRT-native gathers
        # collapses that to one engine: measured 237.8ms -> 34.0ms per restored
        # face. ROOP_EXPR_PATCH_GRIDSAMPLE=0 keeps the stock model and its split.
        # Only worth it on a GPU: measured on CPU the expansion is ~9% SLOWER
        # than onnxruntime's own kernel (2.66s vs 2.44s), because eight gathers
        # beat one fused kernel only when there are cores to spread them over.
        warp_path = os.path.join(model_dir, MODELS["warping"]["file"])
        patched_path = warp_path
        if os.environ.get('ROOP_EXPR_PATCH_GRIDSAMPLE', '1') != '0' and \
                _has_gpu_provider(providers):
            patched_path = ensure_patched_model(warp_path)
        is_patched = patched_path != warp_path

        if is_patched:
            # Nothing 5-D survives the rewrite, so the normal provider list
            # applies — including CUDA, which the stock model has to exclude.
            warp_providers = providers
            print("[Expression] Warping module: fully on the GPU — the two 5-D "
                  "GridSample nodes were rewritten into TRT-native ops, so there "
                  "is no CPU partition.")
        else:
            warp_providers = warping_providers(providers)
            if warp_providers == ['CPUExecutionProvider']:
                print("[Expression] No TensorRT provider — the warping module runs on CPU "
                      "(~1.9s per face). Fine for a preview, far too slow for video.")
            else:
                print("[Expression] Warping module: TensorRT with the two GridSample nodes "
                      "on CPU (~0.25s per face). TensorRT parser warnings about "
                      "'addGridSample ... nbDims == 4' are expected and silenced.")

        for key, spec in MODELS.items():
            path = os.path.join(model_dir, spec["file"])
            if key == "warping":
                # The parser errors only exist on the unpatched path; keep real
                # errors visible once the rewrite has removed their cause.
                ctx = contextlib.nullcontext() if is_patched else _quiet_trt_parser()
                with ctx:
                    self.sessions[key] = onnxruntime.InferenceSession(
                        patched_path, None, providers=warp_providers)
            else:
                self.sessions[key] = onnxruntime.InferenceSession(
                    path, None, providers=providers)

    def Release(self):
        self.sessions = {}
        self.plugin_options = None

    # ── inference helpers ────────────────────────────────────────────────────

    def _run(self, key, feeds):
        """Bind inputs BY NAME, never by position.

        warping_spade declares its inputs as (feature_3d, kp_driving, kp_source)
        — driving before source, which is the reverse of how they read. Zipping a
        positional list against the session's input order silently swapped them,
        applying the expression backwards. Naming them makes that class of
        mistake impossible and self-documenting at the call site."""
        sess = self.sessions[key]
        expected = {i.name for i in sess.get_inputs()}
        missing = expected - set(feeds)
        if missing:
            raise KeyError(f"{key}: model expects inputs {sorted(expected)}, "
                           f"missing {sorted(missing)}")
        return sess.run(None, {k: v for k, v in feeds.items() if k in expected})

    def _motion(self, img):
        """-> (pitch, yaw, roll, t, exp, scale, kp).

        Verified against the FasterLivePortrait export: outputs are declared in
        exactly this order, with exp and kp flattened to (1, 63) = 21 points."""
        out = self._run("motion", {"img": img})
        pitch, yaw, roll, t, exp, scale, kp = out[:7]
        return (headpose_pred_to_degree(pitch), headpose_pred_to_degree(yaw),
                headpose_pred_to_degree(roll),
                np.asarray(t, np.float32).reshape(1, 3),
                np.asarray(exp, np.float32).reshape(1, -1, 3),
                np.asarray(scale, np.float32).reshape(1, 1),
                np.asarray(kp, np.float32).reshape(1, -1, 3))

    # ── main entry ───────────────────────────────────────────────────────────

    def Run(self, swapped_bgr: Frame, driving_bgr: Frame, strength: float = 1.0,
            region: str = "all", use_stitching: bool = True) -> Frame:
        """Re-impose `driving_bgr`'s expression onto `swapped_bgr`.

        Both must be the SAME aligned crop of the same frame — the swapped face
        and the original target face. Returns a crop of the input's size. Any
        failure returns the input unchanged: an expression tweak must never cost
        a frame."""
        if strength == 0.0 or swapped_bgr is None or driving_bgr is None:
            return swapped_bgr
        try:
            h, w = swapped_bgr.shape[:2]
            src_in = _to_input(swapped_bgr)
            drv_in = _to_input(driving_bgr)

            with self._lock:            # ORT sessions are not thread-safe
                feature = self._run("appearance", {"img": src_in})[0]
                p_s, y_s, r_s, t_s, exp_s, scale_s, kp_s = self._motion(src_in)
                _, _, _, _, exp_d, _, _ = self._motion(drv_in)

                rot_s = get_rotation_matrix(p_s, y_s, r_s)
                x_s = transform_keypoint(kp_s, exp_s, scale_s, t_s, rot_s)
                x_d = driving_keypoints(x_s, scale_s, exp_s, exp_d, strength, region)

                if use_stitching and "stitching" in self.sessions:
                    delta = np.asarray(
                        self._run("stitching", {"input": concat_feat(x_s, x_d)})[0],
                        np.float32)
                    n = x_d.shape[1]
                    if delta.size >= n * 3 + 2:
                        flat = delta.reshape(-1)
                        x_d = x_d + flat[:n * 3].reshape(1, n, 3)
                        x_d[:, :, :2] += flat[n * 3:n * 3 + 2].reshape(1, 1, 2)

                out = self._run("warping", {"feature_3d": feature,
                                            "kp_source": x_s,
                                            "kp_driving": x_d})[0]

            img = np.asarray(out, np.float32)
            if img.ndim == 4:
                img = img[0]
            if img.shape[0] in (1, 3):            # CHW -> HWC
                img = np.transpose(img, (1, 2, 0))
            if not np.isfinite(img).all():
                return swapped_bgr
            img = np.clip(img, 0.0, 1.0) * 255.0
            bgr = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
            if bgr.shape[:2] != (h, w):
                bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_CUBIC)
            return bgr
        except Exception as e:
            print(f"[Expression] LivePortrait restore failed: {e}")
            return swapped_bgr
