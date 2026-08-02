from settings import Settings, initial_yaw_align
from typing import List

source_path = None
target_path = None
output_path = None
target_folder_path = None
startup_args = None

cuda_device_id = 0
frame_processors: List[str] = []
keep_fps = None
keep_frames = None
autorotate_faces = None
vr_mode = None
skip_audio = None
wait_after_extraction = None
many_faces = None
use_batch = None
source_face_index = 0
target_face_index = 0
face_position = None
video_encoder = None
video_quality = None
max_memory = None
execution_providers: List[str] = ['CUDAExecutionProvider', 'CPUExecutionProvider']
execution_threads = None
headless = None
log_level = 'error'
selected_enhancer = None
codeformer_fidelity = 0.5
subsample_size = 256
# AI upscale folded into the swap pipeline (runs as the last frame processor,
# strictly after the face is swapped/enhanced, producing one output file).
upscale_after_swap = True
upscale_model_after = 'esrganx2'
face_swap_mode = 'DFL XSeg'
blend_ratio = 0.80
distance_threshold = 1
default_det_size = True
face_detector_size = '640'
face_detector_threshold = 0.50  # Lowered from 0.60: insightface's SCRFD was missing faces at slight angles
face_detector_nms = 0.40
sam2_model_size = 'tiny'   # SAM2 tracked-mask checkpoint: tiny|small|base_plus|large
track_identities = False   # video: lock each tracked person to one source (anti identity-flip)
# Skin-tone / lighting match of the swapped crop to the original crop.
# 'rct' = LAB mean/std (Reinhard, legacy default), 'lct' = LAB covariance
# whitening (handles color casts RCT can't), 'mkl' = Monge-Kantorovitch linear
# (fuller distribution match), 'none' = off.
color_transfer_mode = 'rct'
# Alignment refinement: derive the 5 arcface keypoints from the 68-point
# landmarks (more stable at angles than the detector's raw 5 kps).
refine_landmarks = False
# Profile alignment mode: 'off' | 'stabilize' | 'pose'.
# At near-profile yaw the two eyes project to almost the same point, so the
# 5-point similarity fit against a frontal template is ill-conditioned:
#   'stabilize' takes the rotation from the eye->mouth axis, which stops pitch
#               leaking into in-plane roll (a head nodding +/-25 deg at 90 deg
#               yaw otherwise swings the crop rotation ~30 deg);
#   'pose'      replaces the template with the reference head projected at the
#               estimated yaw, which makes the fit well posed (residual -60%).
# Seeded from ROOP_YAW_ALIGN (1/on/true == 'stabilize', or name a mode);
# the UI selector overrides it per run. See face_util._maybe_constrained.
yaw_align = initial_yaw_align()
# Jaw / chin reshape: warp the target's lower-face silhouette toward the SOURCE
# person's jaw/chin shape after the swap (identity swappers keep the target's
# geometry). strength 0..1 = amount of the shape difference applied.
jaw_reshape = False
jaw_reshape_strength = 0.5
# Skin detail transfer: inject the original footage's high-frequency texture
# (pores, grain) onto the swapped/enhanced face. 0 = off (no-op), 0..1 = amount.
detail_transfer_strength = 0.0
# ── DeepFaceLab merger post-ops (roop/procmgr_merger.py) ─────────────────────
# Cheap CPU passes over the merged crop, applied just before paste-back. Every
# one is a bit-identical no-op at the value below, and the chain short-circuits
# when they all are, so the defaults cost one attribute read each.
#   hist_match  0..1  match the plate's per-channel histogram (a 4th colour
#                     family: rct/lct/mkl match moments, this matches shape)
#   sharpen    -1..1  signed unsharp; negative softens (the denoise direction)
#   motion_blur 0..1  directional blur, axis measured off the plate's own smear
#   grain_match 0..1  add noise matched to the plate's measured noise floor —
#                     the swap comes back cleaner than the footage around it
#   degrade     0..1  bicubic down-and-up, to match a soft/compressed plate
merger_hist_match = 0.0
merger_sharpen = 0.0
merger_motion_blur = 0.0
merger_grain_match = 0.0
merger_degrade = 0.0
# Grow/shrink the pasted face about its own centre (DFL's output_face_scale).
# Identity swappers keep the TARGET's head size, so this is the only lever when
# the source person's face is visibly narrower or broader. 0 = off (no-op).
output_face_scale = 0.0
# Expression restorer (LivePortrait): put the TARGET's expression back onto the
# swapped face. Swappers regress toward their training set's mean expression, so
# laughing/crying/grimacing get flattened; this reads the expression deformation
# off the original frame and re-applies it. 0 = off (bit-exact no-op), 1 = adopt
# the target's expression fully, >1 exaggerates past it. Region limits the
# transfer to 'all' | 'lips' | 'eyes'.
expression_restore_strength = 0.0
expression_restore_region = 'all'
# Small-face rescue: when a frame yields no detections, retry on a 2x upscale
# so tiny/distant faces get picked up (without raising the global det size).
rescue_small_faces = False
# Face detector engine: 'scrfd' (insightface default), 'yoloface' (better on
# steep profiles / occluded faces) or 'retinaface' (highest recall on hard
# poses/lighting). The alternates reuse buffalo_l's aux models for identity +
# landmarks.
detector_engine = 'scrfd'
# Temporal detection (video anti-flicker): pre-pass detects+tracks every frame,
# gap-fills short detection misses (<= ROOP_TEMPORAL_GAP frames) and smooths
# kps/lm106/bbox per track when stabilization is on; the swap pass then reads
# the cached faces instead of re-detecting, so the swap can't blink out.
temporal_detection = False

no_face_action = 1

processing = False
# True for the whole lifetime of a batch_process run (start → end_processing),
# independent of the `processing` stop-signal. `processing` is cleared the moment
# a stop is requested (UI Stop or Ctrl-C), but the background thread still needs a
# moment to wind down and finalize the output video. `batch_active` stays True
# until that teardown completes, so a terminal Ctrl-C can wait for the video to be
# finalized (ffmpeg moov atom written) before exiting — same as the UI Stop path.
batch_active = False
# When True, the per-frame processing loops block (without aborting) so the
# user can pause and later resume from the exact same spot. Stop clears both
# `processing` and `pause`, so an abort always wins over a pause.
pause = False

g_current_face_analysis = None
g_desired_face_analysis = None

FACE_ENHANCER = 'GPEN'

INPUT_FACESETS = []
TARGET_FACES = []
# Parallel to TARGET_FACES: the person/group id each target face belongs to.
# Multiple angles of the same person share a group id; each group maps (by rank)
# to one source faceset. Enables multi-angle target tracking (anti-flicker).
TARGET_FACE_GROUP: List[int] = []
# Optional human-friendly names keyed by RAW group id (e.g. {0: "Bride"}).
# Ranks shift when groups merge/split, but the raw group id is stable, so we key
# names by raw id and resolve to per-rank names in the API payload.
TARGET_FACE_NAMES: dict = {}


IMAGE_CHAIN_PROCESSOR = None
VIDEO_CHAIN_PROCESSOR = None
BATCH_IMAGE_CHAIN_PROCESSOR = None

CFG: Settings = None

use_3d_recon = False


