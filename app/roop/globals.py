from settings import Settings
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
subsample_size = 256
face_swap_mode = 'DFL XSeg'
blend_ratio = 0.80
distance_threshold = 1
default_det_size = True
face_detector_size = '640'
face_detector_threshold = 0.60
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
# Small-face rescue: when a frame yields no detections, retry on a 2x upscale
# so tiny/distant faces get picked up (without raising the global det size).
rescue_small_faces = False

no_face_action = 1

processing = False
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


IMAGE_CHAIN_PROCESSOR = None
VIDEO_CHAIN_PROCESSOR = None
BATCH_IMAGE_CHAIN_PROCESSOR = None

CFG: Settings = None

use_3d_recon = False


