"""FastAPI backend for the React UI (react-ui/).

This runs in a daemon thread alongside the Gradio app (see run.py) and shares the
same roop.globals state. It mirrors the Gradio Face Swap / Face Manager / Settings /
Extras handlers as plain HTTP endpoints so the React front-end can reach full parity.

The Gradio UI (app/ui/) is the frozen legacy/backup UI and is NOT touched here.
"""

import os
import io
import sys
import base64
import shutil
import subprocess
import threading
import traceback
import contextlib

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

import roop.globals as roop_globals
from roop import utilities as util
from roop.face_util import extract_face_images
from roop.FaceSet import FaceSet
from roop.capturer import get_video_frame, get_video_frame_total, get_image_frame
from roop.ProcessEntry import ProcessEntry
import ui.globals as ui_globals

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@contextlib.contextmanager
def temp_mapped_facesets(mapping):
    if not isinstance(mapping, list) or len(mapping) == 0:
        yield
        return
    orig_facesets = list(roop_globals.INPUT_FACESETS)
    mapped = []
    for x in mapping:
        try:
            src_idx = int(x)
        except (ValueError, TypeError):
            src_idx = -1
        if 0 <= src_idx < len(orig_facesets):
            mapped.append(orig_facesets[src_idx])
        else:
            mapped.append(FaceSet())
    roop_globals.INPUT_FACESETS = mapped
    try:
        yield
    finally:
        roop_globals.INPUT_FACESETS = orig_facesets

API_TEMP = os.path.join(os.getcwd(), "temp", "api_uploads")
os.makedirs(API_TEMP, exist_ok=True)

# ── Shared server-side state (mirrors the Gradio module globals) ──────────────
list_files_process: list = []          # list[ProcessEntry] – the target media queue
selected_input_face_index = 0          # which source faceset is "selected"
selected_target_index = 0              # which target file is shown in preview
current_video_fps = 30

# Face Manager state (separate faceset builder)
fm_thumbs: list = []                   # RGB numpy thumbnails
fm_images: list = []                   # BGR numpy full face images
fm_selected_index = -1
fm_files: list = []                    # uploaded source paths for the facemgr tab

# Live progress, polled by the React UI
_progress = {"processing": False, "paused": False, "progress": 0.0, "desc": "", "error": ""}
_last_output = {"path": "", "kind": ""}

no_face_choices = ["Use untouched original frame", "Retry rotated", "Skip Frame",
                   "Skip Frame if no similar face", "Use last swapped"]


# ── helpers ───────────────────────────────────────────────────────────────────
def _save_upload(file: UploadFile) -> str:
    # Recreate the upload dir every time — the Gradio "clean temp" action and
    # prepare_environment() can delete the whole temp/ tree out from under us.
    os.makedirs(API_TEMP, exist_ok=True)
    base = os.path.basename(file.filename)
    path = os.path.join(API_TEMP, base)
    # Never overwrite an earlier upload with the same name — existing target
    # entries keep pointing at the old path, so clobbering it corrupts them.
    stem, ext = os.path.splitext(base)
    n = 1
    while os.path.exists(path):
        path = os.path.join(API_TEMP, f"{stem}_{n}{ext}")
        n += 1
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return path


def _rgb_to_dataurl(rgb) -> str:
    """rgb: HxWx3 RGB numpy (as produced by util.convert_to_gradio) -> data URL."""
    if rgb is None:
        return ""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        return ""
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _bgr_to_dataurl(bgr) -> str:
    if bgr is None:
        return ""
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        return ""
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _bgr_to_jpg_dataurl(bgr) -> str:
    if bgr is None:
        return ""
    try:
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
        if not ok:
            return ""
        return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception:
        return ""


def _mask_offsets_from_cfg():
    c = roop_globals.CFG
    return [c.mask_top, c.mask_bottom, c.mask_left, c.mask_right,
            c.face_mask_blend, c.mouth_mask_blend,
            c.mouth_top_scale, c.mouth_bottom_scale,
            c.mouth_left_scale, c.mouth_right_scale]


def _update_mask_offsets_from_payload(payload: dict):
    global selected_input_face_index
    keys = ["mask_top", "mask_bottom", "mask_left", "mask_right",
            "face_mask_blend", "mouth_mask_blend",
            "mouth_top_scale", "mouth_bottom_scale", "mouth_left_scale", "mouth_right_scale"]
    
    # Update CFG so they are persisted. Only hit the disk when a value actually
    # changed — /api/preview calls this on every scrub/slider debounce.
    if roop_globals.CFG:
        cfg_changed = False
        for k in keys:
            if k in payload and payload[k] is not None:
                if getattr(roop_globals.CFG, k, None) != payload[k]:
                    setattr(roop_globals.CFG, k, payload[k])
                    cfg_changed = True
        if cfg_changed:
            roop_globals.CFG.save()

    # Update current faceset face offsets
    face_index = selected_input_face_index
    if len(roop_globals.INPUT_FACESETS) > face_index:
        faceset = roop_globals.INPUT_FACESETS[face_index]
        if faceset.faces:
            face = faceset.faces[0]
            if not hasattr(face, "mask_offsets") or face.mask_offsets is None:
                face.mask_offsets = _mask_offsets_from_cfg()
            offs = list(face.mask_offsets)
            while len(offs) < 10:
                offs.append(1.0)
            
            mapping = {
                "mask_top": 0,
                "mask_bottom": 1,
                "mask_left": 2,
                "mask_right": 3,
                "face_mask_blend": 4,
                "mouth_mask_blend": 5,
                "mouth_top_scale": 6,
                "mouth_bottom_scale": 7,
                "mouth_left_scale": 8,
                "mouth_right_scale": 9
            }
            updated = False
            for k, idx in mapping.items():
                if k in payload and payload[k] is not None:
                    try:
                        offs[idx] = float(payload[k])
                        updated = True
                    except (ValueError, TypeError):
                        pass
            if updated:
                face.mask_offsets = offs


def translate_swap_mode(text):
    return {"Selected face": "selected", "First found": "first",
            "All input faces": "all_input", "All female": "all_female",
            "All male": "all_male"}.get(text, "all")


def index_of_no_face_action(text):
    try:
        return no_face_choices.index(text)
    except ValueError:
        return 0


def map_mask_engine(selected_mask_engine, clip_text):
    if selected_mask_engine == "Clip2Seg":
        return "mask_clip2seg" if clip_text else None
    if selected_mask_engine == "DFL XSeg":
        return "mask_xseg"
    if selected_mask_engine == "Face Parser (BiSeNet)":
        return "mask_faceparser"
    if selected_mask_engine == "Face Occluder":
        return "mask_occluder"
    if selected_mask_engine == "Face Occluder v3 (XSeg-3)":
        return "mask_xseg3"
    if selected_mask_engine == "Segment Anything (MobileSAM)":
        return "mask_mobilesam"
    if selected_mask_engine == "Segment Anything (FastSAM)":
        return "mask_fastsam"
    if selected_mask_engine == "Segment Anything 2 (tracked)":
        return "mask_sam2"
    return None


class ApiProgress:
    """gradio.Progress-compatible shim that records progress into _progress."""
    def __call__(self, value=0, desc="", total=None, unit=None):
        try:
            if isinstance(value, (tuple, list)) and len(value) == 2 and value[1]:
                _progress["progress"] = float(value[0]) / float(value[1])
            else:
                _progress["progress"] = float(value)
        except Exception:
            pass
        if desc:
            _progress["desc"] = desc

        # Dynamic VRAM Safety Guard & Memory Flush
        try:
            import torch
            if torch.cuda.is_available():
                free_bytes, _ = torch.cuda.mem_get_info(0)
                free_gb = free_bytes / (1024 ** 3)
                if free_gb < 0.8:
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
        except Exception:
            pass

    def tqdm(self, iterable=None, *a, **k):
        return iterable if iterable is not None else []


# ── Settings ─────────────────────────────────────────────────────────────────
@app.get("/api/settings")
def get_settings():
    if roop_globals.CFG:
        return roop_globals.CFG.__dict__
    return {}


@app.post("/api/settings")
def save_settings(settings: dict = Body(...)):
    _update_mask_offsets_from_payload(settings)
    if roop_globals.CFG:
        for k, v in settings.items():
            if hasattr(roop_globals.CFG, k):
                setattr(roop_globals.CFG, k, v)
        roop_globals.CFG.save()
    return {"status": "success"}


def _get_git_version() -> str:
    import subprocess
    try:
        # Get active branch name
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL).decode("ascii").strip()
        # Get tag or short commit
        version = subprocess.check_output(["git", "describe", "--tags", "--always"], stderr=subprocess.DEVNULL).decode("ascii").strip()
        return f"{branch}@{version}"
    except Exception:
        try:
            version = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode("ascii").strip()
            return f"main@{version}"
        except Exception:
            return "main"


@app.get("/api/meta")
def get_meta():
    """Choice lists + current galleries for the React UI to render."""
    try:
        from roop.core import suggest_execution_providers
        providers = suggest_execution_providers()
    except Exception:
        providers = ["cpu"]
    return {
        "git_version": _get_git_version(),
        "providers": providers,
        "trt_precisions": ["fp32", "fp16", "mixed"],
        "enhancers": ["None", "Codeformer", "DMDNet", "GFPGAN", "GPEN",
                       "GPEN 1024", "GPEN 2048", "Restoreformer++"],
        "swap_models": ["inswapper", "reswapper", "hyperswap", "hyperswap_1b",
                         "hyperswap_1c", "ghost_1", "ghost_2", "ghost_3",
                         "simswap", "simswap_512", "hififace", "blendswap", "uniface"],
        "face_detection_modes": ["First found", "All input faces", "All female",
                                  "All male", "All faces", "Selected face"],
        "mask_engines": ["None", "Clip2Seg", "DFL XSeg", "Face Parser (BiSeNet)",
                          "Face Occluder", "Face Occluder v3 (XSeg-3)",
                          "Segment Anything (MobileSAM)", "Segment Anything (FastSAM)",
                          "Segment Anything 2 (tracked)"],
        "sam2_model_sizes": ["tiny", "small", "base_plus", "large"],
        "color_transfer_modes": ["none", "rct", "lct", "mkl"],
        "detector_engines": ["scrfd", "yoloface", "retinaface", "retinaface_r50", "yunet"],
        "encoder_presets": ["auto", "ultrafast", "superfast", "veryfast", "faster",
                             "fast", "medium", "slow", "slower", "veryslow"],
        "pool_sizes": ["auto", "1", "2", "3", "4", "5", "6", "7", "8"],
        "tristate": ["auto", "on", "off"],
        "no_face_actions": no_face_choices,
        "upscale": ["128px", "256px", "512px"],
        "video_methods": ["Extract Frames to media", "In-Memory processing"],
        "output_methods": ["File", "Virtual Camera", "Both"],
        "image_formats": ["jpg", "png", "webp"],
        "video_formats": ["avi", "mkv", "mp4", "webm"],
        "video_codecs": ["libx264", "libx265", "libvpx-vp9", "h264_nvenc", "hevc_nvenc"],
    }


def estimate_face_pose_from_kps(kps):
    try:
        left_eye_x, left_eye_y = kps[0]
        right_eye_x, right_eye_y = kps[1]
        nose_x, nose_y = kps[2]
        left_mouth_x, left_mouth_y = kps[3]
        right_mouth_x, right_mouth_y = kps[4]
        
        dx_left = abs(nose_x - left_eye_x)
        dx_right = abs(right_eye_x - nose_x)
        ratio = dx_left / (dx_right + 1e-6)
        
        yaw_label = "Front"
        if ratio < 0.65:
            yaw_label = "Left Profile"
        elif ratio > 1.55:
            yaw_label = "Right Profile"
            
        eye_y = (left_eye_y + right_eye_y) * 0.5
        mouth_y = (left_mouth_y + right_mouth_y) * 0.5
        dy_eyes = abs(nose_y - eye_y)
        dy_mouth = abs(mouth_y - nose_y)
        v_ratio = dy_eyes / (dy_mouth + 1e-6)
        
        pitch_label = ""
        if v_ratio < 0.65:
            pitch_label = "Up Tilt"
        elif v_ratio > 1.45:
            pitch_label = "Down Tilt"
            
        if pitch_label:
            if yaw_label == "Front":
                return pitch_label
            return f"{yaw_label} + {pitch_label}"
        return yaw_label
    except Exception:
        return "Front"

def _get_source_faces_info():
    source_faces_info = []
    for fs in roop_globals.INPUT_FACESETS:
        faces_poses = []
        for face in fs.faces:
            kps = getattr(face, 'kps', None)
            if kps is None and isinstance(face, dict) and 'kps' in face:
                kps = face['kps']
            poses_str = estimate_face_pose_from_kps(kps) if kps is not None else "Front"
            faces_poses.append(poses_str)
        source_faces_info.append({
            "count": len(fs.faces),
            "poses": faces_poses
        })
    return source_faces_info

def _source_faces_payload():
    return {
        "source_faces": [_rgb_to_dataurl(t) for t in ui_globals.ui_input_thumbs],
        "source_faces_info": _get_source_faces_info(),
        "faceset_count": len(roop_globals.INPUT_FACESETS),
    }


@app.get("/api/state")
def get_state():
    """Rehydrate the UI: current source/target galleries and target queue."""
    targets = [_target_entry_dict(entry) for entry in list_files_process]
    return {
        "source_faces": [_rgb_to_dataurl(t) for t in ui_globals.ui_input_thumbs],
        "source_faces_info": _get_source_faces_info(),
        "target_faces": [_rgb_to_dataurl(t) for t in ui_globals.ui_target_thumbs],
        "target_groups": _target_groups_ranked(),
        "targets": targets,
        "selected_target_index": selected_target_index,
        "faceset_count": len(roop_globals.INPUT_FACESETS),
    }


# ── Source faces ─────────────────────────────────────────────────────────────
@app.post("/api/source/add")
async def source_add(files: list[UploadFile] = File(...)):
    for f in files:
        path = _save_upload(f)
        try:
            if path.lower().endswith("fsz"):
                _ingest_faceset(path)
            elif util.has_image_extension(path):
                roop_globals.source_path = path
                faces_data = extract_face_images(path, (False, 0))
                for fd in faces_data:
                    fs = FaceSet()
                    face = fd[0]
                    face.mask_offsets = _mask_offsets_from_cfg()
                    fs.faces.append(face)
                    ui_globals.ui_input_thumbs.append(util.convert_to_gradio(fd[1]))
                    roop_globals.INPUT_FACESETS.append(fs)
        except Exception:
            traceback.print_exc()
    return _source_faces_payload()


def _ingest_faceset(path):
    unzipfolder = os.path.join(os.environ.get("TEMP", API_TEMP), "faceset")
    if os.path.isdir(unzipfolder):
        shutil.rmtree(unzipfolder, ignore_errors=True)
    os.makedirs(unzipfolder, exist_ok=True)
    util.unzip(path, unzipfolder)
    face_set = FaceSet()
    is_first = True
    for file in os.listdir(unzipfolder):
        if file.endswith(".png"):
            filename = os.path.join(unzipfolder, file)
            for fd in extract_face_images(filename, (False, 0)):
                face = fd[0]
                face.mask_offsets = _mask_offsets_from_cfg()
                face_set.faces.append(face)
                if is_first:
                    ui_globals.ui_input_thumbs.append(util.convert_to_gradio(fd[1]))
                    is_first = False
                face_set.ref_images.append(get_image_frame(filename))
    if len(face_set.faces) > 0:
        if len(face_set.faces) > 1:
            face_set.AverageEmbeddings()
        roop_globals.INPUT_FACESETS.append(face_set)


@app.post("/api/source/remove")
def source_remove(payload: dict = Body(...)):
    idx = int(payload.get("index", -1))
    if 0 <= idx < len(roop_globals.INPUT_FACESETS):
        roop_globals.INPUT_FACESETS.pop(idx)
    if 0 <= idx < len(ui_globals.ui_input_thumbs):
        ui_globals.ui_input_thumbs.pop(idx)
    return _source_faces_payload()


@app.post("/api/source/move")
def source_move(payload: dict = Body(...)):
    idx = int(payload.get("index", -1))
    direction = payload.get("direction", "right")
    offset = -1 if direction == "left" else 1
    new_idx = idx + offset
    if 0 <= idx < len(ui_globals.ui_input_thumbs) and 0 <= new_idx < len(ui_globals.ui_input_thumbs):
        for arr in (roop_globals.INPUT_FACESETS, ui_globals.ui_input_thumbs):
            item = arr.pop(idx)
            arr.insert(new_idx, item)
    return _source_faces_payload()


@app.post("/api/source/clear")
def source_clear():
    ui_globals.ui_input_thumbs.clear()
    roop_globals.INPUT_FACESETS.clear()
    return {"source_faces": [], "source_faces_info": [], "faceset_count": 0}


@app.post("/api/source/select")
def source_select(payload: dict = Body(...)):
    global selected_input_face_index
    selected_input_face_index = int(payload.get("index", 0))
    return {"selected": selected_input_face_index}


# ── Target media ─────────────────────────────────────────────────────────────
@app.post("/api/target/add")
async def target_add(files: list[UploadFile] = File(...)):
    global current_video_fps, selected_target_index
    first_new = len(list_files_process)
    for f in files:
        path = _save_upload(f)
        list_files_process.append(ProcessEntry(path, 0, 0, 0))
    # Refresh every new entry (not just index 0) so videos report their real
    # frame count/fps immediately — the UI's auto-queue and labels rely on it.
    for i in range(first_new, len(list_files_process)):
        _refresh_target_frames(i)
    selected_target_index = first_new if first_new < len(list_files_process) else 0
    return _target_list_payload()


def _refresh_target_frames(idx):
    global current_video_fps
    if idx >= len(list_files_process):
        return
    entry = list_files_process[idx]
    filename = entry.filename
    if util.is_video(filename) or filename.lower().endswith("gif") or util.is_animated_webp(filename):
        total = get_video_frame_total(filename) or 1
        if not filename.lower().endswith(".webp"):
            try:
                current_video_fps = util.detect_fps(filename)
            except Exception:
                current_video_fps = 30
        entry.fps = current_video_fps
    else:
        total = 1
        entry.fps = 0
    entry.total_frames = total
    # Keep the user's trim markers on re-select; only (re)initialise when unset
    # or out of range.
    if not entry.endframe or entry.endframe > total:
        entry.endframe = total
    if entry.startframe > total:
        entry.startframe = 0


def _target_entry_dict(entry):
    total = getattr(entry, "total_frames", 0) or entry.endframe or 1
    return {
        "name": os.path.basename(entry.filename),
        "startframe": entry.startframe,
        "endframe": entry.endframe,
        "start_frame": entry.startframe,
        "end_frame": entry.endframe,
        "frames": total,
        "fps": entry.fps or 0,
    }


def _target_list_payload():
    targets = [_target_entry_dict(entry) for entry in list_files_process]
    return {"targets": targets, "selected_target_index": selected_target_index,
            "fps": current_video_fps}


@app.post("/api/target/select")
def target_select(payload: dict = Body(...)):
    global selected_target_index
    idx = int(payload.get("index", 0))
    # Clamp: a negative index would silently wrap to the last entry via
    # Python list indexing in downstream helpers.
    selected_target_index = max(0, min(idx, max(0, len(list_files_process) - 1)))
    _refresh_target_frames(selected_target_index)
    return _target_list_payload()


@app.post("/api/target/remove")
def target_remove(payload: dict = Body(...)):
    """Remove a single target media item from the queue."""
    global selected_target_index
    idx = int(payload.get("index", -1))
    if 0 <= idx < len(list_files_process):
        list_files_process.pop(idx)
    if selected_target_index >= len(list_files_process):
        selected_target_index = max(0, len(list_files_process) - 1)
    if list_files_process:
        _refresh_target_frames(selected_target_index)
    return _target_list_payload()


@app.post("/api/target/clear")
def target_clear():
    global selected_target_index
    list_files_process.clear()
    roop_globals.TARGET_FACES.clear()
    roop_globals.TARGET_FACE_GROUP.clear()
    ui_globals.ui_target_thumbs.clear()
    selected_target_index = 0
    return _target_list_payload()


@app.post("/api/target/set_frame")
def target_set_frame(payload: dict = Body(...)):
    """Set start/end frame of the selected target (Set as Start / End)."""
    idx = selected_target_index
    which = payload.get("which", "start")
    frame = int(payload.get("frame", 1))
    if idx < len(list_files_process):
        entry = list_files_process[idx]
        total = getattr(entry, "total_frames", 0) or entry.endframe or 0
        if which == "start":
            entry.startframe = min(frame, entry.endframe or frame)
        else:
            if total:
                frame = min(frame, total)
            entry.endframe = max(frame, entry.startframe)
    return _target_list_payload()


@app.get("/api/target/preview")
def target_preview(index: int = 0, frame: int = 1, width: int = 0, fmt: str = "jpg"):
    """Return the raw target frame (no swap).

    Defaults to JPEG — PNG encode of an HD/4K frame is 5-10x slower and much
    larger, which made timeline scrubbing feel sluggish. Pass fmt=png for a
    lossless frame, width=N for a server-side downscale (storyboard / hover
    thumbnails don't need full-resolution frames)."""
    if index < 0 or index >= len(list_files_process):
        return JSONResponse(status_code=404, content={"message": "no target"})
    filename = list_files_process[index].filename
    if util.is_video(filename) or filename.lower().endswith("gif") or util.is_animated_webp(filename):
        frame_img = get_video_frame(filename, frame)
    else:
        frame_img = get_image_frame(filename)
    if frame_img is None:
        return JSONResponse(status_code=404, content={"message": "no frame"})
    if width and width > 0 and frame_img.shape[1] > width:
        h = max(1, round(frame_img.shape[0] * width / frame_img.shape[1]))
        frame_img = cv2.resize(frame_img, (width, h), interpolation=cv2.INTER_AREA)
    if fmt == "png":
        ok, buf = cv2.imencode(".png", frame_img)
        media = "image/png"
    else:
        ok, buf = cv2.imencode(".jpg", frame_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        media = "image/jpeg"
    if not ok:
        return JSONResponse(status_code=500, content={"message": "encode failed"})
    return StreamingResponse(io.BytesIO(buf.tobytes()), media_type=media)


def _target_groups_ranked():
    """Map raw group ids in TARGET_FACE_GROUP to contiguous 0-based person ranks
    (sorted by group id) so person N → source faceset N regardless of removals.
    Also keeps the group list length in sync with TARGET_FACES."""
    faces = roop_globals.TARGET_FACES
    grp = roop_globals.TARGET_FACE_GROUP
    if len(grp) != len(faces):
        grp = list(range(len(faces)))   # default: each face its own person
        roop_globals.TARGET_FACE_GROUP = grp
    uniq = sorted(set(grp))
    rank = {g: r for r, g in enumerate(uniq)}
    return [rank[g] for g in grp]


def _target_faces_payload(extra=None):
    out = {
        "target_faces": [_rgb_to_dataurl(t) for t in ui_globals.ui_target_thumbs],
        "target_groups": _target_groups_ranked(),
    }
    if extra:
        out.update(extra)
    return out


def _faces_from_frame(idx, frame):
    target_path = list_files_process[idx].filename
    roop_globals.target_path = target_path
    if util.is_image(target_path) and not target_path.lower().endswith("gif"):
        return extract_face_images(target_path, (False, 0))
    return extract_face_images(target_path, (True, frame))


@app.post("/api/target/use_face")
def target_use_face(payload: dict = Body(...)):
    """Add target faces from the current frame, each as a NEW person (group)."""
    idx = int(payload.get("index", selected_target_index))
    frame = int(payload.get("frame", 1))
    if idx >= len(list_files_process):
        return {"target_faces": [], "target_groups": []}
    faces_data = _faces_from_frame(idx, frame)
    next_id = (max(roop_globals.TARGET_FACE_GROUP) + 1) if roop_globals.TARGET_FACE_GROUP else 0
    for fd in faces_data:
        roop_globals.TARGET_FACES.append(fd[0])
        roop_globals.TARGET_FACE_GROUP.append(next_id)
        ui_globals.ui_target_thumbs.append(util.convert_to_gradio(fd[1]))
        next_id += 1
    return _target_faces_payload({"count": len(faces_data)})


@app.post("/api/target/add_angle")
def target_add_angle(payload: dict = Body(...)):
    """Add another angle (lateral/profile/upside-down) of an EXISTING target
    person, picking the face in the current frame closest to that person so
    matching survives pose changes (anti-flicker, multi-angle tracking)."""
    person = int(payload.get("person", 0))     # 0-based person rank
    idx = int(payload.get("index", selected_target_index))
    frame = int(payload.get("frame", 1))
    if idx >= len(list_files_process):
        return _target_faces_payload({"count": 0})
    ranks = _target_groups_ranked()
    raw_group = next((roop_globals.TARGET_FACE_GROUP[i] for i, r in enumerate(ranks) if r == person), None)
    if raw_group is None:
        return _target_faces_payload({"count": 0, "message": "no such person"})
    faces_data = _faces_from_frame(idx, frame)
    if not faces_data:
        return _target_faces_payload({"count": 0, "message": "no face in frame"})
    existing = [roop_globals.TARGET_FACES[i] for i, g in enumerate(roop_globals.TARGET_FACE_GROUP) if g == raw_group]
    best_fd, best_d = None, 1e9
    for fd in faces_data:
        d = min(util.compute_cosine_distance(e.embedding, fd[0].embedding) for e in existing)
        if d < best_d:
            best_d, best_fd = d, fd
    if best_fd is not None:
        roop_globals.TARGET_FACES.append(best_fd[0])
        roop_globals.TARGET_FACE_GROUP.append(raw_group)
        ui_globals.ui_target_thumbs.append(util.convert_to_gradio(best_fd[1]))
    return _target_faces_payload({"count": 1, "distance": round(float(best_d), 3)})


@app.post("/api/target/remove_face")
def target_remove_face(payload: dict = Body(...)):
    idx = int(payload.get("index", -1))
    if 0 <= idx < len(roop_globals.TARGET_FACES):
        roop_globals.TARGET_FACES.pop(idx)
    if 0 <= idx < len(roop_globals.TARGET_FACE_GROUP):
        roop_globals.TARGET_FACE_GROUP.pop(idx)
    if 0 <= idx < len(ui_globals.ui_target_thumbs):
        ui_globals.ui_target_thumbs.pop(idx)
    return _target_faces_payload()


@app.post("/api/target/group")
def target_group(payload: dict = Body(...)):
    groups = payload.get("groups")
    if isinstance(groups, list):
        parsed = []
        for x in groups[:len(roop_globals.TARGET_FACES)]:
            try:
                parsed.append(int(x))
            except (ValueError, TypeError):
                parsed.append(0)
        roop_globals.TARGET_FACE_GROUP = parsed
    return _target_faces_payload()


# ── Live preview swap ────────────────────────────────────────────────────────
@app.post("/api/preview")
def preview(payload: dict = Body(...)):
    """Render the selected target frame, optionally with a live face swap."""
    global selected_target_index
    _update_mask_offsets_from_payload(payload)
    idx = int(payload.get("index", selected_target_index))
    frame = int(payload.get("frame", 1))
    fake = bool(payload.get("fake_preview", False))

    if idx >= len(list_files_process):
        return JSONResponse(status_code=404, content={"message": "no target"})

    filename = list_files_process[idx].filename
    if util.is_video(filename) or filename.lower().endswith("gif") or util.is_animated_webp(filename):
        current_frame = get_video_frame(filename, frame)
    else:
        current_frame = get_image_frame(filename)
    if current_frame is None:
        return JSONResponse(status_code=404, content={"message": "no frame"})

    # Apply detection resolution before any detection so the face-box overlay and
    # the swap both use the chosen det_size (640 accurate / 320 fast).
    roop_globals.default_det_size = bool(payload.get("default_det_size", roop_globals.CFG.default_det_size))
    roop_globals.face_detector_size = str(payload.get("face_detector_size", roop_globals.CFG.face_detector_size))
    roop_globals.face_detector_threshold = float(payload.get("face_detector_threshold", roop_globals.CFG.face_detector_threshold))
    roop_globals.face_detector_nms = float(payload.get("face_detector_nms", roop_globals.CFG.face_detector_nms))
    roop_globals.refine_landmarks = bool(payload.get("refine_landmarks", getattr(roop_globals.CFG, "refine_landmarks", False)))
    roop_globals.rescue_small_faces = bool(payload.get("rescue_small_faces", getattr(roop_globals.CFG, "rescue_small_faces", False)))
    roop_globals.detector_engine = payload.get("detector_engine", getattr(roop_globals.CFG, "detector_engine", "scrfd"))

    faces_list = []
    try:
        from roop.face_util import get_all_faces
        faces = get_all_faces(current_frame)
        if faces:
            for f in faces:
                bbox = f["bbox"].astype(int).tolist()
                faces_list.append(bbox)
    except Exception:
        pass

    if not fake or len(roop_globals.INPUT_FACESETS) < 1:
        return {"image": _bgr_to_dataurl(current_frame), "faces": faces_list}

    try:
        from roop.core import live_swap, get_processing_plugins
        from roop.ProcessOptions import ProcessOptions

        roop_globals.face_swap_mode = translate_swap_mode(payload.get("detection", "All faces"))
        roop_globals.selected_enhancer = payload.get("enhancer", "None")
        roop_globals.distance_threshold = float(payload.get("face_distance", 0.85))
        roop_globals.blend_ratio = float(payload.get("blend_ratio", 0.8))
        roop_globals.no_face_action = index_of_no_face_action(payload.get("no_face_action", "Retry rotated"))
        roop_globals.vr_mode = bool(payload.get("vr_mode", False))
        roop_globals.autorotate_faces = bool(payload.get("autorotate", True))
        roop_globals.subsample_size = int(str(payload.get("upscale", "256px"))[:3])
        roop_globals.execution_threads = roop_globals.CFG.max_threads
        roop_globals.color_transfer_mode = payload.get("color_transfer_mode", getattr(roop_globals.CFG, "color_transfer_mode", "rct"))
        roop_globals.sam2_model_size = payload.get("sam2_model_size", getattr(roop_globals.CFG, "sam2_model_size", "tiny"))
        # refine_landmarks / rescue_small_faces / detector_engine already set
        # above (before the box-overlay detection) so both paths agree.

        swap_model = payload.get("swap_model", "inswapper")
        mask_engine = map_mask_engine(payload.get("mask_engine", "None"), payload.get("clip_text", ""))
        face_index = selected_input_face_index
        if len(roop_globals.INPUT_FACESETS) <= face_index:
            face_index = 0

        options = ProcessOptions(
            get_processing_plugins(mask_engine, swap_model=swap_model),
            roop_globals.distance_threshold, roop_globals.blend_ratio,
            roop_globals.face_swap_mode, face_index, payload.get("clip_text", ""),
            None, int(payload.get("num_swap_steps", 1)), roop_globals.subsample_size,
            bool(payload.get("show_mask_offsets", False)),
            bool(payload.get("restore_original_mouth", False)),
            use_3d_recon=bool(payload.get("use_3d_recon", False)),
            use_source_bank=bool(payload.get("use_source_bank", False)),
            use_frontalization=bool(payload.get("use_frontalization", False)),
            frontalization_threshold=float(payload.get("frontalization_threshold", 30.0)),
            swap_model=swap_model,
            stabilize_method=payload.get("stabilize_method", "one_euro"),
            stabilize_face=bool(payload.get("stabilize_face", False)))

        with temp_mapped_facesets(payload.get("face_mapping")):
            swapped = live_swap(current_frame, options)
        if swapped is None:
            return {"image": _bgr_to_dataurl(current_frame), "faces": faces_list}
        return {"image": _bgr_to_dataurl(swapped), "faces": faces_list}
    except Exception:
        traceback.print_exc()
        return {"image": _bgr_to_dataurl(current_frame), "faces": faces_list, "error": "swap failed"}


# ── Run the swap ─────────────────────────────────────────────────────────────
@app.post("/api/swap")
def trigger_swap(payload: dict = Body(...)):
    if _progress["processing"]:
        return JSONResponse(status_code=409, content={"message": "already processing"})
    if len(list_files_process) < 1:
        return JSONResponse(status_code=400, content={"message": "no target media"})
    if len(roop_globals.INPUT_FACESETS) < 1:
        return JSONResponse(status_code=400, content={"message": "no source faces"})

    # Claim the processing flag synchronously — the worker thread also sets it,
    # but only after it starts, so two rapid POSTs could otherwise both pass
    # the guard above and run concurrently.
    _progress.update({"processing": True, "paused": False, "progress": 0.0, "desc": "Starting…", "error": ""})
    threading.Thread(target=_run_swap, args=(payload,), daemon=True).start()
    return {"status": "started"}


def _run_swap(payload):
    global selected_input_face_index
    from ui.main import prepare_environment
    from roop.core import batch_process_regular

    roop_globals.pause = False
    _progress.update({"processing": True, "paused": False, "progress": 0.0, "desc": "Starting…", "error": ""})
    if hasattr(roop_globals, "latest_swapped_frame"):
        try:
            delattr(roop_globals, "latest_swapped_frame")
        except Exception:
            pass
    _live_cache.update({"seq": -1, "data": ""})
    try:
        # Inside the try so any failure (e.g. CFG.save() I/O error) still hits
        # the finally block and clears the processing flag.
        _update_mask_offsets_from_payload(payload)
        prepare_environment()
        if roop_globals.CFG.clear_output:
            shutil.rmtree(roop_globals.output_path, ignore_errors=True)
            os.makedirs(roop_globals.output_path, exist_ok=True)

        enhancer = payload.get("enhancer", roop_globals.CFG.selected_enhancer)
        detection = payload.get("detection", roop_globals.CFG.face_detection_mode)
        output_method = payload.get("output_method", roop_globals.CFG.output_method)
        processing_method = payload.get("video_method", roop_globals.CFG.video_swapping_method)
        upsample = payload.get("upscale", roop_globals.CFG.subsample_upscale)
        selected_mask_engine = payload.get("mask_engine", roop_globals.CFG.mask_engine)
        clip_text = payload.get("clip_text", roop_globals.CFG.mask_clip_text)

        roop_globals.selected_enhancer = enhancer
        roop_globals.target_path = None
        roop_globals.distance_threshold = float(payload.get("face_distance", roop_globals.CFG.max_face_distance))
        roop_globals.blend_ratio = float(payload.get("blend_ratio", roop_globals.CFG.blend_ratio))
        roop_globals.keep_frames = bool(payload.get("keep_frames", roop_globals.CFG.keep_frames))
        roop_globals.wait_after_extraction = bool(payload.get("wait_after_extraction", roop_globals.CFG.wait_after_extraction))
        roop_globals.skip_audio = bool(payload.get("skip_audio", roop_globals.CFG.skip_audio))
        roop_globals.face_swap_mode = translate_swap_mode(detection)
        roop_globals.default_det_size = bool(payload.get("default_det_size", roop_globals.CFG.default_det_size))
        roop_globals.face_detector_size = str(payload.get("face_detector_size", roop_globals.CFG.face_detector_size))
        roop_globals.face_detector_threshold = float(payload.get("face_detector_threshold", roop_globals.CFG.face_detector_threshold))
        roop_globals.face_detector_nms = float(payload.get("face_detector_nms", roop_globals.CFG.face_detector_nms))
        roop_globals.sam2_model_size = payload.get("sam2_model_size", getattr(roop_globals.CFG, "sam2_model_size", "tiny"))
        roop_globals.track_identities = bool(payload.get("track_identities", getattr(roop_globals.CFG, "track_identities", False)))
        roop_globals.no_face_action = index_of_no_face_action(payload.get("no_face_action", roop_globals.CFG.no_face_action))
        roop_globals.vr_mode = bool(payload.get("vr_mode", roop_globals.CFG.vr_mode))
        roop_globals.autorotate_faces = bool(payload.get("autorotate", roop_globals.CFG.autorotate_faces))
        roop_globals.subsample_size = int(str(upsample)[:3])
        roop_globals.execution_threads = roop_globals.CFG.max_threads
        roop_globals.color_transfer_mode = payload.get("color_transfer_mode", roop_globals.CFG.color_transfer_mode)
        roop_globals.refine_landmarks = bool(payload.get("refine_landmarks", roop_globals.CFG.refine_landmarks))
        roop_globals.rescue_small_faces = bool(payload.get("rescue_small_faces", roop_globals.CFG.rescue_small_faces))
        roop_globals.detector_engine = payload.get("detector_engine", roop_globals.CFG.detector_engine)
        roop_globals.temporal_detection = bool(payload.get("temporal_detection", getattr(roop_globals.CFG, "temporal_detection", False)))
        roop_globals.video_encoder = roop_globals.CFG.output_video_codec
        roop_globals.video_quality = roop_globals.CFG.video_quality
        roop_globals.max_memory = roop_globals.CFG.memory_limit if roop_globals.CFG.memory_limit > 0 else None

        mask_engine = map_mask_engine(selected_mask_engine, clip_text)

        if roop_globals.face_swap_mode == "selected" and len(roop_globals.TARGET_FACES) < 1:
            _progress.update({"processing": False, "error": "No target face selected"})
            return

        roop_globals.processing = True
        target_idx = payload.get("target_index")
        if target_idx is not None:
            try:
                target_idx = int(target_idx)
                if 0 <= target_idx < len(list_files_process):
                    files_to_process = [list_files_process[target_idx]]
                else:
                    files_to_process = list_files_process
            except Exception:
                files_to_process = list_files_process
        else:
            files_to_process = list_files_process

        # Flush VRAM and Python garbage collector before starting execution to ensure max headroom
        try:
            import torch
            if torch.cuda.is_available():
                import gc
                gc.collect()
                torch.cuda.empty_cache()
        except Exception:
            pass

        with temp_mapped_facesets(payload.get("face_mapping")):
            batch_process_regular(
                output_method, files_to_process, mask_engine, clip_text,
                processing_method == "In-Memory processing", None,
                bool(payload.get("restore_original_mouth", roop_globals.CFG.restore_original_mouth)),
                int(payload.get("num_swap_steps", roop_globals.CFG.num_swap_steps)),
                ApiProgress(), selected_input_face_index,
                use_3d_recon=bool(payload.get("use_3d_recon", roop_globals.CFG.use_3d_recon)),
                mask_per_frame_json="",
                use_source_bank=bool(payload.get("use_source_bank", roop_globals.CFG.use_source_bank)),
                use_frontalization=bool(payload.get("use_frontalization", roop_globals.CFG.use_frontalization)),
                frontalization_threshold=float(payload.get("frontalization_threshold", roop_globals.CFG.frontalization_threshold)),
                swap_model=payload.get("swap_model", roop_globals.CFG.swap_model),
                stabilize_face=bool(payload.get("stabilize_face", roop_globals.CFG.stabilize_face)),
                stabilize_method=payload.get("stabilize_method", roop_globals.CFG.stabilize_method),
                stabilize_min_cutoff=float(payload.get("stabilize_min_cutoff", roop_globals.CFG.stabilize_min_cutoff)),
                stabilize_beta=float(payload.get("stabilize_beta", roop_globals.CFG.stabilize_beta)),
                stabilize_enhancer=bool(payload.get("stabilize_enhancer", roop_globals.CFG.stabilize_enhancer)),
                stabilize_enhancer_strength=float(payload.get("stabilize_enhancer_strength", roop_globals.CFG.stabilize_enhancer_strength)))

        _progress["progress"] = 1.0
        _progress["desc"] = "Done"
        _record_last_output()
    except Exception as e:
        traceback.print_exc()
        _progress["error"] = str(e)
    finally:
        roop_globals.pause = False
        _progress["processing"] = False
        _progress["paused"] = False


def _record_last_output():
    out = roop_globals.output_path
    if not out or not os.path.isdir(out):
        return
    files = [os.path.join(out, f) for f in os.listdir(out)
             if not f.startswith(".") and os.path.isfile(os.path.join(out, f))]
    if not files:
        return
    latest = max(files, key=os.path.getmtime)
    kind = "video" if util.is_video(latest) else ("image" if util.is_image(latest) else "file")
    _last_output.update({"path": latest, "kind": kind})


@app.post("/api/stop")
def stop_swap():
    # Clear pause too so a stop while paused fully aborts (wait loop checks both).
    roop_globals.pause = False
    roop_globals.processing = False
    _progress["paused"] = False
    _progress["desc"] = "Aborting…"
    return {"status": "stopping"}


@app.post("/api/pause")
def pause_swap():
    if not _progress["processing"]:
        return JSONResponse(status_code=409, content={"message": "not processing"})
    roop_globals.pause = True
    _progress["paused"] = True
    _progress["desc"] = "Paused"
    return {"status": "paused"}


@app.post("/api/resume")
def resume_swap():
    roop_globals.pause = False
    _progress["paused"] = False
    _progress["desc"] = "Resuming…"
    return {"status": "resumed"}


# Cache of the last encoded live frame, keyed by its publish sequence number,
# so a 1 Hz poll doesn't re-encode a frame that hasn't changed.
_live_cache = {"seq": -1, "data": ""}


@app.get("/api/progress")
def get_progress():
    live_frame = ""
    live_seq = 0
    if _progress["processing"] and hasattr(roop_globals, "latest_swapped_frame"):
        frame = getattr(roop_globals, "latest_swapped_frame", None)
        live_seq = getattr(roop_globals, "latest_swapped_seq", 0)
        if frame is not None:
            if _live_cache["seq"] == live_seq and _live_cache["data"]:
                live_frame = _live_cache["data"]
            else:
                live_frame = _bgr_to_jpg_dataurl(frame)
                _live_cache.update({"seq": live_seq, "data": live_frame})
    return {**_progress, "output": _last_output, "live_frame": live_frame, "live_seq": live_seq}


@app.get("/api/output")
def list_output():
    out = getattr(roop_globals, "output_path", None)
    items = []
    if out and os.path.isdir(out):
        for f in sorted(os.listdir(out), reverse=True):
            full = os.path.join(out, f)
            if os.path.isfile(full) and not f.startswith("."):
                kind = "video" if util.is_video(full) else ("image" if util.is_image(full) else "file")
                items.append({"name": f, "kind": kind, "mtime": os.path.getmtime(full),
                              "size": os.path.getsize(full)})
    return {"output_path": out, "files": items[:50]}


@app.post("/api/output/delete")
def delete_output(payload: dict = Body(...)):
    filename = payload.get("name")
    out = getattr(roop_globals, "output_path", None)
    if not filename or not out or not os.path.isdir(out):
        return JSONResponse(status_code=400, content={"message": "invalid parameters"})
    filename = os.path.basename(filename)
    full_path = os.path.join(out, filename)
    if os.path.isfile(full_path):
        try:
            os.remove(full_path)
            global _last_output
            if _last_output.get("path") == full_path:
                _last_output.update({"path": "", "kind": ""})
            return {"status": "success"}
        except Exception as e:
            return JSONResponse(status_code=500, content={"message": f"failed to delete file: {e}"})
    return JSONResponse(status_code=404, content={"message": "file not found"})


@app.post("/api/reveal")
def reveal_output(payload: dict = Body(default={})):
    """Open the OS file manager at the output folder (optionally selecting a file)."""
    target = payload.get("path") or getattr(roop_globals, "output_path", None)
    if not target:
        return JSONResponse(status_code=404, content={"message": "no output folder"})
    target = os.path.abspath(target)
    is_file = os.path.isfile(target)
    folder = os.path.dirname(target) if is_file else target
    if not os.path.isdir(folder):
        return JSONResponse(status_code=404, content={"message": "folder not found"})
    try:
        if sys.platform.startswith("win"):
            if is_file:
                subprocess.Popen(["explorer", "/select,", target])
            else:
                os.startfile(folder)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-R", target] if is_file else ["open", folder])
        else:
            subprocess.Popen(["xdg-open", folder])
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})
    return {"status": "ok", "folder": folder}


def _open_shared(path: str):
    """Open *path* for reading in a way that does NOT lock it against move/delete.

    A normal open() on Windows omits FILE_SHARE_DELETE, so for the life of the
    handle the OS refuses to move/rename/delete the file ("the file is open in
    Python"). While a <video> element streams a finished output, that handle is
    alive — so the user can't move the result out of the output folder. We open
    via CreateFileW with all three share flags (READ|WRITE|DELETE) so Explorer can
    still move or delete the file even while we're streaming it. On non-Windows,
    POSIX already allows unlink/rename of open files, so a plain open() is fine.
    """
    if os.name != "nt":
        return open(path, "rb")
    import ctypes
    import msvcrt
    GENERIC_READ = 0x80000000
    FILE_SHARE_READ, FILE_SHARE_WRITE, FILE_SHARE_DELETE = 0x1, 0x2, 0x4
    OPEN_EXISTING = 3
    FILE_ATTRIBUTE_NORMAL = 0x80
    INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
    CreateFileW = ctypes.windll.kernel32.CreateFileW
    CreateFileW.restype = ctypes.c_void_p
    CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32,
                            ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
                            ctypes.c_void_p]
    handle = CreateFileW(path, GENERIC_READ,
                         FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                         None, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, None)
    if not handle or handle == INVALID_HANDLE_VALUE:
        # Fall back to a normal open rather than failing the request outright.
        return open(path, "rb")
    fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    return os.fdopen(fd, "rb")


@app.get("/api/file")
def get_file(path: str, request: Request):
    """Serve an output/temp file by absolute path (guarded to known dirs).

    Streams the file (with HTTP Range support, so video seeking is instant) using
    a share-delete handle (_open_shared). This matches FileResponse's smooth
    seeking but, unlike FileResponse, never locks the output against move/delete —
    so the finished video can be moved out of the output folder while it's still
    showing in the player.
    """
    roots = [API_TEMP, os.path.join(os.getcwd(), "temp")]
    out_dir = getattr(roop_globals, "output_path", "") or ""
    if out_dir:
        roots.append(out_dir)
    allowed = [os.path.normcase(os.path.abspath(r)) for r in roots]
    ap = os.path.abspath(path)
    ap_n = os.path.normcase(ap)

    def _within(child, parent):
        # commonpath (unlike startswith) can't be fooled by sibling dirs that
        # share a prefix, e.g. "output_evil" vs "output".
        try:
            return os.path.commonpath([child, parent]) == parent
        except ValueError:
            return False

    if not any(_within(ap_n, a) for a in allowed) or not os.path.isfile(ap):
        return JSONResponse(status_code=403, content={"message": "forbidden"})

    import mimetypes
    file_size = os.path.getsize(ap)
    media_type = mimetypes.guess_type(ap)[0] or "application/octet-stream"
    range_header = request.headers.get("range") or request.headers.get("Range")

    def _iter(start: int, length: int, chunk: int = 1024 * 1024):
        remaining = length
        f = _open_shared(ap)
        try:
            f.seek(start)
            while remaining > 0:
                data = f.read(min(chunk, remaining))
                if not data:
                    break
                remaining -= len(data)
                yield data
        finally:
            f.close()

    if range_header and range_header.strip().lower().startswith("bytes="):
        spec = range_header.split("=", 1)[1].split(",", 1)[0].strip()
        start_s, _, end_s = spec.partition("-")
        try:
            start = int(start_s) if start_s else 0
        except ValueError:
            start = 0
        try:
            end = int(end_s) if end_s else file_size - 1
        except ValueError:
            end = file_size - 1
        start = max(0, min(start, file_size - 1))
        end = max(start, min(end, file_size - 1))
        length = end - start + 1
        headers = {
            "Accept-Ranges": "bytes",
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Content-Length": str(length),
        }
        return StreamingResponse(_iter(start, length), status_code=206,
                                 media_type=media_type, headers=headers)

    headers = {"Accept-Ranges": "bytes", "Content-Length": str(file_size)}
    return StreamingResponse(_iter(0, file_size), media_type=media_type, headers=headers)


# ── Face Manager (faceset builder) ───────────────────────────────────────────
@app.post("/api/facemgr/add")
async def facemgr_add(files: list[UploadFile] = File(...)):
    global current_video_fps
    video_path = None
    for f in files:
        path = _save_upload(f)
        if util.has_image_extension(path):
            for fd in extract_face_images(path, (False, 0), 0.5):
                fm_images.append(fd[1])
                fm_thumbs.append(util.convert_to_gradio(fd[1]))
        elif util.is_video(path) or path.lower().endswith("gif"):
            fm_files.append(path)
            video_path = path
            try:
                current_video_fps = util.detect_fps(path)
            except Exception:
                current_video_fps = 30
    resp = {"faces": [_rgb_to_dataurl(t) for t in fm_thumbs]}
    if video_path:
        resp["video"] = os.path.basename(video_path)
        resp["frames"] = get_video_frame_total(video_path) or 1
    return resp


@app.post("/api/facemgr/faceset")
async def facemgr_faceset(file: UploadFile = File(...)):
    path = _save_upload(file)
    fm_thumbs.clear()
    fm_images.clear()
    if path.lower().endswith("fsz"):
        unzipfolder = os.path.join(os.environ.get("TEMP", API_TEMP), "faceset")
        if os.path.isdir(unzipfolder):
            shutil.rmtree(unzipfolder, ignore_errors=True)
        os.makedirs(unzipfolder, exist_ok=True)
        util.unzip(path, unzipfolder)
        for file_ in os.listdir(unzipfolder):
            if file_.endswith(".png"):
                for fd in extract_face_images(os.path.join(unzipfolder, file_), (False, 0), 0.5):
                    fm_images.append(fd[1])
                    fm_thumbs.append(util.convert_to_gradio(fd[1]))
    return {"faces": [_rgb_to_dataurl(t) for t in fm_thumbs]}


@app.get("/api/facemgr/frame")
def facemgr_frame(frame: int = 1):
    if not fm_files:
        return JSONResponse(status_code=404, content={"message": "no video"})
    img = get_video_frame(fm_files[-1], frame)
    if img is None:
        return JSONResponse(status_code=404, content={"message": "no frame"})
    ok, buf = cv2.imencode(".jpg", img)
    return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/jpeg")


@app.post("/api/facemgr/cut")
def facemgr_cut(payload: dict = Body(...)):
    frame = int(payload.get("frame", 1))
    if not fm_files:
        return {"faces": [_rgb_to_dataurl(t) for t in fm_thumbs]}
    for fd in extract_face_images(fm_files[-1], (True, frame), 0.5):
        fm_images.append(fd[1])
        fm_thumbs.append(util.convert_to_gradio(fd[1]))
    return {"faces": [_rgb_to_dataurl(t) for t in fm_thumbs]}


@app.post("/api/facemgr/remove")
def facemgr_remove(payload: dict = Body(...)):
    idx = int(payload.get("index", -1))
    if 0 <= idx < len(fm_thumbs):
        fm_thumbs.pop(idx)
        fm_images.pop(idx)
    return {"faces": [_rgb_to_dataurl(t) for t in fm_thumbs]}


@app.post("/api/facemgr/clear")
def facemgr_clear():
    fm_thumbs.clear()
    fm_images.clear()
    fm_files.clear()
    return {"faces": []}


@app.post("/api/facemgr/build")
def facemgr_build():
    if len(fm_images) < 1:
        return JSONResponse(status_code=400, content={"message": "no faces"})
    from ui.main import prepare_environment
    prepare_environment()
    imgnames = []
    for index, img in enumerate(fm_images):
        filename = os.path.join(roop_globals.output_path, f"{index}.png")
        cv2.imwrite(filename, img)
        imgnames.append(filename)
    finalzip = os.path.join(roop_globals.output_path, "faceset.fsz")
    util.zip(imgnames, finalzip)
    return {"path": finalzip, "name": "faceset.fsz"}


# ── Extras: media editor (resize / rotate / fps / crop) ──────────────────────
@app.post("/api/extras/apply")
async def extras_apply(file: UploadFile = File(...),
                       resolution: str = Form("Original"),
                       rotation: str = Form("None"),
                       fps: float = Form(30.0),
                       crop_left: int = Form(0), crop_right: int = Form(0),
                       crop_top: int = Form(0), crop_bottom: int = Form(0)):
    from ui.main import prepare_environment
    prepare_environment()
    path = _save_upload(file)
    is_video = util.is_video(path) or path.lower().endswith("gif")

    def _process_frame(img):
        h, w = img.shape[:2]
        x0 = int(w * crop_left / 100)
        x1 = w - int(w * crop_right / 100)
        y0 = int(h * crop_top / 100)
        y1 = h - int(h * crop_bottom / 100)
        if x1 > x0 and y1 > y0:
            img = img[y0:y1, x0:x1]
        if rotation == "90° Clockwise":
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == "90° Counter-Clockwise":
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif rotation == "180°":
            img = cv2.rotate(img, cv2.ROTATE_180)
        if resolution and resolution != "Original":
            try:
                target_w = int(str(resolution).split("x")[0])
                scale = target_w / img.shape[1]
                img = cv2.resize(img, (target_w, int(img.shape[0] * scale)))
            except Exception:
                pass
        return img

    out_dir = roop_globals.output_path
    if not is_video:
        img = get_image_frame(path)
        out = _process_frame(img)
        outpath = os.path.join(out_dir, "edited_" + os.path.splitext(os.path.basename(path))[0] + ".png")
        cv2.imwrite(outpath, out)
        return {"path": outpath, "kind": "image"}

    total = get_video_frame_total(path) or 1
    first_frame = get_video_frame(path, 1)
    if first_frame is None:
        return JSONResponse(status_code=400, content={"message": "could not read the video's first frame"})
    first = _process_frame(first_frame)
    oh, ow = first.shape[:2]
    outpath = os.path.join(out_dir, "edited_" + os.path.splitext(os.path.basename(path))[0] + ".mp4")
    writer = cv2.VideoWriter(outpath, cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, oh))
    for i in range(1, total + 1):
        fr = get_video_frame(path, i)
        if fr is None:
            continue
        writer.write(_process_frame(fr))
    writer.release()
    return {"path": outpath, "kind": "video"}


# ── Extras: frame post-processors (AI upscale / colorize / stylize filters) ──
# These reuse the roop frame processors that the swap pipeline never wired in.
_FRAME_FILTERS = ["stylize", "detailenhance", "pencil", "cartoon", "C64"]
_FRAME_UPSCALERS = ["esrganx2", "esrganx4", "esrgan_anime_x4", "ultrasharp_x4", "lsdirx4"]
_FRAME_COLORIZERS = ["deoldify_artistic", "deoldify_stable"]


def _make_frame_processor(operation: str, subtype: str):
    """Instantiate + initialize the requested frame processor. Returns a class
    whose Run(frame)->frame does the work."""
    from roop.utilities import get_device
    devicename = get_device()
    if operation == "upscale":
        from roop.processors.Frame_Upscale import Frame_Upscale
        proc = Frame_Upscale()
    elif operation == "colorize":
        from roop.processors.Frame_Colorizer import Frame_Colorizer
        proc = Frame_Colorizer()
    elif operation == "filter":
        from roop.processors.Frame_Filter import Frame_Filter
        proc = Frame_Filter()
    else:
        raise ValueError(f"unknown operation {operation}")
    proc.Initialize({"devicename": devicename, "subtype": subtype})
    return proc


@app.get("/api/extras/frame_ops")
def get_frame_ops():
    return {"upscale": _FRAME_UPSCALERS, "colorize": _FRAME_COLORIZERS, "filter": _FRAME_FILTERS}


@app.post("/api/extras/enhance")
async def extras_enhance(file: UploadFile = File(...),
                         operation: str = Form("upscale"),
                         subtype: str = Form("esrganx2")):
    """Run a frame post-processor (AI upscale / colorize / stylize) over an
    uploaded image or video and write the result to the output folder."""
    from ui.main import prepare_environment
    prepare_environment()   # ensures the Frame/* models are downloaded

    valid = {"upscale": _FRAME_UPSCALERS, "colorize": _FRAME_COLORIZERS, "filter": _FRAME_FILTERS}
    if operation not in valid or subtype not in valid[operation]:
        return JSONResponse(status_code=400, content={"message": "invalid operation/subtype"})

    path = _save_upload(file)
    is_video = util.is_video(path) or path.lower().endswith("gif")
    out_dir = roop_globals.output_path
    stem = os.path.splitext(os.path.basename(path))[0]

    try:
        proc = _make_frame_processor(operation, subtype)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": f"failed to load model: {e}"})

    try:
        if not is_video:
            img = get_image_frame(path)
            if img is None:
                return JSONResponse(status_code=400, content={"message": "could not read image"})
            out = proc.Run(img)
            outpath = os.path.join(out_dir, f"{operation}_{subtype}_{stem}.png")
            cv2.imwrite(outpath, out)
            return {"path": outpath, "kind": "image"}

        total = get_video_frame_total(path) or 1
        first = get_video_frame(path, 1)
        if first is None:
            return JSONResponse(status_code=400, content={"message": "could not read the video's first frame"})
        out_first = proc.Run(first)
        oh, ow = out_first.shape[:2]
        outpath = os.path.join(out_dir, f"{operation}_{subtype}_{stem}.mp4")
        try:
            fps = util.detect_fps(path)
        except Exception:
            fps = 30
        writer = cv2.VideoWriter(outpath, cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, oh))
        writer.write(out_first)
        for i in range(2, total + 1):
            fr = get_video_frame(path, i)
            if fr is None:
                continue
            res = proc.Run(fr)
            if res.shape[:2] != (oh, ow):
                res = cv2.resize(res, (ow, oh))
            writer.write(res)
        writer.release()
        return {"path": outpath, "kind": "video"}
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": f"processing failed: {e}"})
    finally:
        try:
            proc.Release()
        except Exception:
            pass


@app.get("/api/system/telemetry")
def get_telemetry():
    telemetry = {
        "gpu": "CPU Only",
        "vram_total": 0.0,
        "vram_used": 0.0,
        "cpu_percent": 0.0,
        "ram_total": 0.0,
        "ram_used": 0.0,
        "threads": 0
    }
    try:
        import psutil
        cpu_percent = psutil.cpu_percent()
        ram = psutil.virtual_memory()
        telemetry["cpu_percent"] = cpu_percent
        telemetry["ram_total"] = round(ram.total / (1024 ** 3), 2)
        telemetry["ram_used"] = round(ram.used / (1024 ** 3), 2)
    except Exception:
        pass

    try:
        import torch
        if torch.cuda.is_available():
            telemetry["gpu"] = torch.cuda.get_device_name(0)
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(0)
                telemetry["vram_total"] = round(total_bytes / (1024 ** 3), 2)
                telemetry["vram_used"] = round((total_bytes - free_bytes) / (1024 ** 3), 2)
            except Exception:
                telemetry["vram_total"] = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
                telemetry["vram_used"] = round(torch.cuda.memory_allocated(0) / (1024 ** 3), 2)
    except Exception:
        pass

    import threading
    telemetry["threads"] = threading.active_count()
    return telemetry


def run_api():
    try:
        port = int(os.environ.get("ROOP_API_PORT", 8001))
    except ValueError:
        port = 8001
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="error")

