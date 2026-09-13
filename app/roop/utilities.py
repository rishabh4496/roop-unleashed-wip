import glob
import json
import logging
import mimetypes
import os
import platform
import shutil
import ssl
import subprocess
import sys
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
import torch
import gradio
import tempfile
import cv2
import numpy as np
import threading
import zipfile
import traceback

from pathlib import Path
from typing import List, Any, Optional, Tuple, Dict
from tqdm import tqdm
from scipy.spatial import distance
from datetime import datetime

import roop.template_parser as template_parser

import roop.globals
from roop.offline import offline_enabled, mark_offline

TEMP_FILE = "temp.mp4"
TEMP_DIRECTORY = "temp"

# ---------------------------------------------------------------------------
# CUDA transport and compositing primitives
# ---------------------------------------------------------------------------

class CudaOrtIOBinding:
    """Reusable CUDA buffers for an ONNX Runtime session.

    ORT's normal ``session.run`` path accepts NumPy arrays and returns NumPy
    arrays. With CUDA/TensorRT that means an upload and download at *every*
    call, even when the next stage is also on CUDA. This helper owns
    contiguous torch allocations, exposes their pointers to ORT through
    ``bind_input``/``bind_output`` and reuses those allocations for every
    compatible call.

    It is deliberately a best-effort acceleration. CPU-only installs, a
    provider that rejects a user allocation, dynamic output shapes we cannot
    prove, and any ORT error all return ``None`` so callers retain their
    established ``session.run`` path. The class is per-session: TensorRT
    contexts must never share an I/O binding or an output allocation.
    """

    def __init__(self, session, device_id: int = 0):
        self.session = session
        self.device_id = int(device_id)
        self.enabled = self._has_cuda_provider(session)
        self._inputs: Dict[Tuple[str, Tuple[int, ...]], torch.Tensor] = {}
        self._outputs: Dict[Tuple[str, Tuple[int, ...]], torch.Tensor] = {}
        self._lock = threading.RLock()
        self._failure_reported = False

    @staticmethod
    def _has_cuda_provider(session) -> bool:
        try:
            providers = session.get_providers()
            return (torch.cuda.is_available() and any(
                name in ('CUDAExecutionProvider', 'TensorrtExecutionProvider')
                for name in providers))
        except Exception:
            return False

    @staticmethod
    def _shape_for(meta, batch: int) -> Optional[Tuple[int, ...]]:
        shape = list(meta.shape or [])
        if not shape:
            return None
        result = []
        for index, dimension in enumerate(shape):
            if isinstance(dimension, int) and dimension > 0:
                result.append(int(dimension))
            elif index == 0:
                result.append(int(batch))
            else:
                return None
        return tuple(result)

    @staticmethod
    def _as_float32(value) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        return np.ascontiguousarray(array)

    def _buffer(self, cache, name: str, shape: Tuple[int, ...]) -> torch.Tensor:
        key = (name, shape)
        tensor = cache.get(key)
        if tensor is None:
            tensor = torch.empty(shape, dtype=torch.float32,
                                 device=f'cuda:{self.device_id}').contiguous()
            cache[key] = tensor
        return tensor

    def run_gpu(self, feed: Dict[str, np.ndarray]) -> Optional[List[torch.Tensor]]:
        """Run one concrete-shape request and leave every output on CUDA."""
        if not self.enabled or not feed:
            return None
        try:
            with self._lock, torch.cuda.device(self.device_id):
                binding = self.session.io_binding()
                batch = None
                for name, value in feed.items():
                    host = self._as_float32(value)
                    if host.ndim < 1:
                        return None
                    batch = int(host.shape[0]) if batch is None else batch
                    if int(host.shape[0]) != batch:
                        return None
                    device_tensor = self._buffer(self._inputs, name, tuple(host.shape))
                    device_tensor.copy_(torch.from_numpy(host), non_blocking=False)
                    binding.bind_input(name, 'cuda', self.device_id, np.float32,
                                       tuple(host.shape), device_tensor.data_ptr())

                outputs: List[torch.Tensor] = []
                for meta in self.session.get_outputs():
                    shape = self._shape_for(meta, batch or 1)
                    if shape is None:
                        return None
                    output_tensor = self._buffer(self._outputs, meta.name, shape)
                    binding.bind_output(meta.name, 'cuda', self.device_id, np.float32,
                                        shape, output_tensor.data_ptr())
                    outputs.append(output_tensor)
                self.session.run_with_iobinding(binding)
                return outputs
        except Exception as exc:
            self.enabled = False
            if not self._failure_reported:
                self._failure_reported = True
                print(f'[CUDA I/O binding] disabled for this session: {exc}', flush=True)
            return None

    def run(self, feed: Dict[str, np.ndarray]) -> Optional[List[np.ndarray]]:
        """Compatibility form of :meth:`run_gpu` for NumPy-based callers."""
        outputs = self.run_gpu(feed)
        if outputs is None:
            return None
        return [tensor.detach().cpu().numpy().copy() for tensor in outputs]


def cuda_warp_affine(image: np.ndarray, matrix: np.ndarray,
                     output_size: Tuple[int, int], *,
                     border_mode: str = 'zeros',
                     interpolation: str = 'bilinear') -> Optional[np.ndarray]:
    """GPU equivalent of the common OpenCV affine warp, with safe fallback."""
    if not torch.cuda.is_available() or image is None:
        return None
    try:
        data = np.asarray(image)
        if data.ndim not in (2, 3):
            return None
        h, w = data.shape[:2]
        out_w, out_h = int(output_size[0]), int(output_size[1])
        if min(h, w, out_h, out_w) <= 0:
            return None
        channels = 1 if data.ndim == 2 else data.shape[2]
        source = torch.from_numpy(np.ascontiguousarray(data)).to(
            device='cuda', dtype=torch.float32).permute(
                2, 0, 1).unsqueeze(0) if channels > 1 else torch.from_numpy(
                    np.ascontiguousarray(data)).to(device='cuda', dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        affine = np.asarray(matrix, dtype=np.float32).reshape(2, 3)
        inverse = cv2.invertAffineTransform(affine)
        ys, xs = torch.meshgrid(
            torch.arange(out_h, device='cuda', dtype=torch.float32),
            torch.arange(out_w, device='cuda', dtype=torch.float32), indexing='ij')
        src_x = inverse[0, 0] * xs + inverse[0, 1] * ys + inverse[0, 2]
        src_y = inverse[1, 0] * xs + inverse[1, 1] * ys + inverse[1, 2]
        grid = torch.stack((2.0 * (src_x + 0.5) / w - 1.0,
                            2.0 * (src_y + 0.5) / h - 1.0), dim=-1).unsqueeze(0)
        padding = 'border' if border_mode == 'replicate' else 'zeros'
        result = torch.nn.functional.grid_sample(
            source, grid, mode=interpolation, padding_mode=padding,
            align_corners=False)
        result = result.squeeze(0).permute(1, 2, 0) if channels > 1 else result[0, 0]
        if np.issubdtype(data.dtype, np.integer):
            result = result.clamp(0, 255).to(torch.uint8)
        else:
            result = result.to(torch.float32)
        return result.cpu().numpy()
    except Exception:
        return None


def cuda_laplacian_pyramid_blend(target: np.ndarray, swap: np.ndarray,
                                 mask: np.ndarray, levels: int = 3) -> Optional[np.ndarray]:
    """Blend BGR buffers on CUDA using a bounded Laplacian pyramid."""
    if not torch.cuda.is_available():
        return None
    try:
        target_a = np.ascontiguousarray(np.asarray(target, dtype=np.float32))
        swap_a = np.ascontiguousarray(np.asarray(swap, dtype=np.float32))
        mask_a = np.ascontiguousarray(np.asarray(mask, dtype=np.float32))
        if target_a.ndim != 3 or target_a.shape[2] < 3:
            return None
        h, w = target_a.shape[:2]
        if swap_a.shape[:2] != (h, w):
            return None
        if mask_a.ndim == 3:
            mask_a = mask_a[..., 0]
        if mask_a.shape != (h, w):
            return None
        to_t = lambda a: torch.from_numpy(a).to('cuda', non_blocking=False)
        a = to_t(target_a[..., :3]).permute(2, 0, 1).unsqueeze(0)
        b = to_t(swap_a[..., :3]).permute(2, 0, 1).unsqueeze(0)
        m = to_t(np.clip(mask_a, 0.0, 1.0)).unsqueeze(0).unsqueeze(0)
        ga, gb, gm = [a], [b], [m]
        for _ in range(1, max(1, min(int(levels), 3))):
            if min(ga[-1].shape[-2:]) < 2:
                break
            ga.append(torch.nn.functional.avg_pool2d(ga[-1], 2, 2))
            gb.append(torch.nn.functional.avg_pool2d(gb[-1], 2, 2))
            gm.append(torch.nn.functional.avg_pool2d(gm[-1], 2, 2))
        la, lb = [], []
        for index in range(len(ga) - 1):
            size = ga[index].shape[-2:]
            la.append(ga[index] - torch.nn.functional.interpolate(ga[index + 1], size=size,
                                                                    mode='bilinear', align_corners=False))
            lb.append(gb[index] - torch.nn.functional.interpolate(gb[index + 1], size=size,
                                                                    mode='bilinear', align_corners=False))
        blended = ga[-1] * (1.0 - gm[-1]) + gb[-1] * gm[-1]
        for index in range(len(la) - 1, -1, -1):
            blended = torch.nn.functional.interpolate(blended, size=la[index].shape[-2:],
                                                       mode='bilinear', align_corners=False)
            blended = blended + la[index] * (1.0 - gm[index]) + lb[index] * gm[index]
        return blended.squeeze(0).permute(1, 2, 0).clamp(0, 255).to(torch.uint8).cpu().numpy()
    except Exception:
        return None

_LOGGER = logging.getLogger(__name__)

# monkey patch ssl for mac
if platform.system().lower() == "darwin":
    ssl._create_default_https_context = ssl._create_unverified_context


# https://github.com/facefusion/facefusion/blob/master/facefusion
def _plausible_fps(value) -> bool:
    """True only for a frame rate we would be willing to hand to ffmpeg.

    Rejects None, 0, negatives, NaN (NaN fails every comparison, so the
    `0 < v` test excludes it without a special case) and absurd values — a
    corrupt header can report 1e6 fps, which is as unusable as 0.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    return 0.0 < v <= 1000.0


def detect_fps(target_path: str) -> float:
    # Animated WebP: OpenCV returns 0 FPS — derive from PIL frame durations instead
    if target_path and target_path.lower().endswith('.webp'):
        try:
            from PIL import Image
            with Image.open(target_path) as img:
                n = getattr(img, 'n_frames', 1)
                if n > 1:
                    durations = []
                    for i in range(n):
                        img.seek(i)
                        d = img.info.get('duration', None)
                        durations.append(d)
                    print(f"[detect_fps] WebP '{os.path.basename(target_path)}': "
                          f"{n} frames, raw durations (ms) = {durations}")
                    # Treat None or 0 as 100 ms (browsers use ~100 ms as the
                    # effective minimum for animated WebP, similar to GIF).
                    cleaned = [(d if d and d > 0 else 100) for d in durations]
                    avg_ms = sum(cleaned) / len(cleaned)
                    fps = round(1000.0 / avg_ms, 2)
                    print(f"[detect_fps] avg_ms={avg_ms:.1f} → fps={fps}")
                    return fps
        except Exception as exc:
            print(f"[detect_fps] WebP duration read failed: {exc}")
        return 10.0  # safe fallback: 100 ms per frame
    # cv2 first (cheap, no subprocess), but never trust its answer unchecked.
    #
    # cap.get(CAP_PROP_FPS) returns 0.0 whenever OpenCV cannot read a frame rate
    # from the container — routine for VFR MKV/WebM, fragmented MP4, and files
    # with a damaged header — and NaN on some backends. The old code assigned
    # that straight over the 24.0 default, because isOpened() was still True, so
    # detect_fps returned 0.0/NaN for exactly the malformed inputs the default
    # existed for.
    #
    # That value is load-bearing downstream: FFMPEG_VideoWriter interpolates it
    # into '-r', so fps=0 makes ffmpeg reject the command and exit before a
    # single frame is written, and restore_audio/create_gif mis-time the result.
    #
    # capturer._probe_video already solves this properly — ffprobe's
    # avg_frame_rate, parsed as the rational it is ('30000/1001'), cached per
    # path — so fall through to it instead of inventing a second probe. Imported
    # inside the function to keep the module-level import graph acyclic, the same
    # way capturer imports utilities.
    fps = 0.0
    cap = cv2.VideoCapture(target_path)
    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if _plausible_fps(fps):
        return float(fps)

    try:
        from roop.capturer import _probe_video
        info = _probe_video(target_path)
        if info and _plausible_fps(info.get('fps')):
            probed = float(info['fps'])
            print(f"[detect_fps] cv2 reported {fps!r} for "
                  f"'{os.path.basename(target_path)}'; using ffprobe's {probed:.3f}")
            return probed
    except Exception as exc:
        print(f"[detect_fps] ffprobe fallback failed: {exc}")

    print(f"[detect_fps] no usable frame rate for "
          f"'{os.path.basename(target_path)}' (cv2 gave {fps!r}) — defaulting to 24.0")
    return 24.0


def detect_dimensions(target_path: str):
    """Returns (width, height) for images and videos. Returns (0, 0) on failure."""
    if is_image(target_path):
        img = cv2.imread(target_path)
        if img is not None:
            return img.shape[1], img.shape[0]
        return 0, 0
    # Animated WebP: OpenCV VideoCapture returns 0x0 — use PIL instead
    if target_path and target_path.lower().endswith('.webp') and is_animated_webp(target_path):
        try:
            from PIL import Image
            with Image.open(target_path) as img:
                return img.width, img.height
        except Exception:
            return 0, 0
    cap = cv2.VideoCapture(target_path)
    if cap.isOpened():
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return w, h
    cap.release()
    return 0, 0


# Gradio wants Images in RGB
def convert_to_gradio(image):
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def sort_filenames_ignore_path(filenames):
    """Sorts a list of filenames containing a complete path by their filename,
    while retaining their original path.

    Args:
      filenames: A list of filenames containing a complete path.

    Returns:
      A sorted list of filenames containing a complete path.
    """
    filename_path_tuples = [
        (os.path.split(filename)[1], filename) for filename in filenames
    ]
    sorted_filename_path_tuples = sorted(filename_path_tuples, key=lambda x: x[0])
    return [
        filename_path_tuple[1] for filename_path_tuple in sorted_filename_path_tuples
    ]


def sort_rename_frames(path: str):
    filenames = os.listdir(path)
    filenames.sort()
    for i in range(len(filenames)):
        of = os.path.join(path, filenames[i])
        newidx = i + 1
        new_filename = os.path.join(
            path, f"{newidx:06d}." + roop.globals.CFG.output_image_format
        )
        os.rename(of, new_filename)


def get_temp_frame_paths(target_path: str) -> List[str]:
    temp_directory_path = get_temp_directory_path(target_path)
    return glob.glob(
        (
            os.path.join(
                glob.escape(temp_directory_path),
                f"*.{roop.globals.CFG.output_image_format}",
            )
        )
    )


def get_temp_frame_paths_from_dir(directory: str) -> List[str]:
    """Return sorted frame image paths from an arbitrary directory.

    Used to get originals from _frames_orig/ for per-frame mask re-processing.
    Tries the configured output_image_format first, then falls back to common formats.
    """
    if not directory or not os.path.isdir(directory):
        return []
    fmt = roop.globals.CFG.output_image_format
    paths = sorted(glob.glob(os.path.join(glob.escape(directory), f'*.{fmt}')))
    if not paths:
        for fallback in ('png', 'jpg', 'jpeg'):
            paths = sorted(glob.glob(os.path.join(glob.escape(directory), f'*.{fallback}')))
            if paths:
                break
    return paths


def get_temp_directory_path(target_path: str) -> str:
    target_name, _ = os.path.splitext(os.path.basename(target_path))
    target_directory_path = os.path.dirname(target_path)
    return os.path.join(target_directory_path, TEMP_DIRECTORY, target_name)


def get_temp_output_path(target_path: str) -> str:
    temp_directory_path = get_temp_directory_path(target_path)
    return os.path.join(temp_directory_path, TEMP_FILE)


def normalize_output_path(source_path: str, target_path: str, output_path: str) -> Any:
    if source_path and target_path:
        source_name, _ = os.path.splitext(os.path.basename(source_path))
        target_name, target_extension = os.path.splitext(os.path.basename(target_path))
        if os.path.isdir(output_path):
            return os.path.join(
                output_path, source_name + "-" + target_name + target_extension
            )
    return output_path


def get_destfilename_from_path(
    srcfilepath: str, destfilepath: str, extension: str
) -> str:
    fn, ext = os.path.splitext(os.path.basename(srcfilepath))
    if "." in extension:
        return os.path.join(destfilepath, f"{fn}{extension}")
    return os.path.join(destfilepath, f"{fn}{extension}{ext}")


def replace_template(file_path: str, index: int = 0) -> str:
    fn, ext = os.path.splitext(os.path.basename(file_path))

    # Remove the "__temp" placeholder that was used as a temporary filename
    fn = fn.replace("__temp", "")

    template = roop.globals.CFG.output_template
    replaced_filename = template_parser.parse(
        template, {"index": str(index), "file": fn, "timestamp": datetime.now().strftime('%Y%m%d%H%M%S')}
    )

    return os.path.join(roop.globals.output_path, f"{replaced_filename}{ext}")


def create_temp(target_path: str) -> None:
    temp_directory_path = get_temp_directory_path(target_path)
    Path(temp_directory_path).mkdir(parents=True, exist_ok=True)


def move_temp(target_path: str, output_path: str) -> None:
    temp_output_path = get_temp_output_path(target_path)
    if os.path.isfile(temp_output_path):
        if os.path.isfile(output_path):
            os.remove(output_path)
        shutil.move(temp_output_path, output_path)


def clean_temp(target_path: str) -> None:
    temp_directory_path = get_temp_directory_path(target_path)
    parent_directory_path = os.path.dirname(temp_directory_path)
    if not roop.globals.keep_frames and os.path.isdir(temp_directory_path):
        shutil.rmtree(temp_directory_path)
    if os.path.exists(parent_directory_path) and not os.listdir(parent_directory_path):
        os.rmdir(parent_directory_path)


def delete_temp_frames(filename: str) -> None:
    frames_dir = os.path.abspath(os.path.dirname(filename))
    temp_root = os.path.abspath(os.path.dirname(frames_dir))
    if os.path.basename(temp_root) != TEMP_DIRECTORY:
        raise ValueError(f"Refusing to delete non-temp frame directory: {frames_dir}")
    if os.path.isdir(frames_dir):
        shutil.rmtree(frames_dir)
    if os.path.isdir(temp_root) and not os.listdir(temp_root):
        os.rmdir(temp_root)


def get_frames_output_path(target_path: str) -> str:
    """Return the directory where extracted frames are saved when keep_frames is enabled.
    Frames are placed in a <videoname>_frames sub-folder inside the configured output directory."""
    target_name, _ = os.path.splitext(os.path.basename(target_path))
    return os.path.join(roop.globals.output_path, f"{target_name}_frames")


def move_frames_to_output(target_path: str, fps: float = 0.0) -> None:
    """Move the extracted temp frames to a persistent sub-folder in the output directory.

    When fps > 0 a meta.json sidecar is written inside the frames folder so the
    Frame Editor tab can auto-populate FPS and image format without user input.
    """
    temp_dir = get_temp_directory_path(target_path)
    frames_out_dir = get_frames_output_path(target_path)
    if not os.path.isdir(temp_dir):
        return
    # Remove any stale frames folder from a previous run before moving
    if os.path.isdir(frames_out_dir):
        shutil.rmtree(frames_out_dir)
    shutil.move(temp_dir, frames_out_dir)
    # Write metadata sidecar for the Frame Editor
    if fps > 0:
        write_frames_metadata(
            frames_out_dir,
            fps=fps,
            source_name=target_path,
            image_format=roop.globals.CFG.output_image_format,
        )
    # Clean up the now-empty parent temp directory if nothing else uses it
    parent = os.path.dirname(temp_dir)
    if os.path.exists(parent) and not os.listdir(parent):
        os.rmdir(parent)


def write_frames_metadata(frames_dir: str, fps: float, source_name: str, image_format: str) -> None:
    """Write a meta.json sidecar inside *frames_dir* for use by the Frame Editor."""
    meta = {
        "fps": fps,
        "source": os.path.basename(source_name),
        "source_path": source_name,
        "image_format": image_format,
    }
    try:
        with open(os.path.join(frames_dir, 'meta.json'), 'w') as fh:
            json.dump(meta, fh)
    except Exception as exc:
        print(f"write_frames_metadata: {exc}")


def read_frames_metadata(frames_dir: str) -> dict:
    """Read meta.json from *frames_dir*; return empty dict if absent or corrupt."""
    meta_path = os.path.join(frames_dir, 'meta.json')
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, 'r') as fh:
                return json.load(fh)
        except Exception:
            pass
    return {}


def get_frames_orig_path(target_path: str) -> str:
    """Return the directory where unswapped original frames are stored when keep_frames is enabled.
    Stored alongside the processed frames as <videoname>_frames_orig/ in the output directory."""
    target_name, _ = os.path.splitext(os.path.basename(target_path))
    return os.path.join(roop.globals.output_path, f"{target_name}_frames_orig")


def save_original_frames(target_path: str) -> None:
    """Copy the extracted temp frames to a _frames_orig/ folder BEFORE run_batch overwrites them.

    Called from core.py when keep_frames is True, so the Frame Editor always has
    access to the unswapped source frames for per-frame reprocessing.
    """
    temp_dir = get_temp_directory_path(target_path)
    frames_orig_dir = get_frames_orig_path(target_path)
    if not os.path.isdir(temp_dir):
        return
    if os.path.isdir(frames_orig_dir):
        shutil.rmtree(frames_orig_dir)
    shutil.copytree(temp_dir, frames_orig_dir)


def get_frame_mask_path(frames_orig_dir: str, frame_filename: str) -> str:
    """Return the path for the per-frame mask JSON sidecar.

    frame_filename is the basename of the frame image (e.g. '000001.png').
    The sidecar is stored as '000001_mask.json' in the same _frames_orig/ directory.
    """
    base, _ = os.path.splitext(frame_filename)
    return os.path.join(frames_orig_dir, f"{base}_mask.json")


def save_frame_mask(frames_orig_dir: str, frame_filename: str, mask_data: dict) -> None:
    """Persist per-frame mask settings to a JSON sidecar inside *frames_orig_dir*.

    mask_data is a dict containing any combination of:
      - slider keys: top, bottom, left, right, face_mask_blend,
                     mouth_mask_blend, mouth_top, mouth_bottom,
                     mouth_left, mouth_right (all floats)
      - 'mask_json': the canvas mask JSON string from the mask editor
    """
    mask_path = get_frame_mask_path(frames_orig_dir, frame_filename)
    try:
        with open(mask_path, 'w') as fh:
            json.dump(mask_data, fh)
    except Exception as exc:
        print(f"save_frame_mask: {exc}")


def load_frame_mask(frames_orig_dir: str, frame_filename: str) -> dict:
    """Load per-frame mask settings from the JSON sidecar; return {} if absent or corrupt."""
    mask_path = get_frame_mask_path(frames_orig_dir, frame_filename)
    if os.path.isfile(mask_path):
        try:
            with open(mask_path, 'r') as fh:
                return json.load(fh)
        except Exception:
            pass
    return {}


def has_image_extension(image_path: str) -> bool:
    return image_path.lower().endswith(("png", "jpg", "jpeg", "webp"))


def has_extension(filepath: str, extensions: List[str]) -> bool:
    return filepath.lower().endswith(tuple(extensions))


def is_animated_webp(image_path: str) -> bool:
    """Return True if the file is an animated (multi-frame) WebP."""
    if not image_path or not image_path.lower().endswith(".webp"):
        return False
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            return getattr(img, "n_frames", 1) > 1
    except Exception:
        return False


def is_animated_gif(image_path: str) -> bool:
    """Return True if the file is an animated (multi-frame) GIF."""
    if not image_path or not image_path.lower().endswith(".gif"):
        return False
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            return getattr(img, "n_frames", 1) > 1
    except Exception:
        return False


def is_image(image_path: str) -> bool:
    if image_path and os.path.isfile(image_path):
        if image_path.lower().endswith(".webp"):
            # Animated webp is not a static image
            return not is_animated_webp(image_path)
        if image_path.lower().endswith(".gif"):
            # Animated gif is not a static image
            return not is_animated_gif(image_path)
        mimetype, _ = mimetypes.guess_type(image_path)
        return bool(mimetype and mimetype.startswith("image/"))
    return False


def is_video(video_path: str) -> bool:
    if video_path and os.path.isfile(video_path):
        mimetype, _ = mimetypes.guess_type(video_path)
        return bool(mimetype and mimetype.startswith("video/"))
    return False


DOWNLOAD_SOCKET_TIMEOUT = 3.0
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
_ONLINE_STATE = None
_ONLINE_STATE_AT = 0.0
_ONLINE_CACHE_SECONDS = 5.0
_RUNTIME_DOWNLOADS_LOCKED = False


class OfflineModelError(RuntimeError):
    """Actionable model/cache error shown by the API instead of a traceback."""


def network_downloads_allowed() -> bool:
    """Return whether a model loader may initiate a remote download."""

    return not _RUNTIME_DOWNLOADS_LOCKED and not offline_enabled()


def lock_runtime_downloads() -> None:
    """Prevent new model downloads after a processing job begins."""

    global _RUNTIME_DOWNLOADS_LOCKED
    _RUNTIME_DOWNLOADS_LOCKED = True


def unlock_runtime_downloads() -> None:
    """Allow model preflight for the next job while the app is idle."""

    global _RUNTIME_DOWNLOADS_LOCKED
    _RUNTIME_DOWNLOADS_LOCKED = False


def runtime_downloads_locked() -> bool:
    return _RUNTIME_DOWNLOADS_LOCKED


def require_local_model(path: str, feature: str, *, only_when_offline: bool = False) -> str:
    """Require one concrete local model file and explain how to fix a miss.

    Direct ONNX/PyTorch consumers use ``only_when_offline=True`` because they do
    not own a downloader; their normal online startup path may still be filling
    the cache. The processing lock makes that check strict before a render.
    Download-owning callers leave it false so they are always local-first.
    """

    path = os.path.abspath(path)
    if only_when_offline and network_downloads_allowed():
        return path
    if os.path.isfile(path):
        return path
    if os.path.exists(path):
        detail = "The path exists but is not a regular file."
    else:
        detail = "The file is missing."
    raise OfflineModelError(
        f"{feature} cannot start because its local model is unavailable. "
        f"{detail} Place the exact model file at '{path}', then retry."
    )


def is_online(timeout: float = 2.5) -> bool:
    """Best-effort connectivity probe used only before runtime processing.

    A short cache avoids probing once per pre-warm asset, while the expiry is
    important for an online -> offline transition. Runtime processing does not
    call this helper; it is protected by ``lock_runtime_downloads`` instead.
    """
    global _ONLINE_STATE, _ONLINE_STATE_AT
    if offline_enabled():
        return False
    now = time.monotonic()
    if (_ONLINE_STATE is not None
            and now - _ONLINE_STATE_AT < _ONLINE_CACHE_SECONDS):
        return _ONLINE_STATE
    for host in ("huggingface.co", "github.com"):
        try:
            with socket.create_connection((host, 443), timeout=timeout):
                _ONLINE_STATE = True
                _ONLINE_STATE_AT = now
                return True
        except OSError:
            continue
    _ONLINE_STATE = False
    _ONLINE_STATE_AT = now
    return False


def _handle_missing_model(download_file_path: str, download_directory_path: str,
                          required: bool, reason: str) -> None:
    """Shared policy for a model that is absent and could not be downloaded.

    required=True  -> raise a clear, actionable error. A feature the user just
                      selected needs this model and it is not on disk.
    required=False -> warn and continue. Used by the startup pre-warm so the app
                      still boots offline with whatever partial model set exists;
                      the missing model only surfaces if its feature is used."""
    name = os.path.basename(download_file_path)
    detail = "you appear to be offline" if reason == "offline" else f"download failed: {reason}"
    msg = (
        f"Model '{name}' is not available locally and could not be downloaded "
        f"({detail}). Place the file in '{download_directory_path}' to use this "
        f"feature offline."
    )
    if required:
        raise OfflineModelError(msg)
    try:
        print(f"\033[93m[OFFLINE] {msg}\033[0m")
    except Exception:
        print(f"[OFFLINE] {msg}")


def conditional_download(download_directory_path: str, urls: List[str], required: bool = True) -> None:
    os.makedirs(download_directory_path, exist_ok=True)

    if hasattr(ssl, '_create_unverified_context'):
        ssl._create_default_https_context = ssl._create_unverified_context

    for url in urls:
        # URL query strings are not part of the filename on disk.
        filename = os.path.basename(urllib.parse.urlparse(url).path)
        if not filename:
            _handle_missing_model(
                os.path.join(download_directory_path, "<unknown>"),
                download_directory_path,
                required,
                reason="the download URL has no filename",
            )
            continue
        download_file_path = os.path.join(
            download_directory_path, filename
        )
        if os.path.isfile(download_file_path):
            continue
        if os.path.exists(download_file_path):
            _handle_missing_model(
                download_file_path,
                download_directory_path,
                required,
                reason="the existing path is not a regular file",
            )
            continue

        # Offline mode and the runtime lock must never enter urllib. This is the
        # key boundary that keeps an internet loss from stalling a render.
        if not network_downloads_allowed():
            reason = "offline" if offline_enabled() else "runtime downloads are disabled"
            _handle_missing_model(download_file_path, download_directory_path, required, reason=reason)
            continue

        # Auto offline mode: with no connectivity, don't block on repeated socket
        # timeouts. Fall back to local files and let the required policy decide.
        if not is_online(timeout=DOWNLOAD_SOCKET_TIMEOUT):
            mark_offline("model host is unreachable")
            _handle_missing_model(download_file_path, download_directory_path, required, reason="offline")
            continue

        # Download to a .part file and rename only on success. Writing the
        # final filename directly means an interrupted download leaves a
        # truncated file that the exists() check above then treats as a
        # complete model forever (cryptic ONNX load error until the user
        # deletes it by hand).
        partial_path = download_file_path + ".part"
        try:
            # urlretrieve has no timeout parameter and can leave a truncated
            # final file. Read through urlopen with a bounded socket timeout and
            # rename the .part file only after the response closes successfully.
            with urllib.request.urlopen(url, timeout=DOWNLOAD_SOCKET_TIMEOUT) as response:
                try:
                    total = int(response.headers.get("Content-Length", 0) or 0)
                except (TypeError, ValueError):
                    total = 0
                with tqdm(
                    total=total,
                    desc=f"Downloading {filename}",
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as progress, open(partial_path, "wb") as output:
                    while True:
                        chunk = response.read(DOWNLOAD_CHUNK_SIZE)
                        if not chunk:
                            break
                        output.write(chunk)
                        progress.update(len(chunk))
            size = os.path.getsize(partial_path)
            if size == 0:
                raise IOError("empty response")
            if total and size < total:
                raise IOError(f"incomplete download: got {size} of {total} bytes")
            os.replace(partial_path, download_file_path)
        except (urllib.error.URLError, OSError, socket.timeout, TimeoutError, ssl.SSLError) as exc:
            if os.path.exists(partial_path):
                try:
                    os.remove(partial_path)
                except OSError:
                    pass
            # A failed download (transient network error, host down, partial
            # transfer) is handled the same way as offline: clear error if the
            # model is required now, otherwise warn and move on.
            if not isinstance(exc, urllib.error.HTTPError):
                mark_offline(str(exc))
            _handle_missing_model(download_file_path, download_directory_path, required, reason=str(exc))
        except Exception as exc:
            if os.path.exists(partial_path):
                try:
                    os.remove(partial_path)
                except OSError:
                    pass
            _handle_missing_model(download_file_path, download_directory_path, required, reason=str(exc))


def get_local_files_from_folder(folder: str) -> List[str]:
    if not os.path.exists(folder) or not os.path.isdir(folder):
        return None
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
    ]
    return files


def resolve_relative_path(path: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))


def get_device() -> str:
    import onnxruntime as ort
    available_providers = ort.get_available_providers()

    if len(roop.globals.execution_providers) < 1:
        if 'CUDAExecutionProvider' in available_providers:
            roop.globals.execution_providers = ['CUDAExecutionProvider']
        else:
            roop.globals.execution_providers = ["CPUExecutionProvider"]

    prov = roop.globals.execution_providers[0]
    if "CoreMLExecutionProvider" in prov:
        return "mps"
    if "CUDAExecutionProvider" in prov or "ROCMExecutionProvider" in prov or "TensorrtExecutionProvider" in prov:
        return "cuda"
    if "OpenVINOExecutionProvider" in prov:
        return "mkl"
    return "cpu"


def str_to_class(module_name, class_name) -> Any:
    from importlib import import_module

    class_ = None
    try:
        module_ = import_module(module_name)
        try:
            class_ = getattr(module_, class_name)()
        except AttributeError:
            print(f"Class {class_name} does not exist")
    except ImportError:
        print(f"Module {module_name} does not exist")
    return class_

def is_installed(name:str) -> bool:
    return shutil.which(name);

# Taken from https://stackoverflow.com/a/68842705
def get_platform() -> str:
    if sys.platform == "linux":
        try:
            proc_version = open("/proc/version").read()
            if "Microsoft" in proc_version:
                return "wsl"
        except OSError as exc:
            _LOGGER.debug("Could not inspect /proc/version: %s", exc)
    return sys.platform

def open_with_default_app(filename:str):
    if filename == None:
        return
    platform = get_platform()
    if platform == "darwin":
        subprocess.call(("open", filename))
    elif platform in ["win64", "win32"]:        os.startfile(filename.replace("/", "\\"))
    elif platform == "wsl":
        subprocess.call("cmd.exe /C start".split() + [filename])
    else:  # linux variants
        subprocess.call("xdg-open", filename)


def prepare_for_batch(target_files) -> str:
    print("Preparing temp files")
    tempfolder = os.path.join(tempfile.gettempdir(), "rooptmp")
    if os.path.exists(tempfolder):
        shutil.rmtree(tempfolder)
    Path(tempfolder).mkdir(parents=True, exist_ok=True)
    for f in target_files:
        newname = os.path.basename(f.name)
        shutil.move(f.name, os.path.join(tempfolder, newname))
    return tempfolder


def zip(files, zipname):
    with zipfile.ZipFile(zipname, "w") as zip_file:
        for f in files:
            zip_file.write(f, os.path.basename(f))


def unzip(zipfilename: str, target_path: str):
    with zipfile.ZipFile(zipfilename, "r") as zip_file:
        zip_file.extractall(target_path)


def mkdir_with_umask(directory):
    oldmask = os.umask(0)
    # mode needs octal
    os.makedirs(directory, mode=0o775, exist_ok=True)
    os.umask(oldmask)


def open_folder(path: str):
    platform = get_platform()
    try:
        if platform == "darwin":
            subprocess.call(("open", path))
        elif platform in ["win64", "win32"]:
            open_with_default_app(path)
        elif platform == "wsl":
            subprocess.call("cmd.exe /C start".split() + [path])
        else:  # linux variants
            subprocess.Popen(["xdg-open", path])
    except Exception as e:
        traceback.print_exc()
        pass
        # import webbrowser
        # webbrowser.open(url)


def create_version_html() -> str:
    python_version = ".".join([str(x) for x in sys.version_info[0:3]])
    versions_html = f"""
python: <span title="{sys.version}">{python_version}</span>
•
torch: {getattr(torch, '__long_version__',torch.__version__)}
•
gradio: {gradio.__version__}
"""
    return versions_html


def compute_cosine_distance(emb1, emb2) -> float:
    return distance.cosine(emb1, emb2)

def has_cuda_device():
    return torch.cuda is not None and torch.cuda.is_available()


def print_cuda_info():
    try:
        print(f'Number of CUDA devices: {torch.cuda.device_count()} Currently used Id: {torch.cuda.current_device()} Device Name: {torch.cuda.get_device_name(torch.cuda.current_device())}')
    except (AssertionError, RuntimeError) as exc:
        _LOGGER.info('No CUDA device found: %s', exc)

print_cuda_info()
