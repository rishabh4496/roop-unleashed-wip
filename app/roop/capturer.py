from typing import Optional
import threading
import cv2
import numpy as np

from roop.typing import Frame

current_video_path = None
current_frame_total = 0
current_capture = None
# Frame index the next read() on current_capture will return, or None if unknown.
# Lets sequential requests (playback, forward scrubbing) skip the expensive
# CAP_PROP_POS_FRAMES seek — for long-GOP codecs (H.264/HEVC) a random seek
# re-decodes from the nearest keyframe and can cost seconds per frame, whereas
# reading the next frame in order is near real-time.
current_next_pos = None

# Serialises all access to the shared VideoCapture and animated-WebP cache.
# cv2.VideoCapture is NOT thread-safe: concurrent reads from multiple Gradio
# worker threads corrupt FFmpeg's internal codec context, triggering:
#   "Assertion fctx->async_lock failed at libavcodec/pthread_frame.c:173"
_capture_lock = threading.Lock()

# PIL-based cache for animated webp (PIL is already in the env via clip)
_awebp_path = None
_awebp_frames = None   # list of BGR numpy arrays, populated lazily


def _load_animated_webp(video_path: str):
    """Load all frames of an animated WebP into memory as BGR arrays.

    Must be called while holding _capture_lock."""
    global _awebp_path, _awebp_frames
    if _awebp_path == video_path and _awebp_frames is not None:
        return
    from PIL import Image
    frames = []
    with Image.open(video_path) as img:
        for i in range(getattr(img, "n_frames", 1)):
            img.seek(i)
            frame_rgb = np.array(img.convert("RGB"))
            frames.append(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    _awebp_path = video_path
    _awebp_frames = frames


def get_image_frame(filename: str):
    try:
        return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)
    except:
        print(f"Exception reading {filename}")
    return None


def get_video_frame(video_path: str, frame_number: int = 0) -> Optional[Frame]:
    global current_video_path, current_capture, current_frame_total, current_next_pos

    with _capture_lock:
        # Animated WebP — use PIL-based reader (no FFmpeg involved)
        if video_path.lower().endswith(".webp"):
            _load_animated_webp(video_path)
            if _awebp_frames:
                idx = max(0, min(len(_awebp_frames) - 1, frame_number - 1))
                return _awebp_frames[idx]
            return None

        # `or current_capture is None` matters: release_video() can be called
        # directly (shutdown, between runs) and leaves the path set, so without
        # this a second request for the SAME file would skip re-opening and then
        # dereference a None capture.
        if video_path != current_video_path or current_capture is None:
            release_video()
            current_capture = cv2.VideoCapture(video_path)
            current_video_path = video_path
            current_frame_total = current_capture.get(cv2.CAP_PROP_FRAME_COUNT)
            current_next_pos = 0  # a fresh capture reads frame 0 on the next read()

        target = max(0, min(int(current_frame_total) - 1, frame_number - 1))
        # Fast path: the requested frame is exactly the next one in sequence, so
        # decode it in order without a seek (see current_next_pos note above).
        if target != current_next_pos:
            current_capture.set(cv2.CAP_PROP_POS_FRAMES, target)
        has_frame, frame = current_capture.read()
        if has_frame:
            current_next_pos = target + 1
            return frame
        current_next_pos = None  # position unknown after a failed read
    return None


def release_video():
    global current_capture, current_next_pos, current_video_path, current_frame_total

    # Caller must hold _capture_lock when called from get_video_frame;
    # direct callers (shutdown, etc.) should also acquire it.
    if current_capture is not None:
        current_capture.release()
        current_capture = None
    current_next_pos = None
    # Clear the identity too, so the next request re-opens instead of matching a
    # path whose capture is gone (and re-reads the frame count, which was stale).
    current_video_path = None
    current_frame_total = 0


def get_video_frame_total(video_path: str) -> int:
    # Animated WebP — use PIL
    if video_path.lower().endswith(".webp"):
        try:
            from PIL import Image
            with Image.open(video_path) as img:
                return getattr(img, "n_frames", 1)
        except Exception:
            return 1
    capture = cv2.VideoCapture(video_path)
    video_frame_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return video_frame_total
