"""The most recent processed frame, for the UI's processing view.

The pipeline used to hand every finished frame to the UI by keeping a full-frame
COPY of it (`latest_swapped_frame`) — a ~6 MB memcpy per frame, on the hot swap
path, whether or not anyone was looking. That is why it was removed and
`_publish_live` became a no-op: the cost was per frame while the UI could only
ever display a couple of frames a second, so the box showed a still and the
preview looked frozen for the whole run.

This does the same job for a fixed, tiny budget:

  * THROTTLED — at most one frame every ROOP_LIVE_PREVIEW_MS (default 500), so
    the cost does not scale with frame rate. At 20 fps that is 1 frame in 10.
  * DOWNSCALED — to ROOP_LIVE_PREVIEW_WIDTH (default 960) before anything else
    touches it. The first version used 480 on the reasoning that it was "more
    than the box can show", which is only true at devicePixelRatio 1: the box is
    ~480 CSS px wide, so on the HiDPI displays these renders are watched on it
    was being stretched 2x and looked heavily pixelated. 960 covers a 2x box
    with nothing to spare, which is why it is the default rather than more.
  * ENCODED ONCE — to JPEG here rather than per HTTP poll, so N pollers cost
    nothing extra and the API thread never touches a numpy array (no lock held
    across an encode, no copy needed for thread safety).

Measured shape of the work (this machine, 40 iterations): resize plus imencode
is 3.7 ms per publish from 1080p and 5.3 ms from 4K, twice a second — about 1%
of one thread, and OpenCV releases the GIL for both. The frame it produces is
20-33 KB, i.e. under 70 KB/s over loopback. Set ROOP_LIVE_PREVIEW=0 to switch
it off entirely (publish becomes a bare return).
"""

import os
import threading
import time

import cv2
import numpy as np

_ENABLED = os.environ.get('ROOP_LIVE_PREVIEW', '1').strip().lower() not in ('0', 'false', 'off')

try:
    _INTERVAL = max(0.05, float(os.environ.get('ROOP_LIVE_PREVIEW_MS', '500')) / 1000.0)
except ValueError:
    _INTERVAL = 0.5

try:
    _MAX_W = max(160, int(os.environ.get('ROOP_LIVE_PREVIEW_WIDTH', '960')))
except ValueError:
    _MAX_W = 960

# JPEG quality for that frame. 72 is a web-photo default and shows its blocking
# on the flat skin the swap is judged on; 88 costs ~1.5x the bytes of a frame
# that is already tiny and is what stops the live view looking worse than the
# render it is reporting on.
try:
    _QUALITY = min(100, max(40, int(os.environ.get('ROOP_LIVE_PREVIEW_QUALITY', '88'))))
except ValueError:
    _QUALITY = 88

_lock = threading.Lock()
# jpeg: encoded bytes, seq: bumped on every publish (the UI's cache key),
# t: last publish time (throttle), size: source frame size for the caption.
_state = {'jpeg': None, 'seq': 0, 't': 0.0, 'size': (0, 0)}


def reset():
    """Drop the previous run's frame so a new run never shows a stale one."""
    with _lock:
        _state.update({'jpeg': None, 'seq': 0, 't': 0.0, 'size': (0, 0)})


def enabled():
    return _ENABLED


def publish(frame):
    """Offer a finished frame. Cheap and safe to call for every frame — most
    calls return on the throttle check without touching the pixels."""
    if not _ENABLED or frame is None:
        return
    now = time.time()
    # Read without the lock: a torn read here can only mean one extra (or one
    # skipped) publish, and both are harmless. Taking the lock per frame would
    # put every worker thread through a mutex for a preview.
    if now - _state['t'] < _INTERVAL:
        return
    _state['t'] = now
    try:
        h, w = frame.shape[:2]
        if w > _MAX_W:
            scale = _MAX_W / float(w)
            small = cv2.resize(frame, (_MAX_W, max(1, int(round(h * scale)))),
                               interpolation=cv2.INTER_AREA)
        else:
            small = frame
        ok, buf = cv2.imencode('.jpg', small, [int(cv2.IMWRITE_JPEG_QUALITY), _QUALITY])
        if not ok:
            return
        data = np.asarray(buf).tobytes()
    except Exception:
        return          # a preview must never be able to break a render
    with _lock:
        _state['jpeg'] = data
        _state['seq'] += 1
        _state['size'] = (int(w), int(h))


def seq():
    """Publication counter — 0 until the first frame. The UI uses it as the
    image URL's cache key, so it refetches exactly when there is something new."""
    return _state['seq']


def snapshot():
    """(jpeg_bytes | None, seq, (src_w, src_h))"""
    with _lock:
        return _state['jpeg'], _state['seq'], _state['size']
