"""NVDEC (GPU) video decode via an ffmpeg raw pipe.

The batch pipeline decodes with cv2.VideoCapture, which is CPU software decode.
On long videos decode shows up twice (the temporal pre-pass's track_decode and
the swap pass's decode stage), while the GPU's dedicated NVDEC engine sits
idle — the mirror image of the NVENC encode work already done.

FFmpegVideoReader speaks just enough of the cv2.VideoCapture protocol
(set(CAP_PROP_POS_FRAMES) / read() / get() / release()) to be a drop-in for the
sequential readers in ProcessMgr. It spawns ffmpeg with `-hwaccel cuda`, which
decodes H.264/HEVC/VP9/AV1 on NVDEC and hands bgr24 frames back over stdout;
codecs NVDEC can't do (GIF, old formats) silently decode in software inside the
same pipe, which still fixes cv2's known HEVC quirks. Frame seeking uses
ffmpeg's accurate input seeking at (start-0.5)/fps so trims/resume line up with
cv2's frame numbering. `-noautorotate` matches cv2's raw (unrotated) output.

Rollout: ROOP_NVDEC=0 disables, =1 or unset (auto) enables behind a one-time
per-file probe — if `-hwaccel cuda` can't decode the first frame, the caller
keeps its cv2 reader, so this can never break a run.
"""

import os
import subprocess
import threading

import cv2
import numpy as np

from roop.ffmpeg_writer import FFMPEG_BINARY

_probe_cache = {}
_probe_lock = threading.Lock()


def _popen_kwargs():
    kwargs = {}
    if os.name == "nt":
        kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
    return kwargs


def nvdec_wanted() -> bool:
    return os.environ.get("ROOP_NVDEC", "").strip() != "0"


def _probe(video_path: str) -> bool:
    """Can ffmpeg -hwaccel cuda decode one frame of this file? Cached per path."""
    with _probe_lock:
        if video_path in _probe_cache:
            return _probe_cache[video_path]
    ok = False
    try:
        cmd = [FFMPEG_BINARY, "-hide_banner", "-loglevel", "error", "-nostdin",
               "-hwaccel", "cuda", "-i", video_path,
               "-frames:v", "1", "-f", "null", "-"]
        proc = subprocess.run(cmd, capture_output=True, timeout=30, **_popen_kwargs())
        ok = proc.returncode == 0
    except Exception:
        ok = False
    with _probe_lock:
        _probe_cache[video_path] = ok
    return ok


class FFmpegVideoReader:
    """Sequential ffmpeg-pipe frame reader, cv2.VideoCapture-compatible for the
    set(POS_FRAMES) → read()* → release() pattern ProcessMgr's readers use."""

    def __init__(self, video_path, width, height, fps, hwaccel="cuda"):
        self.path = video_path
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps) if fps else 0.0
        self.hwaccel = hwaccel
        self.proc = None
        self._start_frame = 0
        self._frame_bytes = self.width * self.height * 3

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_POS_FRAMES and self.proc is None:
            self._start_frame = int(value)

    def get(self, prop):
        return {cv2.CAP_PROP_FRAME_WIDTH: float(self.width),
                cv2.CAP_PROP_FRAME_HEIGHT: float(self.height),
                cv2.CAP_PROP_FPS: self.fps}.get(prop, 0.0)

    def _spawn(self):
        cmd = [FFMPEG_BINARY, "-hide_banner", "-loglevel", "error", "-nostdin"]
        if self.hwaccel:
            cmd += ["-hwaccel", self.hwaccel]
        if self._start_frame > 0 and self.fps > 0:
            # Accurate input seek: lands on the first frame whose PTS >= t.
            # Aiming half a frame early makes frame numbering match cv2's
            # CAP_PROP_POS_FRAMES without off-by-one drift from float PTS.
            t = max(0.0, (self._start_frame - 0.5) / self.fps)
            cmd += ["-ss", f"{t:.6f}"]
        cmd += ["-noautorotate", "-i", self.path,
                "-f", "rawvideo", "-pix_fmt", "bgr24", "-an", "-sn", "pipe:1"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL,
                                     bufsize=self._frame_bytes * 4,
                                     **_popen_kwargs())

    def read(self):
        if self.proc is None:
            self._spawn()
        buf = bytearray()
        want = self._frame_bytes
        stdout = self.proc.stdout
        while len(buf) < want:
            chunk = stdout.read(want - len(buf))
            if not chunk:
                return False, None
            buf.extend(chunk)
        # frombuffer over the bytearray is zero-copy AND writable (bytes would
        # be read-only, and downstream code mutates frames in place).
        frame = np.frombuffer(buf, np.uint8).reshape(self.height, self.width, 3)
        return True, frame

    def release(self):
        if self.proc is not None:
            try:
                self.proc.stdout.close()
            except Exception:
                pass
            try:
                self.proc.terminate()
                self.proc.wait(timeout=3)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
            self.proc = None


def wrap_capture(cap, video_path, width, height, fps, tag="decode"):
    """Swap a cv2.VideoCapture for the NVDEC pipe reader when enabled and the
    file probes OK; otherwise return the cv2 capture untouched. The returned
    object always supports set/read/get/release."""
    if not nvdec_wanted() or width <= 0 or height <= 0:
        return cap
    if not _probe(video_path):
        if os.environ.get("ROOP_NVDEC", "").strip() == "1":
            print(f"[NVDEC] -hwaccel cuda probe failed for {os.path.basename(video_path)} "
                  f"— using CPU (cv2) decode for {tag}")
        return cap
    try:
        cap.release()
    except Exception:
        pass
    print(f"[NVDEC] GPU decode active for {tag} ({os.path.basename(video_path)})")
    return FFmpegVideoReader(video_path, width, height, fps)
