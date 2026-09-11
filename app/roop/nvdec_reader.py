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
cv2's frame numbering. Rotation side-data is applied (ffmpeg's default), which
is what cv2.VideoCapture does too — CAP_PROP_ORIENTATION_AUTO defaults to on —
so a portrait phone clip comes back as the same (h, w) picture from both
readers. It used to pass `-noautorotate` on the belief that cv2 hands back the
raw coded picture; it does not, and on a 90-degree-tagged file the unrotated
byte stream reshaped into cv2's rotated dimensions is a silently scrambled
frame (same byte count, mean abs error ~126 against the real frame).
`-fps_mode passthrough` keeps the pipe at one output frame per DECODED frame —
ffmpeg's default would re-time the raw stream to the container's r_frame_rate
and duplicate/drop frames, which breaks that same frame numbering (see _spawn).

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


# ── Which YUV->RGB matrix a reader used ──────────────────────────────────────
# cv2.VideoCapture always converts with libswscale's default (BT.601) and has
# no way to be told otherwise. This pipe can be: `scale=in_color_matrix=<m>`
# decodes with the file's own matrix, so a BT.709 HD source reaches the models
# as the colours the camera recorded rather than a 601 reading of them. Whoever
# ENCODES the frames afterwards must use the same matrix for the round trip to
# stay an identity, so the reader publishes it as `decode_matrix` and
# ProcessMgr hands it to the writer (ffmpeg_writer.color_filter_chain). cv2
# captures have no attribute, and getattr(cap, 'decode_matrix', 'bt601')
# is therefore right for both.
DEFAULT_DECODE_MATRIX = "bt601"
# swscale's spelling for each tag ffprobe can report. Anything not listed
# (unknown, gbr, ycgco, ...) decodes with the default, which is also what the
# decoder would have done with no instruction at all.
_TAG_TO_SWS = {
    "bt709": "bt709",
    "smpte170m": "bt601",
    "bt470bg": "bt601",
    "smpte240m": "smpte240m",
    "bt2020nc": "bt2020",
    "bt2020c": "bt2020",
    "fcc": "fcc",
}


def decode_matrix_for(video_path: str) -> str:
    """The swscale matrix this pipe will decode *video_path* with."""
    try:
        from roop.capturer import probe_color_tags
        tags = probe_color_tags(video_path) or {}
    except Exception:
        tags = {}
    return _TAG_TO_SWS.get(str(tags.get("colorspace") or "").lower(), DEFAULT_DECODE_MATRIX)


_fps_mode_flag = None
_fps_mode_lock = threading.Lock()


def _fps_mode_args():
    """Output args that force 1:1 frame passthrough (no dup/drop re-timing).

    `-fps_mode passthrough` is the modern spelling (ffmpeg >= 5.1); older builds
    only know the deprecated `-vsync 0`. Probed once and cached.
    """
    global _fps_mode_flag
    with _fps_mode_lock:
        if _fps_mode_flag is None:
            try:
                proc = subprocess.run([FFMPEG_BINARY, "-hide_banner", "-h", "full"],
                                      capture_output=True, timeout=30, **_popen_kwargs())
                blob = (proc.stdout or b"") + (proc.stderr or b"")
                _fps_mode_flag = ["-fps_mode", "passthrough"] if b"-fps_mode" in blob else ["-vsync", "0"]
            except Exception:
                _fps_mode_flag = ["-vsync", "0"]
        return list(_fps_mode_flag)


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

    def __init__(self, video_path, width, height, fps, hwaccel="cuda",
                 decode_matrix=DEFAULT_DECODE_MATRIX):
        self.path = video_path
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps) if fps else 0.0
        self.hwaccel = hwaccel
        # See decode_matrix_for. Always explicit — never "whatever swscale
        # picks" — so the writer can be told exactly what to undo.
        self.decode_matrix = decode_matrix if decode_matrix in _TAG_TO_SWS.values() \
            else DEFAULT_DECODE_MATRIX
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
        cmd += ["-i", self.path,
                # CRITICAL: pass every decoded frame through 1:1. Without this,
                # ffmpeg's default (CFR) re-times the raw output to the stream's
                # r_frame_rate, which for a lot of files is a multiple of the real
                # rate (e.g. r_frame_rate=48000/1001 on true-24fps content). ffmpeg
                # then DUPLICATES frames to fill the gap ("dup=N") and the pipeline,
                # which stops after frame_count frames, only ever sees the first
                # half of the video with every frame doubled — a full-length output
                # holding half the content at an apparent half frame rate.
                # passthrough makes the pipe match cv2's decoded-frame numbering,
                # which is what frame_start / frame_count / seeking assume.
                *_fps_mode_args(),
                "-vf", f"scale=in_color_matrix={self.decode_matrix}:in_range=tv,format=bgr24",
                "-f", "rawvideo", "-pix_fmt", "bgr24", "-an", "-sn", "pipe:1"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL,
                                     bufsize=self._frame_bytes * 4,
                                     **_popen_kwargs())

    def read(self):
        if self.proc is None:
            self._spawn()
        want = self._frame_bytes
        stdout = self.proc.stdout
        # Fill a pre-sized buffer in place. The old path built the frame with
        # bytearray() + read() + extend(), which copies every frame TWICE — once
        # into the bytes object read() allocates, once again into the bytearray,
        # plus the regrowth as it extends. At 1080p that is 6.2MB per copy and it
        # showed: measured over 600 frames, 123.5 fps against a raw pipe that
        # delivers 138.4, i.e. ~0.94ms per frame of pure memcpy. readinto brings
        # it to 7.23 ms/frame against the pipe's own 7.16 ceiling — the reader
        # stops being a factor.
        #
        # A FRESH buffer per frame is required, not a reused one: frames go into
        # a queue and are held by the pre-pass and the swap loop, so a shared
        # buffer would alias the previous frame and rewrite it under them.
        buf = bytearray(want)
        mv = memoryview(buf)
        n = 0
        while n < want:
            k = stdout.readinto(mv[n:])
            if not k:
                return False, None
            n += k
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


def wrap_capture(cap, video_path, width, height, fps, tag="decode", tag_aware=False):
    """Swap a cv2.VideoCapture for the NVDEC pipe reader when enabled and the
    file probes OK; otherwise return the cv2 capture untouched. The returned
    object always supports set/read/get/release.

    `tag_aware=True` decodes with the file's own colour matrix (see
    decode_matrix_for) and is only for a reader whose frames are ENCODED
    again — the caller must pass `getattr(cap, 'decode_matrix', 'bt601')` to
    the writer. Detection-only readers (the tracking / temporal pre-passes)
    leave it off: the detector was trained on 601-decoded footage like
    everything else, and nothing they read is written back."""
    if not nvdec_wanted() or width <= 0 or height <= 0:
        return cap
    if not _probe(video_path):
        if os.environ.get("ROOP_NVDEC", "").strip() == "1":
            print(f"[NVDEC] -hwaccel cuda probe failed for {os.path.basename(video_path)} "
                  f"— using CPU (cv2) decode for {tag}")
        return cap
    # The raw pipe has no framing: if ffmpeg's picture size disagrees with the
    # size the caller measured (cv2's, rotation applied), every frame would be
    # reshaped wrong with no error anywhere. Refuse rather than guess.
    try:
        from roop.capturer import _probe_video
        info = _probe_video(video_path)
    except Exception:
        info = None
    if info and (int(info.get("w", 0)), int(info.get("h", 0))) != (int(width), int(height)):
        print(f"[NVDEC] ffprobe reports {info.get('w')}x{info.get('h')} for "
              f"{os.path.basename(video_path)} but the caller measured {width}x{height} "
              f"— keeping cv2 decode for {tag} so frames cannot be mis-sized.")
        return cap
    try:
        cap.release()
    except Exception:
        pass
    matrix = decode_matrix_for(video_path) if tag_aware else DEFAULT_DECODE_MATRIX
    print(f"[NVDEC] GPU decode active for {tag} ({os.path.basename(video_path)}, "
          f"matrix={matrix})")
    return FFmpegVideoReader(video_path, width, height, fps, decode_matrix=matrix)
