"""Crash/abort-resumable video encoding via finalized segment files.

Problem: the in-memory pipeline encodes straight into one long ffmpeg pipe. If
the process dies mid-render (power loss, driver reset, accidental window kill),
the temp file has no trailer (moov atom) and every already-encoded frame is
lost — a 10-hour run restarts from zero.

Fix: SegmentedVideoWriter is a drop-in replacement for FFMPEG_VideoWriter that
rotates the encode into numbered segment files, closing (finalizing) each one
every ROOP_RESUME_CHUNK frames (default 1000) and recording it in a manifest
JSON next to the output. Because each completed segment is a valid playable
file, a crash can only ever lose the in-progress segment. On the next run with
the same source/trim/settings, run_batch_inmem reads the manifest, skips the
already-encoded frames, and continues; when the run completes, the segments are
losslessly concatenated (ffmpeg concat demuxer, stream copy) into the expected
single temp video and cleaned up, so everything downstream (audio restore, the
upscale second pass, template naming) is unchanged.

A deliberate Stop keeps its existing behavior — the partial output is still
concatenated into a playable temp file — but the segments and manifest are kept
so the run can also be *resumed* later instead of only kept.

Disable with ROOP_RESUME=0. Segment files start with '.' so the output-folder
scans (e.g. the upscale pass's _outputs_since) ignore them.
"""

import json
import os
import subprocess

import roop.globals
from roop.ffmpeg_writer import FFMPEG_VideoWriter, FFMPEG_BINARY

MANIFEST_VERSION = 1


def manifest_path(target_video: str) -> str:
    return target_video + ".resume.json"


def _segments_that_exist(m: dict, seg_dir: str):
    """Contiguous prefix of manifest segments whose files are actually on disk.
    (A gap means someone deleted a file — everything after it is unusable
    because concat order would break.)"""
    segs, done = [], 0
    for s in m.get("segments", []):
        fn = s.get("file", "")
        n = int(s.get("frames", 0) or 0)
        if n <= 0 or not fn or not os.path.isfile(os.path.join(seg_dir, fn)):
            break
        segs.append({"file": fn, "frames": n})
        done += n
    return segs, done


class SegmentedVideoWriter:
    """FFMPEG_VideoWriter-compatible writer (write_frame/close) with rotation,
    a resume manifest, and final lossless concatenation."""

    def __init__(self, target_video, size, fps, codec="libx264", crf=14,
                 source_video="", frame_start=0, frame_end=0, signature=""):
        self.target_video = target_video
        self.size = size
        self.fps = float(fps)
        self.codec = codec
        self.crf = crf
        self._dir = os.path.dirname(target_video) or "."
        base, ext = os.path.splitext(os.path.basename(target_video))
        self._seg_prefix = f".{base}.seg"
        self._seg_ext = ext or ".mp4"
        try:
            self.chunk = max(50, int(os.environ.get("ROOP_RESUME_CHUNK", "1000")))
        except ValueError:
            self.chunk = 1000

        # Everything a resume must match on — resuming into a run with different
        # settings/trim/dims would silently mix outputs.
        self._identity = {
            "version": MANIFEST_VERSION,
            "source": os.path.abspath(source_video) if source_video else "",
            "frame_start": int(frame_start),
            "frame_end": int(frame_end),
            "width": int(size[0]),
            "height": int(size[1]),
            "fps": round(self.fps, 3),
            "codec": codec,
            "crf": crf,
            "signature": signature or "",
        }

        self.segments, self.resume_frames = self._load_resume()
        self._seg_index = len(self.segments)
        self._writer = None
        self._cur_seg_file = None
        self._cur_frames = 0
        self._write_manifest()

    # ── resume detection ─────────────────────────────────────────────────────
    def _load_resume(self):
        try:
            with open(manifest_path(self.target_video), "r", encoding="utf-8") as fh:
                m = json.load(fh)
            for key, want in self._identity.items():
                have = m.get(key)
                if key == "fps":
                    if abs(float(have or 0) - want) > 0.01:
                        return [], 0
                elif have != want:
                    return [], 0
            return _segments_that_exist(m, self._dir)
        except Exception:
            return [], 0

    def _write_manifest(self):
        try:
            m = dict(self._identity)
            m["segments"] = self.segments
            tmp = manifest_path(self.target_video) + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(m, fh, indent=1)
            os.replace(tmp, manifest_path(self.target_video))
        except Exception as e:
            print(f"[Resume] could not write manifest: {e}")

    # ── writing ──────────────────────────────────────────────────────────────
    def _open_next_segment(self):
        self._cur_seg_file = f"{self._seg_prefix}{self._seg_index:04d}{self._seg_ext}"
        path = os.path.join(self._dir, self._cur_seg_file)
        self._writer = FFMPEG_VideoWriter(path, self.size, self.fps,
                                          codec=self.codec, crf=self.crf,
                                          audiofile=None)
        self._cur_frames = 0

    def write_frame(self, img_array):
        if self._writer is None:
            self._open_next_segment()
        self._writer.write_frame(img_array)
        self._cur_frames += 1
        if self._cur_frames >= self.chunk:
            self._finalize_segment()

    def _finalize_segment(self):
        """Close the current segment (writes its trailer → playable) and commit
        it to the manifest. From this moment its frames survive a crash."""
        if self._writer is None:
            return
        self._writer.close()
        self._writer = None
        if self._cur_frames > 0:
            self.segments.append({"file": self._cur_seg_file, "frames": self._cur_frames})
            self._seg_index += 1
            self._write_manifest()
        else:
            try:
                os.remove(os.path.join(self._dir, self._cur_seg_file))
            except OSError:
                pass
        self._cur_seg_file = None
        self._cur_frames = 0

    # ── finish ───────────────────────────────────────────────────────────────
    def close(self):
        try:
            self._finalize_segment()
        except Exception as e:
            print(f"[Resume] finalizing last segment failed: {e}")
        if not self.segments:
            return
        # Completed run (nothing signalled a stop) → concat, then clean up.
        # Aborted run → still concat so the partial temp video is playable
        # (mirrors the old Stop behavior), but KEEP segments + manifest so the
        # run can be resumed later.
        completed = bool(roop.globals.processing)
        ok = self._concat()
        if ok and completed:
            self.cleanup()

    def _concat(self) -> bool:
        list_path = os.path.join(self._dir, f"{self._seg_prefix}list.txt")
        try:
            with open(list_path, "w", encoding="utf-8") as fh:
                for s in self.segments:
                    p = os.path.abspath(os.path.join(self._dir, s["file"]))
                    fh.write("file '" + p.replace("'", "'\\''") + "'\n")
            cmd = [FFMPEG_BINARY, "-hide_banner", "-loglevel", "error", "-y",
                   "-f", "concat", "-safe", "0", "-i", list_path,
                   "-c", "copy", self.target_video]
            kwargs = {}
            if os.name == "nt":
                kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
            proc = subprocess.run(cmd, capture_output=True, **kwargs)
            if proc.returncode != 0:
                err = (proc.stderr or b"").decode("utf-8", "replace")[:400]
                print(f"[Resume] segment concat failed (ffmpeg exit {proc.returncode}): {err}")
                return False
            return True
        except Exception as e:
            print(f"[Resume] segment concat failed: {e}")
            return False
        finally:
            try:
                os.remove(list_path)
            except OSError:
                pass

    def cleanup(self):
        """Remove all segments + the manifest (after a fully completed run)."""
        for s in self.segments:
            try:
                os.remove(os.path.join(self._dir, s["file"]))
            except OSError:
                pass
        try:
            os.remove(manifest_path(self.target_video))
        except OSError:
            pass
        self.segments = []
