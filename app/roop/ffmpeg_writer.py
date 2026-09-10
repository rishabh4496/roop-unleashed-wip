"""
FFMPEG_Writer - write set of frames to video file

original from
https://github.com/Zulko/moviepy/blob/master/moviepy/video/io/ffmpeg_writer.py

removed unnecessary dependencies

The MIT License (MIT)

Copyright (c) 2015 Zulko
Copyright (c) 2023 Janvarev Vladislav
"""

import os
import subprocess as sp
import threading
from collections import deque

PIPE = -1
STDOUT = -2
DEVNULL = -3

FFMPEG_BINARY = "ffmpeg"


def probe_encoder(codec="libx265", crf=14, timeout=30):
    """Pre-flight check that the ffmpeg encoder can actually launch and encode.

    Runs a tiny synthetic encode (a 3-frame lavfi source) with the *same* codec
    that the real render will use, capturing exit code and stderr with a hard
    timeout. This catches failures in seconds — BEFORE the long analysis pass —
    so a broken encoder aborts the run early instead of silently hanging the
    frame pipe mid-render.

    The classic Windows trigger: Smart App Control blocks an unsigned ffmpeg DLL
    (e.g. avdevice-62.dll) at process startup, killing the encoder. Because that
    block happens at launch/DLL-load time, this lavfi probe reproduces it without
    needing the real stdin pipe.

    Returns (ok: bool, message: str). message is empty on success.
    """
    import tempfile
    from roop.util_ffmpeg import _rate_control, clamp_quality
    crf = clamp_quality(codec, crf)
    tmp = os.path.join(tempfile.gettempdir(), f"roop_encoder_probe_{os.getpid()}.mp4")
    cmd = [
        FFMPEG_BINARY, '-hide_banner', '-loglevel', 'error', '-y',
        # 256x256 stays above NVENC's minimum supported frame dimensions
        # (smaller sizes fail hevc_nvenc's encoder init and false-negative here).
        '-f', 'lavfi', '-i', 'testsrc=size=256x256:rate=25:duration=1',
        '-frames:v', '3', '-vcodec', codec,
    ]
    cmd.extend(_rate_control(codec, crf))
    cmd.extend(['-pix_fmt', 'yuv420p', tmp])

    popen_params = {"stdout": sp.PIPE, "stderr": sp.PIPE, "stdin": DEVNULL}
    if os.name == "nt":
        popen_params["creationflags"] = 0x08000000  # CREATE_NO_WINDOW

    try:
        proc = sp.Popen(cmd, **popen_params)
    except FileNotFoundError:
        return (False, f"ffmpeg binary '{FFMPEG_BINARY}' was not found on PATH.")
    except Exception as e:
        return (False, f"could not launch ffmpeg: {e}")

    try:
        _, err = proc.communicate(timeout=timeout)
    except sp.TimeoutExpired:
        proc.kill()
        try:
            proc.communicate()
        except Exception:
            pass
        return (False, "the ffmpeg encoder timed out during warm-up — it launched "
                       "but never made progress. On Windows this is usually Smart "
                       "App Control blocking an unsigned ffmpeg DLL.")
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass

    if proc.returncode != 0:
        detail = (err or b"").decode('utf-8', 'replace').strip()
        return (False, detail or f"the ffmpeg encoder exited with code {proc.returncode}.")
    return (True, "")


class FFMPEG_VideoWriter:
    """ A class for FFMPEG-based video writing.

    A class to write videos using ffmpeg. ffmpeg will write in a large
    choice of formats.

    Parameters
    -----------

    filename
      Any filename like 'video.mp4' etc. but if you want to avoid
      complications it is recommended to use the generic extension
      '.avi' for all your videos.

    size
      Size (width,height) of the output video in pixels.

    fps
      Frames per second in the output video file.

    codec
      FFMPEG codec. It seems that in terms of quality the hierarchy is
      'rawvideo' = 'png' > 'mpeg4' > 'libx264'
      'png' manages the same lossless quality as 'rawvideo' but yields
      smaller files. Type ``ffmpeg -codecs`` in a terminal to get a list
      of accepted codecs.

      Note for default 'libx264': by default the pixel format yuv420p
      is used. If the video dimensions are not both even (e.g. 720x405)
      another pixel format is used, and this can cause problem in some
      video readers.

    audiofile
      Optional: The name of an audio file that will be incorporated
      to the video.

    preset
      Sets the time that FFMPEG will take to compress the video. The slower,
      the better the compression rate. Possibilities are: ultrafast,superfast,
      veryfast, faster, fast, medium (default), slow, slower, veryslow,
      placebo.

    bitrate
      Only relevant for codecs which accept a bitrate. "5000k" offers
      nice results in general.

    """

    def __init__(self, filename, size, fps, codec="libx265", crf=14, audiofile=None,
                 preset="faster", bitrate=None,
                 logfile=None, threads=None, ffmpeg_params=None):

        if logfile is None:
            logfile = sp.PIPE

        self.filename = filename
        self.codec = codec
        self.ext = self.filename.split(".")[-1]
        w = size[0] - 1 if size[0] % 2 != 0 else size[0]
        h = size[1] - 1 if size[1] % 2 != 0 else size[1]


        # order is important
        cmd = [
            FFMPEG_BINARY,
            '-hide_banner',
            '-hwaccel', 'auto',
            '-y',
            '-loglevel', 'error' if logfile == sp.PIPE else 'info',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', '%dx%d' % (size[0], size[1]),
            #'-pix_fmt', 'rgba' if withmask else 'rgb24',
            '-pix_fmt', 'bgr24',
            '-r', str(fps),
            '-an', '-i', '-' 
        ]

        if audiofile is not None:
            cmd.extend([
                '-i', audiofile,
                '-acodec', 'copy'
            ])

        cmd.extend(['-vcodec', codec])
        # Out-of-range quality makes ffmpeg exit before a single frame is written,
        # so clamp to what this codec accepts rather than failing the whole render.
        from roop.util_ffmpeg import clamp_quality
        crf = clamp_quality(codec, crf)
        is_nvenc = codec in ('h264_nvenc', 'hevc_nvenc')
        if is_nvenc:
            # NVENC has no -crf; -cq is the constant-quality equivalent (same 0-51
            # scale), so reuse the configured quality directly. Preset p1(fastest)
            # ..p7(slowest/best); p5 + "-tune hq" under VBR is a balanced default.
            # Encoding runs on the GPU's dedicated NVENC engine — off the CPU and
            # separate from CUDA inference. Override the preset with ROOP_NVENC_PRESET.
            nvenc_preset = os.environ.get('ROOP_NVENC_PRESET', 'p5').strip().lower()
            if nvenc_preset not in {f'p{i}' for i in range(1, 8)}:
                nvenc_preset = 'p5'
            cmd.extend(['-rc', 'vbr', '-cq', str(crf), '-b:v', '0',
                        '-preset', nvenc_preset, '-tune', 'hq'])
        else:
            cmd.extend(['-crf', str(crf)])

        # For libx264 / libx265 the preset trades encode SPEED for FILE SIZE at a
        # fixed CRF — the rate control holds perceptual quality constant, so a
        # faster preset speeds up encoding with no visible quality loss (just
        # slightly larger files). Only these encoders use x264-style preset
        # names; vp9 (-deadline) and nvenc (p1-p7) are left untouched. Override
        # the default with env ROOP_ENCODER_PRESET.
        if codec in ('libx264', 'libx265'):
            _valid = {'ultrafast', 'superfast', 'veryfast', 'faster', 'fast',
                      'medium', 'slow', 'slower', 'veryslow', 'placebo'}
            _preset = os.environ.get('ROOP_ENCODER_PRESET', preset).strip().lower()
            if _preset not in _valid:
                _preset = 'faster'
            cmd.extend(['-preset', _preset])
        if ffmpeg_params is not None:
            cmd.extend(ffmpeg_params)
        if bitrate is not None:
            cmd.extend([
                '-b', bitrate
            ])

        # scale to a resolution divisible by 2 if not even
        cmd.extend(['-vf', f'scale={w}:{h}' if w != size[0] or h != size[1] else 'colorspace=bt709:iall=bt601-6-625:fast=1'])

        if threads is not None:
            cmd.extend(["-threads", str(threads)])

        cmd.extend([
            '-pix_fmt', 'yuv420p',

        ])
        cmd.extend([
            filename
        ])

        test = str(cmd)
        print(test)

        popen_params = {"stdout": DEVNULL,
                        "stderr": logfile,
                        "stdin": sp.PIPE}

        # This was added so that no extra unwanted window opens on windows
        # when the child process is created
        if os.name == "nt":
            popen_params["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
        
        self.proc = sp.Popen(cmd, **popen_params)

        # Drain stderr continuously, in a thread, for as long as the encoder
        # lives.
        #
        # `logfile` defaults to sp.PIPE and every call site in the codebase takes
        # that default (ProcessMgr:1298, segment_writer:211, post_swap:257/658),
        # so ffmpeg's stderr is a pipe that nothing read until the process was
        # already dead. An OS pipe holds ~64 KB. Once ffmpeg has written that
        # much and no one has drained it, ffmpeg BLOCKS inside its own write to
        # stderr; a blocked ffmpeg stops reading stdin; the stdin pipe then fills
        # within a single 1080p frame (6.2 MB of bgr24 against a 64 KB buffer),
        # and write_frame() blocks forever.
        #
        # That is a hard hang of the encode with no error anywhere — and it gets
        # MORE likely the longer the render, which is exactly when the whole
        # analysis pass is on the line. It also produces the corrupt-output path
        # ProcessMgr already documents at run_batch_inmem's teardown: the write
        # thread is wedged in write_frame, its join(timeout=10) expires, and
        # close() then shuts stdin underneath an in-flight write.
        #
        # A bounded deque keeps the tail of what ffmpeg said (that is all the
        # error paths below ever quote) while making the pipe impossible to fill.
        self._err_lines = deque(maxlen=200)
        self._err_lock = threading.Lock()
        self._drain = None
        if self.proc.stderr is not None:
            self._drain = threading.Thread(
                target=self._drain_stderr, name='ffmpeg_stderr', daemon=True)
            self._drain.start()

    def _drain_stderr(self):
        """Read ffmpeg's stderr to EOF so the pipe can never back-pressure."""
        stream = self.proc.stderr if self.proc is not None else None
        if stream is None:
            return
        try:
            for line in iter(stream.readline, b''):
                with self._err_lock:
                    self._err_lines.append(line)
        except Exception:
            # Closed underneath us by close() — nothing left to drain.
            pass

    def _stderr_text(self):
        """The retained tail of ffmpeg's stderr, decoded.

        Replaces the old `self.proc.stderr.read()` / `communicate()` reads: with
        the drain thread running, those would return b'' (the thread already
        consumed every byte) and the error messages below would be empty.
        """
        with self._err_lock:
            raw = b''.join(self._err_lines)
        return raw.decode('utf-8', 'replace').strip()

    def write_frame(self, img_array):
        """ Writes one frame in the file."""
        # Fail fast if the encoder process has already died (e.g. Windows Smart
        # App Control blocked an unsigned ffmpeg DLL at launch, or the codec
        # failed to initialise). Without this, writing into the dead pipe can
        # block forever and the whole render hangs silently — losing the entire
        # analysis pass with no error.
        if self.proc is None or self.proc.poll() is not None:
            rc = None if self.proc is None else self.proc.returncode
            # The process is already gone, so the drain thread is at EOF and about
            # to exit; give it a moment so the quoted tail is complete.
            if self._drain is not None:
                self._drain.join(timeout=1.0)
            ffmpeg_error = self._stderr_text()
            raise IOError(
                "roop unleashed error: the ffmpeg encoder process exited "
                f"unexpectedly (code {rc}) before the video was finished.\n\n"
                "On Windows this is usually Smart App Control blocking an "
                "unsigned ffmpeg DLL (e.g. avdevice-62.dll). Re-run the job, or "
                "turn off Smart App Control in Windows Security → App & "
                "browser control.\n\nffmpeg said:\n" + ffmpeg_error)
        try:
            #if PY3:
            self.proc.stdin.write(img_array.tobytes())
            # else:
            #    self.proc.stdin.write(img_array.tostring())
        except IOError as err:
            # Was self.proc.communicate(): that both re-reads a pipe the drain
            # thread already owns (two readers on one fd) and blocks until the
            # process exits. Wait for the exit with a bound, then quote the tail
            # the drain thread collected.
            try:
                self.proc.wait(timeout=10)
            except Exception:
                pass
            if self._drain is not None:
                self._drain.join(timeout=1.0)
            ffmpeg_error = self._stderr_text()
            error = (str(err) + ("\n\nroop unleashed error: FFMPEG encountered "
                                 "the following error while writing file %s:"
                                 "\n\n %s" % (self.filename, ffmpeg_error)))

            if "Unknown encoder" in ffmpeg_error:

                error = error+("\n\nThe video export "
                  "failed because FFMPEG didn't find the specified "
                  "codec for video encoding (%s). Please install "
                  "this codec or change the codec when calling "
                  "write_videofile. For instance:\n"
                  "  >>> clip.write_videofile('myvid.webm', codec='libvpx')")%(self.codec)

            elif "incorrect codec parameters ?" in ffmpeg_error:

                 error = error+("\n\nThe video export "
                  "failed, possibly because the codec specified for "
                  "the video (%s) is not compatible with the given "
                  "extension (%s). Please specify a valid 'codec' "
                  "argument in write_videofile. This would be 'libx264' "
                  "or 'mpeg4' for mp4, 'libtheora' for ogv, 'libvpx for webm. "
                  "Another possible reason is that the audio codec was not "
                  "compatible with the video codec. For instance the video "
                  "extensions 'ogv' and 'webm' only allow 'libvorbis' (default) as a"
                  "video codec."
                  )%(self.codec, self.ext)

            elif  "encoder setup failed" in ffmpeg_error:

                error = error+("\n\nThe video export "
                  "failed, possibly because the bitrate you specified "
                  "was too high or too low for the video codec.")

            elif "Invalid encoder type" in ffmpeg_error:

                error = error + ("\n\nThe video export failed because the codec "
                  "or file extension you provided is not a video")


            raise IOError(error)

    def close(self, timeout=120):
        """Finish the encode and reap the process. Never raises, never hangs.

        The previous version was three unguarded calls in a row:

            self.proc.stdin.close()   # BrokenPipeError if ffmpeg already died
            self.proc.stderr.close()  # cuts the trailer messages off mid-flush
            self.proc.wait()          # unbounded — a wedged ffmpeg hangs forever

        A raise from the first line skipped wait() AND the `self.proc = None`
        below it, so the ffmpeg child was never reaped: one orphan per failed
        render, each still holding the output file open. Over a batch that is a
        pile of stuck processes and a set of files Windows will not let you
        delete. close() is called from `finally` blocks (ProcessMgr:1437, :4016),
        so a raise here also masks whatever real exception sent us there.

        Order matters: EOF on stdin, THEN wait for ffmpeg to write its trailer
        (still draining stderr throughout), and only then tear the pipes down.
        """
        proc, self.proc = self.proc, None
        if proc is None:
            return

        # 1. EOF on stdin tells ffmpeg to finalise the file.
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except Exception:
            pass

        # 2. Let it write the trailer. Muxing a long file is not instant, so the
        #    bound is generous — but it IS a bound.
        try:
            proc.wait(timeout=timeout)
        except sp.TimeoutExpired:
            print(f"[ffmpeg] encoder did not exit within {timeout}s while closing "
                  f"{os.path.basename(self.filename)} — terminating it. The output "
                  f"may be truncated.")
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except Exception:
                try:
                    proc.kill()
                    proc.wait(timeout=10)
                except Exception:
                    pass
        except Exception:
            pass

        # 3. The process is gone, so the drain thread is at EOF; join it before
        #    closing the fd it reads, or it raises on a closed file.
        if self._drain is not None:
            self._drain.join(timeout=5.0)
            self._drain = None
        try:
            if proc.stderr is not None:
                proc.stderr.close()
        except Exception:
            pass

    # Support the Context Manager protocol, to ensure that resources are cleaned up.

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()



    
