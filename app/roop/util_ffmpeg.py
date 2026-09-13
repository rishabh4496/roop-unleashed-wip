
import os
import logging
import math
import subprocess
import tempfile
import threading
from collections import deque
import roop.globals
import roop.utilities as util

from typing import Any, List, Optional, Sequence


_LOGGER = logging.getLogger(__name__)
_FFMPEG_TAIL_LINES = 200

def run_ffmpeg(args: Sequence[str]) -> bool:
    """Run FFmpeg without a shell while retaining only a bounded output tail."""
    commands = [
        'ffmpeg', '-hide_banner', '-hwaccel', 'auto', '-y', '-loglevel',
        str(roop.globals.log_level or 'error'),
        *(str(arg) for arg in args),
    ]
    _LOGGER.info("Running ffmpeg")
    process = None
    try:
        from roop.process_lifecycle import process_lifecycle_manager
    except Exception:
        process_lifecycle_manager = None
    try:
        kwargs = {
            'stdout': subprocess.PIPE,
            'stderr': subprocess.STDOUT,
        }
        # CREATE_NO_WINDOW prevents the asyncio ProactorEventLoop on Windows from
        # raising ConnectionResetError (WinError 10054) when the subprocess pipe closes.
        if os.name == 'nt':
            kwargs['creationflags'] = 0x08000000
        process = subprocess.Popen(commands, **kwargs)
        if process_lifecycle_manager is not None:
            process_lifecycle_manager.register_process(process, "ffmpeg")
        output_tail = deque(maxlen=_FFMPEG_TAIL_LINES)
        if process.stdout is not None:
            for line in iter(process.stdout.readline, b''):
                output_tail.append(line)
            process.stdout.close()
        returncode = process.wait()
        if returncode != 0:
            output = b''.join(output_tail).decode(errors='replace')
            _LOGGER.error("FFmpeg failed with exit code %s. Command: %s\n%s",
                          returncode, subprocess.list2cmdline(commands), output)
            return False
        return True
    except (OSError, ValueError) as exc:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)
        _LOGGER.error("Could not launch FFmpeg. Command: %s; error: %s",
                      subprocess.list2cmdline(commands), exc, exc_info=True)
        return False
    finally:
        if process_lifecycle_manager is not None and process is not None:
            process_lifecycle_manager.unregister_process(process)



# Highest quality value each encoder family accepts. x264/x265 and NVENC's -cq
# top out at 51; the VP9/AV1 family goes to 63. The settings slider used to offer
# 0-100 with nothing clamping it.
#
# Measured, because the encoders do NOT agree on what happens past the limit:
#   libx264   -crf 80 -> exits 0, output byte-identical to -crf 51 (clamps itself)
#   libx265   -crf 80 -> FAILS, no output          <- the default encoder
#   h264_nvenc/hevc_nvenc -cq 80 -> FAILS, no output
# So on the default (libx265) and on both GPU encoders an out-of-range slider
# value killed the render outright; on libx264 it silently did nothing. Clamping
# makes all of them behave like libx264.
_QUALITY_MAX = 51
_QUALITY_MAX_WIDE = 63
_WIDE_RANGE_CODECS = ('libvpx', 'vp9', 'aom', 'av1', 'svtav1')


def quality_max(codec: str) -> int:
    """Largest quality value *codec* accepts (0 = best quality for all of them)."""
    c = (codec or '').lower()
    return _QUALITY_MAX_WIDE if any(k in c for k in _WIDE_RANGE_CODECS) else _QUALITY_MAX


def clamp_quality(codec: str, quality) -> int:
    """Coerce *quality* into the range *codec* actually accepts."""
    try:
        q = int(round(float(quality)))
    except (TypeError, ValueError):
        q = 14
    return max(0, min(q, quality_max(codec)))


def _rate_control(codec: str, quality) -> List[str]:
    """Return the encoder rate-control args for *codec* at *quality*.

    libx264/libx265 (and most CPU encoders) use -crf; NVENC has no -crf and uses
    -cq under VBR with p1..p7 presets instead. Keeping this in one place means
    every re-encode path supports h264_nvenc/hevc_nvenc (GPU encode) correctly.
    """
    q = clamp_quality(codec, quality)
    if codec in ('h264_nvenc', 'hevc_nvenc'):
        preset = os.environ.get('ROOP_NVENC_PRESET', 'p5').strip().lower()
        if preset not in {f'p{i}' for i in range(1, 8)}:
            preset = 'p5'
        # -cq selects the target quality, while -b:v 0 removes FFmpeg's default
        # bitrate ceiling. Without it, detailed/high-resolution footage can be
        # bitrate-starved even at a low CQ value.
        return ['-rc', 'vbr', '-cq', str(q), '-b:v', '0',
                '-preset', preset, '-tune', 'hq']
    args = ['-crf', str(q)]
    if codec in ('libx264', 'libx265'):
        valid = {'ultrafast', 'superfast', 'veryfast', 'faster', 'fast',
                 'medium', 'slow', 'slower', 'veryslow', 'placebo'}
        preset = os.environ.get('ROOP_ENCODER_PRESET', 'faster').strip().lower()
        args.extend(['-preset', preset if preset in valid else 'faster'])
    return args


def cut_video(original_video: str, cut_video: str, start_frame: int,
              end_frame: int, reencode: bool) -> bool:
    fps = util.detect_fps(original_video)
    start_time = start_frame / fps
    num_frames = end_frame - start_frame
    if num_frames <= 0:
        _LOGGER.error("Refusing to cut an empty frame range: %s..%s",
                      start_frame, end_frame)
        return False

    if reencode:
        return run_ffmpeg(['-ss', format(start_time, ".9f"), '-i', original_video,
                           '-c:v', roop.globals.video_encoder, '-c:a', 'aac',
                           '-frames:v', str(num_frames), cut_video])
    return run_ffmpeg(['-ss', format(start_time, ".9f"), '-i', original_video,
                       '-frames:v', str(num_frames), '-c:v', 'copy', '-c:a', 'copy',
                       cut_video])

def join_videos(videos: List[str], dest_filename: str, simple: bool) -> bool:
    if not videos:
        _LOGGER.error("Cannot join an empty video list")
        return False
    if simple:
        temp_dir = util.resolve_relative_path('../temp')
        os.makedirs(temp_dir, exist_ok=True)
        list_path = None
        try:
            with tempfile.NamedTemporaryFile(
                    mode='w', suffix='.ffconcat', prefix='roop_join_',
                    dir=temp_dir, encoding='utf-8', delete=False) as handle:
                list_path = handle.name
                for video in videos:
                    normalized = os.path.abspath(video).replace('\\', '/')
                    escaped = normalized.replace("'", "'\\''")
                    handle.write(f"file '{escaped}'\n")
            return run_ffmpeg(['-f', 'concat', '-safe', '0', '-i', list_path,
                               '-c', 'copy', dest_filename])
        except OSError as exc:
            _LOGGER.error("Could not create FFmpeg concat list: %s", exc, exc_info=True)
            return False
        finally:
            if list_path:
                try:
                    os.remove(list_path)
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    _LOGGER.warning("Could not remove concat list %s: %s", list_path, exc)

    inputs = [arg for video in videos for arg in ('-i', video)]
    filter_graph = ''.join(f'[{i}:v:0][{i}:a:0]' for i in range(len(videos)))
    filter_graph += f'concat=n={len(videos)}:v=1:a=1[outv][outa]'
    return run_ffmpeg([
        *inputs, '-filter_complex', filter_graph,
        '-map', '[outv]', '-map', '[outa]', dest_filename,
    ])



def _get_output_image_format() -> str:
    cfg = getattr(roop.globals, 'CFG', None)
    if cfg is not None and getattr(cfg, 'output_image_format', None):
        return str(cfg.output_image_format)
    return 'png'


def _extract_frames_from_animated_webp(target_path: str, trim_frame_start, trim_frame_end, temp_directory_path: str) -> bool:
    """Extract individual frames from an animated WebP file using Pillow.

    FFmpeg's native webp_pipe demuxer skips ANIM/ANMF chunks and cannot decode
    animated WebP files, producing zero frames.  Pillow handles them correctly.
    Frames are written as the configured output_image_format (typically png).
    """
    import numpy as np
    import cv2
    from PIL import Image

    try:
        with Image.open(target_path) as img:
            n_frames = getattr(img, 'n_frames', 1)
            start = int(trim_frame_start) if trim_frame_start is not None else 0
            end   = int(trim_frame_end)   if trim_frame_end   is not None else n_frames
            end   = min(end, n_frames)

            fmt = _get_output_image_format()
            frame_num = 1
            for i in range(start, end):
                img.seek(i)
                frame_rgb = np.array(img.convert('RGB'))
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                out_path = os.path.join(
                    temp_directory_path,
                    f'{frame_num:06d}.{fmt}',
                )
                cv2.imwrite(out_path, frame_bgr)
                frame_num += 1

        extracted = frame_num - 1
        print(f'Extracted {extracted} frames from animated WebP via PIL')
        return extracted > 0
    except Exception as e:
        print(f'PIL animated WebP frame extraction failed: {e}')
        return False


def _parse_fps_fraction(val: Any) -> float:
    """Safely parse an FFmpeg frame rate string like '30000/1001' or '29.97' to a float."""
    try:
        s = str(val or '').strip()
        if '/' in s:
            num, den = s.split('/', 1)
            den_f = float(den)
            return float(num) / den_f if den_f != 0 else 0.0
        return float(s)
    except (TypeError, ValueError):
        return 0.0


def is_variable_frame_rate(video_path: str) -> bool:
    """Pre-check whether input media has a Variable Frame Rate (VFR).

    Compares the nominal container rate (r_frame_rate) against the calculated
    average stream frame rate (avg_frame_rate). A non-trivial difference indicates
    variable frame timing (common in smartphone recordings and screen captures).
    """
    if not video_path or not os.path.isfile(video_path):
        return False
    try:
        from roop.capturer import _ffprobe_binary, _popen_kwargs
        cmd = [
            _ffprobe_binary(), '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=r_frame_rate,avg_frame_rate',
            '-of', 'json', video_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, timeout=10, **_popen_kwargs())
        import json
        blob = json.loads((proc.stdout or b'{}').decode('utf-8', 'replace'))
        streams = blob.get('streams') or []
        if not streams:
            return False
        st = streams[0]
        r_fps = _parse_fps_fraction(st.get('r_frame_rate', '0'))
        avg_fps = _parse_fps_fraction(st.get('avg_frame_rate', '0'))
        if r_fps > 0 and avg_fps > 0 and abs(r_fps - avg_fps) > 0.05:
            return True
    except Exception as exc:
        _LOGGER.debug("VFR pre-check error for %s: %s", video_path, exc)
    return False


def extract_frames(
    target_path: str,
    trim_frame_start: Any = None,
    trim_frame_end: Any = None,
    fps: float = 24.0,
    **kwargs: Any,
) -> bool:
    util.create_temp(target_path)
    temp_directory_path = util.get_temp_directory_path(target_path)

    try:
        from roop.process_lifecycle import process_lifecycle_manager
        process_lifecycle_manager.register_temp_dir(temp_directory_path)
    except Exception:
        pass

    # FFmpeg's native webp_pipe demuxer cannot decode animated WebP (ANIM/ANMF chunks).
    # Detect animated webp and fall back to PIL-based extraction.
    if target_path.lower().endswith('.webp') and util.is_animated_webp(target_path):
        return _extract_frames_from_animated_webp(target_path, trim_frame_start, trim_frame_end, temp_directory_path)

    vfr_detected = is_variable_frame_rate(target_path)
    if vfr_detected:
        _LOGGER.info(
            "Target media '%s' has Variable Frame Rate (VFR); conforming to Constant Frame Rate (CFR: %.3f fps)",
            os.path.basename(target_path), fps
        )
        print(f"[FFmpeg] Conforming VFR input '{os.path.basename(target_path)}' to CFR ({fps:.2f} fps) via -fps_mode cfr")

    commands = ['-i', target_path, '-q:v', '1', '-pix_fmt', 'rgb24']
    if trim_frame_start is not None and trim_frame_end is not None:
        commands.extend(['-vf', f'trim=start_frame={trim_frame_start}:end_frame={trim_frame_end},fps={fps}'])
    else:
        commands.extend(['-vf', f'fps={fps}'])

    # Conforming to CFR with -fps_mode cfr guarantees exact constant interval frame extraction
    fmt = _get_output_image_format()
    commands.extend(['-fps_mode', 'cfr', os.path.join(temp_directory_path, f'%06d.{fmt}')])
    return run_ffmpeg(commands)


# Aliases for directory resolution
get_temp_frame_path = util.get_temp_directory_path
get_temp_directory_path = util.get_temp_directory_path
get_temp_frame_paths = util.get_temp_frame_paths


def _frames_dir_vf(source_video: Optional[str] = None) -> str:
    """-vf for encoding a directory of decoded PNG/JPG frames.

    Same rule as FFMPEG_VideoWriter (see ffmpeg_writer.color_filter_chain):
    even dimensions (yuv420p needs them), encode with the BT.601 matrix the
    frames were decoded with, stamp the SOURCE's own colour tags and convert
    nothing. The previous chain, `colorspace=bt709:iall=bt601-6-625`, converted
    601->709 on top of the decoder's 601 assumption and shifted every output a
    measured mean 6.1 / max 27 (8-bit) from a BT.709 source."""
    from roop.ffmpeg_writer import color_filter_chain
    tags = None
    if source_video:
        try:
            from roop.capturer import probe_color_tags
            tags = probe_color_tags(source_video)
        except Exception as exc:
            _LOGGER.debug("colour-tag probe failed for %s: %s", source_video, exc)
    return color_filter_chain('trunc(iw/2)*2', 'trunc(ih/2)*2', tags)


def create_video(target_path: str, dest_filename: str, fps: float = 24.0,
                 temp_directory_path: str = None) -> bool:
    if temp_directory_path is None:
        temp_directory_path = util.get_temp_directory_path(target_path)
    vf = _frames_dir_vf(target_path)
    fmt = _get_output_image_format()
    return run_ffmpeg([
        '-framerate', format(float(fps), '.12g'),
        '-i', os.path.join(temp_directory_path,
                           f'%06d.{fmt}'),
        '-c:v', roop.globals.video_encoder,
        *_rate_control(roop.globals.video_encoder, roop.globals.video_quality),
        '-pix_fmt', 'yuv420p', '-vf', vf, '-y', dest_filename,
    ])


def create_gif_from_video(video_path: str, gif_path: str,
                          target_fps: float = None) -> bool:
    """Convert a video file to an optimised animated GIF.

    target_fps — if provided, use this frame rate instead of detecting it from
    the file.  Pass the known fps when converting from an intermediate temp MP4
    so we don't lose the original source timing through a second detect_fps call.
    """
    fps = target_fps if target_fps is not None else util.detect_fps(video_path)
    width, height = util.detect_dimensions(video_path)

    # Keep the larger dimension at its original size; auto-scale the other.
    if width >= height:
        scale = f'{width}:-1'
    else:
        scale = f'-1:{height}'

    return run_ffmpeg([
        '-i', video_path, '-vf',
        f'fps={fps},scale={scale}:flags=lanczos,split[s0][s1];'
        f'[s0]palettegen[p];[s1][p]paletteuse',
        '-loop', '0', gif_path,
    ])


def apply_media_transforms_gif(input_path: str, output_path: str,
                                vf_filters: list, target_fps=None) -> bool:
    """Re-encode an animated GIF with correct palette generation.

    FFmpeg's default GIF encoder uses a poor global palette that introduces
    colour artifacts on grayscale content.  This function uses the two-pass
    palettegen+paletteuse pipeline so the output palette is optimised for the
    actual frame content — exactly the same approach used by create_gif_from_video.

    vf_filters  - list of video filters to apply BEFORE palette generation
                  (e.g. crop, scale, transpose).  May be empty.
    target_fps  - if not None, a fps= filter is prepended so the frame-rate
                  of the output GIF matches the requested value.
    """
    all_filters = list(vf_filters)
    if target_fps is not None:
        all_filters.insert(0, f'fps={target_fps}')

    if all_filters:
        user_chain = ','.join(all_filters) + ','
    else:
        user_chain = ''

    # Two-pass palette approach in a single ffmpeg invocation using filtergraph.
    # The filter chain is: [user filters] → split → palettegen / paletteuse
    vf = f'{user_chain}split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse'
    return run_ffmpeg(['-i', input_path, '-vf', vf, '-loop', '0', output_path])



def create_video_from_gif(gif_path: str, output_path: str) -> bool:
    fps = util.detect_fps(gif_path)
    video_filter = (
        "scale='trunc(in_w/2)*2':'trunc(in_h/2)*2',"
        f"format=yuv420p,fps={format(float(fps), '.12g')}"
    )
    # With shell=False quotes are data, not grouping syntax. Passing literal
    # quotes made FFmpeg parse an unknown filter name beginning with `"`.
    return run_ffmpeg(['-i', gif_path, '-vf', video_filter,
                       '-movflags', '+faststart', '-shortest', output_path])



def resize_video(input_path: str, output_path: str, width: int, height: int) -> bool:
    scale_filter = (
        f'scale={width}:{height}:force_original_aspect_ratio=decrease,'
        f'pad={width}:{height}:(ow-iw)/2:(oh-ih)/2'
    )
    return run_ffmpeg(['-i', input_path, '-vf', scale_filter,
                       '-c:v', roop.globals.video_encoder,
                       *_rate_control(roop.globals.video_encoder, roop.globals.video_quality),
                       '-c:a', 'copy', output_path])


def rotate_media(input_path: str, output_path: str, transform: str) -> bool:
    transform_map = {
        "90° Clockwise":        "transpose=1",
        "90° Counter-clockwise": "transpose=2",
        "180°":                  "transpose=1,transpose=1",
        "Flip Horizontal":       "hflip",
        "Flip Vertical":         "vflip",
    }
    vf = transform_map.get(transform, "transpose=1")
    return run_ffmpeg(['-i', input_path, '-vf', vf, '-c:a', 'copy', output_path])


def change_fps(input_path: str, output_path: str, fps: float) -> bool:
    return run_ffmpeg(['-i', input_path, '-vf', f'fps={fps}',
                       '-c:v', roop.globals.video_encoder,
                       *_rate_control(roop.globals.video_encoder, roop.globals.video_quality),
                       '-c:a', 'copy', output_path])


def crop_media(input_path: str, output_path: str,
               left_pct: float, right_pct: float,
               top_pct: float,  bottom_pct: float) -> bool:
    l, r, t, b = left_pct / 100, right_pct / 100, top_pct / 100, bottom_pct / 100
    crop_filter = (
        f"crop=in_w*(1-{l:.4f}-{r:.4f}):in_h*(1-{t:.4f}-{b:.4f})"
        f":in_w*{l:.4f}:in_h*{t:.4f}"
    )
    return run_ffmpeg(['-i', input_path, '-vf', crop_filter, '-c:a', 'copy', output_path])


def apply_media_transforms(input_path: str, output_path: str,
                           vf_filters: list, is_video: bool) -> bool:
    """Apply a list of -vf filters in a single ffmpeg pass."""
    if not vf_filters:
        return False
    codec   = roop.globals.video_encoder   or 'libx264'
    quality = roop.globals.video_quality   if roop.globals.video_quality is not None else 14
    vf = ','.join(vf_filters)
    args = ['-i', input_path, '-vf', vf]
    if is_video:
        args += ['-c:v', codec, *_rate_control(codec, quality), '-c:a', 'copy']
    args.append(output_path)
    return run_ffmpeg(args)


def extract_audio_wav(input_path: str, output_wav_path: str, sample_rate: int = 16000) -> bool:
    """Mono PCM WAV at *sample_rate* Hz — the driving audio for lip-sync's
    feature-extraction pass. 16kHz is Whisper's native input rate.

    The only audio-decode entry point in this codebase: every other audio
    operation here is a stream-copy mux (restore_audio), never a real decode.
    Staying ffmpeg-subprocess-based keeps that the same for extraction too,
    rather than adding a librosa/soundfile dependency for one caller.
    """
    return run_ffmpeg(['-i', input_path, '-vn', '-ac', '1', '-ar', str(sample_rate),
                       '-f', 'wav', output_wav_path])


def crop_to_fill(source_w: int, source_h: int,
                 target_ratio_w: float, target_ratio_h: float) -> tuple:
    """Centered crop percentages (left, right, top, bottom) that bring
    source_w x source_h to the target_ratio_w:target_ratio_h aspect ratio,
    trimming whichever axis is oversized relative to that ratio — the
    "crop-to-fill" behaviour social platforms use, as opposed to letterboxing.
    All-zero when the source already matches the ratio, or on degenerate input.
    """
    if source_w <= 0 or source_h <= 0 or target_ratio_w <= 0 or target_ratio_h <= 0:
        return (0.0, 0.0, 0.0, 0.0)
    source_ar = source_w / source_h
    target_ar = target_ratio_w / target_ratio_h
    if source_ar > target_ar:
        # Wider than the target ratio: trim width, keep full height.
        keep_w = source_h * target_ar
        trim_pct = max(0.0, (source_w - keep_w) / source_w * 100.0) / 2
        return (trim_pct, trim_pct, 0.0, 0.0)
    if source_ar < target_ar:
        # Taller than the target ratio: trim height, keep full width.
        keep_h = source_w / target_ar
        trim_pct = max(0.0, (source_h - keep_h) / source_h * 100.0) / 2
        return (0.0, 0.0, trim_pct, trim_pct)
    return (0.0, 0.0, 0.0, 0.0)


def export_preset(input_path: str, output_path: str,
                  target_w: int, target_h: int) -> bool:
    """Crop *input_path* to fill the target_w:target_h aspect ratio (centered,
    no letterboxing), scale to that exact resolution, and re-encode in one
    ffmpeg pass using the app's own encoder/quality settings."""
    from roop.capturer import probe_media_dimensions
    dims = probe_media_dimensions(input_path)
    if not dims:
        return False
    source_w, source_h = dims
    l, r, t, b = crop_to_fill(source_w, source_h, target_w, target_h)
    l, r, t, b = l / 100, r / 100, t / 100, b / 100
    filters = [
        f"crop=in_w*(1-{l:.4f}-{r:.4f}):in_h*(1-{t:.4f}-{b:.4f}):in_w*{l:.4f}:in_h*{t:.4f}",
        f"scale={target_w}:{target_h}",
    ]
    return apply_media_transforms(input_path, output_path, filters, is_video=True)


def apply_media_transforms_webp(input_path: str, output_path: str,
                                vf_filters: list, fps: float) -> bool:
    """Process animated webp: decode frames via PIL, pipe through ffmpeg with vf filters.

    FFmpeg cannot reliably decode animated webp files with malformed Exif headers.
    This function bypasses that by loading frames with Pillow and feeding raw BGR
    video into ffmpeg via stdin, applying any vf filters in a single pass.
    Output is always an mp4 (caller must ensure output_path has .mp4 extension).
    """
    import numpy as np
    import cv2
    from PIL import Image

    try:
        fps = float(fps)
    except (TypeError, ValueError):
        fps = 0.0
    if not math.isfinite(fps) or fps <= 0:
        _LOGGER.error("apply_media_transforms_webp: invalid fps %r", fps)
        return False

    # video_encoder/quality may be None if faceswap tab hasn't run yet — use safe defaults
    codec   = roop.globals.video_encoder   or 'libx264'
    quality = roop.globals.video_quality   if roop.globals.video_quality is not None else 14

    # yuv420p requires even dimensions — round odd width/height down before encoding.
    even_scale = 'scale=trunc(iw/2)*2:trunc(ih/2)*2'
    user_vf = ','.join(vf_filters)
    vf = f'{user_vf},{even_scale}' if user_vf else even_scale
    process = None
    drain_thread = None
    stderr_tail = deque(maxlen=_FFMPEG_TAIL_LINES)
    try:
        with Image.open(input_path) as image:
            width, height = int(image.width), int(image.height)
            frame_count = int(getattr(image, 'n_frames', 1))
            if width <= 0 or height <= 0 or frame_count <= 0:
                _LOGGER.error("apply_media_transforms_webp: no frames or zero dimensions")
                return False

            cmd = [
                'ffmpeg', '-hide_banner', '-y',
                '-loglevel', str(roop.globals.log_level or 'error'),
                '-f', 'rawvideo', '-vcodec', 'rawvideo',
                '-s', f'{width}x{height}', '-pix_fmt', 'bgr24',
                '-framerate', format(fps, '.12g'), '-an', '-i', '-',
                '-vf', vf, '-c:v', codec, *_rate_control(codec, quality),
                '-pix_fmt', 'yuv420p', output_path,
            ]
            popen_params = {
                'stdin': subprocess.PIPE,
                'stdout': subprocess.DEVNULL,
                'stderr': subprocess.PIPE,
            }
            if os.name == 'nt':
                popen_params['creationflags'] = 0x08000000  # CREATE_NO_WINDOW
            process = subprocess.Popen(cmd, **popen_params)

            def _drain_stderr() -> None:
                if process.stderr is None:
                    return
                for line in iter(process.stderr.readline, b''):
                    stderr_tail.append(line)
                process.stderr.close()

            drain_thread = threading.Thread(
                target=_drain_stderr, name='webp_ffmpeg_stderr', daemon=True)
            drain_thread.start()
            _LOGGER.info("Streaming %d WebP frames at %.6g fps", frame_count, fps)

            if process.stdin is None:
                raise RuntimeError("FFmpeg stdin pipe was not created")
            for index in range(frame_count):
                image.seek(index)
                frame_rgb = np.asarray(image.convert('RGB'), dtype=np.uint8)
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                process.stdin.write(frame_bgr.tobytes())
            process.stdin.close()
            process.stdin = None

        try:
            finalize_timeout = float(os.environ.get(
                'ROOP_FFMPEG_FINALIZE_TIMEOUT', '60'))
        except ValueError:
            finalize_timeout = 60.0
        returncode = process.wait(timeout=max(1.0, min(finalize_timeout, 600.0)))
        if drain_thread is not None:
            drain_thread.join(timeout=2.0)
        if returncode != 0:
            output = b''.join(stderr_tail).decode(errors='replace')
            _LOGGER.error("Animated WebP FFmpeg export failed with exit code %s:\n%s",
                          returncode, output)
        return returncode == 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        output = b''.join(stderr_tail).decode(errors='replace')
        _LOGGER.error("Animated WebP export failed: %s\n%s", exc, output,
                      exc_info=True)
        return False
    finally:
        if process is not None:
            if process.stdin is not None:
                try:
                    process.stdin.close()
                except OSError:
                    pass
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5.0)
        if drain_thread is not None and drain_thread.is_alive():
            drain_thread.join(timeout=2.0)


def create_video_from_frames_dir(frames_dir: str, output_path: str, fps: float,
                                  image_format: str = 'png',
                                  source_video: Optional[str] = None) -> bool:
    """Re-assemble a video from a directory of sequentially named frame images.

    Frames must follow the %06d.<image_format> naming convention that
    extract_frames() produces (e.g. 000001.png, 000002.png …). *source_video*,
    when known, supplies the colour tags stamped on the output.
    """
    codec   = roop.globals.video_encoder   or 'libx264'
    quality = roop.globals.video_quality   if roop.globals.video_quality is not None else 14
    vf = _frames_dir_vf(source_video)
    return run_ffmpeg([
        '-framerate', format(float(fps), '.12g'),
        '-i',    os.path.join(frames_dir, f'%06d.{image_format}'),
        '-c:v',  codec,
    ] + _rate_control(codec, quality) + [
        '-pix_fmt', 'yuv420p',
        '-vf',   vf,
        '-y',    output_path,
    ])


def create_gif_from_frames_dir(frames_dir: str, output_path: str, fps: float,
                                width: int, height: int,
                                image_format: str = 'png') -> bool:
    """Re-assemble an animated GIF from a directory of sequentially named frame images.

    Uses the two-pass palettegen+paletteuse pipeline for accurate colour reproduction.
    Frames must follow the %06d.<image_format> naming convention.
    """
    if width and height:
        scale = f'{width}:-1' if width >= height else f'-1:{height}'
    else:
        scale = 'iw:ih'   # no-op scale if dimensions are unknown
    vf = (
        f'scale={scale}:flags=lanczos,'
        f'split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse'
    )
    return run_ffmpeg([
        '-framerate', format(float(fps), '.12g'),
        '-i',   os.path.join(frames_dir, f'%06d.{image_format}'),
        '-vf',  vf,
        '-loop', '0',
        output_path,
    ])


def _probe_audio_properties(video_path: str) -> dict:
    """Probe audio sample rate, channels, and bitrate from source media."""
    props = {'sample_rate': None, 'bit_rate': None, 'has_subtitles': False}
    try:
        from roop.capturer import _ffprobe_binary, _popen_kwargs
        cmd = [
            _ffprobe_binary(), '-v', 'error',
            '-show_entries', 'stream=index,codec_type,sample_rate,bit_rate',
            '-of', 'json', video_path
        ]
        proc = subprocess.run(cmd, capture_output=True, timeout=10, **_popen_kwargs())
        import json
        blob = json.loads((proc.stdout or b'{}').decode('utf-8', 'replace'))
        for st in blob.get('streams') or []:
            ctype = st.get('codec_type')
            if ctype == 'audio' and props['sample_rate'] is None:
                sr = st.get('sample_rate')
                if sr:
                    props['sample_rate'] = str(sr)
                br = st.get('bit_rate')
                if br:
                    props['bit_rate'] = str(br)
            elif ctype == 'subtitle':
                props['has_subtitles'] = True
    except Exception as exc:
        _LOGGER.debug("Audio probe error for %s: %s", video_path, exc)
    return props


def restore_audio(
    intermediate_video: Optional[str] = None,
    original_video: Optional[str] = None,
    trim_frame_start: Any = None,
    trim_frame_end: Any = None,
    final_video: Optional[str] = None,
    source_fps: Optional[float] = None,
    *,
    source_path: Optional[str] = None,
    target_path: Optional[str] = None,
    output_path: Optional[str] = None,
    fps: Optional[float] = None,
    **kwargs: Any,
) -> bool:
    """Mux audio, subtitles, and metadata from *original_video* into *intermediate_video*, writing *final_video*.

    Retains:
    1. ALL original audio streams (-map 1:a?)
    2. Original audio sample rates and bitrates without desync or quality loss
    3. Subtitle tracks (-map 1:s?) with stream copy
    4. All global and stream metadata tags (-map_metadata 1)
    5. Conforms output to constant frame rate (-fps_mode cfr) to eliminate drift
    """
    # Map keyword aliases if passed
    if intermediate_video is None and target_path is not None:
        intermediate_video = target_path
    if original_video is None and source_path is not None:
        original_video = source_path
    if final_video is None and output_path is not None:
        final_video = output_path
    if source_fps is None and fps is not None:
        source_fps = fps

    if not intermediate_video or not original_video or not final_video:
        _LOGGER.error("restore_audio missing required paths: intermediate=%r, original=%r, final=%r",
                      intermediate_video, original_video, final_video)
        return False

    try:
        from roop.process_lifecycle import process_lifecycle_manager
        process_lifecycle_manager.register_incomplete_file(final_video)
    except Exception:
        process_lifecycle_manager = None

    effective_fps = float(source_fps if source_fps is not None
                          else util.detect_fps(original_video))
    if not math.isfinite(effective_fps) or effective_fps <= 0:
        _LOGGER.error("Cannot restore audio with invalid source fps %r", effective_fps)
        return False

    # Seek the audio source to match any trim applied before processing
    start_frame = int(trim_frame_start or 0)
    audio_seek = ['-ss', format(start_frame / effective_fps, '.9f')]
    if trim_frame_end is not None:
        duration_frames = int(trim_frame_end) - start_frame
        if duration_frames <= 0:
            _LOGGER.error("Cannot restore audio for empty frame range %s..%s",
                          start_frame, trim_frame_end)
            return False
        audio_seek += ['-t', format(duration_frames / effective_fps, '.9f')]

    extension = os.path.splitext(final_video)[1].lower()
    audio_props = _probe_audio_properties(original_video)

    transcode = []
    if extension in ('.mp4', '.m4v', '.mov'):
        transcode = ['-c:a', 'aac', '-b:a', '256k']
        sub_codec = ['-c:s', 'mov_text']
    elif extension == '.webm':
        transcode = ['-c:a', 'libopus', '-b:a', '160k']
        sub_codec = ['-c:s', 'copy']
    else:
        transcode = ['-c:a', 'copy']
        sub_codec = ['-c:s', 'copy']

    if audio_props.get('sample_rate'):
        transcode.extend(['-ar', audio_props['sample_rate']])

    def _mux(audio_codec: List[str], subtitle_opts: Optional[List[str]] = None) -> bool:
        cmd = (
            ['-i', intermediate_video]
            + audio_seek
            + ['-i', original_video,
               '-c:v', 'copy']
            + audio_codec
            + ['-map', '0:v:0',
               '-map', '1:a?',
               '-map_metadata', '1']
        )
        if subtitle_opts:
            cmd.extend(subtitle_opts)
        cmd.extend(['-fps_mode', 'cfr', '-shortest', final_video])
        return run_ffmpeg(cmd)

    has_subtitles = audio_props.get('has_subtitles', False)
    sub_opts = ['-map', '1:s?', '-c:s', 'copy'] if has_subtitles else None

    # Stream copy first: lossless and instant, and the common case.
    if _mux(['-c:a', 'copy'], sub_opts):
        if process_lifecycle_manager is not None:
            process_lifecycle_manager.unregister_incomplete_file(final_video)
        return True

    # If stream copy failed with subtitles, retry stream copy without subtitles only if subtitles were present
    if has_subtitles and _mux(['-c:a', 'copy']):
        if process_lifecycle_manager is not None:
            process_lifecycle_manager.unregister_incomplete_file(final_video)
        return True

    if not transcode:
        return False

    _LOGGER.warning("Audio stream copy failed for %s; re-encoding audio to %s",
                    os.path.basename(final_video), transcode[1])
    try:
        if os.path.isfile(final_video):
            os.remove(final_video)
    except OSError:
        pass

    sub_transcode = ['-map', '1:s?'] + sub_codec if (has_subtitles and sub_codec) else None
    if _mux(transcode, sub_transcode):
        if process_lifecycle_manager is not None:
            process_lifecycle_manager.unregister_incomplete_file(final_video)
        return True

    if has_subtitles and _mux(transcode):
        if process_lifecycle_manager is not None:
            process_lifecycle_manager.unregister_incomplete_file(final_video)
        return True

    return False
