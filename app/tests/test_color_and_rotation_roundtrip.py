"""Two silent media-path corruptions, each pinned by a measurement.

1. Colour. Frames come off cv2.VideoCapture / the ffmpeg bgr24 pipe through a
   BT.601 matrix regardless of the file's tag. The writer used to re-interpret
   them as 601 and CONVERT them to 709 (`colorspace=bt709:iall=bt601-6-625`),
   which on a genuine BT.709 source left the output a mean 6.1 / max 27 (8-bit)
   from the source in a tag-aware player, and every optional re-encode of our
   own tagged output (post_swap upscale / interpolation) stacked another ~6.
   The fix encodes with the decoder's matrix and only COPIES the tags. This
   test performs the actual round trip through ffmpeg and bounds the error.

2. Rotation. cv2 applies a file's rotation side-data (CAP_PROP_ORIENTATION_AUTO
   defaults to on); nvdec_reader passed `-noautorotate`, so on a 90-degree
   tagged portrait clip the pipe delivered the unrotated byte stream and
   reshaped it into cv2's rotated dimensions — same byte count, mean abs error
   ~126 against the real frame, no exception anywhere. The test builds such a
   file and requires the two readers to agree pixel for pixel.

Both need the ffmpeg/ffprobe binaries; they skip cleanly without them.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cv2  # noqa: E402

from roop import capturer  # noqa: E402
from roop.ffmpeg_writer import FFMPEG_VideoWriter, color_filter_chain  # noqa: E402
from roop.nvdec_reader import FFmpegVideoReader  # noqa: E402

_HAVE_FFMPEG = shutil.which('ffmpeg') is not None and shutil.which('ffprobe') is not None

W, H = 256, 256


def _pattern():
    img = np.zeros((H, W, 3), np.uint8)
    img[..., 0] = np.linspace(0, 255, W)[None, :]
    img[..., 1] = np.linspace(0, 255, H)[:, None]
    img[..., 2] = 128
    img[64:192, 64:192] = (140, 170, 220)        # a skin-ish patch
    return img


def _ffmpeg(*args, stdin=None):
    proc = subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', *args],
                          input=stdin, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode('utf-8', 'replace'))
    return proc.stdout


def _decode_as(path, matrix):
    """One frame, decoded the way a tag-aware player would for *matrix*."""
    out = _ffmpeg('-i', path, '-frames:v', '1', '-vf', f'scale=in_color_matrix={matrix}',
                  '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-')
    return np.frombuffer(out, np.uint8).reshape(H, W, 3)


@unittest.skipUnless(_HAVE_FFMPEG, 'ffmpeg/ffprobe not on PATH')
class ColourRoundTrip(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix='roop_colour_')
        capturer._probe_cache.clear()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _bt709_source(self):
        """A file the way an HD camera writes one: 709 matrix, 709 tags."""
        path = os.path.join(self.dir, 'src709.mp4')
        img = _pattern()
        _ffmpeg('-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', '25', '-i', '-',
                '-vcodec', 'libx264', '-crf', '0', '-preset', 'ultrafast',
                '-vf', 'scale=out_color_matrix=bt709,format=yuv444p,'
                       'setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709',
                path, stdin=img.tobytes() * 3)
        return path, img

    def test_writer_preserves_a_bt709_source_within_chroma_rounding(self):
        src, truth = self._bt709_source()
        tags = capturer.probe_color_tags(src)
        self.assertEqual(tags.get('colorspace'), 'bt709')

        cap = cv2.VideoCapture(src)
        ok, frame = cap.read()
        cap.release()
        self.assertTrue(ok)
        # What the pipeline works on: cv2's 601 reading of a 709 file.
        self.assertGreater(np.abs(frame.astype(int) - truth.astype(int)).mean(), 4.0)

        out = os.path.join(self.dir, 'out.mp4')
        writer = FFMPEG_VideoWriter(out, (W, H), 25, codec='libx264', crf=0,
                                    color_tags=tags)
        for _ in range(3):
            writer.write_frame(frame)
        writer.close()

        shown = _decode_as(out, 'bt709')
        err = np.abs(shown.astype(int) - truth.astype(int)).mean()
        # 4:2:0 chroma subsampling on hard synthetic edges costs ~2.6 here; the
        # old convert path measured 6.4 on this exact pattern.
        self.assertLess(err, 3.5, f'player-visible error {err:.2f} vs source')

        # ...and the tags travelled.
        self.assertEqual(capturer.probe_color_tags(out).get('colorspace'), 'bt709')

    def test_second_pass_over_own_output_does_not_drift(self):
        """post_swap re-reads our tagged output with cv2 and re-encodes it."""
        src, _ = self._bt709_source()
        tags = capturer.probe_color_tags(src)
        cap = cv2.VideoCapture(src)
        _, frame = cap.read()
        cap.release()
        first = os.path.join(self.dir, 'pass1.mp4')
        w = FFMPEG_VideoWriter(first, (W, H), 25, codec='libx264', crf=0, color_tags=tags)
        for _ in range(3):
            w.write_frame(frame)
        w.close()
        pass1 = _decode_as(first, 'bt709')

        cap = cv2.VideoCapture(first)
        _, frame2 = cap.read()
        cap.release()
        second = os.path.join(self.dir, 'pass2.mp4')
        w = FFMPEG_VideoWriter(second, (W, H), 25, codec='libx264', crf=0,
                               color_tags=capturer.probe_color_tags(first))
        for _ in range(3):
            w.write_frame(frame2)
        w.close()
        pass2 = _decode_as(second, 'bt709')
        drift = np.abs(pass2.astype(int) - pass1.astype(int)).mean()
        # 4:2:0 re-subsampling of the hard-edged pattern costs ~2.2; the old
        # convert chain measured 5.57 per extra pass (11.93 vs source after
        # two) on this exact input.
        self.assertLess(drift, 3.0, f'second pass drifted {drift:.2f} levels')

    def test_tag_aware_pipe_reader_hands_the_models_the_true_colours(self):
        """The swap pass reads through nvdec_reader with tag_aware=True: a
        BT.709 file is decoded with the 709 matrix (cv2 cannot do this), and the
        writer is told to encode with the same one."""
        from roop.nvdec_reader import FFmpegVideoReader, decode_matrix_for
        src, truth = self._bt709_source()
        matrix = decode_matrix_for(src)
        self.assertEqual(matrix, 'bt709')
        reader = FFmpegVideoReader(src, W, H, 25.0, hwaccel=None, decode_matrix=matrix)
        try:
            ok, frame = reader.read()
        finally:
            reader.release()
        self.assertTrue(ok)
        seen = np.abs(frame.astype(int) - truth.astype(int)).mean()
        self.assertLess(seen, 1.0, f'the pipeline sees {seen:.2f} off the source (cv2: ~6.0)')

        out = os.path.join(self.dir, 'out709.mp4')
        writer = FFMPEG_VideoWriter(out, (W, H), 25, codec='libx264', crf=0,
                                    color_tags=capturer.probe_color_tags(src),
                                    decode_matrix=matrix)
        for _ in range(3):
            writer.write_frame(frame)
        writer.close()
        shown = _decode_as(out, 'bt709')
        err = np.abs(shown.astype(int) - truth.astype(int)).mean()
        self.assertLess(err, 3.0, f'player-visible error {err:.2f} vs source')

    def test_untagged_source_stays_untagged(self):
        chain = color_filter_chain(W, H, None)
        self.assertNotIn('setparams', chain)
        self.assertIn('out_color_matrix=bt601', chain)
        self.assertIn('out_color_matrix=bt709', color_filter_chain(W, H, None, decode_matrix='bt709'))
        self.assertIn('out_color_matrix=bt601', color_filter_chain(W, H, None, decode_matrix='nonsense'))
        chain = color_filter_chain('trunc(iw/2)*2', 'trunc(ih/2)*2',
                                   {'colorspace': 'bt709', 'color_trc': 'garbage'})
        self.assertIn('setparams=colorspace=bt709', chain)
        self.assertNotIn('garbage', chain)

    def test_writer_refuses_a_mis_sized_frame(self):
        out = os.path.join(self.dir, 'bad.mp4')
        writer = FFMPEG_VideoWriter(out, (W, H), 25, codec='libx264', crf=0)
        try:
            with self.assertRaises(ValueError):
                writer.write_frame(np.zeros((H // 2, W, 3), np.uint8))
        finally:
            writer.close()


@unittest.skipUnless(_HAVE_FFMPEG, 'ffmpeg/ffprobe not on PATH')
class RotationTagAgreement(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix='roop_rot_')
        capturer._probe_cache.clear()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_pipe_reader_matches_cv2_on_a_rotated_clip(self):
        plain = os.path.join(self.dir, 'plain.mp4')
        rotated = os.path.join(self.dir, 'rot90.mp4')
        _ffmpeg('-f', 'lavfi', '-i', 'testsrc=size=320x180:rate=25:duration=1',
                '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', plain)
        # The way phones write portrait video: landscape samples + a rotation tag.
        _ffmpeg('-display_rotation', '90', '-i', plain, '-c', 'copy', rotated)

        cap = cv2.VideoCapture(rotated)
        cw, ch = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ok, cv_frame = cap.read()
        cap.release()
        self.assertTrue(ok)
        self.assertEqual((cw, ch), (180, 320), 'cv2 is expected to apply the rotation tag')

        info = capturer._probe_video(rotated)
        self.assertEqual((info['w'], info['h']), (cw, ch),
                         'the probe must report DISPLAY dimensions, like cv2')
        self.assertEqual(info['rotation'] % 180, 90)

        reader = FFmpegVideoReader(rotated, cw, ch, 25.0, hwaccel=None)
        try:
            ok, pipe_frame = reader.read()
        finally:
            reader.release()
        self.assertTrue(ok)
        self.assertEqual(pipe_frame.shape, cv_frame.shape)
        err = np.abs(pipe_frame.astype(int) - cv_frame.astype(int)).mean()
        self.assertLess(err, 1.0, f'pipe and cv2 disagree by {err:.1f} levels (was ~126)')


if __name__ == '__main__':
    unittest.main()
