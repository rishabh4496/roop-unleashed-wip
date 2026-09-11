"""FFmpeg argv, timestamp, cleanup, and bounded-buffer regression tests."""

import inspect
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop import util_ffmpeg  # noqa: E402


class TestConcatArguments(unittest.TestCase):
    def test_filter_concat_passes_each_argument_separately(self):
        with mock.patch.object(util_ffmpeg, 'run_ffmpeg', return_value=True) as run:
            self.assertTrue(util_ffmpeg.join_videos(
                ['first clip.mp4', 'second.mp4'], 'joined.mp4', simple=False))
        args = run.call_args.args[0]
        self.assertEqual(args[:4], ['-i', 'first clip.mp4', '-i', 'second.mp4'])
        filter_graph = args[args.index('-filter_complex') + 1]
        self.assertIn('[0:v:0][0:a:0][1:v:0][1:a:0]', filter_graph)
        self.assertNotIn('"[outv]"', args)
        self.assertIn('[outv]', args)

    def test_concat_list_is_unique_quoted_and_removed(self):
        observed = {}

        def fake_run(args):
            list_path = args[args.index('-i') + 1]
            observed['path'] = list_path
            with open(list_path, encoding='utf-8') as handle:
                observed['content'] = handle.read()
            return True

        with tempfile.TemporaryDirectory() as temp_dir, \
             mock.patch.object(util_ffmpeg.util, 'resolve_relative_path',
                               return_value=temp_dir), \
             mock.patch.object(util_ffmpeg, 'run_ffmpeg', side_effect=fake_run):
            self.assertTrue(util_ffmpeg.join_videos(
                ["clip one's.mp4"], 'joined.mp4', simple=True))
            self.assertFalse(os.path.exists(observed['path']))
        self.assertIn("file '", observed['content'])
        self.assertIn("'\\''", observed['content'])


class TestMediaTimingAndFailurePropagation(unittest.TestCase):
    def test_restore_audio_uses_video_fps_duration_and_container_codec(self):
        with mock.patch.object(util_ffmpeg, 'run_ffmpeg', return_value=True) as run, \
             mock.patch.object(util_ffmpeg.util, 'detect_fps') as detect:
            ok = util_ffmpeg.restore_audio(
                'silent.mp4', 'voice.mp3', 100, 300, 'final.mp4',
                source_fps=30000 / 1001)
        self.assertTrue(ok)
        detect.assert_not_called()
        args = run.call_args.args[0]
        self.assertNotIn('-to', args)
        self.assertEqual(args[args.index('-ss') + 1], '3.336666667')
        self.assertEqual(args[args.index('-t') + 1], '6.673333333')
        # Lossless stream copy is tried first; the container codec is only
        # used when the copy is refused (see the next test).
        self.assertEqual(args[args.index('-c:a') + 1], 'copy')
        self.assertEqual(args[args.index('-c:v') + 1], 'copy')

    def test_restore_audio_transcodes_only_when_stream_copy_is_refused(self):
        # Opus/Vorbis/PCM from a WebM or MKV source cannot be stream-copied
        # into MP4: the first mux fails and the audio is re-encoded to AAC.
        with mock.patch.object(util_ffmpeg, 'run_ffmpeg', side_effect=[False, True]) as run,              mock.patch.object(util_ffmpeg.util, 'detect_fps', return_value=25.0):
            ok = util_ffmpeg.restore_audio('silent.mp4', 'voice.webm', 0, 50, 'final.mp4')
        self.assertTrue(ok)
        self.assertEqual(run.call_count, 2)
        first = run.call_args_list[0].args[0]
        second = run.call_args_list[1].args[0]
        self.assertEqual(first[first.index('-c:a') + 1], 'copy')
        self.assertEqual(second[second.index('-c:a') + 1], 'aac')
        self.assertEqual(second[second.index('-c:v') + 1], 'copy')

    def test_gif_filter_is_not_shell_quoted(self):
        with mock.patch.object(util_ffmpeg.util, 'detect_fps', return_value=12.5), \
             mock.patch.object(util_ffmpeg, 'run_ffmpeg', return_value=True) as run:
            self.assertTrue(util_ffmpeg.create_video_from_gif('in.gif', 'out.mp4'))
        value = run.call_args.args[0][run.call_args.args[0].index('-vf') + 1]
        self.assertFalse(value.startswith('"'))
        self.assertIn('fps=12.5', value)

    def test_create_video_returns_ffmpeg_failure(self):
        cfg = mock.Mock(output_image_format='png')
        with mock.patch.object(util_ffmpeg.roop.globals, 'CFG', cfg), \
             mock.patch.object(util_ffmpeg.roop.globals, 'video_encoder', 'libx264'), \
             mock.patch.object(util_ffmpeg.roop.globals, 'video_quality', 14), \
             mock.patch.object(util_ffmpeg, 'run_ffmpeg', return_value=False):
            self.assertFalse(util_ffmpeg.create_video(
                'source.mp4', 'output.mp4', 30000 / 1001, 'frames'))

    def test_webp_export_has_no_full_animation_accumulator(self):
        source = inspect.getsource(util_ffmpeg.apply_media_transforms_webp)
        self.assertNotIn('frames = []', source)
        self.assertNotIn("b''.join(f.tobytes()", source)
        self.assertIn('process.stdin.write(frame_bgr.tobytes())', source)
        self.assertIn("deque(maxlen=_FFMPEG_TAIL_LINES)", source)


if __name__ == '__main__':
    unittest.main(verbosity=2)
