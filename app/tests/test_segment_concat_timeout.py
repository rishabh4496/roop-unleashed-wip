"""Regression test for the unbounded concat subprocess in SegmentedVideoWriter.

_concat ran `subprocess.run(cmd, capture_output=True)` with no timeout, unlike
every other subprocess.run in the codebase (capturer:235 and nvdec_reader:82 use
30s, bench:1325 uses 300s). A concat that never returns — a corrupt segment, a
stalled network drive, an ffmpeg wedged on a bad moov — hung the process forever
at the very END of a long render, with every frame already encoded.

The right failure is ok=False, because that keeps the segments AND the manifest
on disk, so the run stays resumable rather than being silently discarded.
"""
import os
import subprocess
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import roop.globals                                        # noqa: E402
from roop.segment_writer import SegmentedVideoWriter       # noqa: E402


def _writer(tmpdir, segments):
    """A SegmentedVideoWriter with `segments` already committed, without
    encoding anything (no ffmpeg, no GPU)."""
    w = SegmentedVideoWriter.__new__(SegmentedVideoWriter)
    w.target_video = os.path.join(tmpdir, 'out.mp4')
    w._dir = tmpdir
    w._seg_prefix = '.out.seg'
    w._seg_ext = '.mp4'
    w.segments = segments
    w._writer = None
    return w


class TestConcatTimeout(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp(prefix='roop_seg_')
        # Two plausible segment files so the size-based budget has something
        # real to measure.
        self.segs = []
        for i in range(2):
            name = f'.out.seg{i:04d}.mp4'
            with open(os.path.join(self.tmp, name), 'wb') as fh:
                fh.write(b'\0' * 4096)
            self.segs.append({'file': name, 'frames': 500})

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_a_timeout_is_passed(self):
        seen = {}

        def fake_run(cmd, **kw):
            seen.update(kw)
            return subprocess.CompletedProcess(cmd, 0, b'', b'')

        w = _writer(self.tmp, self.segs)
        with mock.patch('roop.segment_writer.subprocess.run', side_effect=fake_run):
            self.assertTrue(w._concat())
        self.assertIn('timeout', seen,
                      'concat must not run an unbounded subprocess')
        # Generous enough that a real stream copy never trips it.
        self.assertGreaterEqual(seen['timeout'], 300.0)

    def test_timeout_budget_scales_with_output_size(self):
        """A flat timeout would be wrong: this is a copy whose duration scales
        with the bytes being copied."""
        seen = {}

        def fake_run(cmd, **kw):
            seen['timeout'] = kw.get('timeout')
            return subprocess.CompletedProcess(cmd, 0, b'', b'')

        big = os.path.join(self.tmp, '.out.seg0002.mp4')
        with open(big, 'wb') as fh:
            fh.write(b'\0' * (32 * 1024 * 1024))            # 32 MB
        segs = self.segs + [{'file': '.out.seg0002.mp4', 'frames': 500}]

        w = _writer(self.tmp, segs)
        with mock.patch('roop.segment_writer.subprocess.run', side_effect=fake_run):
            w._concat()
        small_budget = 300.0
        self.assertGreaterEqual(seen['timeout'], small_budget)

    def test_a_hung_concat_fails_closed_and_keeps_the_segments(self):
        """TimeoutExpired must become ok=False, not an escaped exception —
        close() uses that to decide whether to delete the parts."""
        def fake_run(cmd, **kw):
            raise subprocess.TimeoutExpired(cmd, kw.get('timeout', 0))

        w = _writer(self.tmp, self.segs)
        with mock.patch('roop.segment_writer.subprocess.run', side_effect=fake_run):
            self.assertFalse(w._concat(), 'a timed-out concat must report failure')
        # The whole point: the frames are still on disk to resume from.
        for s in self.segs:
            self.assertTrue(os.path.isfile(os.path.join(self.tmp, s['file'])),
                            f"{s['file']} was removed after a failed concat")

    def test_failed_concat_leaves_parts_for_resume_via_close(self):
        """close() must not call cleanup() when the concat failed."""
        w = _writer(self.tmp, list(self.segs))
        roop.globals.processing = True                      # 'completed' run
        with mock.patch('roop.segment_writer.subprocess.run',
                        side_effect=subprocess.TimeoutExpired('ffmpeg', 300)):
            w.close()
        for s in self.segs:
            self.assertTrue(os.path.isfile(os.path.join(self.tmp, s['file'])),
                            'a failed concat must never delete the segments')

    def test_the_list_file_is_cleaned_up_even_on_timeout(self):
        w = _writer(self.tmp, self.segs)
        with mock.patch('roop.segment_writer.subprocess.run',
                        side_effect=subprocess.TimeoutExpired('ffmpeg', 300)):
            w._concat()
        leftovers = [f for f in os.listdir(self.tmp) if f.endswith('list.txt')]
        self.assertEqual(leftovers, [], f'concat list file left behind: {leftovers}')


if __name__ == '__main__':
    unittest.main(verbosity=2)
