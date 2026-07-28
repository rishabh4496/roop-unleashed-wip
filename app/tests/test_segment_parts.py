"""The output-parts registry behind the console's part tabs.

A resumable render writes numbered part files into the output folder and merges
them at the end. Those parts are the run's chapters in the UI, so what matters
here is that their FRAME NUMBERING is right — a part labelled "frames
6001-7000" is how a user finds the stretch of video that went wrong, and a
resumed run must continue the numbering rather than restart at 1.

The registry is observational only: nothing it records feeds encoding or resume
(those stay driven by the manifest), so these tests drive it directly rather
than spawning ffmpeg.
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop import segment_writer as sw                      # noqa: E402


def _writer(tmp, **kw):
    return sw.SegmentedVideoWriter(os.path.join(tmp, "out.mp4"), (64, 64), 30.0,
                                   source_video=os.path.join(tmp, "src.mp4"), **kw)


class PartsRegistryTest(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        sw.reset_parts()

    def tearDown(self):
        self._tmp.cleanup()
        sw.reset_parts()

    def test_starts_empty(self):
        self.assertEqual(sw.parts_snapshot(), [])
        self.assertEqual(sw.current_part_index(), 0)

    def test_finalized_parts_number_frames_continuously(self):
        w = _writer(self.tmp)
        w._register(1, "a.mp4", 1000, done=True)
        w._register(2, "b.mp4", 1000, done=True)
        w._register(3, "c.mp4", 640, done=True)
        got = [(p["index"], p["first"], p["last"]) for p in sw.parts_snapshot()]
        self.assertEqual(got, [(1, 1, 1000), (2, 1001, 2000), (3, 2001, 2640)])
        self.assertTrue(all(p["done"] for p in sw.parts_snapshot()))
        self.assertEqual(sw.current_part_index(), 3)

    def test_open_part_reports_frames_written_so_far(self):
        w = _writer(self.tmp)
        w._register(1, "a.mp4", 1000, done=True)
        sw._current = {"index": 2, "file": "b.mp4", "first": 1001, "_written": 412,
                       "bytes": 0, "done": False, "inherited": False}
        try:
            live = sw.parts_snapshot()[-1]
            self.assertEqual((live["index"], live["frames"], live["first"], live["last"]),
                             (2, 412, 1001, 1412))
            self.assertFalse(live["done"])
            self.assertEqual(sw.current_part_index(), 2)   # log lines tag part 2
        finally:
            sw._current = None

    def test_resumed_run_continues_the_numbering(self):
        """A resume inherits finished parts; the next one must not restart at 1."""
        w = _writer(self.tmp)
        with open(sw.manifest_path(w.target_video), encoding="utf-8") as fh:
            manifest = json.load(fh)
        for name, n in (("s0.mp4", 1000), ("s1.mp4", 1000)):
            open(os.path.join(self.tmp, name), "wb").close()
            manifest["segments"].append({"file": name, "frames": n})
        with open(sw.manifest_path(w.target_video), "w", encoding="utf-8") as fh:
            json.dump(manifest, fh)

        w2 = _writer(self.tmp)                        # same identity → resumes
        self.assertEqual(w2.resume_frames, 2000)
        inherited = sw.parts_snapshot()
        self.assertEqual([(p["index"], p["first"], p["last"]) for p in inherited],
                         [(1, 1, 1000), (2, 1001, 2000)])
        self.assertTrue(all(p["inherited"] for p in inherited))

        w2._register(3, "s2.mp4", 500, done=True)
        self.assertEqual(sw.parts_snapshot()[-1]["first"], 2001)

    def test_a_new_run_clears_the_previous_run_s_parts(self):
        w = _writer(self.tmp)
        w._register(1, "a.mp4", 1000, done=True)
        sw.reset_parts()
        self.assertEqual(sw.parts_snapshot(), [])
        self.assertEqual(sw.current_part_index(), 0)


class CounterLineTest(unittest.TestCase):
    """The rule that keeps per-frame counters OUT of the console history.

    Duplicated from api.py's regex rather than imported, because importing api
    loads the whole model stack. The point of the test is the classification: a
    line that changes every poll is pinned, anything that happens once is kept.
    """

    import re as _re
    RE = _re.compile(r"\d[\d,]*\s*/\s*\d[\d,]*|\bfps\b|\d+\s*%", _re.I)

    def assert_pinned(self, msg):
        self.assertTrue(self.RE.search(msg), f"expected pinned: {msg!r}")

    def assert_logged(self, msg):
        self.assertIsNone(self.RE.search(msg), f"expected logged: {msg!r}")

    def test_frame_counters_are_pinned(self):
        for m in ("Processing frame 6412 / 109376 (20.3 FPS)",
                  "Upscaling frame 12 / 300",
                  "Interpolating frame 5 / 60",
                  "Extracting frames 45%"):
            self.assert_pinned(m)

    def test_one_off_events_are_logged(self):
        for m in ("▶ Starting job…", "✓ Done", "⚠ ffmpeg failed",
                  "Combining (encode + audio)…", "Analyzing faces", "Paused",
                  "✓ part 3 written · frames 2001–3000 · 41 MB"):
            self.assert_logged(m)

    def test_regex_matches_the_one_api_uses(self):
        """Keep the duplicate honest — if api.py's rule moves, this must too."""
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               '..', 'api.py'), encoding='utf-8') as fh:
            src = fh.read()
        self.assertIn(r"_COUNTER_RE = _re.compile(r\"\d[\d,]*\s*/\s*\d[\d,]*|\bfps\b|\d+\s*%\""
                      .replace('\\"', '"'), src)


if __name__ == '__main__':
    unittest.main()
