"""Lip-sync — the parts that are provable without the MuseTalk weights.

roop.lipsync_audio's frame/time mapping and Lipsync_MuseTalk.face_bbox_crop
are plain functions kept out of the model class specifically so this suite
can exercise them with no onnxruntime/torch session — same rationale as
test_expression_restore.py testing Expression_LivePortrait's geometry helpers
directly rather than through a live model.

The default-off no-op guarantee is proven the same way
test_enhancer_align.py proves ProcessMgr composites against the frame: by
inspecting ProcessMgr's own source for the gate, since actually running
process_face() needs a full detector + swapper pipeline this suite has no
business standing up just to prove a boolean short-circuits.
"""

import os
import sys
import unittest

import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)

from roop.lipsync_audio import (                                # noqa: E402
    AudioFeatureCache, audio_index_for_frame, frame_time)
from roop.processors.Lipsync_MuseTalk import (                  # noqa: E402
    INPUT_SIZE, face_bbox_crop)


class TestFrameTime(unittest.TestCase):
    def test_zero_offset_at_25fps(self):
        self.assertAlmostEqual(frame_time(0, 0, 25.0), 0.0)
        self.assertAlmostEqual(frame_time(0, 25, 25.0), 1.0)
        self.assertAlmostEqual(frame_time(0, 50, 25.0), 2.0)

    def test_frame_start_offsets_into_the_original_clip(self):
        # A trimmed clip starting at frame 100 of a 30fps source: frame_idx 0
        # of the trim is 100/30s into the original audio, not 0s.
        self.assertAlmostEqual(frame_time(100, 0, 30.0), 100 / 30)
        self.assertAlmostEqual(frame_time(100, 30, 30.0), 130 / 30)

    def test_zero_or_negative_fps_does_not_explode(self):
        self.assertEqual(frame_time(0, 10, 0.0), 0.0)
        self.assertEqual(frame_time(0, 10, -5.0), 0.0)


class TestAudioIndexForFrame(unittest.TestCase):
    def test_maps_and_clamps_within_range(self):
        self.assertEqual(audio_index_for_frame(0.0, 50.0, 100), 0)
        self.assertEqual(audio_index_for_frame(1.0, 50.0, 100), 50)
        self.assertEqual(audio_index_for_frame(10.0, 50.0, 100), 99)   # clamped high

    def test_negative_time_clamps_to_zero(self):
        self.assertEqual(audio_index_for_frame(-1.0, 50.0, 100), 0)

    def test_degenerate_inputs_return_zero(self):
        self.assertEqual(audio_index_for_frame(1.0, 50.0, 0), 0)
        self.assertEqual(audio_index_for_frame(1.0, 0.0, 100), 0)

    def test_rounds_to_nearest_not_truncates(self):
        # 0.99 chunks in should round to chunk 1, not floor to 0.
        self.assertEqual(audio_index_for_frame(0.0198, 50.0, 100), 1)


class TestAudioFeatureCache(unittest.TestCase):
    def test_features_for_time_indexes_correctly(self):
        feats = np.arange(10 * 4).reshape(10, 4).astype(np.float32)
        cache = AudioFeatureCache(feats, audio_fps=10.0)
        self.assertEqual(cache.num_chunks, 10)
        np.testing.assert_array_equal(cache.features_for_time(0.0), feats[0])
        np.testing.assert_array_equal(cache.features_for_time(0.5), feats[5])
        np.testing.assert_array_equal(cache.features_for_time(100.0), feats[9])  # clamped

    def test_empty_cache_returns_none(self):
        cache = AudioFeatureCache(None, audio_fps=10.0)
        self.assertEqual(cache.num_chunks, 0)
        self.assertIsNone(cache.features_for_time(1.0))


class TestFaceBboxCrop(unittest.TestCase):
    def test_crop_is_resized_to_input_size(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        crop = face_bbox_crop(frame, (100, 50, 300, 250))
        self.assertEqual(crop.shape, (INPUT_SIZE, INPUT_SIZE, 3))

    def test_bbox_is_clamped_to_frame_bounds(self):
        # A bbox that overruns the frame on every side must still produce a
        # valid crop rather than an empty/negative slice.
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        crop = face_bbox_crop(frame, (-50, -50, 150, 150))
        self.assertIsNotNone(crop)
        self.assertEqual(crop.shape, (INPUT_SIZE, INPUT_SIZE, 3))

    def test_degenerate_bbox_returns_none(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        self.assertIsNone(face_bbox_crop(frame, (200, 200, 250, 250)))  # entirely outside
        self.assertIsNone(face_bbox_crop(frame, (50, 50, 40, 40)))      # x2<x1, y2<y1


class TestDefaultOffIsANoOp(unittest.TestCase):
    """Mirrors test_enhancer_align.py's source-inspection approach: proving
    the gate exists and that nothing upstream of it (Lipsync_MuseTalk
    construction, the audio cache, the mouth-crop math) runs unless the
    setting is explicitly on."""

    def setUp(self):
        src_path = os.path.join(APP, 'roop', 'ProcessMgr.py')
        self.src = open(src_path, encoding='utf-8').read()

    def test_default_is_off(self):
        import roop.globals as g
        self.assertFalse(g.lipsync_enabled)
        self.assertEqual(g.lipsync_audio_source, 'original')

    def test_processface_gates_on_the_global(self):
        block = self.src.split('# ── Lip-sync (post-composite)', 1)[1][:2200]
        self.assertIn("getattr(roop.globals, 'lipsync_enabled', False)", block)

    def test_processface_is_mutually_exclusive_with_restore_original_mouth(self):
        block = self.src.split('# ── Lip-sync (post-composite)', 1)[1][:2200]
        self.assertIn('not self.options.restore_original_mouth', block)

    def test_audio_cache_setup_is_itself_gated(self):
        block = self.src.split("self._lipsync_fps = fps", 1)[1][:1200]
        self.assertIn("getattr(self.options, 'lipsync_enabled', False)", block)

    def test_lipsync_stage_never_costs_a_frame(self):
        # The whole block must be inside a try/except that falls back silently
        # — same contract every other opt-in post-composite stage here follows.
        block = self.src.split('# ── Lip-sync (post-composite)', 1)[1][:2200]
        self.assertIn('try:', block)
        self.assertIn('except Exception as e:', block)
        self.assertIn('bar_write', block)


if __name__ == "__main__":
    unittest.main()
