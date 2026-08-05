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
    INPUT_SIZE, crop_box_to_local, face_bbox_crop, resolve_device)


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
        crop, crop_box = face_bbox_crop(frame, (100, 50, 300, 250))
        self.assertEqual(crop.shape, (INPUT_SIZE, INPUT_SIZE, 3))

    def test_crop_box_reflects_the_margin_and_is_in_frame_coords(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        crop, crop_box = face_bbox_crop(frame, (100, 50, 300, 250), extra_margin=10)
        self.assertEqual(crop_box, (100, 50, 300, 260))  # y2 + extra_margin

    def test_bbox_is_clamped_to_frame_bounds(self):
        # A bbox that overruns the frame on every side must still produce a
        # valid crop rather than an empty/negative slice.
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        crop, crop_box = face_bbox_crop(frame, (-50, -50, 150, 150))
        self.assertIsNotNone(crop)
        self.assertEqual(crop.shape, (INPUT_SIZE, INPUT_SIZE, 3))
        self.assertEqual(crop_box, (0, 0, 100, 100))

    def test_degenerate_bbox_returns_none(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        self.assertEqual(face_bbox_crop(frame, (200, 200, 250, 250)), (None, None))  # entirely outside
        self.assertEqual(face_bbox_crop(frame, (50, 50, 40, 40)), (None, None))      # x2<x1, y2<y1


class TestCropBoxToLocal(unittest.TestCase):
    """The geometry that keeps the composited mouth actually mouth-shaped:
    mapping the full-frame mouth bounding box into the local pixel space of
    the 256x256 face crop MuseTalk generated, so only that sub-region gets
    pasted — not the whole generated face squashed into the mouth box."""

    def test_maps_a_region_at_the_crop_origin(self):
        # crop_box is a 200x200 region resized to 256x256 -> scale 1.28x.
        local = crop_box_to_local((100, 100, 300, 300), (100, 100, 150, 150))
        self.assertEqual(local, (0, 0, 64, 64))

    def test_maps_a_region_in_the_middle_of_the_crop(self):
        local = crop_box_to_local((0, 0, 256, 256), (64, 64, 192, 192))
        self.assertEqual(local, (64, 64, 192, 192))  # identity scale (256/256)

    def test_region_partly_outside_the_crop_is_clamped_not_dropped(self):
        # Mouth padding can extend slightly below a tight detector bbox even
        # with extra_margin; this must degrade gracefully, not explode.
        local = crop_box_to_local((0, 0, 256, 256), (200, 200, 300, 300))
        self.assertIsNotNone(local)
        lx1, ly1, lx2, ly2 = local
        self.assertLessEqual(lx2, INPUT_SIZE)
        self.assertLessEqual(ly2, INPUT_SIZE)

    def test_region_entirely_outside_the_crop_returns_none(self):
        self.assertIsNone(crop_box_to_local((0, 0, 256, 256), (300, 300, 400, 400)))

    def test_degenerate_crop_box_returns_none(self):
        self.assertIsNone(crop_box_to_local((100, 100, 100, 100), (0, 0, 50, 50)))
        self.assertIsNone(crop_box_to_local(None, (0, 0, 50, 50)))
        self.assertIsNone(crop_box_to_local((0, 0, 100, 100), None))


class TestResolveDevice(unittest.TestCase):
    """roop.utilities.get_device() names ONNX execution providers ('cuda',
    'mps', 'mkl' for OpenVINO, 'cpu'), not torch devices — 'mkl' passed
    straight to torch.device() raises, and since Initialize() never sets
    _ready, that failure repeats on every subsequent frame rather than once."""

    def test_cuda_and_mps_pass_through(self):
        self.assertEqual(resolve_device('cuda', cuda_available=True), 'cuda')
        self.assertEqual(resolve_device('mps', cuda_available=False), 'mps')

    def test_onnx_only_provider_names_fall_back_to_a_real_torch_device(self):
        self.assertEqual(resolve_device('mkl', cuda_available=True), 'cuda')
        self.assertEqual(resolve_device('mkl', cuda_available=False), 'cpu')

    def test_none_falls_back_to_a_real_torch_device(self):
        self.assertEqual(resolve_device(None, cuda_available=True), 'cuda')
        self.assertEqual(resolve_device(None, cuda_available=False), 'cpu')


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
        block = self.src.split('# ── Lip-sync (post-composite)', 1)[1][:2600]
        self.assertIn("getattr(roop.globals, 'lipsync_enabled', False)", block)

    def test_processface_is_mutually_exclusive_with_restore_original_mouth(self):
        block = self.src.split('# ── Lip-sync (post-composite)', 1)[1][:2600]
        self.assertIn('not self.options.restore_original_mouth', block)

    def test_audio_cache_setup_is_itself_gated(self):
        block = self.src.split("self._lipsync_fps = fps", 1)[1][:1200]
        self.assertIn("getattr(roop.globals, 'lipsync_enabled', False)", block)

    def test_both_gates_read_the_same_global_not_processoptions(self):
        """Regression guard: the audio-cache setup (run_batch_inmem) originally
        gated on self.options.lipsync_enabled, which ProcessOptions never
        defines — getattr's False default meant the cache was NEVER built, so
        turning the toggle on did nothing, silently, forever. Both gates must
        read the same roop.globals attribute or they can drift apart again."""
        cache_block = self.src.split("self._lipsync_fps = fps", 1)[1][:1200]
        face_block = self.src.split('# ── Lip-sync (post-composite)', 1)[1][:2600]
        self.assertNotIn("self.options.lipsync_enabled", cache_block)
        for block in (cache_block, face_block):
            self.assertIn("getattr(roop.globals, 'lipsync_enabled', False)", block)

    def test_lipsync_stage_never_costs_a_frame(self):
        # The whole block must be inside a try/except that falls back silently
        # — same contract every other opt-in post-composite stage here follows.
        block = self.src.split('# ── Lip-sync (post-composite)', 1)[1][:2600]
        self.assertIn('try:', block)
        self.assertIn('except Exception as e:', block)
        self.assertIn('bar_write', block)


if __name__ == "__main__":
    unittest.main()
