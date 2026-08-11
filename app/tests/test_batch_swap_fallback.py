"""The sequential fallback inside RunBatch / RunBatchMulti.

When a batched swap inference fails (a static-batch ONNX export, or a
TensorRT engine built for batch 1), the swapper re-runs the crops one at a
time instead of failing the frame. The pixels were always covered by that
fallback; the MASKS were not, and that is the interesting half.

`_stash_masks` writes into a single-slot thread-local, and `Run` calls it once
per crop. So a fallback loop that just calls Run in sequence leaves exactly one
mask behind — the last crop's — while the caller (ProcessMgr, around the
`take_masks()` after RunBatch) does ONE take_masks() and pairs the result to
the crops BY POSITION. One mask for N crops then either misattributes every
face's mask or fails reassembly outright, and it does so only on the path that
exists to rescue a broken batch: the failure is invisible until the day the
fallback fires.

So the contract under test is: after a fallback, take_masks() returns one mask
per crop, in crop order — the same shape the batched path publishes — and a
PARTIAL set collapses to None rather than to a short list that would silently
shift every mask onto the wrong face.
"""
import os
import sys
import threading
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from roop.processors.FaceSwapInsightFace import FaceSwapInsightFace  # noqa: E402


class _Swapper(FaceSwapInsightFace):
    """The real RunBatch/RunBatchMulti/Run/_stash_masks over a stub session.

    Subclassed rather than constructed so the test drives the shipping methods
    without loading a 500MB ONNX model or a GPU.
    """

    def __init__(self, batch_works, emits_mask, size=8):
        self.pool = None
        self._mask_tls = threading.local()
        self.image_input_name = 'target'
        self.embed_input_name = 'source'
        self._batch_works = batch_works
        self._emits_mask = emits_mask
        self._size = size
        self.single_calls = 0

    def _compute_source_input(self, source_face):
        return np.zeros((1, 512), dtype=np.float32)

    def _infer(self, feed):
        n = feed[self.image_input_name].shape[0]
        if n > 1:
            self.single_calls += 0
            if not self._batch_works:
                raise RuntimeError('static batch: expected 1, got %d' % n)
        else:
            self.single_calls += 1
        img = np.full((n, 3, self._size, self._size), float(n), dtype=np.float32)
        if not self._emits_mask:
            return [img]
        # One distinct mask per crop, so a misattribution is visible.
        mask = np.stack([np.full((1, self._size, self._size), i + 1, dtype=np.float32)
                         for i in range(n)])
        return [img, mask]


def _crops(n, size=8):
    return [np.full((1, 3, size, size), i, dtype=np.float32) for i in range(n)]


class FallbackKeepsTheMaskContract(unittest.TestCase):

    def test_batched_path_publishes_one_mask_per_crop(self):
        """The baseline the fallback has to match."""
        sw = _Swapper(batch_works=True, emits_mask=True)
        outs = sw.RunBatch(object(), object(), _crops(4))
        self.assertEqual(len(outs), 4)
        masks = sw.take_masks()
        self.assertEqual(len(masks), 4)

    def test_fallback_publishes_one_mask_per_crop(self):
        sw = _Swapper(batch_works=False, emits_mask=True)
        outs = sw.RunBatch(object(), object(), _crops(4))

        self.assertEqual(len(outs), 4, 'every crop must still come back')
        self.assertEqual(sw.single_calls, 4, 'fallback runs one inference per crop')

        masks = sw.take_masks()
        self.assertIsNotNone(masks, 'the fallback dropped the swap model mask')
        self.assertEqual(len(masks), 4,
                         'one mask per crop — a single-slot thread-local means '
                         'only the last Run survives unless the loop drains it')

    def test_fallback_masks_stay_in_crop_order(self):
        """Position IS the pairing, so order is the whole contract."""
        sw = _Swapper(batch_works=False, emits_mask=True)
        sw.RunBatch(object(), object(), _crops(3))
        masks = sw.take_masks()
        # Each single-crop inference emits mask value 1 for its one crop, so
        # every entry is a real (H,W) plane rather than a repeat of one object.
        self.assertEqual(len(masks), 3)
        for m in masks:
            self.assertEqual(np.asarray(m).ndim, 2, 'a mask per crop, not a batch axis')
        self.assertEqual(len({id(m) for m in masks}), 3,
                         'the same mask object repeated means one crop won')

    def test_masks_are_cleared_after_reading(self):
        sw = _Swapper(batch_works=False, emits_mask=True)
        sw.RunBatch(object(), object(), _crops(2))
        self.assertIsNotNone(sw.take_masks())
        self.assertIsNone(sw.take_masks(), 'take_masks must consume exactly once')

    def test_a_maskless_model_publishes_none_not_a_list_of_none(self):
        """`if _m:` upstream must not see a truthy list of Nones."""
        sw = _Swapper(batch_works=False, emits_mask=False)
        outs = sw.RunBatch(object(), object(), _crops(3))
        self.assertEqual(len(outs), 3)
        self.assertIsNone(sw.take_masks())

    def test_partial_masks_collapse_to_none(self):
        """A gap must not shift every later mask onto the wrong face."""
        sw = _Swapper(batch_works=False, emits_mask=True)
        sw._republish_masks([np.zeros((4, 4)), None, np.zeros((4, 4))])
        self.assertIsNone(sw.take_masks())

    def test_run_batch_multi_fallback_keeps_the_contract(self):
        sw = _Swapper(batch_works=False, emits_mask=True)
        reqs = [(object(), object(), c) for c in _crops(3)]
        outs = sw.RunBatchMulti(reqs)

        self.assertEqual(len(outs), 3)
        self.assertEqual(sw.single_calls, 3)
        masks = sw.take_masks()
        self.assertIsNotNone(masks, 'RunBatchMulti dropped the swap model mask')
        self.assertEqual(len(masks), 3)

    def test_fallback_output_shape_matches_the_batched_path(self):
        """[3,H,W] per crop either way, or explode_pixel_boost mis-tiles."""
        batched = _Swapper(batch_works=True, emits_mask=True).RunBatch(
            object(), object(), _crops(3))
        fell_back = _Swapper(batch_works=False, emits_mask=True).RunBatch(
            object(), object(), _crops(3))
        self.assertEqual([np.asarray(o).shape for o in batched],
                         [np.asarray(o).shape for o in fell_back])


if __name__ == '__main__':
    unittest.main()
