"""The processing box's live frame — throttled, downscaled, encoded once.

This exists on the hot swap path: `_publish_live` is called for every frame the
pipeline finishes. The previous version of that idea kept a full-frame copy per
frame and was removed for cost, so the properties that make it affordable are
the ones worth pinning:

  * a publish never costs more than a throttle check unless the interval elapsed;
  * what is stored is small (downscaled + JPEG), so the API hands back bytes;
  * a bad frame can never raise into the render;
  * a new run starts blank rather than showing the previous run's last frame.
"""

import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop import live_preview as lp                       # noqa: E402


def _frame(w=1920, h=1080):
    y, x = np.mgrid[0:h, 0:w].astype('float32')
    return np.dstack([(x / 8) % 255, (y / 8) % 255, ((x + y) / 12) % 255]).astype('uint8')


class LivePreviewTest(unittest.TestCase):

    def setUp(self):
        lp.reset()

    def tearDown(self):
        lp.reset()

    def _publish_now(self, frame):
        """Publish ignoring the throttle (which the next test covers directly)."""
        lp._state['t'] = 0.0
        lp.publish(frame)

    def test_starts_blank(self):
        self.assertEqual(lp.seq(), 0)
        self.assertIsNone(lp.snapshot()[0])

    def test_publishes_a_decodable_jpeg(self):
        self._publish_now(_frame())
        data, seq, size = lp.snapshot()
        self.assertTrue(data)
        self.assertEqual(seq, 1)
        self.assertEqual(size, (1920, 1080))          # source size, for the caption
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        self.assertIsNotNone(img, 'stored bytes are not a valid JPEG')

    def test_downscales(self):
        self._publish_now(_frame())
        img = cv2.imdecode(np.frombuffer(lp.snapshot()[0], np.uint8), cv2.IMREAD_COLOR)
        self.assertLessEqual(img.shape[1], lp._MAX_W)
        self.assertGreater(img.shape[1], 100)
        # aspect preserved (16:9 within a pixel of rounding)
        self.assertAlmostEqual(img.shape[1] / img.shape[0], 1920 / 1080, delta=0.02)

    def test_small_frames_are_not_upscaled(self):
        self._publish_now(_frame(w=320, h=180))
        img = cv2.imdecode(np.frombuffer(lp.snapshot()[0], np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual((img.shape[1], img.shape[0]), (320, 180))

    def test_throttles(self):
        """Back-to-back publishes must collapse to one — this is called per frame."""
        f = _frame(w=640, h=360)
        self._publish_now(f)
        for _ in range(50):
            lp.publish(f)
        self.assertEqual(lp.seq(), 1)

    def test_publish_never_raises(self):
        for bad in (None, np.zeros((0, 0, 3), np.uint8), 'not a frame', np.zeros((4,), np.uint8)):
            lp._state['t'] = 0.0
            lp.publish(bad)                      # must not raise
        self.assertEqual(lp.seq(), 0, 'a malformed frame must not be published')

    def test_reset_clears_for_the_next_run(self):
        self._publish_now(_frame(w=640, h=360))
        self.assertEqual(lp.seq(), 1)
        lp.reset()
        self.assertEqual(lp.seq(), 0)
        self.assertIsNone(lp.snapshot()[0])

    def test_seq_is_what_the_ui_keys_on(self):
        """It must advance once per published frame, so the <img> URL changes."""
        f = _frame(w=640, h=360)
        for expected in (1, 2, 3):
            self._publish_now(f)
            self.assertEqual(lp.seq(), expected)


if __name__ == '__main__':
    unittest.main()
