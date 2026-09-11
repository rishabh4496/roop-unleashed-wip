"""The live preview must not build session pools it can never drive.

core.live_swap runs one frame at a time under _preview_lock, so a pool of N
sessions in the preview's ProcessMgr is N-1 idle TensorRT engines per model.
On a shared 12GB card that oversubscription paged the process's allocations to
system RAM and turned a ~0.8s preview into 5-70s. session_pool.single_context
is what stops it; these pin its shape:

- every per-ProcessMgr query answers 1 inside it (swapper/enhancer, mask,
  expression pools), so Initialize() takes the unpooled path;
- the process-wide pools (FaceAnalysis, hybrid detectors) keep the configured
  size — a 1-wide analyser built by the first preview after boot would
  serialise every later consumer, and _ensure_face_analyser does not rebuild
  on a size mismatch;
- it is thread-local and restored on exit, so a render's workers never see it.
"""

import os
import sys
import threading
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop import session_pool  # noqa: E402


CONFIGURED = {'trt': 4, 'detmask': 4}


class TestPreviewSingleContext(unittest.TestCase):
    def setUp(self):
        self._pools = mock.patch.object(session_pool, '_resolve_pools',
                                        return_value=CONFIGURED)
        self._expr = mock.patch.dict(os.environ, {'ROOP_EXPR_POOL': '2'})
        self._pools.start()
        self._expr.start()

    def tearDown(self):
        self._expr.stop()
        self._pools.stop()

    def test_per_processmgr_pools_collapse_to_one(self):
        self.assertTrue(session_pool.pooling_enabled())
        self.assertTrue(session_pool.detmask_pooling_enabled())
        self.assertTrue(session_pool.expression_pooling_enabled())
        with session_pool.single_context():
            self.assertEqual(session_pool.pool_size(), 1)
            self.assertFalse(session_pool.pooling_enabled())
            self.assertEqual(session_pool.detmask_pool_size(), 1)
            self.assertFalse(session_pool.detmask_pooling_enabled())
            self.assertEqual(session_pool.expression_pool_size(), 1)
            self.assertFalse(session_pool.expression_pooling_enabled())
        self.assertEqual(session_pool.pool_size(), 4)
        self.assertEqual(session_pool.detmask_pool_size(), 4)
        self.assertEqual(session_pool.expression_pool_size(), 2)

    def test_shared_pools_keep_their_configured_size(self):
        with session_pool.single_context():
            self.assertEqual(session_pool.detmask_pool_size(shared=True), 4)
            self.assertTrue(session_pool.detmask_pooling_enabled(shared=True))
            self.assertEqual(session_pool.detector_pool_size(), 4)

    def test_override_is_thread_local(self):
        seen = {}
        inside = threading.Event()
        release = threading.Event()

        def worker():
            seen['worker'] = session_pool.pool_size()
            inside.set()
            release.wait(2.0)

        with session_pool.single_context():
            thread = threading.Thread(target=worker)
            thread.start()
            self.assertTrue(inside.wait(2.0))
            seen['main'] = session_pool.pool_size()
            release.set()
            thread.join(2.0)
        self.assertEqual(seen, {'worker': 4, 'main': 1})

    def test_nested_use_restores_the_outer_state(self):
        with session_pool.single_context():
            with session_pool.single_context():
                self.assertEqual(session_pool.pool_size(), 1)
            self.assertEqual(session_pool.pool_size(), 1)
        self.assertEqual(session_pool.pool_size(), 4)

    def test_restored_after_an_exception(self):
        with self.assertRaises(RuntimeError):
            with session_pool.single_context():
                raise RuntimeError('preview failed')
        self.assertEqual(session_pool.pool_size(), 4)


if __name__ == '__main__':
    unittest.main()
