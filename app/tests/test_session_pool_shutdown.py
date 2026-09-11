"""SessionPool cancellation/release races must never strand CUDA workers."""

import os
import sys
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop.session_pool import SessionPool  # noqa: E402


class TestSessionPoolShutdown(unittest.TestCase):
    def test_invalid_size_is_rejected_instead_of_deadlocking(self):
        with self.assertRaises(ValueError):
            SessionPool(lambda index: index, 0)

    def test_waiter_is_unblocked_when_pool_is_released(self):
        pool = SessionPool(lambda index: object(), 1)
        started = threading.Event()
        finished = threading.Event()
        errors = []

        def waiter():
            started.set()
            try:
                with pool.lease():
                    self.fail('a waiter acquired a released pool')
            except RuntimeError as exc:
                errors.append(str(exc))
            finally:
                finished.set()

        with pool.lease():
            thread = threading.Thread(target=waiter)
            thread.start()
            self.assertTrue(started.wait(1.0))
            pool.release()
            self.assertTrue(finished.wait(2.0),
                            'blocked Queue.get did not observe pool release')
        thread.join(timeout=1.0)
        self.assertFalse(thread.is_alive())
        self.assertTrue(errors)

    def test_active_item_is_not_returned_after_release(self):
        pool = SessionPool(lambda index: {'slot': index}, 1)
        with pool.lease() as item:
            self.assertEqual(item['slot'], 0)
            pool.release()
        with self.assertRaises(RuntimeError):
            with pool.lease():
                pass

    def test_acquire_timeout_is_bounded(self):
        pool = SessionPool(lambda index: object(), 1)
        with pool.lease():
            with self.assertRaises(TimeoutError):
                with pool.lease(timeout=0.01):
                    pass
        pool.release()


if __name__ == '__main__':
    unittest.main(verbosity=2)
