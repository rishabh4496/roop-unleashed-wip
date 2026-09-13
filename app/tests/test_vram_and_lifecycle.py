"""Unit and Integration Tests for VRAM Guard, Offloading, Process Lifecycle, and Abort Recovery.

Verifies:
1. ModelLifecycleManager VRAM threshold checking, LRU inactive model offloading.
2. execution_guard & recursive batch-halving fallback on OutOfMemoryError.
3. ProcessLifecycleManager process tree termination (taskkill on Windows), temp cleanup,
   and GPU VRAM recovery after abort.
"""

import gc
import os
import sys
import tempfile
import time
import unittest
import subprocess
from unittest.mock import MagicMock, patch

# Ensure app root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from roop.model_lifecycle import ModelLifecycleManager, ModelMetadata
from roop.process_lifecycle import ProcessLifecycleManager


class TestVRAMGuardAndModelOffloading(unittest.TestCase):
    """Test suite for Dynamic VRAM Guard and LRU Model Offloading."""

    def setUp(self):
        self.mgr = ModelLifecycleManager(vram_threshold_gb=1.5)

    def test_vram_detection(self):
        """Verify free VRAM reporting returns a non-negative float."""
        free_gb = self.mgr.get_free_vram_gb()
        self.assertIsInstance(free_gb, float)
        self.assertGreaterEqual(free_gb, 0.0)

    def test_model_registration_and_access(self):
        """Verify models can be registered, tracked, and accessed."""
        unload_mock = MagicMock()
        self.mgr.register_model("inswapper", unload_mock, device="cuda")

        meta = self.mgr.get_model_meta("inswapper")
        self.assertIsNotNone(meta)
        self.assertEqual(meta.name, "inswapper")
        self.assertTrue(meta.is_loaded)

        self.mgr.touch("inswapper")
        meta_updated = self.mgr.get_model_meta("inswapper")
        self.assertGreater(meta_updated.last_used, 0.0)

    def test_sequential_offloading_under_vram_pressure(self):
        """Simulate low VRAM and verify that inactive models are sequentially offloaded LRU-first."""
        unloaded_order = []

        def make_unload_cb(name):
            def cb():
                unloaded_order.append(name)
            return cb

        self.mgr.register_model("codeformer", make_unload_cb("codeformer"), device="cuda")
        time.sleep(0.02)
        self.mgr.register_model("gfpgan", make_unload_cb("gfpgan"), device="cuda")
        time.sleep(0.02)
        self.mgr.register_model("inswapper", make_unload_cb("inswapper"), device="cuda")

        # Touch codeformer so gfpgan is the oldest (LRU)
        time.sleep(0.02)
        self.mgr.touch("codeformer")

        # Mock free VRAM as 0.8 GB (below 1.5 GB threshold) until offloaded
        with patch.object(self.mgr, "get_free_vram_gb", side_effect=[0.8, 0.8, 1.2, 1.8, 1.8, 1.8]):
            # Request 1.5 GB free VRAM excluding 'inswapper'
            freed = self.mgr.ensure_vram(required_gb=1.5, exclude={"inswapper"})
            self.assertTrue(freed)
            # gfpgan was oldest, then codeformer
            self.assertIn("gfpgan", unloaded_order)
            self.assertEqual(unloaded_order[0], "gfpgan")
            self.assertNotIn("inswapper", unloaded_order)

    def test_batch_halving_on_oom(self):
        """Verify run_with_batch_fallback halves batch size recursively on OOM."""
        inputs = list(range(16))
        call_batches = []

        def mock_infer(batch):
            call_batches.append(len(batch))
            if len(batch) > 4:
                raise torch.cuda.OutOfMemoryError("Simulated CUDA OOM")
            return [x * 2 for x in batch]

        results = self.mgr.run_with_batch_fallback(mock_infer, inputs, initial_batch_size=16, min_batch_size=1)
        self.assertEqual(len(results), 16)
        self.assertEqual(results, [x * 2 for x in range(16)])
        # Batch of 16 failed -> split into 8 & 8 -> both 8 failed -> split into 4s -> succeeded
        self.assertIn(16, call_batches)
        self.assertIn(8, call_batches)
        self.assertIn(4, call_batches)

    def test_execution_guard_context(self):
        """Verify execution_guard marks active state and updates timestamps."""
        self.mgr.register_model("test_model", lambda: None, device="cuda")

        with self.mgr.execution_guard("test_model", required_gb=1.5):
            meta = self.mgr.get_model_meta("test_model")
            self.assertTrue(meta.is_active)

        meta = self.mgr.get_model_meta("test_model")
        self.assertFalse(meta.is_active)


class TestProcessLifecycleAndAbort(unittest.TestCase):
    """Test suite for Clean Process Lifecycle, Abort, and Garbage Collection."""

    def setUp(self):
        self.proc_mgr = ProcessLifecycleManager()

    def test_process_tracking_and_tree_termination(self):
        """Spawn child processes and verify tree termination terminates all children."""
        if sys.platform == "win32":
            proc = subprocess.Popen(["ping", "127.0.0.1", "-n", "30"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            proc = subprocess.Popen(["sleep", "30"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        self.proc_mgr.register_process(proc, label="test_sleep")
        self.assertIn(proc, self.proc_mgr.active_processes)

        # Terminate all
        self.proc_mgr.terminate_all(reason="Test abort")
        time.sleep(0.5)

        # Verify process has terminated
        poll_res = proc.poll()
        self.assertIsNotNone(poll_res, "Child process was not terminated")
        self.assertEqual(len(self.proc_mgr.active_processes), 0)

    def test_temp_cleanup_and_incomplete_file_purge(self):
        """Verify temp directories and incomplete files are purged on abort."""
        temp_dir = tempfile.mkdtemp(prefix="roop_test_frames_")
        test_file = os.path.join(temp_dir, "frame_0001.png")
        with open(test_file, "w") as f:
            f.write("dummy")

        incomplete_file = tempfile.mktemp(prefix="roop_incomplete_", suffix=".mp4")
        with open(incomplete_file, "w") as f:
            f.write("partial video data")

        self.proc_mgr.register_temp_dir(temp_dir)
        self.proc_mgr.register_incomplete_file(incomplete_file)

        self.assertTrue(os.path.isdir(temp_dir))
        self.assertTrue(os.path.isfile(incomplete_file))

        self.proc_mgr.terminate_all(reason="Test cleanup")

        self.assertFalse(os.path.exists(temp_dir), "Temp directory was not purged")
        self.assertFalse(os.path.exists(incomplete_file), "Incomplete output file was not purged")

    def test_vram_recovery_after_abort(self):
        """Verify GPU VRAM is completely recovered when terminate_all is called."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA GPU not available for VRAM test")

        device_id = 0
        torch.cuda.empty_cache()
        gc.collect()

        initial_free = torch.cuda.mem_get_info(device_id)[0]

        # Allocate 500 MB tensor on GPU
        big_tensor = torch.empty((125, 1024, 1024), dtype=torch.float32, device="cuda")
        allocated_free = torch.cuda.mem_get_info(device_id)[0]
        self.assertLess(allocated_free, initial_free - (400 * 1024 * 1024))

        # Register a release callback that drops the tensor
        def mock_release():
            nonlocal big_tensor
            del big_tensor
            gc.collect()

        self.proc_mgr.register_release_hook(mock_release)

        # Trigger termination & resource release
        self.proc_mgr.terminate_all(reason="Test VRAM abort")

        recovered_free = torch.cuda.mem_get_info(device_id)[0]
        diff_mb = abs(recovered_free - initial_free) / (1024 * 1024)
        self.assertLess(diff_mb, 50.0, f"VRAM did not recover: diff {diff_mb:.1f} MB")


if __name__ == "__main__":
    unittest.main(verbosity=2)
