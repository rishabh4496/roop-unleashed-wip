"""Comprehensive Verification Suite for the Zero-Overhead High-Throughput Performance Stack.

Tests:
1. PinnedBufferPool & get_crop_buffer: zero-copy pinned host memory allocation and reuse.
2. CudaOrtIOBinding: pre-allocated contiguous GPU tensor binding helper.
3. cuda_warp_affine & cuda_laplacian_pyramid_blend: CUDA PyTorch-accelerated warping & blending.
4. FaceSet V2 & Caching: deterministic 512-D embedding serialization and .fsz reading.
5. diou_nms: Distance-IoU suppression with center-distance penalty.
6. HardwareProfiler & RuntimeOptimizer: hardware probe, auto-tuning, and NVENC/worker policies.
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock
import numpy as np
import cv2

# Add app root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from roop.buffer_pool import PinnedBufferPool, allocate_pinned_buffer, get_crop_buffer
from roop.utilities import (
    CudaOrtIOBinding,
    cuda_warp_affine,
    cuda_laplacian_pyramid_blend,
)
from roop.face_detector import compute_diou, compute_diou_matrix, diou_nms
from roop.faceset_v2 import (
    write_faceset_v2,
    read_faceset_archive,
    FORMAT_NAME,
    FORMAT_VERSION,
)
from roop.FaceSet import FaceSet
from roop.runtime_optimizer import (
    HardwareProfiler,
    WorkloadProfiler,
    RuntimeOptimizer,
)


class TestBufferPoolAndPinnedMemory(unittest.TestCase):
    """Test suite for pinned buffer allocation and thread-safe buffer reuse."""

    def test_allocate_pinned_buffer(self):
        buf = allocate_pinned_buffer((1080, 1920, 3), dtype=np.uint8)
        self.assertIsInstance(buf, np.ndarray)
        self.assertEqual(buf.shape, (1080, 1920, 3))
        self.assertEqual(buf.dtype, np.uint8)

    def test_pinned_buffer_pool_lease_and_release(self):
        pool = PinnedBufferPool((512, 512, 3), capacity=2, dtype=np.uint8)
        self.assertEqual(pool.available_count, 2)

        buf1 = pool.acquire()
        self.assertEqual(buf1.shape, (512, 512, 3))
        self.assertEqual(pool.available_count, 1)

        buf2 = pool.acquire()
        self.assertEqual(pool.available_count, 0)

        # Releasing buffer restores capacity
        pool.release(buf1)
        self.assertEqual(pool.available_count, 1)

        # Acquiring again yields a buffer from the pool
        buf3 = pool.acquire()
        self.assertEqual(pool.available_count, 0)

        pool.release(buf2)
        pool.release(buf3)
        self.assertEqual(pool.available_count, 2)

    def test_get_crop_buffer(self):
        b512 = get_crop_buffer(512)
        self.assertEqual(b512.shape, (512, 512, 3))
        b256 = get_crop_buffer(256)
        self.assertEqual(b256.shape, (256, 256, 3))
        b128 = get_crop_buffer(128)
        self.assertEqual(b128.shape, (128, 128, 3))


class TestCudaOrtIOBinding(unittest.TestCase):
    """Test suite for CudaOrtIOBinding helper functions and fallback behavior."""

    def test_cuda_ort_io_binding_has_cuda_provider(self):
        mock_session_cuda = MagicMock()
        mock_session_cuda.get_providers.return_value = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.assertTrue(CudaOrtIOBinding._has_cuda_provider(mock_session_cuda))

        mock_session_cpu = MagicMock()
        mock_session_cpu.get_providers.return_value = ['CPUExecutionProvider']
        self.assertFalse(CudaOrtIOBinding._has_cuda_provider(mock_session_cpu))

    def test_shape_for_concrete_and_dynamic(self):
        meta_concrete = MagicMock()
        meta_concrete.shape = [1, 3, 512, 512]
        self.assertEqual(CudaOrtIOBinding._shape_for(meta_concrete, batch=1), (1, 3, 512, 512))

        meta_dynamic = MagicMock()
        meta_dynamic.shape = ['batch', 3, 512, 512]
        self.assertEqual(CudaOrtIOBinding._shape_for(meta_dynamic, batch=4), (4, 3, 512, 512))


class TestGpuAcceleratedWarpAndBlend(unittest.TestCase):
    """Test suite for PyTorch CUDA grid_sample warping and Laplacian pyramid blending."""

    def setUp(self):
        self.has_cuda = torch.cuda.is_available()

    def test_cuda_warp_affine_correctness(self):
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (150, 150), (255, 128, 64), -1)

        # Affine translation matrix (dx=20, dy=10)
        M = np.array([[1.0, 0.0, 20.0],
                      [0.0, 1.0, 10.0]], dtype=np.float32)

        # OpenCV reference
        ref = cv2.warpAffine(img, M, (256, 256), flags=cv2.INTER_LINEAR)

        # CUDA accelerated warp
        res = cuda_warp_affine(img, M, (256, 256))

        self.assertEqual(res.shape, (256, 256, 3))
        self.assertEqual(res.dtype, np.uint8)

        # Difference should be minimal (bilinear sampling rounding difference <= 5)
        diff = np.abs(res.astype(np.int32) - ref.astype(np.int32))
        self.assertLess(np.mean(diff), 2.0)

    def test_cuda_laplacian_pyramid_blend(self):
        # Create base image and target template
        img1 = np.full((256, 256, 3), 50, dtype=np.uint8)
        img2 = np.full((256, 256, 3), 200, dtype=np.uint8)

        # Smooth center mask
        mask = np.zeros((256, 256), dtype=np.float32)
        cv2.circle(mask, (128, 128), 64, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (31, 31), 10.0)

        blended = cuda_laplacian_pyramid_blend(img1, img2, mask, levels=3)

        self.assertIsNotNone(blended)
        self.assertEqual(blended.shape, (256, 256, 3))
        self.assertEqual(blended.dtype, np.uint8)
        # Center should be close to img2 (200), outer region close to img1 (50)
        self.assertGreater(blended[128, 128, 0], 150)
        self.assertLess(blended[10, 10, 0], 70)


class TestFaceSetV2MetadataCaching(unittest.TestCase):
    """Test suite for deterministic ArcFace 512-D embedding serialization and .fsz reading."""

    def test_faceset_v2_archive_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = os.path.join(tmpdir, "test_faceset.fsz")

            # Create synthetic FaceSet with image and embedding
            fs = FaceSet()
            rng = np.random.RandomState(42)
            img = (rng.rand(128, 128, 3) * 255).astype(np.uint8)
            emb = rng.randn(512).astype(np.float32)
            emb /= np.linalg.norm(emb)

            face = {
                'embedding': emb,
                'normed_embedding': emb,
                'bbox': [20, 20, 100, 100],
                'kps': [[30, 40], [70, 40], [50, 60], [35, 80], [65, 80]],
                'det_score': 0.95,
                'pose': [0.0, 0.0, 0.0],
            }
            fs.faces = [face]
            fs.ref_images = [img]

            # Write archive
            write_faceset_v2(archive_path, fs, [img], source_name="test_identity")
            self.assertTrue(os.path.isfile(archive_path))

            # Read back archive
            read_meta = read_faceset_archive(archive_path)
            self.assertIsNotNone(read_meta)
            self.assertEqual(read_meta.get("schema"), FORMAT_NAME)
            self.assertEqual(int(read_meta.get("version")), FORMAT_VERSION)

            # Attach to a new FaceSet
            fs2 = FaceSet()
            fs2.attach_v2_metadata(read_meta)
            self.assertIsNotNone(fs2.faceset_metadata)
            self.assertIsNotNone(fs2.default_embedding)
            self.assertEqual(fs2.default_embedding.shape, (512,))

            # Cosine similarity between original and cached embedding
            cos_sim = float(np.dot(emb, fs2.default_embedding))
            self.assertAlmostEqual(cos_sim, 1.0, places=5)


class TestDiouNms(unittest.TestCase):
    """Test suite for Distance-IoU suppression and matrix computation."""

    def test_compute_diou_identical_and_disjoint(self):
        box1 = np.array([0, 0, 100, 100], dtype=np.float32)
        box2 = np.array([0, 0, 100, 100], dtype=np.float32)
        # Identical boxes: IoU=1.0, center_dist=0 -> DIoU=1.0
        self.assertAlmostEqual(compute_diou(box1, box2), 1.0, places=5)

        box3 = np.array([200, 200, 300, 300], dtype=np.float32)
        # Disjoint boxes: IoU=0.0, center_penalty > 0 -> DIoU < 0
        self.assertLess(compute_diou(box1, box3), 0.0)

    def test_diou_nms_suppression(self):
        # Two highly overlapping boxes: second should be suppressed
        dets = np.array([
            [10.0, 10.0, 100.0, 100.0, 0.95],
            [12.0, 11.0, 102.0, 101.0, 0.85],
        ], dtype=np.float32)
        kpss = np.zeros((2, 5, 2), dtype=np.float32)

        keep_dets, keep_kpss, indices = diou_nms(dets, kpss, iou_thresh=0.5)
        self.assertEqual(len(indices), 1)
        self.assertEqual(indices[0], 0)

    def test_diou_nms_adjacent_faces_preserved(self):
        # Two side-by-side touching faces: IoU is low and centers are separated
        dets = np.array([
            [10.0, 10.0, 100.0, 100.0, 0.95],
            [95.0, 10.0, 185.0, 100.0, 0.90],
        ], dtype=np.float32)
        kpss = np.zeros((2, 5, 2), dtype=np.float32)

        keep_dets, keep_kpss, indices = diou_nms(dets, kpss, iou_thresh=0.5)
        self.assertEqual(len(indices), 2)


class TestHardwareProfilerAndOptimizer(unittest.TestCase):
    """Test suite for HardwareProfiler, RuntimeOptimizer, and hardware-aware tuning."""

    def test_hardware_profiler(self):
        profiler = HardwareProfiler()
        hw = profiler.profile()
        self.assertIsNotNone(hw)
        self.assertGreater(hw.cpu_logical_cores, 0)
        self.assertIsInstance(hw.nvenc_available, bool)
        self.assertIsInstance(hw.nvdec_available, bool)

    def test_runtime_optimizer_tuning(self):
        optimizer = RuntimeOptimizer()
        workload = WorkloadProfiler().profile(
            source_video="",
            frame_count=300,
            resolution=(1920, 1080),
            faces_per_frame=1.0,
        )
        profile = optimizer.build_profile(workload, save=False)
        self.assertIsNotNone(profile)
        self.assertIsNotNone(profile.tuning)
        self.assertGreater(profile.tuning.worker_count, 0)
        self.assertIn(profile.tuning.encoder, ("hevc_nvenc", "h264_nvenc", "libx264"))

    def test_apply_environment(self):
        optimizer = RuntimeOptimizer()
        workload = WorkloadProfiler().profile(source_video="", resolution=(1280, 720))
        profile = optimizer.build_profile(workload, save=False)
        env = RuntimeOptimizer.apply_environment(profile)
        self.assertIn("ROOP_RUNTIME_WORKER_COUNT", env)
        self.assertIn("ROOP_RUNTIME_ENCODER", env)


if __name__ == "__main__":
    unittest.main()
