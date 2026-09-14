"""Unit tests for Stage 4: React UI Communication, Telemetry & Memory Safety.

Validates:
1. Compilation progress event schema:
   {"status": "compiling_engine", "model": "GPEN-Realistic", "provider": "TensorRT", "estimated_time": "2-4 minutes"}
2. Standardized runtime telemetry logging format:
   [INFO] Enhancer session active: GPEN Realistic | Provider: TensorrtExecutionProvider (FP16: True, Engine Cache: HIT)
3. Memory leak prevention on re-initialization (explicit session deletion + gc.collect)
4. Silent CPU fallback prevention (forces CUDA fallback if TRT drops to CPU)
5. REST and WebSocket compilation endpoints in FastAPI backend
"""

import gc
import json
import unittest
from unittest.mock import MagicMock, patch

from roop.trt_events import (
    TensorRTCompilationMonitor,
    broadcast_compilation_event,
    get_compilation_event,
    normalize_model_label,
)
from roop.face_enhancer import (
    FaceEnhancer,
    log_enhancer_telemetry,
    create_resilient_session,
)


class TestStage4CompilationSignaling(unittest.TestCase):
    """Test WebSocket/REST compilation events and telemetry formatting."""

    def test_01_normalize_model_label(self):
        self.assertEqual(normalize_model_label("GPENRealistic-512"), "GPEN-Realistic")
        self.assertEqual(normalize_model_label("gpen_bfr_1024"), "GPEN-Realistic")
        self.assertEqual(normalize_model_label("ultramax"), "UltraMax")
        self.assertEqual(normalize_model_label("codeformer_fp16"), "CodeFormer")
        self.assertEqual(normalize_model_label("restoreformer++"), "RestoreFormer++")
        self.assertEqual(normalize_model_label("gfpgan_v1.4"), "GFPGAN")

    def test_02_compilation_start_event_payload(self):
        """Assert exact payload dispatched on compilation start."""
        monitor = TensorRTCompilationMonitor(label="GPENRealistic-512", interval_sec=10.0)
        monitor.start()
        try:
            ev = get_compilation_event()
            self.assertIsNotNone(ev)
            self.assertEqual(ev.get("status"), "compiling_engine")
            self.assertEqual(ev.get("model"), "GPEN-Realistic")
            self.assertEqual(ev.get("provider"), "TensorRT")
            self.assertEqual(ev.get("estimated_time"), "2-4 minutes")
            self.assertIn("elapsed_sec", ev)
        finally:
            monitor.finish(success=True)

        finished_ev = get_compilation_event()
        self.assertIsNotNone(finished_ev)
        self.assertEqual(finished_ev.get("status"), "engine_ready")
        self.assertEqual(finished_ev.get("model"), "GPEN-Realistic")
        self.assertEqual(finished_ev.get("provider"), "TensorRT")

    def test_03_compilation_failure_event_payload(self):
        """Assert payload dispatched on compilation abort/failure."""
        monitor = TensorRTCompilationMonitor(label="UltraMax", interval_sec=10.0)
        monitor.start()
        monitor.finish(success=False, error_msg="TRT out of memory")
        ev = get_compilation_event()
        self.assertIsNotNone(ev)
        self.assertEqual(ev.get("status"), "compilation_failed")
        self.assertEqual(ev.get("model"), "UltraMax")
        self.assertEqual(ev.get("error"), "TRT out of memory")

    def test_04_runtime_telemetry_logging_format(self):
        """Assert exact console telemetry format required by operational directives."""
        # TRT Hit
        line1 = log_enhancer_telemetry("GPEN Realistic", ["TensorrtExecutionProvider"], fp16=True, cache_hit=True)
        self.assertEqual(
            line1,
            "[INFO] Enhancer session active: GPEN Realistic | Provider: TensorrtExecutionProvider (FP16: True, Engine Cache: HIT)",
        )

        # TRT Miss
        line2 = log_enhancer_telemetry("gpen_realistic", ["TensorrtExecutionProvider"], fp16=True, cache_hit=False)
        self.assertEqual(
            line2,
            "[INFO] Enhancer session active: GPEN Realistic | Provider: TensorrtExecutionProvider (FP16: True, Engine Cache: MISS)",
        )

        # UltraMax with CUDA
        line3 = log_enhancer_telemetry("ultramax", ["CUDAExecutionProvider"], fp16=True, cache_hit=False)
        self.assertEqual(
            line3,
            "[INFO] Enhancer session active: UltraMax | Provider: CUDAExecutionProvider (FP16: True, Engine Cache: HIT)",
        )

        # FP32 CPU
        line4 = log_enhancer_telemetry("codeformer", ["CPUExecutionProvider"], fp16=False, cache_hit=False)
        self.assertEqual(
            line4,
            "[INFO] Enhancer session active: CodeFormer | Provider: CPUExecutionProvider (FP16: False, Engine Cache: HIT)",
        )


class TestStage4MemorySafetyAndFallbacks(unittest.TestCase):
    """Test memory deallocation, garbage collection, and silent CPU guard."""

    def test_05_face_enhancer_explicit_release(self):
        enh = FaceEnhancer(enhancer_type="gpen_realistic", model_size=512)
        dummy_sess = MagicMock()
        enh.session = dummy_sess
        enh.active_providers = ["TensorrtExecutionProvider"]

        with patch("gc.collect") as mock_gc, patch("roop.face_enhancer.clear_cuda_cache") as mock_cuda:
            enh.release()
            self.assertIsNone(enh.session)
            self.assertEqual(enh.active_providers, [])
            mock_cuda.assert_called_once()
            mock_gc.assert_called_once()

    def test_06_silent_cpu_fallback_prevention(self):
        """Verify that when a session silently drops to CPU while CUDA is available, it forces clean CUDA fallback."""
        fake_cpu_sess = MagicMock()
        fake_cpu_sess.get_providers.return_value = ["CPUExecutionProvider"]

        fake_cuda_sess = MagicMock()
        fake_cuda_sess.get_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        available = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        with patch("roop.face_enhancer.onnxruntime.get_available_providers", return_value=available), \
             patch("roop.face_enhancer.onnxruntime.InferenceSession") as mock_sess:
            # First attempt (Tier 1 TRT) returns a session that only bound CPU (silent degradation)
            # Second attempt (Tier 2 CUDA) returns CUDA session
            mock_sess.side_effect = [fake_cpu_sess, fake_cuda_sess]

            sess, active = create_resilient_session(
                model_path="dummy.onnx",
                requested_providers=[("TensorrtExecutionProvider", {})],
                label="GPENRealistic-512",
                warmup=False,
            )
            # Must have rejected the silent CPU session and successfully fallen back to CUDA
            self.assertEqual(active[0], "CUDAExecutionProvider")


class TestStage4FastAPIEndpoints(unittest.TestCase):
    """Test REST and polling endpoints for compilation event propagation."""

    @classmethod
    def setUpClass(cls):
        from starlette.testclient import TestClient
        from app.api import app
        cls.client = TestClient(app)

    def test_07_rest_compilation_status_endpoint(self):
        # Broadcast synthetic compilation event
        test_ev = {
            "status": "compiling_engine",
            "model": "GPEN-Realistic",
            "provider": "TensorRT",
            "estimated_time": "2-4 minutes",
            "elapsed_sec": 42,
        }
        broadcast_compilation_event(test_ev)

        resp = self.client.get("/api/compilation/status")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("status"), "compiling_engine")
        self.assertEqual(data.get("model"), "GPEN-Realistic")
        self.assertEqual(data.get("provider"), "TensorRT")

    def test_08_progress_endpoint_surfaces_compilation(self):
        test_ev = {
            "status": "compiling_engine",
            "model": "GPEN-Realistic",
            "provider": "TensorRT",
            "estimated_time": "2-4 minutes",
            "elapsed_sec": 15,
        }
        broadcast_compilation_event(test_ev)

        resp = self.client.get("/api/progress")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("compilation", data)
        self.assertEqual(data["compilation"].get("status"), "compiling_engine")
        self.assertTrue(data.get("compiling_engine"))

        # Clean up by broadcasting finished event
        broadcast_compilation_event({
            "status": "engine_ready",
            "model": "GPEN-Realistic",
            "provider": "TensorRT",
        })


if __name__ == "__main__":
    unittest.main()
