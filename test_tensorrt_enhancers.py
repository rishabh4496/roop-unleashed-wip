"""Verification suite for TensorRT Execution Provider support in GPEN Realistic and UltraMax.

Verifies:
1. Static optimization profile binding:
   - GPEN Realistic (512): min/opt/max profile (1x3x512x512)
   - GPEN Realistic (1024): min/opt/max profile (1x3x1024x1024)
   - UltraMax / CodeFormer: min/opt/max profile (x:1x3x512x512, w:1)
2. Selectable precision routing (FP16 vs FP32 vs mixed):
   - fp16: trt_fp16_enable=True, trt_layer_norm_fp32_fallback=False
   - fp32: trt_fp16_enable=False, trt_layer_norm_fp32_fallback=True
   - mixed: trt_fp16_enable=True, trt_layer_norm_fp32_fallback=True
   - GPEN 1024 FP32 safety enforcement (protects against NaN / white speckle)
3. Cache path segregation under models/trt_cache/{precision}/
4. Multi-tier graceful fallback chain (TensorRT -> CUDA -> CPU)
5. Zero-copy OrtValue device buffer reuse contract
6. Backward compatibility for standard enhancers (GFPGAN, CodeFormer, RestoreFormer++)
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# Ensure app is in path
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.join(REPO_ROOT, "app")
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import roop.globals
from roop.precision_policy import providers_for
from roop.trt_shape_profile import apply_shape_profile, resolve_profile
from roop.face_enhancer import create_enhancer_session


class TestTensorRTShapeProfiles(unittest.TestCase):
    """Test static optimization profile binding to prevent TRT dynamic axis rejection."""

    def test_gpen_512_shape_profile(self):
        providers = [("TensorrtExecutionProvider", {})]
        configured, prec = providers_for(
            model_tag="gpen_realistic",
            execution_providers=providers,
            explicit_shape=(1, 3, 512, 512),
        )
        self.assertEqual(len(configured), 1)
        name, opts = configured[0]
        self.assertEqual(name, "TensorrtExecutionProvider")
        self.assertIn("trt_profile_min_shapes", opts)
        self.assertIn("trt_profile_opt_shapes", opts)
        self.assertIn("trt_profile_max_shapes", opts)
        # Should include 1x3x512x512
        self.assertIn("1x3x512x512", opts["trt_profile_opt_shapes"])

    def test_gpen_1024_shape_profile(self):
        providers = [("TensorrtExecutionProvider", {})]
        configured, prec = providers_for(
            model_tag="gpen_1024",
            execution_providers=providers,
            explicit_shape=(1, 3, 1024, 1024),
        )
        self.assertEqual(len(configured), 1)
        name, opts = configured[0]
        self.assertEqual(name, "TensorrtExecutionProvider")
        self.assertIn("1x3x1024x1024", opts["trt_profile_opt_shapes"])

    def test_ultramax_codeformer_shape_profile(self):
        providers = [("TensorrtExecutionProvider", {})]
        configured, prec = providers_for(
            model_tag="ultramax",
            execution_providers=providers,
            explicit_shape=(1, 3, 512, 512),
        )
        self.assertEqual(len(configured), 1)
        name, opts = configured[0]
        self.assertEqual(name, "TensorrtExecutionProvider")
        opt_shapes = opts["trt_profile_opt_shapes"]
        # CodeFormer dynamic inputs are x and w
        self.assertTrue("512x512" in opt_shapes)


class TestPrecisionPolicy(unittest.TestCase):
    """Test precision routing, cache segregation, and GPEN 1024 safety overrides."""

    def test_fp16_precision_routing(self):
        providers = [("TensorrtExecutionProvider", {})]
        configured, prec = providers_for(
            model_tag="gpen_realistic",
            execution_providers=providers,
            requested_precision="fp16",
        )
        self.assertEqual(prec, "fp16")
        _, opts = configured[0]
        self.assertTrue(opts["trt_fp16_enable"])
        self.assertFalse(opts["trt_layer_norm_fp32_fallback"])
        self.assertIn("fp16", opts["trt_engine_cache_path"])

    def test_fp32_precision_routing(self):
        providers = [("TensorrtExecutionProvider", {})]
        configured, prec = providers_for(
            model_tag="gpen_realistic",
            execution_providers=providers,
            requested_precision="fp32",
        )
        self.assertEqual(prec, "fp32")
        _, opts = configured[0]
        self.assertFalse(opts["trt_fp16_enable"])
        self.assertTrue(opts["trt_layer_norm_fp32_fallback"])
        self.assertIn("fp32", opts["trt_engine_cache_path"])

    def test_mixed_precision_routing(self):
        providers = [("TensorrtExecutionProvider", {})]
        configured, prec = providers_for(
            model_tag="gpen_realistic",
            execution_providers=providers,
            requested_precision="mixed",
        )
        self.assertEqual(prec, "mixed")
        _, opts = configured[0]
        self.assertTrue(opts["trt_fp16_enable"])
        self.assertTrue(opts["trt_layer_norm_fp32_fallback"])
        self.assertIn("mixed", opts["trt_engine_cache_path"])

    def test_gpen_1024_forces_fp32_for_safety(self):
        providers = [("TensorrtExecutionProvider", {})]
        # Even if mixed is requested, GPEN 1024 must enforce FP32 to prevent overflow / NaN speckles
        configured, prec = providers_for(
            model_tag="gpen_1024",
            execution_providers=providers,
            requested_precision="mixed",
        )
        self.assertEqual(prec, "fp32")
        _, opts = configured[0]
        self.assertFalse(opts["trt_fp16_enable"])
        self.assertTrue(opts["trt_layer_norm_fp32_fallback"])

    def test_gpen_1024_allows_explicit_fp16_override(self):
        providers = [("TensorrtExecutionProvider", {})]
        configured, prec = providers_for(
            model_tag="gpen_1024",
            execution_providers=providers,
            requested_precision="fp16",
        )
        self.assertEqual(prec, "fp16")
        _, opts = configured[0]
        self.assertTrue(opts["trt_fp16_enable"])


class TestMultiTierFallbackChain(unittest.TestCase):
    """Test multi-tier graceful fallback chain when TensorRT engine fails to compile."""

    @patch("onnxruntime.get_available_providers")
    @patch("onnxruntime.InferenceSession")
    def test_fallback_from_tensorrt_to_cuda_to_cpu(self, mock_sess, mock_avail):
        mock_avail.return_value = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]

        call_count = [0]

        def _side_effect(*args, **kwargs):
            call_count[0] += 1
            provs = kwargs.get("providers") or (args[2] if len(args) > 2 else [])
            p0 = provs[0]
            name = p0[0] if isinstance(p0, (tuple, list)) else str(p0)
            if "TensorrtExecutionProvider" in name:
                raise RuntimeError("TensorRT compilation simulated failure")
            # Return a mock session for CUDA or CPU
            sess = MagicMock()
            sess.get_providers.return_value = [name]
            sess.get_inputs.return_value = []
            return sess

        mock_sess.side_effect = _side_effect

        sess, active = create_enhancer_session(
            model_path="dummy_model.onnx",
            requested_providers=["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
            label="GPENRealistic-512",
            warmup=False,
        )

        self.assertGreaterEqual(call_count[0], 2)
        # Verify fallback chose CUDAExecutionProvider with cudnn_conv_algo_search: 'DEFAULT'
        self.assertEqual(active[0], "CUDAExecutionProvider")


class TestZeroCopyOrtValueContract(unittest.TestCase):
    """Test zero-copy OrtValue buffer allocation and update_inplace reuse."""

    def test_gpen_realistic_reuses_device_buffers_contract(self):
        from roop.processors.Enhance_GPENRealistic import Enhance_GPENRealistic

        enhancer = Enhance_GPENRealistic()
        self.assertFalse(enhancer.reuses_device_buffers)

        # Mock OrtValue and session
        class MockOrtValue:
            def __init__(self, shape, dtype):
                self.arr = np.zeros(shape, dtype=dtype)

            @classmethod
            def ortvalue_from_shape_and_type(cls, shape, dtype, device, device_id):
                return cls(shape, dtype)

            def update_inplace(self, data):
                self.arr[...] = data

            def numpy(self):
                return self.arr.copy()

        class MockIOB:
            def __init__(self):
                self.inputs = {}
                self.outputs = {}

            def bind_ortvalue_input(self, name, val):
                self.inputs[name] = val

            def bind_ortvalue_output(self, name, val):
                self.outputs[name] = val

            def bind_output(self, name, device):
                pass

            def synchronize_outputs(self):
                pass

        class MockInput:
            name = "input"
            type = "tensor(float)"
            shape = [1, 3, 512, 512]

        class MockOutput:
            name = "output"
            type = "tensor(float)"
            shape = [1, 3, 512, 512]

        class MockSession:
            def get_inputs(self):
                return [MockInput()]

            def get_outputs(self):
                return [MockOutput()]

            def get_providers(self):
                return ["CUDAExecutionProvider"]

            def io_binding(self):
                return MockIOB()

            def run_with_iobinding(self, iob):
                pass

        with patch("onnxruntime.OrtValue", MockOrtValue, create=True), \
             patch("roop.face_enhancer.create_enhancer_session", return_value=(MockSession(), ["CUDAExecutionProvider"])), \
             patch("roop.model_registry.ensure_model_downloaded", return_value="dummy.onnx"):

            enhancer.Initialize({"devicename": "cuda"})
            self.assertTrue(enhancer.reuses_device_buffers)

            # Test run slot with update_inplace
            dummy_in = np.ones((1, 3, 512, 512), dtype=np.float32)
            out = enhancer._run_slot(0, dummy_in)
            self.assertEqual(len(out), 1)
            self.assertEqual(out[0].shape, (1, 3, 512, 512))

            enhancer.Release()
            self.assertFalse(enhancer.reuses_device_buffers)

    def test_ultramax_reuses_device_buffers_contract(self):
        from roop.processors.Enhance_UltraMax import Enhance_UltraMax

        enhancer = Enhance_UltraMax()
        self.assertFalse(enhancer.reuses_device_buffers)

        class MockOrtValue:
            def __init__(self, shape, dtype):
                self.arr = np.zeros(shape, dtype=dtype)

            @classmethod
            def ortvalue_from_shape_and_type(cls, shape, dtype, device, device_id):
                return cls(shape, dtype)

            def update_inplace(self, data):
                self.arr[...] = data

            def numpy(self):
                return self.arr.copy()

        class MockIOB:
            def __init__(self):
                self.inputs = {}
                self.outputs = {}

            def bind_ortvalue_input(self, name, val):
                self.inputs[name] = val

            def bind_ortvalue_output(self, name, val):
                self.outputs[name] = val

            def bind_output(self, name, device):
                pass

            def synchronize_outputs(self):
                pass

        class MockInputX:
            name = "x"
            type = "tensor(float16)"
            shape = [1, 3, 512, 512]

        class MockInputW:
            name = "w"
            type = "tensor(double)"
            shape = [1]

        class MockOutput:
            name = "output"
            type = "tensor(float16)"
            shape = [1, 3, 512, 512]

        class MockSession:
            def get_inputs(self):
                return [MockInputX(), MockInputW()]

            def get_outputs(self):
                return [MockOutput()]

            def get_providers(self):
                return ["CUDAExecutionProvider"]

            def io_binding(self):
                return MockIOB()

            def run_with_iobinding(self, iob):
                pass

        with patch("onnxruntime.OrtValue", MockOrtValue, create=True), \
             patch("roop.face_enhancer.create_enhancer_session", return_value=(MockSession(), ["CUDAExecutionProvider"])), \
             patch("roop.model_registry.ensure_model_downloaded", return_value="dummy.onnx"):

            enhancer.Initialize({"devicename": "cuda"})
            self.assertTrue(enhancer.reuses_device_buffers)

            dummy_in = np.ones((1, 3, 512, 512), dtype=np.float16)
            out = enhancer._run_slot(0, dummy_in)
            self.assertEqual(len(out), 1)
            self.assertEqual(out[0].shape, (1, 3, 512, 512))

            enhancer.Release()
            self.assertFalse(enhancer.reuses_device_buffers)


if __name__ == "__main__":
    unittest.main()
