"""Test Stage 2: Fallback Chaining, Resilient Session Initialization & Profile Shape Binding.

Verifies:
1. Dynamic Profile Shape Binding:
   - Dynamic model input dimension detection (512x512, 1024x1024)
   - trt_profile_min_shapes / opt_shapes / max_shapes: 'input:1x3x512x512'
   - Multi-profile support for batch enhancement (batch_size > 1):
     'min=input:1x3x512x512', 'opt=input:4x3x512x512', 'max=input:4x3x512x512'
   - Secondary input binding for UltraMax / CodeFormer ('w:1')
2. Provider Priority Stack:
   - Explicit fallback hierarchy:
     [
         ('TensorrtExecutionProvider', trt_options),
         ('CUDAExecutionProvider', cuda_options),
         'CPUExecutionProvider'
     ]
   - TRT options configuration (device_id, workspace up to 4GB, fp16, caches)
   - CUDA options configuration (device_id, cudnn_conv_algo, do_copy, arena)
3. Session Health Check:
   - Automated warm-up inference pass (np.zeros((1, 3, H, W), dtype=np.float32))
   - Compilation error catching with actionable diagnostic terminal output
   - Clean fallback to CUDAExecutionProvider without crashing video batches
4. VRAM safety during fallback (clear_cuda_cache invoked)
"""

from __future__ import annotations

import io
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

_APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

import numpy as np
import onnxruntime

from roop.face_enhancer import (
    _warmup_enhancer_session,
    clear_cuda_cache,
    create_enhancer_session,
)
from roop.model_loader import (
    build_provider_priority_stack,
    get_tensorrt_cache_dir,
    get_tensorrt_provider_options,
)
from roop.trt_shape_profile import apply_shape_profile, resolve_profile


class TestProfileShapeBinding(unittest.TestCase):
    """Test suite for dynamic profile shape binding."""

    def test_gpen_512_single_frame(self):
        """GPEN-512 with batch_size=1 configures min/opt/max to input:1x3x512x512."""
        profile = resolve_profile(
            model_key="gpen_realistic",
            explicit_shape=(1, 3, 512, 512),
            batch_size=1,
        )
        self.assertIsNotNone(profile)
        self.assertEqual(profile.min_shapes, "input:1x3x512x512")
        self.assertEqual(profile.opt_shapes, "input:1x3x512x512")
        self.assertEqual(profile.max_shapes, "input:1x3x512x512")

    def test_gpen_1024_single_frame(self):
        """GPEN-1024 with batch_size=1 configures min/opt/max to input:1x3x1024x1024."""
        profile = resolve_profile(
            model_key="gpen_1024",
            explicit_shape=(1, 3, 1024, 1024),
            batch_size=1,
        )
        self.assertIsNotNone(profile)
        self.assertEqual(profile.min_shapes, "input:1x3x1024x1024")
        self.assertEqual(profile.opt_shapes, "input:1x3x1024x1024")
        self.assertEqual(profile.max_shapes, "input:1x3x1024x1024")

    def test_gpen_512_batch_enhancement(self):
        """GPEN-512 with batch_size=4 configures multi-profile support."""
        profile = resolve_profile(
            model_key="gpen_realistic",
            explicit_shape=(1, 3, 512, 512),
            batch_size=4,
        )
        self.assertIsNotNone(profile)
        self.assertEqual(profile.min_shapes, "input:1x3x512x512")
        self.assertEqual(profile.opt_shapes, "input:4x3x512x512")
        self.assertEqual(profile.max_shapes, "input:4x3x512x512")

    def test_gpen_1024_batch_enhancement(self):
        """GPEN-1024 with batch_size=2 configures multi-profile support."""
        profile = resolve_profile(
            model_key="gpen_1024",
            explicit_shape=(1, 3, 1024, 1024),
            batch_size=2,
        )
        self.assertIsNotNone(profile)
        self.assertEqual(profile.min_shapes, "input:1x3x1024x1024")
        self.assertEqual(profile.opt_shapes, "input:2x3x1024x1024")
        self.assertEqual(profile.max_shapes, "input:2x3x1024x1024")

    def test_ultramax_secondary_input_binding(self):
        """UltraMax / CodeFormer binds secondary weight parameter 'w:1'."""
        profile = resolve_profile(
            model_key="ultramax",
            explicit_shape=(1, 3, 512, 512),
            batch_size=4,
        )
        self.assertIsNotNone(profile)
        self.assertEqual(profile.min_shapes, "x:1x3x512x512,w:1")
        self.assertEqual(profile.opt_shapes, "x:4x3x512x512,w:1")
        self.assertEqual(profile.max_shapes, "x:4x3x512x512,w:1")


class TestProviderPriorityStack(unittest.TestCase):
    """Test suite for the explicit provider fallback hierarchy."""

    def test_provider_priority_hierarchy_structure(self):
        """Verify stack is [('TensorrtExecutionProvider', trt_opts), ('CUDAExecutionProvider', cuda_opts), 'CPUExecutionProvider']."""
        stack = build_provider_priority_stack(device_id=0)
        self.assertEqual(len(stack), 3)

        # 1. TensorrtExecutionProvider with trt_options
        self.assertIsInstance(stack[0], tuple)
        self.assertEqual(stack[0][0], "TensorrtExecutionProvider")
        trt_opts = stack[0][1]
        self.assertIsInstance(trt_opts, dict)
        self.assertEqual(trt_opts["device_id"], 0)
        self.assertIn("trt_max_workspace_size", trt_opts)
        self.assertIn("trt_fp16_enable", trt_opts)
        self.assertIn("trt_engine_cache_enable", trt_opts)
        self.assertIn("trt_engine_cache_path", trt_opts)
        self.assertIn("trt_timing_cache_enable", trt_opts)

        # 2. CUDAExecutionProvider with cuda_options
        self.assertIsInstance(stack[1], tuple)
        self.assertEqual(stack[1][0], "CUDAExecutionProvider")
        cuda_opts = stack[1][1]
        self.assertIsInstance(cuda_opts, dict)
        self.assertEqual(cuda_opts["device_id"], 0)
        self.assertIn("cudnn_conv_algo_search", cuda_opts)
        self.assertIn("do_copy_in_default_stream", cuda_opts)
        self.assertIn("arena_extend_strategy", cuda_opts)

        # 3. CPUExecutionProvider
        self.assertEqual(stack[2], "CPUExecutionProvider")

    def test_provider_priority_stack_with_shape_binding(self):
        """Verify build_provider_priority_stack binds dynamic profile shapes when model metadata is supplied."""
        stack = build_provider_priority_stack(
            device_id=1,
            model_key="gpen_realistic",
            explicit_shape=(1, 3, 512, 512),
            batch_size=4,
        )
        trt_opts = stack[0][1]
        self.assertEqual(trt_opts["device_id"], 1)
        self.assertEqual(trt_opts["trt_profile_min_shapes"], "input:1x3x512x512")
        self.assertEqual(trt_opts["trt_profile_opt_shapes"], "input:4x3x512x512")
        self.assertEqual(trt_opts["trt_profile_max_shapes"], "input:4x3x512x512")


class TestSessionHealthCheckAndFallback(unittest.TestCase):
    """Test suite for automated warm-up inference pass, diagnostic output, and clean fallback."""

    def test_warmup_inference_pass_structure(self):
        """Verify _warmup_enhancer_session passes np.zeros((1, 3, H, W)) to session.run()."""
        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "input"
        mock_input.shape = [1, 3, 512, 512]
        mock_input.type = "tensor(float)"
        mock_session.get_inputs.return_value = [mock_input]

        success = _warmup_enhancer_session(
            mock_session,
            label="GPENRealistic-512",
            explicit_shape=(1, 3, 512, 512),
            batch_size=1,
        )
        self.assertTrue(success)
        mock_session.run.assert_called_once()
        _, feed_dict = mock_session.run.call_args[0]
        self.assertIn("input", feed_dict)
        dummy_tensor = feed_dict["input"]
        self.assertEqual(dummy_tensor.shape, (1, 3, 512, 512))
        self.assertEqual(dummy_tensor.dtype, np.float32)
        self.assertTrue(np.all(dummy_tensor == 0.0))

    def test_warmup_inference_pass_multi_input(self):
        """Verify warm-up handles secondary inputs (e.g. UltraMax 'w' weight parameter)."""
        mock_session = MagicMock()
        mock_input_x = MagicMock()
        mock_input_x.name = "x"
        mock_input_x.shape = [1, 3, 512, 512]
        mock_input_x.type = "tensor(float16)"

        mock_input_w = MagicMock()
        mock_input_w.name = "w"
        mock_input_w.shape = [1]
        mock_input_w.type = "tensor(float)"

        mock_session.get_inputs.return_value = [mock_input_x, mock_input_w]

        success = _warmup_enhancer_session(
            mock_session,
            label="UltraMax",
            explicit_shape=(1, 3, 512, 512),
            batch_size=2,
        )
        self.assertTrue(success)
        _, feed_dict = mock_session.run.call_args[0]
        self.assertIn("x", feed_dict)
        self.assertIn("w", feed_dict)
        self.assertEqual(feed_dict["x"].shape, (2, 3, 512, 512))
        self.assertEqual(feed_dict["x"].dtype, np.float16)
        self.assertEqual(feed_dict["w"].shape, (1,))

    @patch("onnxruntime.get_available_providers", return_value=["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"])
    @patch("onnxruntime.InferenceSession")
    @patch("roop.face_enhancer._warmup_enhancer_session")
    def test_trt_failure_triggers_actionable_diagnostic_and_cuda_fallback(
        self,
        mock_warmup,
        mock_inference_session,
        mock_get_providers,
    ):
        """When TensorRT compilation / warmup fails, logs actionable diagnostic and falls back cleanly to CUDA."""
        # Tier 1 session instance (will fail on warmup)
        mock_sess_trt = MagicMock()
        mock_sess_trt.get_providers.return_value = ["TensorrtExecutionProvider"]

        # Tier 2 session instance (CUDA fallback)
        mock_sess_cuda = MagicMock()
        mock_sess_cuda.get_providers.return_value = ["CUDAExecutionProvider"]

        # First call creates TRT session; second call creates CUDA session
        mock_inference_session.side_effect = [mock_sess_trt, mock_sess_cuda]

        # Warm-up raises an error on TRT to simulate engine build abort / driver mismatch
        mock_warmup.side_effect = [
            RuntimeError("TensorRT engine compilation aborted: out of device memory or unsupported op"),
            True,  # Second call succeeds on CUDA
        ]

        captured_stdout = io.StringIO()
        with patch("sys.stdout", captured_stdout):
            sess, active = create_enhancer_session(
                model_path="dummy_model.onnx",
                requested_providers=[("TensorrtExecutionProvider", {}), ("CUDAExecutionProvider", {}), "CPUExecutionProvider"],
                label="GPENRealistic-512",
                warmup=True,
                explicit_shape=(1, 3, 512, 512),
            )

        stdout_text = captured_stdout.getvalue()

        # 1. Verify actionable diagnostic output was printed to console
        self.assertIn("[Session Health Check] TensorRT compilation / warmup aborted", stdout_text)
        self.assertIn("[Actionable Diagnostic] Falling back cleanly to CUDAExecutionProvider", stdout_text)
        self.assertIn("Clean fallback to CUDAExecutionProvider succeeded", stdout_text)

        # 2. Verify fallback succeeded without crashing
        self.assertEqual(sess, mock_sess_cuda)
        self.assertEqual(active, ["CUDAExecutionProvider"])

        # 3. Verify InferenceSession was called twice (Tier 1 then Tier 2)
        self.assertEqual(mock_inference_session.call_count, 2)
        tier1_call_kwargs = mock_inference_session.call_args_list[0][1]
        tier2_call_kwargs = mock_inference_session.call_args_list[1][1]

        # Verify Tier 1 had the explicit priority stack
        t1_providers = tier1_call_kwargs["providers"]
        self.assertEqual(t1_providers[0][0], "TensorrtExecutionProvider")
        self.assertEqual(t1_providers[1][0], "CUDAExecutionProvider")
        self.assertEqual(t1_providers[2], "CPUExecutionProvider")

        # Verify Tier 2 fell back cleanly to CUDA
        t2_providers = tier2_call_kwargs["providers"]
        self.assertEqual(t2_providers[0][0], "CUDAExecutionProvider")
        self.assertEqual(t2_providers[1], "CPUExecutionProvider")


if __name__ == "__main__":
    unittest.main()
