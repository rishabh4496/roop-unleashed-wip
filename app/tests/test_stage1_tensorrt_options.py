"""Test Stage 1: TensorRT Provider Options & Engine Cache Architecture.

Verifies:
1. get_tensorrt_provider_options() definition and availability inside model loader modules:
   - roop.model_loader
   - roop.face_enhancer
   - roop.model_registry
   - roop.precision_policy
2. Configuration defaults and behavior:
   - device_id: Configurable GPU index (default: 0)
   - trt_max_workspace_size: Dynamic allocation up to 4GB (4 * 1024 * 1024 * 1024)
   - trt_fp16_enable: Boolean flag (default: True)
   - trt_engine_cache_enable: Set to True
   - trt_engine_cache_path: Dedicated directory under models/cache/tensorrt/
   - trt_timing_cache_enable: Set to True for faster subsequent rebuilds
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

_APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from roop.model_loader import (
    calculate_dynamic_trt_workspace_size,
    get_models_directory,
    get_tensorrt_cache_dir,
    get_tensorrt_provider_options,
)


class TestStage1TensorRTOptions(unittest.TestCase):
    """Test suite for TensorRT provider options and engine cache architecture."""

    def test_module_availability(self):
        """Verify get_tensorrt_provider_options is importable from model loader modules."""
        import roop.model_loader as ml
        self.assertTrue(callable(getattr(ml, "get_tensorrt_provider_options", None)))

        import roop.face_enhancer as fe
        self.assertTrue(callable(getattr(fe, "get_tensorrt_provider_options", None)))

        import roop.model_registry as mr
        self.assertTrue(callable(getattr(mr, "get_tensorrt_provider_options", None)))

        import roop.precision_policy as pp
        self.assertTrue(callable(getattr(pp, "get_tensorrt_provider_options", None)))

    def test_default_options(self):
        """Verify defaults for all 6 required parameters."""
        opts = get_tensorrt_provider_options()

        # 1. device_id defaults to 0
        self.assertIn("device_id", opts)
        self.assertEqual(opts["device_id"], 0)

        # 2. trt_max_workspace_size: dynamic allocation up to 4GB
        self.assertIn("trt_max_workspace_size", opts)
        max_4gb = 4 * 1024 * 1024 * 1024
        self.assertGreaterEqual(opts["trt_max_workspace_size"], 256 * 1024 * 1024)
        self.assertLessEqual(opts["trt_max_workspace_size"], max_4gb)

        # 3. trt_fp16_enable defaults to True
        self.assertIn("trt_fp16_enable", opts)
        self.assertIs(opts["trt_fp16_enable"], True)

        # 4. trt_engine_cache_enable is True
        self.assertIn("trt_engine_cache_enable", opts)
        self.assertIs(opts["trt_engine_cache_enable"], True)

        # 5. trt_engine_cache_path is dedicated directory under models/cache/tensorrt/
        self.assertIn("trt_engine_cache_path", opts)
        norm_path = os.path.normpath(opts["trt_engine_cache_path"]).replace("\\", "/")
        self.assertTrue(
            "models/cache/tensorrt" in norm_path,
            f"Expected 'models/cache/tensorrt' in path, got: {norm_path}",
        )
        self.assertTrue(os.path.isdir(opts["trt_engine_cache_path"]))

        # 6. trt_timing_cache_enable is True
        self.assertIn("trt_timing_cache_enable", opts)
        self.assertIs(opts["trt_timing_cache_enable"], True)

    def test_configurable_device_id(self):
        """Verify device_id can be configured to any GPU index."""
        opts_gpu1 = get_tensorrt_provider_options(device_id=1)
        self.assertEqual(opts_gpu1["device_id"], 1)

        opts_gpu2 = get_tensorrt_provider_options(device_id=2)
        self.assertEqual(opts_gpu2["device_id"], 2)

    def test_dynamic_workspace_allocation_cap(self):
        """Verify dynamic allocation respects the 4GB ceiling."""
        max_4gb = 4 * 1024 * 1024 * 1024

        # When GPU has massive VRAM (e.g. 24GB on RTX 4090 or 80GB on A100)
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.device_count", return_value=1), \
             patch("torch.cuda.get_device_properties") as mock_props:
            mock_props.return_value = MagicMock(total_memory=24 * 1024 * 1024 * 1024)
            ws_size = calculate_dynamic_trt_workspace_size(device_id=0)
            self.assertEqual(ws_size, max_4gb)

        # When GPU has smaller VRAM (e.g. 6GB on RTX 3060 Mobile)
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.device_count", return_value=1), \
             patch("torch.cuda.get_device_properties") as mock_props:
            mock_props.return_value = MagicMock(total_memory=6 * 1024 * 1024 * 1024)
            ws_size = calculate_dynamic_trt_workspace_size(device_id=0)
            # 50% of 6GB is 3GB
            self.assertEqual(ws_size, 3 * 1024 * 1024 * 1024)

        # Custom explicit workspace size cannot exceed 4GB cap
        opts_custom = get_tensorrt_provider_options(max_workspace_size=8 * 1024 * 1024 * 1024)
        self.assertEqual(opts_custom["trt_max_workspace_size"], max_4gb)

    def test_fp16_toggle(self):
        """Verify trt_fp16_enable can be set to False (e.g. for FP32 mode)."""
        opts_fp32 = get_tensorrt_provider_options(fp16_enable=False)
        self.assertIs(opts_fp32["trt_fp16_enable"], False)

        opts_fp16 = get_tensorrt_provider_options(fp16_enable=True)
        self.assertIs(opts_fp16["trt_fp16_enable"], True)

    def test_dedicated_cache_path_creation(self):
        """Verify dedicated directory under models/cache/tensorrt/ is created."""
        cache_dir = get_tensorrt_cache_dir()
        self.assertTrue(os.path.isdir(cache_dir))
        self.assertTrue("cache" in cache_dir and "tensorrt" in cache_dir)

        # Subdirectory (e.g. fp16 or fp32)
        fp16_cache = get_tensorrt_cache_dir("fp16")
        self.assertTrue(os.path.isdir(fp16_cache))
        self.assertTrue(fp16_cache.endswith("fp16"))


if __name__ == "__main__":
    unittest.main()
