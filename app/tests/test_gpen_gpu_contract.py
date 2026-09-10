"""Fail-fast provider and execution-order contracts for GPEN."""

import importlib
import os
import sys
import threading
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from typing import ClassVar

import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)


class _Options:
    def __init__(self):
        self.entries = {}
        self.log_severity_level = None

    def add_session_config_entry(self, key, value):
        self.entries[key] = value


class _Session:
    calls: ClassVar[list] = []

    def __init__(self, path, options, providers):
        self.path = path
        self.options = options
        self.providers = providers
        type(self).calls.append(self)

    def get_providers(self):
        names = [
            provider[0] if isinstance(provider, (tuple, list)) else provider
            for provider in self.providers
        ]
        # ORT appends its CPU EP internally even when it was not explicitly
        # registered. The session option is what forbids node assignment to it.
        return names + ["CPUExecutionProvider"]

    def disable_fallback(self):
        self.runtime_fallback_disabled = True


class GpenGpuContract(unittest.TestCase):
    def setUp(self):
        self.module = importlib.import_module(
            "roop.processors.Enhance_GPEN"
        )
        self.saved_ort = self.module.onnxruntime
        _Session.calls = []

    def tearDown(self):
        self.module.onnxruntime = self.saved_ort

    def test_gpu_is_first_and_explicit_cpu_provider_is_removed(self):
        self.module.onnxruntime = types.SimpleNamespace(
            SessionOptions=_Options,
            InferenceSession=_Session,
            get_available_providers=lambda: [
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        )
        requested = [
            "CPUExecutionProvider",
            ("TensorrtExecutionProvider", {"device_id": 0}),
            "CUDAExecutionProvider",
        ]
        session = self.module.create_gpen_session(
            "gpen.onnx", requested, "cuda"
        )

        self.assertEqual(
            [self.module._provider_name(p) for p in session.providers],
            ["TensorrtExecutionProvider", "CUDAExecutionProvider"],
        )
        self.assertEqual(
            session.options.entries["session.disable_cpu_ep_fallback"],
            "1",
        )
        self.assertEqual(session.get_providers()[0],
                         "TensorrtExecutionProvider")
        self.assertTrue(session.runtime_fallback_disabled)

    def test_missing_gpu_provider_fails_before_session_creation(self):
        self.module.onnxruntime = types.SimpleNamespace(
            SessionOptions=_Options,
            InferenceSession=_Session,
            get_available_providers=lambda: ["CPUExecutionProvider"],
        )
        with self.assertRaisesRegex(RuntimeError, "GPU acceleration was requested"):
            self.module.create_gpen_session(
                "gpen.onnx",
                ["CUDAExecutionProvider", "CPUExecutionProvider"],
                "cuda",
            )
        self.assertEqual(_Session.calls, [])

    def test_explicit_cpu_mode_remains_supported(self):
        self.module.onnxruntime = types.SimpleNamespace(
            SessionOptions=_Options,
            InferenceSession=_Session,
            get_available_providers=lambda: ["CPUExecutionProvider"],
        )
        session = self.module.create_gpen_session(
            "gpen.onnx", ["CPUExecutionProvider"], "cpu"
        )
        self.assertNotIn(
            "session.disable_cpu_ep_fallback",
            session.options.entries,
        )

    def test_fixed_shape_slot_binds_device_buffers_only_once(self):
        class ValueInfo:
            name = "value"
            type = "tensor(float)"
            shape = (1, 3, 2, 2)

        class OrtValue:
            def __init__(self, shape, dtype):
                self.array = np.empty(shape, dtype=dtype)

            @classmethod
            def ortvalue_from_shape_and_type(
                cls, shape, dtype, _device, _device_id
            ):
                return cls(shape, dtype)

            def update_inplace(self, value):
                self.array[...] = value

            def numpy(self):
                return self.array.copy()

        class Binding:
            def bind_ortvalue_input(self, _name, value):
                self.input = value

            def bind_ortvalue_output(self, _name, value):
                self.output = value

            def synchronize_outputs(self):
                pass

        class Session:
            bindings = 0

            def __init__(self):
                self.barrier = None

            def get_inputs(self):
                return [ValueInfo()]

            def get_outputs(self):
                return [ValueInfo()]

            def io_binding(self):
                type(self).bindings += 1
                return Binding()

            def run_with_iobinding(self, binding):
                if self.barrier is not None:
                    self.barrier.wait(timeout=2)
                binding.output.array[...] = binding.input.array * 2

        self.module.onnxruntime = types.SimpleNamespace(OrtValue=OrtValue)
        session = Session()
        slot = self.module._GPENSlot(
            session,
            "input",
            "output",
            "cuda",
            [("CUDAExecutionProvider", {"device_id": 0})],
        )
        first = np.ones((1, 3, 2, 2), dtype=np.float32)
        second = np.full((1, 3, 2, 2), 3.0, dtype=np.float32)

        self.assertTrue(slot.reuses_device_buffers)
        np.testing.assert_array_equal(slot.run(first), first * 2)
        np.testing.assert_array_equal(slot.run(second), second * 2)
        self.assertEqual(Session.bindings, 1)

        # Two simultaneous CUDA callers get independent mutable buffers and do
        # not inherit TensorRT's one-context lock.
        session.barrier = threading.Barrier(2)
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(slot.run, (first, second)))
        np.testing.assert_array_equal(results[0], first * 2)
        np.testing.assert_array_equal(results[1], second * 2)
        self.assertEqual(Session.bindings, 3)


if __name__ == "__main__":
    unittest.main()
