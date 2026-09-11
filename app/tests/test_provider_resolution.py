"""Execution-provider ordering, fallback, and TensorRT batch profile guards."""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import onnx
from onnx import TensorProto, helper

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop import core  # noqa: E402
from roop.processors.FaceSwapInsightFace import _with_trt_batch_profiles  # noqa: E402


def _names(providers):
    return [provider[0] if isinstance(provider, tuple) else provider
            for provider in providers]


class TestProviderResolution(unittest.TestCase):
    def setUp(self):
        self.old_cfg = core.roop.globals.CFG
        self.old_device = core.roop.globals.cuda_device_id
        core.roop.globals.CFG = SimpleNamespace(trt_precision='mixed')
        core.roop.globals.cuda_device_id = 0

    def tearDown(self):
        core.roop.globals.CFG = self.old_cfg
        core.roop.globals.cuda_device_id = self.old_device

    def test_preserves_requested_precedence_and_appends_cpu(self):
        available = ['TensorrtExecutionProvider', 'CUDAExecutionProvider',
                     'CPUExecutionProvider']
        with mock.patch.object(core.ort, 'get_available_providers', return_value=available), \
             mock.patch.object(core.torch.cuda, 'is_available', return_value=False):
            providers = core.decode_execution_providers(['cuda', 'tensorrt'])
        self.assertEqual(_names(providers), [
            'CUDAExecutionProvider', 'TensorrtExecutionProvider',
            'CPUExecutionProvider',
        ])

    def test_unavailable_accelerator_falls_back_to_cpu(self):
        with mock.patch.object(core.ort, 'get_available_providers',
                               return_value=['CPUExecutionProvider']):
            providers = core.decode_execution_providers(['cuda'])
        self.assertEqual(providers, ['CPUExecutionProvider'])

    def test_invalid_cuda_device_is_repaired_before_ort_sees_it(self):
        available = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        core.roop.globals.cuda_device_id = 99
        with mock.patch.object(core.ort, 'get_available_providers', return_value=available), \
             mock.patch.object(core.torch.cuda, 'is_available', return_value=True), \
             mock.patch.object(core.torch.cuda, 'device_count', return_value=1), \
             mock.patch.object(core.torch.cuda, 'set_device') as set_device:
            providers = core.decode_execution_providers(['cuda'])
        self.assertEqual(providers[0][1]['device_id'], 0)
        self.assertEqual(core.roop.globals.cuda_device_id, 0)
        set_device.assert_called_once_with(0)

    def test_one_provider_configuration_failure_does_not_drop_fallbacks(self):
        available = ['TensorrtExecutionProvider', 'CUDAExecutionProvider',
                     'CPUExecutionProvider']
        with mock.patch.object(core.ort, 'get_available_providers', return_value=available), \
             mock.patch.object(core.os, 'makedirs', side_effect=OSError('read only')), \
             mock.patch.object(core.torch.cuda, 'is_available', return_value=False):
            providers = core.decode_execution_providers(['tensorrt', 'cuda'])
        self.assertEqual(_names(providers), available)


class TestTensorRTBatchProfile(unittest.TestCase):
    def test_profile_covers_all_dynamic_swap_inputs(self):
        graph = helper.make_graph(
            [helper.make_node('Identity', ['target'], ['output'])],
            'swap',
            [
                helper.make_tensor_value_info('target', TensorProto.FLOAT,
                                              ['N', 3, 256, 256]),
                helper.make_tensor_value_info('source', TensorProto.FLOAT,
                                              ['N', 512]),
            ],
            [helper.make_tensor_value_info('output', TensorProto.FLOAT,
                                          ['N', 3, 256, 256])],
        )
        model = helper.make_model(graph)
        providers = _with_trt_batch_profiles(
            [('TensorrtExecutionProvider', {'trt_fp16_enable': False}),
             'CUDAExecutionProvider'],
            model, 16, 'hyperswap')
        options = providers[0][1]
        self.assertEqual(
            options['trt_profile_min_shapes'],
            'target:1x3x256x256,source:1x512')
        self.assertEqual(
            options['trt_profile_opt_shapes'],
            'target:4x3x256x256,source:4x512')
        self.assertEqual(
            options['trt_profile_max_shapes'],
            'target:16x3x256x256,source:16x512')
        self.assertEqual(options['trt_engine_cache_prefix'],
                         'hyperswap_batch16')
        self.assertEqual(providers[1], 'CUDAExecutionProvider')


if __name__ == '__main__':
    unittest.main(verbosity=2)
