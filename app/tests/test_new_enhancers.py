"""Registration and finishing-contract tests for the named enhancer profiles."""

import os
import sys
import unittest
import inspect

import cv2
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)

import roop.globals as g                                      # noqa: E402
from roop.core import get_processing_plugins, _trt_precision_options  # noqa: E402
from roop import session_pool                                     # noqa: E402
from roop.processors.enhance_common import inject_reference_detail  # noqa: E402
from roop.processors.Enhance_GPENUltimate import Enhance_GPENUltimate  # noqa: E402
from roop.processors.Enhance_RestoreUltra import Enhance_RestoreUltra  # noqa: E402


class _FakeIOBinding:
    def bind_cpu_input(self, name, value):
        self.input = (name, value)

    def bind_output(self, name, device):
        self.output = (name, device)

    def copy_outputs_to_cpu(self):
        return [np.zeros((1, 3, 256, 256), dtype=np.float32)]


class _FakeSession:
    created = 0
    providers_seen = []

    def __init__(self, path, options, providers):
        type(self).created += 1
        type(self).providers_seen.append(providers)

    def get_inputs(self):
        return [type('Input', (), {'name': 'x'})()]

    def get_outputs(self):
        return [type('Output', (), {'name': 'y'})()]

    def io_binding(self):
        return _FakeIOBinding()

    def run_with_iobinding(self, io_binding):
        return None


class NamedProfiles(unittest.TestCase):
    def setUp(self):
        self.old = g.selected_enhancer

    def tearDown(self):
        g.selected_enhancer = self.old

    def test_gpen_ultimate_is_a_distinct_256_profile(self):
        g.selected_enhancer = 'GPEN Ultimate'
        plugins = get_processing_plugins(None)
        self.assertEqual(plugins['gpen_ultimate'], {})
        self.assertEqual(Enhance_GPENUltimate.processorname, 'gpen_ultimate')
        self.assertIn('Enhance_GPEN', Enhance_GPENUltimate.__mro__[1].__name__)
        self.assertTrue(getattr(Enhance_GPENUltimate, 'force_align', False))
        self.assertEqual(getattr(Enhance_GPENUltimate, 'model_template', None), 'ffhq_512')

    def test_restore_ultra_is_a_distinct_restoreformer_profile(self):
        g.selected_enhancer = 'Restore Ultra'
        plugins = get_processing_plugins(None)
        self.assertEqual(plugins['restore_ultra'], {})
        self.assertEqual(Enhance_RestoreUltra.processorname, 'restore_ultra')
        self.assertIn('Enhance_RestoreFormerPPlus',
                      Enhance_RestoreUltra.__mro__[1].__name__)
        self.assertTrue(getattr(Enhance_RestoreUltra, 'force_align', False))
        self.assertEqual(getattr(Enhance_RestoreUltra, 'model_template', None), 'ffhq_512')


class DetailFinish(unittest.TestCase):
    def test_neutral_strength_is_a_noop(self):
        image = np.full((32, 32, 3), 128, np.uint8)
        self.assertIs(inject_reference_detail(image, image, 0), image)

    def test_finish_preserves_registered_swapped_texture(self):
        reference = np.full((64, 64, 3), 128, np.uint8)
        cv2.line(reference, (8, 32), (56, 32), (240, 240, 240), 2)
        restored = np.full((64, 64, 3), 128, np.uint8)
        finished = inject_reference_detail(restored, reference, 0.16)
        self.assertEqual(finished.shape, restored.shape)
        self.assertEqual(finished.dtype, restored.dtype)
        self.assertTrue(np.isfinite(finished).all())
        self.assertGreater(int(np.abs(finished.astype(np.int16) - restored).max()), 0)

    def test_crispness_is_edge_limited(self):
        reference = np.full((64, 64, 3), 128, np.uint8)
        cv2.line(reference, (8, 32), (56, 32), (240, 240, 240), 2)
        restored = np.full((64, 64, 3), 128, np.uint8)
        cv2.line(restored, (8, 32), (56, 32), (180, 180, 180), 2)
        texture_only = inject_reference_detail(restored, reference,
                                                strength=0.28, crispness=0.0)
        finished = inject_reference_detail(restored, reference,
                                            strength=0.28, crispness=0.15)
        self.assertGreater(int(np.abs(finished.astype(np.int16)
                                     - texture_only.astype(np.int16)).max()), 0)
        self.assertEqual(finished[4, 4].tolist(), restored[4, 4].tolist())

    def test_crispness_tracks_edges_when_reference_is_soft(self):
        # A restored face can have clearer model edges than its swapped input.
        # The output-edge gate must still sharpen those features without touching
        # the flat crop around them.
        reference = np.full((64, 64, 3), 128, np.uint8)
        restored = np.full((64, 64, 3), 128, np.uint8)
        cv2.line(restored, (8, 32), (56, 32), (180, 180, 180), 2)
        texture_only = inject_reference_detail(restored, reference,
                                                strength=0.42, crispness=0.0)
        finished = inject_reference_detail(restored, reference,
                                            strength=0.42, crispness=0.28)
        self.assertGreater(int(np.abs(finished.astype(np.int16)
                                     - texture_only.astype(np.int16)).max()), 0)
        self.assertEqual(finished[4, 4].tolist(), restored[4, 4].tolist())


class TensorRTPrecisionContract(unittest.TestCase):
    def test_all_requested_modes_have_explicit_provider_options(self):
        expected = {
            'mixed': (True, True),
            'fp16': (True, False),
            'fp32': (False, False),
        }
        for mode, (fp16, layer_norm_fp32) in expected.items():
            canonical, options = _trt_precision_options(mode)
            self.assertEqual(canonical, mode)
            self.assertEqual(options['trt_fp16_enable'], fp16)
            self.assertEqual(options['trt_layer_norm_fp32_fallback'],
                             layer_norm_fp32)

    def test_unknown_mode_is_safe_and_uses_mixed(self):
        canonical, options = _trt_precision_options('FP64')
        self.assertEqual(canonical, 'mixed')
        self.assertTrue(options['trt_fp16_enable'])
        self.assertTrue(options['trt_layer_norm_fp32_fallback'])

    def test_gpen_ultimate_forwards_each_mode_without_forcing_fp32(self):
        import importlib
        import types

        module = importlib.import_module('roop.processors.Enhance_GPEN')
        saved = {
            'ort': module.onnxruntime,
            'download': module.conditional_download,
            'resolve': module.resolve_relative_path,
            'providers': g.execution_providers,
            'pool_cache': dict(session_pool._pool_cache),
        }
        try:
            module.onnxruntime = types.SimpleNamespace(InferenceSession=_FakeSession)
            module.conditional_download = lambda _dir, _urls: None
            module.resolve_relative_path = lambda path: path
            session_pool._pool_cache.clear()
            session_pool._pool_cache.update({'trt': 0, 'detmask': 0})
            _FakeSession.providers_seen = []

            for mode in ('mixed', 'fp16', 'fp32'):
                _, precision_opts = _trt_precision_options(mode)
                providers = [
                    ('TensorrtExecutionProvider', precision_opts.copy()),
                    'CUDAExecutionProvider',
                ]
                g.execution_providers = providers
                processor = Enhance_GPENUltimate()
                processor.Initialize({'devicename': 'cpu'})
                self.assertEqual(_FakeSession.providers_seen[-1], providers)
                processor.Release()
        finally:
            module.onnxruntime = saved['ort']
            module.conditional_download = saved['download']
            module.resolve_relative_path = saved['resolve']
            g.execution_providers = saved['providers']
            session_pool._pool_cache.clear()
            session_pool._pool_cache.update(saved['pool_cache'])

    def test_restore_ultra_forwards_each_mode_without_rewriting_precision(self):
        import importlib
        import types

        module = importlib.import_module(
            'roop.processors.Enhance_RestoreFormerPPlus')
        base = module.Enhance_RestoreFormerPPlus
        saved = {
            'ort': module.onnxruntime,
            'resolve': module.resolve_relative_path,
            'providers': g.execution_providers,
            'model': getattr(base, 'model_restoreformerpplus', None),
            'pool': getattr(base, 'pool', None),
            'pool_cache': dict(session_pool._pool_cache),
        }
        try:
            module.onnxruntime = types.SimpleNamespace(InferenceSession=_FakeSession)
            module.resolve_relative_path = lambda path: path
            session_pool._pool_cache.clear()
            session_pool._pool_cache.update({'trt': 0, 'detmask': 0})
            _FakeSession.providers_seen = []

            for mode in ('mixed', 'fp16', 'fp32'):
                _, precision_opts = _trt_precision_options(mode)
                providers = [
                    ('TensorrtExecutionProvider', precision_opts.copy()),
                    'CUDAExecutionProvider',
                ]
                g.execution_providers = providers
                processor = Enhance_RestoreUltra()
                processor.Initialize({'devicename': 'cpu'})
                self.assertEqual(_FakeSession.providers_seen[-1], providers)
                processor.Release()
        finally:
            module.onnxruntime = saved['ort']
            module.resolve_relative_path = saved['resolve']
            g.execution_providers = saved['providers']
            base.model_restoreformerpplus = saved['model']
            base.pool = saved['pool']
            session_pool._pool_cache.clear()
            session_pool._pool_cache.update(saved['pool_cache'])


class ProfileDeclarations(unittest.TestCase):
    @staticmethod
    def _read(path):
        with open(path, encoding='utf-8') as fh:
            return fh.read()

    def test_gpen_ultimate_declares_pool_and_profile(self):
        src = inspect.getsource(Enhance_GPENUltimate)
        self.assertIn('size', src)
        base_src = self._read(os.path.join(APP, 'roop', 'processors',
                                           'Enhance_GPEN.py'))
        self.assertIn('session_pool.SessionPool', base_src)
        self.assertIn("profile == 'ultimate'", base_src)

    def test_restore_ultra_reuses_the_pooled_base(self):
        src = self._read(os.path.join(APP, 'roop', 'processors',
                                      'Enhance_RestoreUltra.py'))
        base_src = self._read(os.path.join(APP, 'roop', 'processors',
                                           'Enhance_RestoreFormerPPlus.py'))
        self.assertIn('super().Run', src)
        self.assertIn('session_pool.SessionPool', base_src)
        self.assertIn('inject_reference_detail', src)


class GpenUltimateRuntime(unittest.TestCase):
    def test_initializes_the_512_model_and_pooled_sessions(self):
        import importlib
        import types

        module = importlib.import_module('roop.processors.Enhance_GPEN')
        saved = {
            'ort': module.onnxruntime,
            'download': module.conditional_download,
            'resolve': module.resolve_relative_path,
            'providers': g.execution_providers,
            'pool_cache': dict(session_pool._pool_cache),
        }
        try:
            _FakeSession.created = 0
            module.onnxruntime = types.SimpleNamespace(InferenceSession=_FakeSession)
            module.conditional_download = lambda _dir, _urls: None
            module.resolve_relative_path = lambda path: path
            g.execution_providers = ['CPUExecutionProvider']
            session_pool._pool_cache.clear()
            session_pool._pool_cache.update({'trt': 3, 'detmask': 0})

            processor = Enhance_GPENUltimate()
            processor.Initialize({'devicename': 'cpu'})
            result, scale = processor.Run(
                None, None, np.zeros((512, 512, 3), dtype=np.uint8))

            self.assertEqual(processor.model_size, 512)
            self.assertTrue(getattr(processor, 'force_align', False))
            self.assertEqual(getattr(processor, 'model_template', None), 'ffhq_512')
            self.assertIsNotNone(processor.pool)
            self.assertEqual(_FakeSession.created, 3)
            self.assertEqual(result.shape, (512, 512, 3))
            self.assertEqual(scale, 1)
            processor.Release()
        finally:
            module.onnxruntime = saved['ort']
            module.conditional_download = saved['download']
            module.resolve_relative_path = saved['resolve']
            g.execution_providers = saved['providers']
            session_pool._pool_cache.clear()
            session_pool._pool_cache.update(saved['pool_cache'])


if __name__ == '__main__':
    unittest.main()
