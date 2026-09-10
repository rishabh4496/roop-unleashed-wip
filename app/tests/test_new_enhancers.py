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
from roop.processors.enhance_common import (                        # noqa: E402
    inject_reference_detail, enhance_eyes_clarity, _eye_region)
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
    io_bindings_created = 0

    def __init__(self, path, options, providers):
        type(self).created += 1
        type(self).providers_seen.append(providers)

    def get_inputs(self):
        return [type('Input', (), {'name': 'x'})()]

    def get_outputs(self):
        return [type('Output', (), {'name': 'y'})()]

    def io_binding(self):
        type(self).io_bindings_created += 1
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


class EyeClarityLeavesTheSkinAlone(unittest.TestCase):
    """The eye finish may add local contrast; it may not change tone.

    The first version ran CLAHE over the whole crop's L channel and blended the
    result into a feathered ellipse pair. CLAHE redistributes a region's levels,
    so its output differs from its input by a large low-frequency term — and
    blending that into an oval writes a tone STEP into the periocular skin with
    the mask's feather as its only edge. That is a halo around the eyes, which
    is what it was reported as. Measured on the reported frames: +12.5 L of
    brightening painted into the ellipse.

    These lock the two properties that make the halo impossible rather than
    merely unlikely — the operation is zero-mean, and it cannot reach skin it
    was never meant to touch.
    """

    W = 512
    # ffhq_512's own eye keypoints (face_util.WARP_TEMPLATES).
    EYES = ((0.37691676, 0.46864664), (0.62285697, 0.46912813))

    @classmethod
    def _face(cls):
        """A face-like luminance field. The eye-socket shadow is the point: it
        is the local structure a histogram equalisation flattens, and flattening
        it is what moved the tone."""
        w = cls.W
        rng = np.random.default_rng(0)
        yy, xx = np.mgrid[0:w, 0:w].astype(np.float32)
        lum = 172 + 14 * (yy / w)
        centres = [(fx * w, fy * w) for fx, fy in cls.EYES]
        iod = centres[1][0] - centres[0][0]
        for cx, cy in centres:
            r = np.hypot((xx - cx) / (iod * 0.42), (yy - cy) / (iod * 0.34))
            lum -= 34 * np.exp(-r * r)
        lum -= 18 * np.exp(-(((xx - w * 0.5) / (w * 0.05)) ** 2
                             + ((yy - w * 0.62) / (w * 0.14)) ** 2))
        lum += rng.normal(0, 1.6, lum.shape)
        img = np.clip(np.dstack([lum * 0.80, lum * 0.89, lum]), 0, 255).astype(np.uint8)
        for cx, cy in centres:
            c = (int(cx), int(cy))
            cv2.ellipse(img, c, (int(iod * .21), int(iod * .11)), 0, 0, 360, (232, 235, 238), -1)
            cv2.circle(img, c, int(iod * .085), (72, 78, 92), -1)
            cv2.circle(img, c, int(iod * .038), (12, 12, 14), -1)
            cv2.ellipse(img, c, (int(iod * .22), int(iod * .12)), 0, 180, 360, (44, 42, 44), 2)
        return img, centres, iod

    @staticmethod
    def _luma(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)

    def test_the_periocular_skin_keeps_its_tone(self):
        img, centres, iod = self._face()
        out = enhance_eyes_clarity(img, template='ffhq_512', strength=0.52)
        delta = self._luma(out) - self._luma(img)

        w = self.W
        yy, xx = np.mgrid[0:w, 0:w]
        skin = np.zeros((w, w), bool)
        for cx, cy in centres:
            rr = np.hypot(xx - cx, yy - cy)
            skin |= (rr > iod * 0.32) & (rr < iod * 0.55)
        far = np.zeros((w, w), bool)
        far[30:110, 200:312] = True

        step = abs(float(delta[skin].mean()) - float(delta[far].mean()))
        self.assertLess(step, 0.25, f'periocular tone moved {step:.2f} L')

    def test_nothing_outside_the_eye_box_moves_at_all(self):
        img = np.random.default_rng(1).integers(
            0, 255, (self.W, self.W, 3), dtype=np.uint8)
        out = enhance_eyes_clarity(img, template='ffhq_512', strength=0.52)
        x0, y0, x1, y1 = _eye_region(self.W, self.W, 'ffhq_512', 0.52)[1]
        outside = np.ones((self.W, self.W), bool)
        outside[y0:y1, x0:x1] = False
        # Not "close to" — identical. The old full-crop BGR->LAB->BGR round trip
        # moved pixels the mask gave zero weight.
        self.assertTrue(np.array_equal(img[outside], out[outside]))

    def test_the_ellipse_is_an_eye_and_not_a_band_across_the_face(self):
        weight, (x0, y0, x1, y1), _ = _eye_region(self.W, self.W, 'ffhq_512', 0.52)
        # The shipped 0.115w x 0.075h pair plus its feather covered ~98 rows of
        # a 512 crop, brows to cheekbones. An eye is ~33 rows.
        self.assertLess(y1 - y0, 72)
        # And the two ellipses must not meet over the nose bridge.
        self.assertLess(float(weight[:, (weight.shape[1] // 2)].max()), 0.05)

    def test_it_still_sharpens_the_eye(self):
        # Softened first, because that is the input this exists for: a restorer
        # that left the eyes milky. A synthetic face drawn with hard vector
        # edges is already at the ceiling the 3x3 envelope allows, so sharpening
        # it further is exactly what must NOT happen.
        img, centres, iod = self._face()
        img = cv2.GaussianBlur(img, (0, 0), sigmaX=1.6)
        out = enhance_eyes_clarity(img, template='ffhq_512', strength=0.52)
        w = self.W
        yy, xx = np.mgrid[0:w, 0:w]
        eye = np.zeros((w, w), bool)
        for cx, cy in centres:
            eye |= np.hypot(xx - cx, yy - cy) < iod * 0.16

        def detail(im):
            luma = self._luma(im)
            return np.abs(luma - cv2.GaussianBlur(luma, (0, 0), 1.2))

        self.assertGreater(detail(out)[eye].mean(), detail(img)[eye].mean() * 1.05)

    def test_neutral_strength_and_a_tiny_crop_are_no_ops(self):
        img = np.full((64, 64, 3), 128, np.uint8)
        self.assertIs(enhance_eyes_clarity(img, strength=0.0), img)
        tiny = np.full((12, 12, 3), 128, np.uint8)
        self.assertIs(enhance_eyes_clarity(tiny, strength=0.5), tiny)


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
        self.assertIn('self.profile == "ultimate"', base_src)
        self.assertIn('create_gpen_session', base_src)

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
            _FakeSession.io_bindings_created = 0
            module.onnxruntime = types.SimpleNamespace(InferenceSession=_FakeSession)
            module.conditional_download = lambda _dir, _urls: None
            module.resolve_relative_path = lambda path: path
            g.execution_providers = [
                'TensorrtExecutionProvider',
                'CUDAExecutionProvider',
                'CPUExecutionProvider',
            ]
            session_pool._pool_cache.clear()
            session_pool._pool_cache.update({'trt': 3, 'detmask': 0})

            processor = Enhance_GPENUltimate()
            processor.Initialize({'devicename': 'cuda'})
            result, scale = processor.Run(
                None, None, np.zeros((512, 512, 3), dtype=np.uint8))
            processor.Run(None, None, np.zeros((512, 512, 3), dtype=np.uint8))

            self.assertEqual(processor.model_size, 512)
            self.assertTrue(getattr(processor, 'force_align', False))
            self.assertEqual(getattr(processor, 'model_template', None), 'ffhq_512')
            self.assertIsNotNone(processor.pool)
            self.assertEqual(_FakeSession.created, 3)
            # Three warm-ups plus two runtime calls. Minimal test doubles do
            # not expose OrtValue, so they correctly exercise fresh bindings.
            self.assertEqual(_FakeSession.io_bindings_created, 5)
            self.assertTrue(all(hasattr(item, 'run')
                                for item in processor.pool._items))
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
