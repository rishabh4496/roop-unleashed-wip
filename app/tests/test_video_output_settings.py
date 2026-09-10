"""Regression coverage for codec-dependent output settings and encode quality."""

import ast
import os
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

APP = Path(__file__).resolve().parents[1]
REPO = APP.parent
sys.path.insert(0, str(APP))

from settings import (Settings, normalize_encoder_preset, normalize_thread_count,
                      normalize_video_quality)  # noqa: E402

# Compile just the dependency-free helper instead of importing util_ffmpeg,
# whose unrelated video utilities import the full Torch/Gradio runtime.
_util_source = (APP / 'roop' / 'util_ffmpeg.py').read_text(encoding='utf-8')
_util_tree = ast.parse(_util_source)
_rate_nodes = [
    node for node in _util_tree.body
    if (isinstance(node, ast.FunctionDef)
        and node.name in {'quality_max', 'clamp_quality', '_rate_control'})
    or (isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id.startswith('_QUALITY_')
                or isinstance(target, ast.Name) and target.id == '_WIDE_RANGE_CODECS'
                for target in node.targets))
]
_rate_namespace = {'os': os, 'List': list}
exec(compile(ast.Module(body=_rate_nodes, type_ignores=[]),
             str(APP / 'roop' / 'util_ffmpeg.py'), 'exec'), _rate_namespace)
_rate_control = _rate_namespace['_rate_control']


class QualityValuesMatchTheSelectedCodec(unittest.TestCase):

    def test_standard_codecs_are_clamped_to_51(self):
        self.assertEqual(normalize_video_quality('libx265', 99), 51)
        self.assertEqual(normalize_video_quality('hevc_nvenc', '63'), 51)

    def test_wide_range_codecs_keep_values_through_63(self):
        self.assertEqual(normalize_video_quality('libvpx-vp9', 63), 63)

    def test_bad_values_fall_back_to_the_high_quality_default(self):
        self.assertEqual(normalize_video_quality('libx264', 'not-a-number'), 14)

    def test_load_normalizes_old_config_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, 'config.yaml')
            path.write_text(yaml.safe_dump({
                'output_video_codec': 'hevc_nvenc',
                'video_quality': 80,
                'perf_encoder_preset': 'faster',
            }), encoding='utf-8')
            cfg = Settings(str(path))
        self.assertEqual(cfg.video_quality, 51)
        self.assertEqual(cfg.perf_encoder_preset, 'auto')

    def test_preset_validation_switches_with_codec_family(self):
        self.assertEqual(normalize_encoder_preset('hevc_nvenc', 'p6'), 'p6')
        self.assertEqual(normalize_encoder_preset('hevc_nvenc', 'medium'), 'auto')
        self.assertEqual(normalize_encoder_preset('libx265', 'medium'), 'medium')
        self.assertEqual(normalize_encoder_preset('libx265', 'p6'), 'auto')

    def test_manual_threads_cannot_oversubscribe_or_fall_below_one(self):
        logical = os.cpu_count() or 4
        self.assertEqual(normalize_thread_count(logical + 100), logical)
        self.assertEqual(normalize_thread_count(0), 1)
        self.assertEqual(normalize_thread_count('bad', default=3), min(3, logical))


class EncoderArgumentsPreserveQuality(unittest.TestCase):

    def setUp(self):
        self.saved = {
            name: os.environ.get(name)
            for name in ('ROOP_ENCODER_PRESET', 'ROOP_NVENC_PRESET')
        }

    def tearDown(self):
        for name, value in self.saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def test_nvenc_constant_quality_has_no_default_bitrate_cap(self):
        os.environ['ROOP_NVENC_PRESET'] = 'p6'
        args = _rate_control('hevc_nvenc', 18)
        self.assertEqual(args[args.index('-cq') + 1], '18')
        self.assertEqual(args[args.index('-b:v') + 1], '0')
        self.assertEqual(args[args.index('-preset') + 1], 'p6')

    def test_secondary_software_encodes_use_the_selected_speed_preset(self):
        os.environ['ROOP_ENCODER_PRESET'] = 'veryfast'
        args = _rate_control('libx265', 18)
        self.assertEqual(args[args.index('-crf') + 1], '18')
        self.assertEqual(args[args.index('-preset') + 1], 'veryfast')


class OutputSettingsStayInsideThePublicContract(unittest.TestCase):

    def test_internal_config_path_is_not_exposed(self):
        cfg = Settings(str(APP / '__no_such_output_settings__.yaml'))
        self.assertNotIn('config_file', cfg.public_dict())
        self.assertIn('output_video_codec', cfg.public_dict())

    def test_backend_publishes_container_codec_compatibility(self):
        source = (APP / 'api.py').read_text(encoding='utf-8')
        self.assertIn('"video_codecs_by_format": video_codecs_by_format', source)
        self.assertIn('"webm": ["libvpx-vp9"]', source)
        self.assertIn("roop_globals.CFG.public_dict()", source)

    def test_formats_without_an_installed_compatible_codec_are_empty(self):
        source = (APP / 'api.py').read_text(encoding='utf-8')
        tree = ast.parse(source)
        nodes = [
            node for node in tree.body
            if (isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name)
                        and target.id in {'_VIDEO_CODECS', '_VIDEO_FORMAT_CODECS'}
                        for target in node.targets))
            or (isinstance(node, ast.FunctionDef)
                and node.name == '_video_codecs_by_format')
        ]
        namespace = {}
        exec(compile(ast.Module(body=nodes, type_ignores=[]),
                     str(APP / 'api.py'), 'exec'), namespace)
        compatible = namespace['_video_codecs_by_format'](['libx264', 'hevc_nvenc'])
        self.assertEqual(compatible['webm'], [])
        self.assertEqual(compatible['mp4'], ['libx264', 'hevc_nvenc'])

    def test_advanced_preset_reaches_both_encoder_families(self):
        source = (APP / 'run.py').read_text(encoding='utf-8')
        self.assertIn("cfg.get('perf_encoder_preset')", source)
        self.assertIn("_set('ROOP_ENCODER_PRESET', encoder_preset)", source)
        self.assertIn("_set('ROOP_NVENC_PRESET', encoder_preset)", source)

    def test_normal_launchers_do_not_enable_diagnostic_profiling(self):
        for name in ('start_react.js', 'start_legacy.js'):
            source = (REPO / name).read_text(encoding='utf-8')
            self.assertIn('ROOP_PROFILE: "0"', source, name)


if __name__ == '__main__':
    unittest.main()
