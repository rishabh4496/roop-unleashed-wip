"""A mistyped ROOP_* tuning knob must cost the tuning, not the app.

The ~21 thresholds in procmgr_runtime were read with a bare
float()/int() around os.environ.get(), at MODULE level. Those raise ValueError
on anything non-numeric, and a raise at module level is an ImportError: the app
does not start, and the user sees a traceback ending inside a constants block
rather than a message naming the variable.

start_react.js — tracked, and shipped to every install — sets about fifteen of
these, and the constants are commented with the measurements behind them
precisely so people retune them. A decimal comma from a locale or a stray
trailing character was enough to brick startup.
"""
import importlib
import os
import subprocess
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop.procmgr_runtime import env_float, env_int    # noqa: E402

_APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestEnvFloat(unittest.TestCase):
    def setUp(self):
        os.environ.pop('ROOP_TEST_KNOB', None)

    tearDown = setUp

    def test_absent_uses_the_default(self):
        self.assertEqual(env_float('ROOP_TEST_KNOB', '0.85'), 0.85)

    def test_a_real_value_wins(self):
        os.environ['ROOP_TEST_KNOB'] = '0.42'
        self.assertAlmostEqual(env_float('ROOP_TEST_KNOB', '0.85'), 0.42)

    def test_junk_falls_back_instead_of_raising(self):
        for junk in ('abc', '0.85.', '0,85', '  ', '', 'NaN%', '--1'):
            os.environ['ROOP_TEST_KNOB'] = junk
            try:
                got = env_float('ROOP_TEST_KNOB', '0.85')
            except Exception as e:                      # noqa: BLE001
                self.fail(f'{junk!r} raised {e!r} instead of falling back')
            self.assertEqual(got, 0.85, f'{junk!r} should fall back')

    def test_whitespace_padding_is_accepted(self):
        os.environ['ROOP_TEST_KNOB'] = '  0.42  '
        self.assertAlmostEqual(env_float('ROOP_TEST_KNOB', '0.85'), 0.42)

    def test_always_returns_a_float(self):
        for v in (None, '', 'abc', '1', '1.5'):
            if v is None:
                os.environ.pop('ROOP_TEST_KNOB', None)
            else:
                os.environ['ROOP_TEST_KNOB'] = v
            self.assertIsInstance(env_float('ROOP_TEST_KNOB', '0.85'), float)


class TestEnvInt(unittest.TestCase):
    def setUp(self):
        os.environ.pop('ROOP_TEST_KNOB', None)

    tearDown = setUp

    def test_absent_uses_the_default(self):
        self.assertEqual(env_int('ROOP_TEST_KNOB', '45'), 45)

    def test_a_float_string_is_taken_as_the_number_meant(self):
        os.environ['ROOP_TEST_KNOB'] = '45.0'
        self.assertEqual(env_int('ROOP_TEST_KNOB', '3'), 45)

    def test_junk_falls_back(self):
        for junk in ('abc', '4 5', '', 'x45'):
            os.environ['ROOP_TEST_KNOB'] = junk
            self.assertEqual(env_int('ROOP_TEST_KNOB', '45'), 45, f'{junk!r}')

    def test_always_returns_an_int(self):
        os.environ['ROOP_TEST_KNOB'] = '45.9'
        self.assertIsInstance(env_int('ROOP_TEST_KNOB', '3'), int)


class TestImportSurvivesBadEnv(unittest.TestCase):
    """The actual regression: importing the module with a junk knob set."""

    def _import_with(self, **env):
        e = dict(os.environ, **env)
        e['PYTHONPATH'] = _APP + os.pathsep + e.get('PYTHONPATH', '')
        return subprocess.run(
            [sys.executable, '-c',
             'import roop.procmgr_runtime as r; print(r._TRACK_VETO_DIST)'],
            capture_output=True, text=True, env=e, cwd=_APP, timeout=180)

    def test_a_junk_threshold_does_not_break_the_import(self):
        r = self._import_with(ROOP_TRACK_VETO='0,85')       # decimal comma
        self.assertEqual(r.returncode, 0,
                         f'import failed with a mistyped knob:\n{r.stderr[-800:]}')
        self.assertIn('0.85', r.stdout, 'should have fallen back to the default')

    def test_it_says_which_variable_was_wrong(self):
        r = self._import_with(ROOP_TRACK_VETO='abc')
        self.assertEqual(r.returncode, 0)
        said = r.stdout + r.stderr
        self.assertIn('ROOP_TRACK_VETO', said,
                      'the warning must name the offending variable')

    def test_a_good_value_is_still_honoured(self):
        r = self._import_with(ROOP_TRACK_VETO='0.5')
        self.assertEqual(r.returncode, 0)
        self.assertIn('0.5', r.stdout)


if __name__ == '__main__':
    unittest.main(verbosity=2)
