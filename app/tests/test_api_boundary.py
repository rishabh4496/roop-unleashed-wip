"""Boundary tests for the API: CORS origin policy and index validation.

CORS was allow_origins=["*"] on a server bound to 127.0.0.1 at a fixed,
guessable port. Loopback binding is no defence against that — the request comes
from the user's own browser — so any page open while the app ran could
enumerate outputs, read finished videos through /api/file, delete them, or start
renders. Nothing legitimate needed it: the UI is same-origin
(api.js: `API = window.location.origin`) and Vite proxies /api server-side.

/api/source/select assigned int(payload["index"]) with no range check. Its
consumers guard only the upper bound (api.py:331, :2460), so a NEGATIVE index
passed straight through and Python indexed from the end — picking source 0 and
silently swapping with the last faceset.
"""
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_API = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'api.py')


def _origin_pattern():
    src = open(_API, encoding='utf-8').read()
    m = re.search(r'allow_origin_regex=r"([^"]+)"', src)
    assert m, 'api.py no longer configures allow_origin_regex'
    return re.compile(m.group(1))


class TestCorsOrigins(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pat = _origin_pattern()

    def test_wildcard_is_gone(self):
        # Code only — the comment above the middleware quotes the old value on
        # purpose, and matching prose would make this test fail on its own
        # explanation.
        code = '\n'.join(line.split('#', 1)[0]
                         for line in open(_API, encoding='utf-8'))
        self.assertNotIn('allow_origins=["*"]', code,
                         'the API must not accept every origin')

    def test_local_origins_still_work(self):
        # Everything the app is legitimately served under, including Pinokio's
        # "<port>.localhost" HTTPS proxy convention and IPv6 loopback.
        for origin in ('http://localhost', 'http://localhost:5173',
                       'http://127.0.0.1:8001', 'https://8001.localhost',
                       'http://[::1]:8001', 'https://localhost:443'):
            self.assertTrue(self.pat.match(origin), f'{origin} should be allowed')

    def test_remote_origins_are_refused(self):
        for origin in ('https://evil.com', 'http://evil.com:8001',
                       'https://roop.example.org', 'http://10.0.0.5:8001'):
            self.assertIsNone(self.pat.match(origin), f'{origin} must be refused')

    def test_lookalike_hosts_are_refused(self):
        """The classic bypasses against a sloppy prefix/suffix match."""
        for origin in ('http://localhost.evil.com', 'https://notlocalhost',
                       'http://127.0.0.1.evil.com', 'http://localhost@evil.com',
                       'http://evil.com/localhost'):
            self.assertIsNone(self.pat.match(origin), f'{origin} must be refused')


class TestSourceSelectBounds(unittest.TestCase):
    """Checked by source inspection: importing api.py builds the whole FastAPI
    app and pulls in the model stack, which these assertions do not need."""

    @classmethod
    def setUpClass(cls):
        src = open(_API, encoding='utf-8').read()
        start = src.index('def source_select(')
        cls.body = src[start:src.index('@app.post', start + 10)]

    def test_rejects_out_of_range_both_ways(self):
        # The whole point: a lower bound, not just an upper one.
        self.assertIn('0 <= idx < len(', self.body,
                      'source_select must range-check on both sides')

    def test_coercion_cannot_500(self):
        self.assertIn('except (TypeError, ValueError)', self.body,
                      'a non-numeric index must be a 400, not an unhandled 500')

    def test_does_not_assign_before_validating(self):
        assign = self.body.index('state.selected_input_face_index = idx')
        check = self.body.index('0 <= idx < len(')
        self.assertLess(check, assign,
                        'the index must be validated before it is stored')


if __name__ == '__main__':
    unittest.main(verbosity=2)
