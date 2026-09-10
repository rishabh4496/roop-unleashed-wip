"""The malformed-payload handler: a bad value must be 400, not 500.

Every endpoint takes `payload: dict = Body(...)`, so FastAPI validates only that
the body is an object; the coercion is by hand, 59 times across api.py, as
int(payload.get(...)) / float(payload.get(...)). Unhandled, each of those
answers 500 with no body the UI can render — api.js's errorMessage() falls back
to bare status text and the user is told "Internal Server Error" for what is
entirely a bad request.

This exercises a stand-alone app wired the same way as api.py rather than
importing api.py itself, which constructs the whole FastAPI app and pulls in the
model stack. A source check below keeps the real registration honest.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import Body, FastAPI, Request                   # noqa: E402
from fastapi.responses import JSONResponse                   # noqa: E402
from fastapi.testclient import TestClient                    # noqa: E402

_API = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'api.py')


def _build_app():
    """Same shape as api.py: hand-coerced payload + the shared handler."""
    app = FastAPI()

    @app.exception_handler(ValueError)
    @app.exception_handler(TypeError)
    async def _bad_payload_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=400,
            content={"message": f"{request.url.path}: the request contained a "
                                f"value this endpoint could not use ({exc})"},
        )

    @app.post("/api/thing")
    def thing(payload: dict = Body(...)):
        # The exact idiom used 59 times in api.py.
        return {"index": int(payload.get("index", 0)),
                "accept": float(payload.get("accept", 1.0))}

    @app.post("/api/boom")
    def boom(payload: dict = Body(...)):
        raise RuntimeError('a genuine server bug')

    return app


class TestBadPayloadHandler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(_build_app(), raise_server_exceptions=False)

    def test_valid_payload_still_works(self):
        r = self.client.post('/api/thing', json={'index': 2, 'accept': 0.5})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {'index': 2, 'accept': 0.5})

    def test_numeric_strings_still_work(self):
        r = self.client.post('/api/thing', json={'index': '3', 'accept': '0.25'})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()['index'], 3)

    def test_non_numeric_string_is_400(self):
        r = self.client.post('/api/thing', json={'index': 'abc'})
        self.assertEqual(r.status_code, 400,
                         'a non-numeric value must not be a 500')

    def test_null_is_400(self):
        # int(None) -> TypeError, the other half of the pair.
        r = self.client.post('/api/thing', json={'index': None})
        self.assertEqual(r.status_code, 400)

    def test_wrong_container_type_is_400(self):
        for bad in ([1, 2], {'a': 1}, True and 'x'):
            r = self.client.post('/api/thing', json={'index': bad})
            self.assertEqual(r.status_code, 400, f'index={bad!r}')

    def test_the_message_is_renderable_by_the_ui(self):
        """api.js reads payload.message; it must be a non-empty string."""
        r = self.client.post('/api/thing', json={'accept': 'not-a-float'})
        body = r.json()
        self.assertIn('message', body)
        self.assertIsInstance(body['message'], str)
        self.assertTrue(body['message'].strip())
        self.assertIn('/api/thing', body['message'],
                      'the message should name the endpoint that refused')

    def test_a_real_server_bug_is_still_a_500(self):
        """The handler must not swallow every failure into 400."""
        r = self.client.post('/api/boom', json={})
        self.assertEqual(r.status_code, 500,
                         'a RuntimeError is a server fault and must stay a 500')

    def test_a_non_object_body_is_still_422(self):
        """FastAPI's own body validation is unchanged."""
        r = self.client.post('/api/thing', content='"just a string"',
                             headers={'content-type': 'application/json'})
        self.assertEqual(r.status_code, 422)


class TestHandlerIsRegisteredInApi(unittest.TestCase):
    """Source check, so the shape above cannot drift from the real thing."""

    @classmethod
    def setUpClass(cls):
        cls.src = open(_API, encoding='utf-8').read()

    def test_both_exception_types_are_registered(self):
        self.assertIn('@app.exception_handler(ValueError)', self.src)
        self.assertIn('@app.exception_handler(TypeError)', self.src)

    def test_it_answers_400(self):
        start = self.src.index('async def _bad_payload_handler')
        body = self.src[start:start + 700]
        self.assertIn('status_code=400', body)

    def test_it_prints_the_traceback(self):
        """A genuine ValueError reported as 400 must stay diagnosable."""
        start = self.src.index('async def _bad_payload_handler')
        body = self.src[start:start + 700]
        self.assertIn('print_exception', body)

    def test_keyerror_is_not_caught(self):
        """KeyError here is far more often internal state than a payload field."""
        self.assertNotIn('@app.exception_handler(KeyError)', self.src)


if __name__ == '__main__':
    unittest.main(verbosity=2)
