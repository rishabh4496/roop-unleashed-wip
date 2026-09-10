"""Regression tests for the FFMPEG_VideoWriter stderr deadlock.

The writer launches ffmpeg with stderr=sp.PIPE (the default `logfile`, taken by
every call site) and, before the fix, never read that pipe while the render was
running. An OS pipe holds ~64 KB; once ffmpeg had written that much it blocked
inside its own stderr write, stopped draining stdin, and the next write_frame()
blocked forever — a silent hang of the whole encode, more likely the longer the
render.

These tests stand a real child process in for ffmpeg: it reads stdin like ffmpeg
does and floods stderr like a chatty encoder does. `test_deadlock_mechanism`
proves the failure is real for an undrained pipe; `test_writer_survives_*` proves
the writer no longer suffers it.

No ffmpeg binary required — the point is the pipe discipline, not the codec.
"""
import os
import subprocess as sp
import sys
import threading
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from roop.ffmpeg_writer import FFMPEG_VideoWriter

# A stand-in encoder: emit STDERR_KB of stderr immediately, then consume stdin.
# The ordering is what matters — a real ffmpeg interleaves the two, but front
# loading the stderr makes the deadlock deterministic instead of timing
# dependent.
FAKE_ENCODER = r'''
import sys, os
kb = int(os.environ.get("FAKE_STDERR_KB", "256"))
line = b"[fake] " + b"x" * 89 + b"\n"          # 96 bytes
for _ in range((kb * 1024) // len(line)):
    sys.stderr.buffer.write(line)
sys.stderr.buffer.flush()
# Now behave like an encoder: swallow every byte of raw video sent to us.
while True:
    chunk = sys.stdin.buffer.read(1 << 20)
    if not chunk:
        break
sys.stderr.buffer.write(b"[fake] done\n")
sys.stderr.buffer.flush()
'''

PIPE_BUFFER_KB = 256      # comfortably over the 64 KB the OS will hold
FRAME = np.zeros((64, 64, 3), dtype=np.uint8)   # 12 KB per frame


def _spawn_fake(**popen_kwargs):
    env = dict(os.environ, FAKE_STDERR_KB=str(PIPE_BUFFER_KB))
    return sp.Popen([sys.executable, '-c', FAKE_ENCODER], env=env, **popen_kwargs)


class TestDeadlockIsReal(unittest.TestCase):
    """Establish the bug exists, so the fix below is not testing a no-op."""

    def test_deadlock_mechanism(self):
        proc = _spawn_fake(stdin=sp.PIPE, stdout=sp.DEVNULL, stderr=sp.PIPE)
        blocked = threading.Event()
        finished = threading.Event()

        def writer():
            blocked.set()
            try:
                # Far more than the stdin pipe can hold while the child is stuck
                # on its own stderr write.
                for _ in range(200):
                    proc.stdin.write(FRAME.tobytes())
                proc.stdin.flush()
            except Exception:
                pass
            finished.set()

        t = threading.Thread(target=writer, daemon=True)
        t.start()
        blocked.wait(5)
        # Nobody is reading stderr, so the child is wedged and cannot drain
        # stdin. The writer must NOT get through.
        self.assertFalse(
            finished.wait(4),
            "expected the undrained stderr pipe to block the writer; if this "
            "passes, the platform's pipe buffer is larger than the test assumes",
        )

        # Draining stderr releases it — proving the pipe was the cause.
        drained = threading.Thread(
            target=lambda: proc.stderr.read(), daemon=True)
        drained.start()
        self.assertTrue(
            finished.wait(15),
            "draining stderr should have unblocked the writer",
        )
        try:
            proc.stdin.close()
            proc.wait(timeout=10)
        except Exception:
            proc.kill()


class TestWriterDrainsStderr(unittest.TestCase):
    """The real class, with a fake encoder swapped in underneath Popen."""

    def _make_writer(self, tmpname):
        real_popen = sp.Popen

        def fake_popen(cmd, **kwargs):
            # Ignore the ffmpeg argv; keep the pipe wiring the writer chose, so
            # the stdin/stderr discipline under test is exactly the real one.
            kwargs.pop('creationflags', None)
            env = dict(os.environ, FAKE_STDERR_KB=str(PIPE_BUFFER_KB))
            return real_popen([sys.executable, '-c', FAKE_ENCODER],
                              env=env, **kwargs)

        with mock.patch('roop.ffmpeg_writer.sp.Popen', side_effect=fake_popen):
            return FFMPEG_VideoWriter(tmpname, (64, 64), 25)

    def test_writer_survives_a_chatty_encoder(self):
        """200 frames through an encoder that front-loads 256 KB of stderr."""
        w = self._make_writer('test_out.mp4')
        done = threading.Event()
        err = []

        def run():
            try:
                for _ in range(200):
                    w.write_frame(FRAME)
            except Exception as e:      # noqa: BLE001 - reported to the assert
                err.append(e)
            done.set()

        threading.Thread(target=run, daemon=True).start()
        self.assertTrue(
            done.wait(30),
            "write_frame() blocked — stderr is not being drained",
        )
        self.assertEqual(err, [], f"write_frame raised: {err}")
        w.close(timeout=20)
        self.assertIsNone(w.proc, "close() must clear the process handle")

    def test_stderr_tail_is_retained_for_error_messages(self):
        """Draining must not throw the text away — the error paths quote it."""
        w = self._make_writer('test_out2.mp4')
        w.write_frame(FRAME)
        time.sleep(0.5)                 # let the drain thread collect
        tail = w._stderr_text()
        self.assertIn('[fake]', tail,
                      "the drained stderr tail should still be readable")
        w.close(timeout=20)

    def test_close_is_idempotent_and_reaps_the_child(self):
        w = self._make_writer('test_out3.mp4')
        w.write_frame(FRAME)
        proc = w.proc
        w.close(timeout=20)
        self.assertIsNotNone(proc.returncode, "child was not reaped")
        w.close(timeout=20)             # second close must be a no-op, not a crash

    def test_close_survives_a_dead_encoder(self):
        """The old close() raised BrokenPipeError here and leaked the process."""
        w = self._make_writer('test_out4.mp4')
        w.proc.kill()
        w.proc.wait(timeout=10)
        w.close(timeout=10)             # must not raise
        self.assertIsNone(w.proc)


if __name__ == '__main__':
    unittest.main(verbosity=2)
