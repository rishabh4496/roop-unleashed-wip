"""Severity is what the caller knows, and status lines go through bar_write.

Two rules, both of which the terminal previously broke.

1. SEVERITY IS NOT RECOVERABLE FROM THE MESSAGE.  update_status used to pick
   its badge by matching the message against keyword lists.  Every interesting
   message interpolates a user filename, so the badge was decided by what the
   user named their file.  Reproduced against the old rule:

       Creating error_clip.mp4 with 30 FPS...     -> [ERROR]
       Creating success_take2.mp4 with 30 FPS...  -> [SUCCESS]
       Creating mistook_scene.mp4 with 30 FPS...  -> [SUCCESS]   ("mis-took")

   The first announced a failure at the moment a render STARTED.  A call site
   knows its own severity for free and cannot be wrong about it, so the level
   is a parameter and the classifier is gone.

2. A STATUS LINE MUST NOT CORRUPT A LIVE PROGRESS BAR.  tqdm draws by
   rewriting the current line; a bare print() lands in the middle of it and
   terminates it.  That is documented at length above ChunkedProgress, which
   exists because one rewritten bar became one 451-character line per frame on
   a real 48,501-frame render.  bar_write (tqdm.write) is the safe form, used
   52 times across five modules -- and none of them was core.py, which owns
   the app's primary status channel and called print() directly.

The third class here is degradation: colour and glyphs must both have an off
switch that does not depend on guessing what the far end of the pipe is.
"""

import os
import re
import sys
import unittest

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE = os.path.join(APP, 'roop', 'core.py')
TUI = os.path.join(APP, 'roop', 'tui.py')
RUNTIME = os.path.join(APP, 'roop', 'procmgr_runtime.py')

if APP not in sys.path:
    sys.path.insert(0, APP)


def _read(path):
    with open(path, encoding='utf-8') as fh:
        return fh.read()


def _func(src, name):
    """The full source of a top-level function, docstring included.

    Stopping at the first blank line would stop INSIDE the docstring, which is
    where these functions explain themselves; the body has to run to the next
    top-level definition.
    """
    m = re.search(
        r'^def ' + re.escape(name) + r'\(.*?(?=^\S|\Z)',
        src, re.S | re.M)
    return m.group(0) if m else ''


def _strip_comments_and_docstrings(src):
    """Crude but sufficient: drop # comments and triple-quoted blocks.

    The docstrings here deliberately QUOTE the old broken code, so a guard that
    cannot tell a quotation from a use would force the explanations out.
    """
    # The delimiters are built rather than written literally, so this file can
    # describe the pattern without containing it.
    dq = chr(34) * 3
    sq = chr(39) * 3
    src = re.sub(dq + '.*?' + dq, '', src, flags=re.S)
    src = re.sub(sq + '.*?' + sq, '', src, flags=re.S)
    return re.sub(r'#[^\n]*', '', src)


class SeverityComesFromTheCallSite(unittest.TestCase):

    def test_update_status_takes_a_level(self):
        import roop.core as core
        import inspect
        params = list(inspect.signature(core.update_status).parameters)
        self.assertEqual(params[:2], ['message', 'level'])

    def test_the_keyword_classifier_is_gone(self):
        body = _strip_comments_and_docstrings(_read(CORE))
        for kw in ('[SUCCESS]', '[ACTION]', '[STATUS]'):
            self.assertNotIn(
                kw, body,
                f'{kw} was produced by matching the message text, which '
                'contains user filenames -- error_clip.mp4 rendered [ERROR] '
                'as its render began.')
        self.assertNotRegex(
            body, r'lower_msg\s*=',
            'severity must not be recovered from the message at all.')

    def test_a_filename_cannot_change_the_badge(self):
        """The regression itself, exercised through the real function."""
        import roop.core as core
        import roop.tui as tui
        seen = []
        orig = tui.status
        tui.status = lambda m, lvl=tui.INFO: seen.append((m, lvl))
        try:
            for name in ('error_clip.mp4', 'success_take2.mp4',
                         'mistook_scene.mp4', 'holiday.mp4'):
                core.update_status(f'Creating {name} with 30 FPS...', core.RUN)
        finally:
            tui.status = orig
        self.assertEqual([lvl for _m, lvl in seen], [core.RUN] * 4,
                         'the badge changed with the filename')

    def test_every_call_site_that_can_fail_says_so(self):
        """The messages naming a failure must not sit at the neutral default."""
        body = _read(CORE)
        for needle in ('ffmpeg is not installed.',
                       'Frame extraction failed for',
                       'is not working, aborting.'):
            m = re.search(re.escape(needle) + r'[^\n]*', body)
            self.assertIsNotNone(m, f'{needle!r} vanished')
        # A user-requested stop is not a failure; it must not be ERR.
        self.assertRegex(
            body, r"end_processing\(msg:str, level:str = WARN\)",
            'a stop the user asked for is a warning, not an error')
        self.assertRegex(body, r"end_processing\('Finished', OK\)")


class StatusLinesDoNotCorruptTheBar(unittest.TestCase):

    def test_core_does_not_print_status_directly(self):
        body = _strip_comments_and_docstrings(_read(CORE))
        m = re.search(r'def update_status\(.*?\n(?=\ndef |\nclass )', body, re.S)
        self.assertIsNotNone(m, 'update_status not found')
        self.assertNotIn(
            'print(', m.group(0),
            'a bare print() lands in the middle of a live tqdm bar and '
            'terminates it -- route through roop.tui, i.e. bar_write.')

    def test_tui_routes_everything_through_bar_write(self):
        body = _strip_comments_and_docstrings(_read(TUI))
        # The only raw stream writes allowed are the transient spinner row and
        # its erase, which bar_write cannot express.
        self.assertNotIn('print(', body,
                         'permanent output goes through bar_write')
        self.assertIn('bar_write', body)

    def test_a_message_mid_spin_clears_the_spinner_row_first(self):
        body = _read(TUI)
        self.assertRegex(
            body, r'def _emit\(', 'permanent output must funnel through _emit')
        self.assertIn(
            '_erase_locked', _func(body, '_emit'),
            'tqdm.write clears the BAR; it cannot know about the spinner row, '
            'so _emit has to erase it explicitly or the message lands on top '
            'of a half-drawn frame.')


class DegradesWithoutGuessing(unittest.TestCase):

    def test_no_color_is_honoured_by_presence(self):
        body = _read(TUI)
        self.assertRegex(
            body, r'environ\.get\("NO_COLOR"\)\s+is not None',
            'no-color.org specifies NO_COLOR applies when PRESENT, whatever '
            'its value -- NO_COLOR=0 must still disable colour.')

    def test_glyphs_degrade_on_encodability_not_on_isatty(self):
        body = _read(TUI)
        fn = _func(body, '_glyphs_ok')
        self.assertTrue(fn, '_glyphs_ok is gone')
        self.assertIn('.encode(enc)', fn,
                      'the question is whether the stream can carry the glyph')
        # Against the CODE, not the prose: the docstring says "rather than to
        # isatty()", which is the opposite of what a naive substring search
        # concludes from it.
        self.assertNotIn(
            'isatty', _func(_strip_comments_and_docstrings(body), '_glyphs_ok'),
            'glyph choice must not depend on whether the far end is a tty -- '
            'Pinokio is a pipe that renders UTF-8 perfectly well.')

    def test_every_level_has_an_ascii_fallback(self):
        import roop.tui as tui
        for level, entry in tui._STYLE.items():
            self.assertEqual(len(entry), 4, f'{level} lost a field')
            _color, _uni, ascii_glyph, tag = entry
            ascii_glyph.encode('ascii')      # raises if it is not ASCII
            tag.encode('ascii')

    def test_terminal_width_is_requeried_not_cached(self):
        """SIGWINCH does not exist on Windows; asking the OS each time does."""
        body = _read(TUI)
        self.assertIn('get_terminal_size', _func(body, '_width'))
        self.assertNotRegex(body, r'^_WIDTH\s*=', 'width must not be cached')


class SpinnerIsForIndeterminateWorkOnly(unittest.TestCase):

    def test_it_does_not_animate_off_a_terminal(self):
        import roop.tui as tui
        body = _read(TUI)
        m = re.search(r'def _run\(self\).*', body, re.S)
        self.assertIn('if self._tty:', m.group(0),
                      'a captured log must get a heartbeat, not ten frames a '
                      'second of cursor codes -- the same split ChunkedProgress '
                      'makes, for the same reason.')

    def test_it_never_swallows_an_exception(self):
        import roop.tui as tui
        with self.assertRaises(RuntimeError):
            with tui.Spinner('probe', every=99):
                raise RuntimeError('boom')

    def test_the_frame_loop_still_uses_chunkedprogress(self):
        """A Spinner where a total exists would throw away the ETA the web UI
        reads. Frames have a total; they stay on ChunkedProgress."""
        mgr = _read(os.path.join(APP, 'roop', 'ProcessMgr.py'))
        self.assertIn('ChunkedProgress(total=self.total_frames', mgr)
        self.assertNotIn('Spinner(', mgr)


if __name__ == '__main__':
    unittest.main()
