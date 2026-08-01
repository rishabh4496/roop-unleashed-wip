"""Light themes must repoint `--color-white`, and every one of them must be listed.

Tailwind v4 compiles the whole `*-white` utility family — text, background,
border, ring, at every alpha, under every variant — into a single expression:
`color-mix(in oklab, var(--color-white) N%, transparent)`. On a dark theme that
variable is `#fff` and everything reads correctly. On a LIGHT theme, anything
still resolving to white is invisible.

The bug this guards was live in both directions:

  * The old fix enumerated `.text-white\\/30 … \\/95` and set `opacity`. It
    therefore missed every alpha nobody thought to list (/15 /20 /25 /35 /45
    /55 /85 — 88 uses) and every variant-prefixed one, so `hover:text-white`
    (55 uses) turned invisible the moment you hovered it.
  * A theme added to `themes.js` with `mode: 'light'` but not to the selector
    list in `index.css` gets NO correction at all, and is white-on-white
    everywhere.

The second is the one that will happen again: the two files are far apart and
nothing but this test connects them. Neither failure shows up in a diff, in
oxlint, or in the build — the CSS is valid and the app renders; the text is
just not there.
"""

import os
import re
import unittest

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(os.path.dirname(APP), 'react-ui', 'src')
CSS = os.path.join(SRC, 'index.css')
THEMES = os.path.join(SRC, 'themes.js')

# The `className` + `mode` pair off one THEMES entry in themes.js.
THEME_ENTRY = re.compile(
    r"className:\s*'(?P<cls>[^']*)'.*?mode:\s*'(?P<mode>\w+)'", re.S)


def _read(path):
    with open(path, encoding='utf-8') as fh:
        return fh.read()


def _light_theme_block(css):
    """The selector list + body of the light-theme correction block."""
    start = css.index('Light Theme Readability')
    # First `{` after the selector list, then match to its balanced close.
    brace = css.index('{', start)
    depth, i = 0, brace
    while i < len(css):
        if css[i] == '{':
            depth += 1
        elif css[i] == '}':
            depth -= 1
            if depth == 0:
                break
        i += 1
    return css[start:brace], css[brace:i + 1]


class LightThemeCorrection(unittest.TestCase):
    def test_every_light_theme_is_corrected(self):
        css = _read(CSS)
        selectors, _body = _light_theme_block(css)

        declared = [m.group('cls') for m in THEME_ENTRY.finditer(_read(THEMES))
                    if m.group('mode') == 'light' and m.group('cls')]
        self.assertTrue(declared, 'no light themes parsed out of themes.js — '
                                  'the THEMES entry shape must have changed')

        missing = [c for c in declared if f'.{c}' not in selectors]
        self.assertEqual(
            missing, [],
            'these themes are mode:"light" in themes.js but are not in the '
            'light-theme block in index.css, so --color-white still resolves '
            'to #fff and their text is white-on-white: ' + ', '.join(missing))

    def test_the_correction_repoints_the_variable(self):
        _selectors, body = _light_theme_block(_read(CSS))
        self.assertRegex(
            body, r'--color-white:\s*var\(--text-main\)',
            'the light-theme block must repoint --color-white at the theme ink '
            'colour; that one variable is what every *-white utility reads')

    def test_dark_surfaces_keep_white_ink(self):
        """Scrims and the HUD stay dark in light themes, so they opt back in."""
        _selectors, body = _light_theme_block(_read(CSS))
        self.assertIn('[class*="bg-black/"]', body)
        self.assertIn('.hud-glass', body)
        self.assertRegex(body, r'--color-white:\s*#fff')

    def test_no_per_alpha_text_enumeration_returns(self):
        """The enumeration is the bug, not the fix — it cannot cover variants."""
        _selectors, body = _light_theme_block(_read(CSS))
        enumerated = re.findall(r'\.text-white\\/\d+', body)
        self.assertEqual(
            enumerated, [],
            'listing individual text-white alphas cannot cover hover:, '
            'group-hover: or placeholder: variants, and silently misses any '
            'alpha not written down — repoint --color-white instead: '
            + ', '.join(enumerated))


if __name__ == '__main__':
    unittest.main()
