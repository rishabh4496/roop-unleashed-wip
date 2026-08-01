"""Light themes must repoint `--color-white`, and none may be able to miss it.

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
  * The fix that replaced it enumerated the eight light theme CLASS names
    instead. A theme added to `themes.js` with `mode: 'light'` but not to that
    list got no correction at all and was white-on-white everywhere — and a
    user-authored theme could never be in the list, because it has no class.

Both were the same shape of mistake: a hand-maintained list, far from the thing
it mirrors, whose failure is silent. The CSS stays valid and the app still
renders; the text is simply not there.

So the correction is now keyed on `data-theme-mode`, which `applyThemeToDom`
derives from the resolved theme's own `mode` field. There is no list to keep in
step, and these tests hold that mechanism in place rather than re-checking a
list that no longer exists.
"""

import os
import re
import unittest

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(os.path.dirname(APP), 'react-ui', 'src')
CSS = os.path.join(SRC, 'index.css')
THEMES = os.path.join(SRC, 'themes.js')


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
    def test_correction_is_keyed_on_the_mode_attribute(self):
        selectors, _body = _light_theme_block(_read(CSS))
        self.assertIn(
            '[data-theme-mode="light"]', selectors,
            'the light-theme block must key off data-theme-mode, which is '
            'derived from each theme\'s own `mode`. Anything else has to be '
            'maintained by hand and will eventually miss a theme.')

    def test_no_per_theme_class_list_returns(self):
        """A list of `.theme-*-light` classes is the bug, not the fix."""
        selectors, _body = _light_theme_block(_read(CSS))
        enumerated = re.findall(r'\.theme-[\w-]*light\b', selectors)
        self.assertEqual(
            enumerated, [],
            'listing light themes by class name cannot cover a user-authored '
            'theme (it has no class) and silently misses any preset nobody '
            'remembered to add — key off data-theme-mode instead: '
            + ', '.join(enumerated))

    def test_the_mode_attribute_is_actually_set_from_the_theme(self):
        """The CSS above is inert unless something writes the attribute."""
        themes = _read(THEMES)
        self.assertRegex(
            themes,
            r"setAttribute\(\s*'data-theme-mode'\s*,[^)]*mode",
            "applyThemeToDom in themes.js must set data-theme-mode from the "
            "theme's `mode` field — without it every light theme renders "
            "white-on-white and the block in index.css never matches")

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


class CustomThemeDerivation(unittest.TestCase):
    """A custom theme must define everything a preset block defines.

    A missing variable does not error — it falls through to whatever the
    :root default left behind, so a custom theme would silently wear one of
    the default theme's colours. The light-only `--input-bg-focus` is the
    subtle one: without it, focusing an input on a light custom theme turns
    its background dark while the ink stays dark (see `.glass-input:focus`).
    """

    def test_derivation_covers_the_root_variable_set(self):
        vars_js = _read(os.path.join(SRC, 'themeVars.js'))
        css = _read(CSS)

        # Everything :root declares is what a theme is made of.
        root = css[css.index(':root {'):]
        root = root[:root.index('\n}')]
        declared = set(re.findall(r'(--[\w-]+)\s*:', root))

        derived = set(re.findall(r"'(--[\w-]+)':", vars_js))
        missing = sorted(declared - derived)
        self.assertEqual(
            missing, [],
            'deriveThemeVars must emit every variable :root defines, or a '
            'custom theme inherits the default theme for these: '
            + ', '.join(missing))

    def test_light_custom_themes_get_the_input_focus_fix(self):
        vars_js = _read(os.path.join(SRC, 'themeVars.js'))
        self.assertIn(
            '--input-bg-focus', vars_js,
            'light themes need --input-bg-focus or focusing an input paints it '
            'dark under dark ink — the preset light blocks set it, so the '
            'derivation must too')


if __name__ == '__main__':
    unittest.main()
