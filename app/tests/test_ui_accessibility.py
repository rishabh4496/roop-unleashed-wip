"""Two accessibility rules that fail silently, so they are checked mechanically.

Neither shows up in a diff, in oxlint, or in the build. Both were live:

  * `Toggle` rendered its real <input type="checkbox"> with Tailwind's `hidden`,
    which is display:none — that takes an element out of the tab order
    ENTIRELY. The visible switch beside it is a <div>, so there was nothing
    focusable at all: every toggle in the app (most of Settings, most of the
    Face Swap panel) could not be reached or operated from a keyboard. The fix
    is `sr-only`, which hides it visually but keeps it focusable, so this
    checks the class rather than trusting the memory of why.

  * An icon-only button whose whole content is an emoji or an SVG has no
    accessible name. `title` is not one: it is a tooltip, exposed
    inconsistently, and never on touch. Such a button is announced as
    "button" — the row of them in the preview HUD was unusable without sight.

Both rules are about controls that LOOK fine and simply cannot be operated, so
the cost of a regression is total for the people affected and invisible to
everyone else.
"""

import os
import re
import unittest

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(os.path.dirname(APP), 'react-ui', 'src')

BUTTON = re.compile(r'<button\b(?P<attrs>(?:[^>]|\n)*?)>(?P<body>(?:.|\n)*?)</button>', re.S)
# A form control hidden with display:none is unreachable; `sr-only` is the
# visually-hidden-but-focusable idiom and is what these must use instead.
HIDDEN_INPUT = re.compile(r'<input\b(?:[^>]|\n)*?className="[^"]*\bhidden\b[^"]*"', re.S)


def _jsx_files():
    for root, _dirs, names in os.walk(SRC):
        for name in sorted(names):
            if name.endswith('.jsx'):
                yield os.path.join(root, name)


def _rel(path):
    return os.path.relpath(path, SRC).replace('\\', '/')


def _visible_text(body):
    """What a sighted user reads on the button, minus markup and expressions."""
    text = re.sub(r'<svg(?:.|\n)*?</svg>', '', body, flags=re.S)
    text = re.sub(r'\{[^{}]*\}', '', text)
    text = re.sub(r'<[^>]*>', '', text)
    return text.strip()


class AccessibleNames(unittest.TestCase):
    def test_icon_only_buttons_have_an_accessible_name(self):
        offenders = []
        for path in _jsx_files():
            with open(path, encoding='utf-8') as fh:
                src = fh.read()
            for m in BUTTON.finditer(src):
                if 'aria-label' in m.group('attrs'):
                    continue
                text = _visible_text(m.group('body'))
                # Any ASCII letter is a readable name already.
                if re.search(r'[A-Za-z]', text):
                    continue
                # A button with no content at all is a layout element, not a
                # control a user is meant to find; the ones that matter here are
                # the emoji/SVG icons.
                if not text and '<svg' not in m.group('body'):
                    continue
                line = src[:m.start()].count('\n') + 1
                offenders.append(f'{_rel(path)}:{line}')
        self.assertEqual(offenders, [],
                         'icon-only buttons with no aria-label (a `title` is a '
                         'tooltip, not an accessible name): ' + ', '.join(offenders))


class FocusableControls(unittest.TestCase):
    def test_no_form_control_is_hidden_with_display_none(self):
        """…unless a real, focusable control drives it.

        The exempt pattern is a `ref`'d file input clicked by a <button> beside
        it: there the button IS the control, it is in the tab order, and hiding
        the input is correct. What is not correct — and is what six file pickers
        here did — is hiding the input inside a bare <label>: a <label> is not a
        tab stop, so nothing focusable was left and the picker could not be
        opened from a keyboard at all.
        """
        offenders = []
        for path in _jsx_files():
            with open(path, encoding='utf-8') as fh:
                src = fh.read()
            for m in HIDDEN_INPUT.finditer(src):
                if 'ref=' in m.group(0):
                    continue
                line = src[:m.start()].count('\n') + 1
                offenders.append(f'{_rel(path)}:{line}')
        self.assertEqual(offenders, [],
                         'these inputs use `hidden` (display:none), which removes '
                         'them from the tab order, and no focusable control drives '
                         'them — use `sr-only` so they stay focusable: '
                         + ', '.join(offenders))

    def test_the_toggle_switch_is_operable_from_the_keyboard(self):
        """Belt and braces on the specific control that was broken: the checkbox
        must be sr-only AND a `peer`, so the focus ring lands on the visible
        switch rather than on an element nobody can see."""
        with open(os.path.join(SRC, 'components', 'ui.jsx'), encoding='utf-8') as fh:
            src = fh.read()
        toggle = src.split('export const Toggle', 1)[1].split('export const', 1)[0]
        self.assertIn('sr-only peer', toggle)
        self.assertIn('peer-focus-visible', toggle,
                      'the switch must show a focus ring when the input behind it '
                      'is focused, or keyboard users cannot see where they are')


class DisclosureState(unittest.TestCase):
    def test_collapsible_sections_report_whether_they_are_open(self):
        with open(os.path.join(SRC, 'components', 'ui.jsx'), encoding='utf-8') as fh:
            src = fh.read()
        section = src.split('export const Section', 1)[1].split('export const', 1)[0]
        self.assertIn('aria-expanded', section)


if __name__ == '__main__':
    unittest.main()
