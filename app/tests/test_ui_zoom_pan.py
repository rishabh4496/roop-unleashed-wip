"""Guards on the shared zoom/pan maths behind the preview stage and the grids.

Every bug this file pins was silent — the zoom still "worked", it just did the
wrong thing, and nothing but a careful eye on a face crop would catch it:

  * `preventDefault()` inside a React `onWheel` is a NO-OP. React attaches
    `wheel` at the root as a passive listener, so the comparison grid zoomed
    AND scrolled the page behind itself on every notch. The only fix is a
    native listener registered with `{ passive: false }`, which is a shape a
    test can check for and a refactor back to `onWheel` cannot fake.

  * Both stages hardcoded `* 1.5` when double-clicking to 2.5x, so the clicked
    point landed 40% off centre. The factor has to be the zoom itself.

  * Pointer deltas are VISUAL pixels (the app-level CSS `zoom` scales them)
    while a transform's `translate()` is in layout pixels, so every drag has to
    divide by the app zoom or the content slides away from the cursor.

These are asserted structurally against the source, in the same style as the
other `test_ui_*` guards — there is no JS runtime in this suite.
"""

import os
import re
import unittest

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(os.path.dirname(APP), 'react-ui', 'src')
FS = os.path.join(SRC, 'components', 'faceswap')
ZOOMPAN = os.path.join(FS, 'zoomPan.js')
GRID = os.path.join(FS, 'CompareGrid.jsx')
PREVIEW = os.path.join(FS, 'InteractivePreview.jsx')
APP_JSX = os.path.join(SRC, 'App.jsx')

STAGES = {'CompareGrid.jsx': GRID, 'InteractivePreview.jsx': PREVIEW}


BLOCK_COMMENT = re.compile(r'/\*.*?\*/', re.S)
LINE_COMMENT = re.compile(r'(?<!:)//[^\n]*')


def _read(path):
    with open(path, encoding='utf-8') as fh:
        return fh.read()


def _code(path):
    """Source with comments stripped.

    These assertions look for things the code must and must not DO, and several
    of them name the very construct being banned. The comment explaining why
    `onWheel` was abandoned contains the word `onWheel`, so a whole-file scan
    fails on the explanation rather than on the code — and the only way to pass
    would be to delete the reasoning. Strip the prose and check the source.
    """
    return LINE_COMMENT.sub('', BLOCK_COMMENT.sub('', _read(path)))


class WheelIsNonPassive(unittest.TestCase):
    def test_neither_stage_zooms_from_a_react_onWheel_prop(self):
        for name, path in STAGES.items():
            src = _code(path)
            # The JSX PROP form specifically — `const onWheel = …` is the
            # native handler these stages now pass to addEventListener.
            self.assertNotRegex(
                src, r'onWheel\s*=\s*\{',
                f'{name} binds wheel through the React onWheel prop, which is '
                'passive — its preventDefault() is ignored and the page '
                'scrolls behind the zoom. Use a native listener instead.')

    def test_both_stages_register_wheel_as_non_passive(self):
        for name, path in STAGES.items():
            src = _code(path)
            self.assertRegex(
                src, r"addEventListener\(\s*'wheel'[^)]*passive:\s*false",
                f'{name} must register its wheel listener with '
                '{ passive: false } or it cannot cancel the page scroll')

    def test_the_wheel_step_can_decline_the_event(self):
        """At a zoom clamp the stage must let the page scroll instead."""
        src = _code(ZOOMPAN)
        self.assertRegex(
            src, r'return\s+Math\.abs\(next - zoom\)\s*<\s*1e-4\s*\?\s*null',
            'wheelZoom must return null when the zoom cannot move, which is '
            'what tells the caller not to swallow the scroll')
        for name, path in STAGES.items():
            src = _code(path)
            self.assertRegex(
                src, r'if\s*\(\s*next === null\s*\)\s*return',
                f'{name} must bail out before preventDefault() when the wheel '
                'cannot change the zoom, or scrolling past a fully zoomed-out '
                'stage traps the page')


class ZoomAnchoring(unittest.TestCase):
    def test_no_stage_hardcodes_the_old_double_click_factor(self):
        """`* 1.5` while zooming to 2.5x put the clicked detail off centre."""
        for name, path in STAGES.items():
            src = _code(path)
            self.assertNotRegex(
                src, r'rect\.(width|height)\s*/\s*2\s*-\s*click',
                f'{name} still centres a double-click with its own inline '
                'arithmetic; use panCenteringAt so the factor is the zoom')

    def test_centering_uses_the_zoom_as_the_factor(self):
        src = _code(ZOOMPAN)
        self.assertRegex(
            src, r'x:\s*-u\.x\s*\*\s*zoom',
            'panCenteringAt must scale the clicked offset by the zoom being '
            'applied — any other constant lands the click off centre')

    def test_wheel_zoom_is_anchored_at_the_pointer(self):
        for name, path in STAGES.items():
            src = _code(path)
            self.assertIn(
                'panAnchoredAt', src,
                f'{name} must anchor its wheel zoom at the cursor; a '
                'centre-anchored zoom pushes the region being inspected off '
                'the edge on the way in')


class AppZoomCompensation(unittest.TestCase):
    """The app-level CSS zoom scales pointer coordinates but not translate()."""

    def test_ui_scale_is_measured_from_the_element(self):
        src = _code(ZOOMPAN)
        self.assertRegex(
            src, r'getBoundingClientRect\(\)\.width\s*/\s*w',
            'uiScale must derive the app zoom by comparing the visual rect to '
            'offsetWidth, not by reading documentElement.style.zoom')

    def test_both_stages_divide_drag_deltas_by_it(self):
        for name, path in STAGES.items():
            src = _code(path)
            self.assertIn(
                'uiScale', src,
                f'{name} must divide its pointer deltas by uiScale(), or the '
                'content lags the cursor at any app zoom but 100%')

    def test_pan_is_bounded(self):
        for name, path in STAGES.items():
            src = _code(path)
            self.assertIn(
                'clampPan', src,
                f'{name} must clamp its pan, or the image can be dragged out '
                'of the box entirely with no way back but the reset button')

    def test_ctrl_wheel_is_routed_to_the_app_zoom(self):
        """Otherwise the browser's page zoom stacks on top of the CSS zoom."""
        src = _code(APP_JSX)
        self.assertRegex(
            src, r"addEventListener\(\s*'wheel'[^)]*passive:\s*false",
            'App must claim Ctrl+wheel as a non-passive listener, or the '
            'browser zooms the page on top of the app zoom and the toolbar '
            'percentage goes stale')


class GridSizing(unittest.TestCase):
    def test_the_grid_height_follows_the_row_count(self):
        """A 2x2 comparison needs twice the box of a single row per cell."""
        src = _code(GRID)
        self.assertRegex(
            src, r'rows\s*=\s*Math\.max\(1,\s*Math\.ceil\(',
            'CompareGrid must derive its row count from the item count')
        self.assertRegex(
            src, r'gridTemplateRows',
            'CompareGrid must lay its rows out explicitly, or a cell still '
            'showing a spinner collapses to the height of that spinner')

    def test_the_grid_is_no_longer_capped_at_45vh(self):
        src = _code(GRID)
        self.assertNotIn(
            'max-h-[45vh]', src,
            'the comparison grid was capped at 45vh, which left each cell of a '
            '2x2 around 200px tall — too small to judge the differences the '
            'grid exists to show')


if __name__ == '__main__':
    unittest.main()
