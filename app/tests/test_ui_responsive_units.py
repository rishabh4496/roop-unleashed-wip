"""Lengths that are physical distances must be measured in pixels, not fractions.

Two families of bug live here, and both only show up on hardware the author of
the code did not have.

1. VIEWPORT UNITS UNDER THE APP ZOOM.  App.jsx drives an app-level CSS `zoom`
   on <html> (0.5x-1.6x).  `zoom` scales the used value of every length, but
   viewport units compute against the UNZOOMED viewport and are not divided
   back out -- so a plain `92vh` paints at 92% * zoom of the real window.  At
   1.6x that is 147%, which put the footer of every `max-h-[92vh]` modal below
   the bottom of the screen with nothing able to scroll it back into view, and
   gave `body { min-height: 100vh }` a permanent scrollbar on every tab.  The
   fix is `--vh`/`--vw`, which divide `--app-zoom` back out; the rule is that
   no raw `Nvh`/`Nvw` survives in a className.

2. PIXEL DISTANCES WRITTEN AS FRACTIONS.  A grab radius, half a label and half
   a hover card are all distances on the user's screen.  Written as a fixed
   share of a container they silently scale with the display: the timeline
   In/Out grab zone was 0.04 of the track, which is 20px on a 500px column but
   196px on a 32:9 screen with both side panels hidden -- wide enough that
   ordinary scrubs near the clip start registered as trims instead.  Each of
   these has to divide a pixel constant by a MEASURED width.

The slider case is the same shape one level down: a range thumb's centre never
reaches 0% or 100% of its track (the browser insets it by half a thumb), so a
fill or a pip drawn at a plain `percent%` disagrees with the thumb by up to
7.5px.  That is 2% of a one-column track and 8% of a six-column one, which is
why it read as a display-dependent bug rather than a constant offset.
"""

import os
import re
import unittest

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(os.path.dirname(APP), 'react-ui', 'src')
CSS = os.path.join(SRC, 'index.css')
APP_JSX = os.path.join(SRC, 'App.jsx')
FACESWAP = os.path.join(SRC, 'components', 'FaceSwap.jsx')
TIMELINE = os.path.join(SRC, 'components', 'faceswap', 'Timeline.jsx')
TRACKER = os.path.join(SRC, 'components', 'faceswap', 'SliderTrackerBar.jsx')
PREVIEW = os.path.join(SRC, 'components', 'faceswap', 'InteractivePreview.jsx')

# `[54vh]`, `[95vw]` -- the Tailwind arbitrary-value form.
RAW_VIEWPORT = re.compile(r'\[[^\]]*?\d+v[hw]\b')


def _read(path):
    with open(path, encoding='utf-8') as fh:
        return fh.read()


def _without_comments(src):
    """Strip // and /* */ comments.

    Comments legitimately quote the OLD code -- CompareGrid explains why it no
    longer uses `aspect-video max-h-[45vh]` -- and a guard that cannot tell a
    quotation from a use would force those explanations to be deleted.
    """
    src = re.sub(r'/\*.*?\*/', '', src, flags=re.S)
    return re.sub(r'//[^\n]*', '', src)


def _jsx_files():
    for root, _dirs, files in os.walk(SRC):
        for name in files:
            if name.endswith('.jsx'):
                yield os.path.join(root, name)


class ViewportUnitsSurviveTheAppZoom(unittest.TestCase):

    def test_the_zoom_factor_is_published_to_css(self):
        src = _read(APP_JSX)
        self.assertIn('style.zoom', src)
        self.assertRegex(
            src, r"setProperty\(\s*'--app-zoom'",
            'App.jsx sets html { zoom } but does not publish the factor, so '
            'nothing in CSS can divide it back out of a viewport unit.')

    def test_css_defines_the_corrected_units_off_the_theme_surface(self):
        css = _read(CSS)
        block = re.search(r'html\s*\{(.*?)\n\}', css, re.S)
        self.assertIsNotNone(
            block, 'index.css must declare --vh/--vw on `html`')
        body = block.group(1)
        for name in ('--app-zoom', '--vh', '--vw', '--range-thumb'):
            self.assertIn(name, body, f'{name} missing from the html block')
        self.assertRegex(body, r'--vh:\s*calc\(\s*1vh\s*/\s*var\(--app-zoom\)')
        self.assertRegex(body, r'--vw:\s*calc\(\s*1vw\s*/\s*var\(--app-zoom\)')

        # These four are layout mechanics, identical under all 38 themes.
        # :root is the THEME surface -- test_ui_light_themes asserts that
        # deriveThemeVars emits everything declared there, so putting a
        # structural variable in :root either breaks that test or forces every
        # theme to restate a constant it has no business overriding.
        root = re.search(r':root\s*\{(.*?)\n\}', css, re.S)
        self.assertIsNotNone(root)
        for name in ('--vh', '--vw', '--app-zoom'):
            self.assertNotIn(name, root.group(1))

    def test_no_raw_viewport_unit_survives_in_a_class_name(self):
        offenders = []
        for path in _jsx_files():
            for i, line in enumerate(_without_comments(_read(path)).split('\n'), 1):
                if RAW_VIEWPORT.search(line):
                    rel = os.path.relpath(path, SRC).replace(os.sep, '/')
                    offenders.append(f'{rel}:{i}')
        self.assertEqual(
            offenders, [],
            'these render at N% * app-zoom of the real screen; use '
            'calc(N*var(--vh)) / calc(N*var(--vw)): ' + ', '.join(offenders))

    def test_the_page_floor_is_corrected_too(self):
        css = _read(CSS)
        self.assertNotRegex(
            css, r'min-height:\s*100vh',
            'body sits inside the zoomed <html>, so a bare 100vh floor is '
            '160% of the window at 1.6x and puts a scrollbar on every tab.')


class PixelDistancesAreNotFractions(unittest.TestCase):

    def test_timeline_grab_radius_is_derived_from_the_measured_track(self):
        src = _read(FACESWAP)
        m = re.search(r'const\s+tolerance\s*=\s*([^;]+);', src)
        self.assertIsNotNone(m, 'the In/Out hit test lost its tolerance')
        expr = m.group(1)
        self.assertIn(
            'rect.width', expr,
            'a grab radius is a distance on screen. As a bare fraction it was '
            '20px on a narrow column and 196px on a 32:9 track, so clicks a '
            'fifth of a second from the In point became trims: ' + expr)

    def test_offscreen_handles_cannot_win_the_hit_test(self):
        src = _read(FACESWAP)
        self.assertRegex(
            src, r'p\s*<\s*0\s*\|\|\s*p\s*>\s*1\s*\?\s*Infinity',
            'pctOfFrame returns <0 or >1 for a handle scrolled outside the '
            'visible window; those have no on-screen position to be near.')

    def test_timeline_label_and_hover_card_clamp_in_pixels(self):
        src = _read(TIMELINE)
        for name in ('edgePct', 'hoverHalfPct'):
            m = re.search(rf'const\s+{name}\s*=\s*([^;]+);', src)
            self.assertIsNotNone(m, f'{name} is gone')
            self.assertIn(
                'trackW', m.group(1),
                f'{name} sizes a fixed-pixel element, so it has to be '
                'converted through the measured track width: ' + m.group(1))

        # The clamp must actually consume it, not just compute it.
        self.assertRegex(
            src, r'clamp\(\s*pctOf\(hoverFrame\)\s*,\s*hoverHalfPct',
            'the hover card is 190px wide and centred on the pointer; a fixed '
            '9%/91% clamp let it hang off a narrow track and detached it by '
            '440px on a wide one.')


class RangeFillFollowsTheThumb(unittest.TestCase):

    def test_thumb_size_is_written_once(self):
        css = _read(CSS)
        self.assertNotRegex(
            css, r'::-webkit-slider-thumb\s*\{[^}]*width:\s*\d+px',
            'the thumb size is spelled in CSS and consumed by the fill maths '
            'in SliderTrackerBar; route it through --range-thumb so the two '
            'cannot drift apart.')
        self.assertIn('--moz-range-thumb'.replace('--', '::-'), css)

    def test_fill_and_pip_both_offset_by_half_a_thumb(self):
        src = _read(TRACKER)
        self.assertRegex(
            src, r'const\s+thumbAt\s*=', 'the thumb-centre helper is gone')
        m = re.search(r'const\s+thumbAt\s*=\s*\(percent\)\s*=>\s*\n?\s*(`[^`]+`)', src)
        self.assertIsNotNone(m, 'thumbAt must return a CSS length')
        self.assertIn('var(--range-thumb)', m.group(1))
        self.assertIn('0.5', m.group(1))

        # Both consumers, or the one left behind is the visible bug.
        gradient = re.search(r'background:\s*`linear-gradient\(to right[^`]*`', src)
        self.assertIsNotNone(gradient)
        self.assertIn(
            'thumbAt(percent)', gradient.group(0),
            'the accent fill must end where the thumb centre actually is')
        self.assertRegex(
            src, r'left:\s*thumbAt\(pct\(s\.defaultVal\)\)',
            'the default-value pip sat visibly off the thumb whenever a '
            'slider was at its default.')


class GridsAskTheRightContainer(unittest.TestCase):
    """Column counts inside the collapsible columns must not key off the window.

    FaceSwap's two side panels hide independently of the viewport, moving the
    centre column by up to ~1020px without moving a single breakpoint. The
    tab-level grids (Extras, FaceManager) genuinely do span the window and are
    left alone; these three do not.
    """

    CONTAINER_SIZED = [
        os.path.join(SRC, 'components', 'faceswap', 'SliderTrackerBar.jsx'),
        os.path.join(SRC, 'components', 'Settings.jsx'),
        os.path.join(SRC, 'components', 'ui.jsx'),
    ]

    def test_no_ultrawide_column_steps_in_container_sized_grids(self):
        offenders = []
        for path in self.CONTAINER_SIZED:
            src = _without_comments(_read(path))
            for hit in re.findall(r'\d?xl:grid-cols-\d+', src):
                offenders.append(
                    f'{os.path.relpath(path, SRC).replace(os.sep, "/")}: {hit}')
        self.assertEqual(
            offenders, [],
            'these grids are narrower than the window by an amount the window '
            'cannot report; state the minimum cell size and let auto-fit '
            'solve the count: ' + ', '.join(offenders))

    def test_they_state_an_intrinsic_minimum_instead(self):
        for path in self.CONTAINER_SIZED:
            self.assertIn(
                'repeat(auto-fit,minmax(', _read(path),
                f'{os.path.basename(path)} lost its intrinsic column sizing')

    def test_the_centre_column_is_not_a_containment_context(self):
        """`container-type` would trap the fixed-position modals inside it."""
        src = _read(FACESWAP)
        self.assertNotIn(
            '@container', _without_comments(src),
            'container-type applies `contain: layout inline-size style`, '
            'which makes the column a containing block for fixed-position '
            'descendants -- the modals would be confined to the column.')


class PreviewStageTakesTheMediaShape(unittest.TestCase):

    def test_the_stage_is_not_pinned_to_16_9(self):
        src = _without_comments(_read(PREVIEW))
        self.assertNotIn(
            'aspect-video max-h', src,
            'given a definite width and a clamped height, aspect-ratio is the '
            'constraint CSS drops -- so `aspect-video max-h-[54vh]` resolved '
            'to roughly 5.5:1 on a 32:9 display rather than to 16:9.')

    def test_the_stage_ratio_comes_from_the_measured_media(self):
        src = _read(PREVIEW)
        self.assertRegex(src, r'const\s+mediaRatio\s*=\s*imgDim')
        m = re.search(r'const\s+stageStyle\s*=\s*(.*?)\n  \};', src, re.S)
        self.assertIsNotNone(m, 'stageStyle is gone')
        self.assertIn('aspectRatio', m.group(1))
        self.assertIn(
            'splitMode', m.group(1),
            'split view shows the media twice side by side, so its natural '
            'stage is twice as wide as one pane.')

    def test_no_min_height_fights_the_max_height(self):
        src = _without_comments(_read(PREVIEW))
        self.assertNotIn(
            'min-h-[260px]', src,
            'min-height wins over max-height, so under ~481px of viewport '
            'height the two swapped roles and the stage overflowed.')


if __name__ == '__main__':
    unittest.main()
