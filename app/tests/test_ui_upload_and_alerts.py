"""Uploads must report progress, and the finish alert must tell you the truth.

Three regressions this pins, none of which a build or a lint can see:

  * `fetch()` CANNOT report request-body progress — there is no event for it.
    So a multi-gigabyte target posted through fetch showed an indeterminate
    spinner and nothing else: no bytes, no percentage, no rate, no way to tell
    a slow upload from a wedged one, and no way to cancel a file dropped by
    mistake. Only XMLHttpRequest exposes `upload.onprogress`, so a well-meant
    "modernise this to fetch" silently deletes the feature. The check is
    therefore on the mechanism, not on the UI it feeds.

  * The run-finished alert fired on the processing -> idle edge without ever
    reading the error, so a four-hour render that died at 90% — or one stopped
    by hand — got a rising chime and "Render Complete!". The alert exists for
    the moment you are NOT watching the screen, which is exactly when that
    reads as good news.

  * The desktop-alerts toggle was plain useState(false). Every comparable
    preference in this app persists; this one reset on each reload, and a
    Pinokio tab switch IS a reload.
"""

import os
import re
import unittest

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(os.path.dirname(APP), 'react-ui', 'src')
API = os.path.join(SRC, 'api.js')
ALERT = os.path.join(SRC, 'components', 'faceswap', 'useRunCompleteAlert.js')
UTILS = os.path.join(SRC, 'components', 'faceswap', 'utils.js')
DROP = os.path.join(SRC, 'components', 'faceswap', 'FileDrop.jsx')
FACESWAP = os.path.join(SRC, 'components', 'FaceSwap.jsx')


def _read(path):
    with open(path, encoding='utf-8') as fh:
        return fh.read()


class UploadsReportProgress(unittest.TestCase):
    def test_the_uploaders_do_not_go_back_to_fetch(self):
        src = _read(API)
        body = src.split('const xhrUpload', 1)
        self.assertEqual(
            len(body), 2,
            'api.js must upload through an XHR helper — fetch() cannot report '
            'request-body progress, so switching back silently removes the '
            'upload bar with no other symptom')
        for fn in ('export const postFiles', 'export const postFile'):
            tail = src.split(fn, 1)[1].split('\n}', 1)[0]
            self.assertIn(
                'xhrUpload', tail,
                f'{fn} must go through xhrUpload; a bare fetch() here loses '
                'progress reporting AND cancellation')

    def test_progress_reports_the_server_side_phase_too(self):
        """A bar parked at 100% while the server decodes reads as a hang."""
        src = _read(API)
        self.assertIn(
            "phase: 'analyse'", src,
            'xhrUpload must report the switch from sending bytes to waiting on '
            "the server; for a long video that second half is the longer one")
        self.assertRegex(
            src, r'xhr\.upload\.onprogress',
            'the whole point of using XHR is upload.onprogress')

    def test_uploads_can_be_cancelled(self):
        self.assertIn(
            'signal.addEventListener', _read(API),
            'xhrUpload must honour an AbortSignal, or a 4 GB file dropped by '
            'mistake can only be stopped by restarting the app')
        src = _read(FACESWAP)
        self.assertIn('AbortController', src,
                      'the upload call sites must create a controller to cancel with')
        self.assertIn('onCancel', src,
                      'the drop zones must offer the cancel they now support')

    def test_a_cancel_is_not_reported_as_a_failure(self):
        src = _read(FACESWAP)
        self.assertRegex(
            src, r"AbortError",
            'a cancelled upload is an outcome the user asked for and must not '
            'raise a red error toast')


class TheAlertTellsTheTruth(unittest.TestCase):
    def test_the_finish_alert_reads_the_error(self):
        src = _read(ALERT)
        self.assertIn(
            'errorRef', src,
            'useRunCompleteAlert must consult the run error at the finish edge, '
            'or a failed render is announced as a completed one')
        self.assertIn(
            'playFailTone', src,
            'a failed run needs a distinguishable sound — the alert is for when '
            'you are not looking at the screen')

    def test_the_failure_sound_exists_and_differs(self):
        src = _read(UTILS)
        self.assertIn('export const playFailTone', src)
        # Same shape as playChime (WebAudio, no asset — the app is offline and
        # CSP-restricted), but it must not simply re-play the success notes.
        chime = src.split('export const playChime', 1)[1].split('export const', 1)[0]
        fail = src.split('export const playFailTone', 1)[1].split('export const', 1)[0]
        notes = re.compile(r'\[([\d.,\s]+)\]\.forEach')
        cn, fn = notes.search(chime), notes.search(fail)
        self.assertTrue(cn and fn, 'both tones should list their notes inline')
        self.assertNotEqual(
            cn.group(1), fn.group(1),
            'the failure tone must not use the success chime\'s notes')

    def test_the_call_site_passes_the_error_through(self):
        src = _read(FACESWAP)
        call = src.split('useRunCompleteAlert({', 1)[1].split('})', 1)[0]
        self.assertIn(
            'error:', call,
            'FaceSwap must hand the alert progress.error; without it the hook '
            'cannot tell a finished run from a failed one')

    def test_the_desktop_alert_preference_persists(self):
        src = _read(ALERT)
        self.assertIn(
            'localStorage', src,
            'the desktop-alerts toggle must survive a reload — switching '
            'Pinokio tabs reloads the frontend, so a session-only preference '
            'is off again every time you come back')


class TheDropZoneIsOperable(unittest.TestCase):
    def test_the_picker_is_focusable_and_precedes_its_label(self):
        src = _read(DROP)
        self.assertIn(
            'sr-only peer', src,
            'the file input must be sr-only (focusable) rather than hidden '
            '(display:none, removed from the tab order)')
        self.assertLess(
            src.index('sr-only peer'), src.index('<motion.div'),
            "Tailwind's peer-* only reaches a PRECEDING sibling, so the input "
            'must come before the box that shows its focus ring')
        self.assertIn(
            'peer-focus-visible', src,
            'a control focusable but with no visible focus ring is still '
            'unusable from a keyboard')


if __name__ == '__main__':
    unittest.main()
