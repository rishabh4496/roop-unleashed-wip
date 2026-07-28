"""Settings wiring — that a control reaching the UI actually reaches the render.

The recurring failure here is a HALF-WIRED setting: a toggle that appears in the
UI, saves fine, and silently changes nothing, because one of the places it has to
be listed was missed. It has bitten at least twice.

The frontend half is now structural rather than tested: FaceSwap.jsx builds the
request body and the cache signature from one `buildPreviewPayload`, and the
signature IS the request body with the frame coordinates zeroed, so a setting
cannot be sent without being keyed or keyed without being sent.

The backend half cannot be collapsed the same way — `preview()` sets globals
while the run path also passes kwargs — so it is guarded here instead: these
tests read the actual source and assert the two sides agree, with every
exception named and justified rather than merely tolerated.

Source parsing (rather than importing) is deliberate: it works across the
Python/JavaScript boundary and needs no browser or running server.
"""

import os
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = Path(__file__).resolve().parents[2]
FACESWAP_JSX = REPO / "react-ui" / "src" / "components" / "FaceSwap.jsx"
API_PY = REPO / "app" / "api.py"


def _match_braces(text, start):
    depth, i = 0, start
    while True:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
        i += 1


def frontend_payload_keys():
    """Every key FaceSwap.jsx sends to /api/preview."""
    src = FACESWAP_JSX.read_text(encoding="utf-8")
    m = re.search(r"const buildPreviewPayload = .*?=> \(", src, re.S)
    assert m, "buildPreviewPayload not found — did the preview builder get renamed?"
    body = _match_braces(src, src.index("{", m.end() - 1))
    keys = set(re.findall(r"(?:^|[\{,]|\n)\s*([a-z_][a-z_0-9]*)\s*:", body))
    keys.add("index")           # shorthand property
    keys.discard("overrides")   # the spread, not a field
    return keys


def _api_source():
    return API_PY.read_text(encoding="utf-8")


def _function_body(src, marker, from_index=0):
    i = src.index(marker, from_index)
    start = src.rindex("def ", 0, i)
    try:
        end = src.index("\n@app.", i)
    except ValueError:
        end = len(src)
    return src[start:end]


def preview_consumed_keys():
    src = _api_source()
    body = _function_body(src, 'def preview(')
    return set(re.findall(r'payload\.get\(\s*["\']([a-z_0-9]+)["\']', body))


def run_consumed_keys():
    src = _api_source()
    body = _function_body(src, "batch_process_regular(")
    return set(re.findall(r'payload\.get\(\s*["\']([a-z_0-9]+)["\']', body))


# Sent by the frontend but not read via payload.get in preview(): these are
# consumed by _update_mask_offsets_from_payload(payload), called at the top of
# preview() and of the run endpoint.
VIA_MASK_OFFSET_HELPER = {
    "mask_top", "mask_bottom", "mask_left", "mask_right",
    "face_mask_blend", "mouth_mask_blend",
    "mouth_top_scale", "mouth_bottom_scale", "mouth_left_scale", "mouth_right_scale",
}

# Preview-only concepts, meaningless to a full render.
PREVIEW_ONLY = {"index", "frame", "fake_preview", "show_mask_offsets"}

# Run-only: temporal settings that need more than one frame, and output/pipeline
# settings that do not affect pixels. A single-frame preview cannot show these,
# so their absence from preview() is correct, not an oversight.
RUN_ONLY_TEMPORAL = {
    "stabilize_enhancer", "stabilize_enhancer_strength", "stabilize_beta",
    "stabilize_min_cutoff", "track_identities", "temporal_detection",
}
RUN_ONLY_OUTPUT = {
    "keep_frames", "skip_audio", "output_method", "video_method",
    "wait_after_extraction", "interp_after_swap",
    "upscale_after_swap", "upscale_model_after", "target_index",
}


class TestFrontendReachesBackend(unittest.TestCase):
    def test_every_sent_setting_is_consumed(self):
        """A key in the request that the backend never reads is a control that
        silently does nothing — the exact bug this suite exists for."""
        unconsumed = frontend_payload_keys() - preview_consumed_keys() - VIA_MASK_OFFSET_HELPER
        self.assertEqual(unconsumed, set(), f"FaceSwap.jsx sends {sorted(unconsumed)} "
                                            f"but /api/preview never reads them")

    def test_mask_offset_helper_is_actually_called(self):
        """VIA_MASK_OFFSET_HELPER above is only a valid excuse while preview()
        really does call the helper."""
        body = _function_body(_api_source(), 'def preview(')
        self.assertIn("_update_mask_offsets_from_payload(payload)", body)

    def test_the_builder_is_the_only_preview_request_site(self):
        """Any hand-written /api/preview body would sit outside the cache
        signature and reintroduce the stale-preview bug."""
        src = FACESWAP_JSX.read_text(encoding="utf-8")
        literals = re.findall(r"postJSON\('/api/preview',\s*\{", src)
        self.assertEqual(literals, [], "found a hand-built /api/preview body; "
                                       "route it through buildPreviewPayload")
        calls = len(re.findall(r"postJSON\('/api/preview',\s*buildPreviewPayload\(", src))
        self.assertGreaterEqual(calls, 5)

    def test_cache_signature_is_derived_from_the_payload(self):
        """If these ever diverge, a setting can be sent without invalidating the
        cache, which is what made toggles appear to do nothing."""
        src = FACESWAP_JSX.read_text(encoding="utf-8")
        self.assertRegex(src, r"const previewSignature = \([^)]*\) =>\s*"
                              r"JSON\.stringify\(buildPreviewPayload\(")
        self.assertIn("const previewKey = previewSignature(", src)
        self.assertNotIn("const previewKey = JSON.stringify({", src)


class TestPreviewAndRunAgree(unittest.TestCase):
    def test_preview_extras_are_preview_only(self):
        extra = preview_consumed_keys() - run_consumed_keys()
        self.assertEqual(extra - PREVIEW_ONLY, set(),
                         f"preview() reads {sorted(extra - PREVIEW_ONLY)} that the "
                         f"run path ignores — a preview would then not match the render")

    def test_run_extras_are_temporal_or_output_only(self):
        extra = run_consumed_keys() - preview_consumed_keys()
        unexplained = extra - RUN_ONLY_TEMPORAL - RUN_ONLY_OUTPUT - VIA_MASK_OFFSET_HELPER
        self.assertEqual(unexplained, set(),
                         f"the run path reads {sorted(unexplained)} that preview() "
                         f"does not. If it affects a single frame, wire it into "
                         f"preview(); if not, add it to RUN_ONLY_* with a reason")

    def test_run_only_settings_are_not_sent_to_preview(self):
        """Sending a temporal/output setting to the preview endpoint would imply
        it changes the frame, and would bloat the cache signature so unrelated
        changes discard cached previews."""
        leaked = frontend_payload_keys() & (RUN_ONLY_TEMPORAL | RUN_ONLY_OUTPUT)
        self.assertEqual(leaked, set(), f"{sorted(leaked)} are run-only but are in "
                                        f"the preview payload")


class TestSettingsPersistence(unittest.TestCase):
    def test_yaw_align_survives_a_round_trip_as_a_mode(self):
        """Regression: api.py wrapped this in bool(), which flattened the 'pose'
        mode to True and silently selected 'stabilize'."""
        src = _api_source()
        self.assertNotIn('bool(payload.get("yaw_align"', src)
        self.assertEqual(len(re.findall(r'payload\.get\("yaw_align"', src)), 2,
                         "both apply sites must set yaw_align")

    def test_settings_defaults_are_exposed_for_the_ui(self):
        """GET /api/settings returns CFG.__dict__, so a field missing from
        Settings.__init__ never reaches the UI."""
        from settings import Settings
        cfg = Settings(str(REPO / "app" / "config.yaml"))
        for field in ("yaw_align", "refine_landmarks", "swap_model", "mask_engine"):
            self.assertIn(field, cfg.__dict__, f"{field} missing from Settings")

    def test_saved_settings_round_trip(self):
        """Every field the UI can POST back must also be written by save(), or
        it silently resets on restart."""
        from settings import Settings
        cfg = Settings(str(REPO / "app" / "config.yaml"))
        saved = re.findall(r"'([a-z_0-9]+)':\s*self\.",
                           _function_body(
                               (REPO / "app" / "settings.py").read_text(encoding="utf-8"),
                               "def save("))
        for field in ("yaw_align", "refine_landmarks", "jaw_reshape"):
            self.assertIn(field, saved, f"{field} is loaded but never saved")


if __name__ == "__main__":
    unittest.main()
