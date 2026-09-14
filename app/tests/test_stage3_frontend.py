"""Unit and regression tests for Stage 3: React Frontend Integration.

Validates:
1. TypeScript interfaces and schemas in react-ui/src/types.ts
2. Enhancer dropdown options in FaceSwap.jsx and BatchSwap.jsx
3. Dynamic slider for Enhancer Blend / Opacity conditionally rendered on active enhancer
4. Preset configurations (defaults.js, PresetStudioModal.jsx, QualityProfilesModal.jsx)
5. Payload builders synchronization with enhancer_type and enhancer_blend
"""

import os
import re
import unittest

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_DIR = os.path.dirname(APP_DIR)
REACT_UI_DIR = os.path.join(REPO_DIR, "react-ui", "src")


def _read_file(*parts):
    path = os.path.join(REACT_UI_DIR, *parts)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class TestStage3FrontendIntegration(unittest.TestCase):

    def test_typescript_definitions_exist_and_cover_enhancers(self):
        content = _read_file("types.ts")
        self.assertIn("'gpen_realistic'", content)
        self.assertIn("'ultramax'", content)
        self.assertIn("'GPEN Realistic'", content)
        self.assertIn("'UltraMax'", content)
        self.assertIn("export interface EnhancerSettings", content)
        self.assertIn("export interface ProcessVideoRequest", content)
        self.assertIn("export interface PreviewRequest", content)
        self.assertIn("export interface FaceSwapState", content)

    def test_faceswap_enhancers_dropdown_and_conditional_slider(self):
        content = _read_file("components", "FaceSwap.jsx")

        # Fallback list has GPEN Realistic and UltraMax
        self.assertIn("'GPEN Realistic'", content)
        self.assertIn("'UltraMax'", content)

        # Post-processing enhancer Select has correct options and wires enhancer_type
        self.assertIn("options={Array.isArray(meta?.enhancers)", content)
        self.assertIn("set('selected_enhancer', v)", content)
        self.assertIn("set('enhancer_type', v)", content)

        # Dynamic slider for Enhancer Blend / Opacity
        self.assertIn("Enhancer Blend / Opacity", content)
        # Conditionally rendered only when an enhancer is selected and != 'None'
        self.assertIn("p.selected_enhancer && p.selected_enhancer !== 'None'", content)
        self.assertIn("set('blend_ratio', v)", content)
        self.assertIn("set('enhancer_blend', v)", content)

    def test_faceswap_payload_builders_include_new_params(self):
        content = _read_file("components", "FaceSwap.jsx")

        # buildSwapPayload
        self.assertIn("enhancer_type: sp.enhancer_type || sp.selected_enhancer", content)
        self.assertIn("enhancer_blend: num(sp.enhancer_blend !== undefined ? sp.enhancer_blend : sp.blend_ratio, 0.85)", content)

        # buildPreviewPayload
        self.assertIn("enhancer_type: activeParams.enhancer_type || activeParams.selected_enhancer", content)
        self.assertIn("enhancer_blend: num(activeParams.enhancer_blend !== undefined ? activeParams.enhancer_blend : activeParams.blend_ratio, 0.85)", content)

    def test_batchswap_enhancers_and_payload_builder(self):
        content = _read_file("components", "BatchSwap.jsx")

        # Fallback arrays
        self.assertIn("'GPEN Realistic'", content)
        self.assertIn("'UltraMax'", content)

        # Dynamic mode1 slider
        self.assertIn("Enhancer Blend / Opacity", content)
        self.assertIn("mode1Enhancer && mode1Enhancer !== 'None'", content)

        # Payload building
        self.assertIn("enhancer_type: overrides.enhancer_type", content)
        self.assertIn("enhancer_blend: parseFloat", content)

    def test_preset_recipes_and_profiles(self):
        preset_content = _read_file("components", "faceswap", "PresetStudioModal.jsx")
        self.assertIn("id: 'ultramax_studio'", preset_content)
        self.assertIn("id: 'gpen_realistic'", preset_content)

        profiles_content = _read_file("components", "QualityProfilesModal.jsx")
        self.assertIn("id: 'ultramax'", profiles_content)
        self.assertIn("id: 'gpen_realistic'", profiles_content)

    def test_faceswap_defaults(self):
        defaults_content = _read_file("components", "faceswap", "defaults.js")
        self.assertIn("enhancer_type:", defaults_content)
        self.assertIn("enhancer_blend: 0.85", defaults_content)


if __name__ == "__main__":
    unittest.main()
