"""End-to-End Verification Script: Face Enhancer API (test_enhancer_api.py).

Tests:
1. Synthetic payload with "enhancer_type": "gpen_realistic" to local backend.
   Asserts HTTP 200 with valid enhanced frame data.
2. Synthetic payload with "enhancer_type": "ultramax" to local backend.
   Asserts HTTP 200 with valid enhanced frame data.
3. VRAM Safety: CUDA OOM error handling.
   Asserts structured JSON error response (HTTP 500, error_type="cuda_oom", success=False)
   without crashing the Python process.
4. Compatibility check for standard enhancers (GFPGAN, CodeFormer, RestoreFormer++).

Usage:
    python test_enhancer_api.py
"""

from __future__ import annotations

import base64
import json
import os
import sys
import unittest
from typing import Any, Dict, Tuple

import cv2
import numpy as np

# Ensure app directory is on path
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.join(REPO_ROOT, "app")
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)


def make_synthetic_face_bgr(width: int = 512, height: int = 512) -> np.ndarray:
    """Generate a synthetic 512x512 face image for reproducible testing."""
    img = np.full((height, width, 3), (175, 190, 215), dtype=np.uint8)
    cx, cy = width // 2, int(height * 0.52)
    # Face oval
    cv2.ellipse(img, (cx, cy), (int(width * 0.28), int(height * 0.38)), 0, 0, 360, (145, 168, 205), -1)
    # Eyes
    cv2.circle(img, (int(width * 0.38), int(height * 0.46)), int(width * 0.03), (50, 45, 40), -1)
    cv2.circle(img, (int(width * 0.62), int(height * 0.46)), int(width * 0.03), (50, 45, 40), -1)
    # Nose
    cv2.circle(img, (cx, int(height * 0.60)), int(width * 0.025), (110, 125, 165), -1)
    # Mouth
    cv2.ellipse(img, (cx, int(height * 0.72)), (int(width * 0.09), int(height * 0.035)), 0, 0, 360, (75, 85, 155), -1)
    return img


def bgr_to_dataurl(img: np.ndarray, format: str = ".jpg") -> str:
    """Convert a BGR numpy image to a base64 data URL."""
    _, buf = cv2.imencode(format, img)
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    mime = "image/jpeg" if format.lower() in (".jpg", ".jpeg") else "image/png"
    return f"data:{mime};base64,{b64}"


def dataurl_to_bgr(dataurl: str) -> np.ndarray:
    """Decode a base64 data URL into a BGR numpy array."""
    assert dataurl and "base64," in dataurl, f"Invalid data URL: {dataurl[:40]}"
    encoded = dataurl.split("base64,")[1]
    raw = base64.b64decode(encoded)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    assert img is not None, "Failed to decode image from data URL"
    return img


class ClientRunner:
    """Dispatches HTTP requests to either a live running server or in-process TestClient."""

    def __init__(self, base_url: str = "http://127.0.0.1:8001"):
        self.base_url = base_url.rstrip("/")
        self.is_live = False
        self._test_client = None

        # Check if live server is reachable
        try:
            import urllib.request
            req = urllib.request.Request(f"{self.base_url}/api/settings", method="GET")
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                if resp.status in (200, 404):
                    self.is_live = True
        except Exception:
            self.is_live = False

        if not self.is_live:
            # Fallback to in-process FastAPI TestClient
            try:
                from starlette.testclient import TestClient
                from app.api import app
                self._test_client = TestClient(app)
            except Exception as e:
                raise RuntimeError(f"Cannot initialize TestClient: {e}") from e

    def post(self, path: str, json_data: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        """Send POST request and return (status_code, response_json)."""
        if self.is_live:
            import urllib.request
            import urllib.error
            url = f"{self.base_url}{path}"
            body_bytes = json.dumps(json_data).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=body_bytes,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=30.0) as resp:
                    status = resp.status
                    data = json.loads(resp.read().decode("utf-8"))
                    return status, data
            except urllib.error.HTTPError as err:
                raw = err.read().decode("utf-8")
                try:
                    data = json.loads(raw)
                except Exception:
                    data = {"error": raw}
                return err.code, data
        else:
            resp = self._test_client.post(path, json=json_data)
            try:
                data = resp.json()
            except Exception:
                data = {"raw": resp.text}
            return resp.status_code, data


class TestEnhancerAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n" + "=" * 60)
        print("  Starting Face Enhancer API & VRAM Safety Verification")
        print("=" * 60)
        try:
            import roop.globals as rg
            from roop.core import decode_execution_providers
            rg.execution_providers = decode_execution_providers(["cuda", "cpu"])
        except Exception:
            pass
        cls.client = ClientRunner()
        mode = "LIVE server on " + cls.client.base_url if cls.client.is_live else "In-Process TestClient"
        print(f"[*] Execution Mode: {mode}\n")

    def test_01_gpen_realistic_synthetic_payload(self):
        """Send synthetic payload with enhancer_type: 'gpen_realistic' and assert HTTP 200 with valid enhanced frame."""
        print("[Test 1] Testing 'gpen_realistic' with synthetic 512x512 frame...")
        synthetic_bgr = make_synthetic_face_bgr(512, 512)
        data_url = bgr_to_dataurl(synthetic_bgr)

        payload = {
            "enhancer_type": "gpen_realistic",
            "enhancer_blend": 0.85,
            "image": data_url,
            "kps": [
                [194.0, 235.0],
                [318.0, 235.0],
                [256.0, 307.0],
                [201.0, 368.0],
                [311.0, 368.0],
            ],
        }

        status, resp = self.client.post("/api/enhance", payload)
        self.assertEqual(
            status,
            200,
            f"Expected HTTP 200 for GPEN Realistic, got {status}: {resp}",
        )
        self.assertTrue(resp.get("success"), f"Expected success=True, got {resp}")
        self.assertIn("enhanced_frame", resp, "Response missing 'enhanced_frame' field")

        # Decode base64 frame and verify dimensions and integrity
        enhanced_bgr = dataurl_to_bgr(resp["enhanced_frame"])
        self.assertEqual(enhanced_bgr.shape, (512, 512, 3), f"Unexpected shape {enhanced_bgr.shape}")
        self.assertGreater(float(enhanced_bgr.std()), 5.0, "Enhanced image appears collapsed or uniform")
        self.assertEqual(resp.get("enhancer_type"), "GPEN Realistic")
        self.assertAlmostEqual(float(resp.get("enhancer_blend", 0.0)), 0.85, places=2)
        print("  [PASS] GPEN Realistic passed (HTTP 200, valid 512x512 frame, std > 5.0)")

    def test_02_ultramax_synthetic_payload(self):
        """Send synthetic payload with enhancer_type: 'ultramax' and assert HTTP 200."""
        print("[Test 2] Testing 'ultramax' with synthetic frame...")
        synthetic_bgr = make_synthetic_face_bgr(512, 512)
        data_url = bgr_to_dataurl(synthetic_bgr)

        payload = {
            "enhancer_type": "ultramax",
            "enhancer_blend": 0.90,
            "image": data_url,
        }

        status, resp = self.client.post("/api/enhance", payload)
        self.assertEqual(
            status,
            200,
            f"Expected HTTP 200 for UltraMax, got {status}: {resp}",
        )
        self.assertTrue(resp.get("success"), f"Expected success=True, got {resp}")
        self.assertIn("enhanced_frame", resp, "Response missing 'enhanced_frame' field")

        enhanced_bgr = dataurl_to_bgr(resp["enhanced_frame"])
        self.assertEqual(enhanced_bgr.shape, (512, 512, 3))
        self.assertGreater(float(enhanced_bgr.std()), 5.0)
        self.assertEqual(resp.get("enhancer_type"), "UltraMax")
        self.assertAlmostEqual(float(resp.get("enhancer_blend", 0.0)), 0.90, places=2)
        print("  [PASS] UltraMax passed (HTTP 200, valid 512x512 frame)")

    def test_03_vram_safety_cuda_oom_handling(self):
        """Verify explicit try-catch blocks for CUDA OOM return structured JSON error instead of crashing."""
        print("[Test 3] Testing VRAM Safety (CUDA OOM error handling)...")
        payload = {
            "enhancer_type": "gpen_realistic",
            "enhancer_blend": 0.85,
            "simulate_oom": True,
        }

        status, resp = self.client.post("/api/enhance", payload)
        self.assertEqual(
            status,
            500,
            f"Expected HTTP 500 for CUDA OOM simulation, got {status}: {resp}",
        )
        self.assertFalse(resp.get("success"), "Expected success=False during CUDA OOM")
        self.assertEqual(resp.get("error_type"), "cuda_oom", f"Expected error_type='cuda_oom', got {resp}")
        self.assertIn("CUDA Out of Memory", resp.get("error", "") + resp.get("message", ""))
        print("  [PASS] VRAM Safety passed (HTTP 500, structured JSON error_type='cuda_oom', no crash)")

    def test_04_existing_enhancers_compatibility(self):
        """Verify negative constraint: existing standard enhancers remain fully functional."""
        print("[Test 4] Testing existing enhancers compatibility (GFPGAN, Codeformer)...")
        synthetic_bgr = make_synthetic_face_bgr(512, 512)
        data_url = bgr_to_dataurl(synthetic_bgr)

        for enh in ["gfpgan", "codeformer", "restoreformer++"]:
            payload = {
                "enhancer_type": enh,
                "enhancer_blend": 0.80,
                "image": data_url,
            }
            status, resp = self.client.post("/api/enhance", payload)
            self.assertEqual(status, 200, f"Expected HTTP 200 for {enh}, got {status}: {resp}")
            self.assertTrue(resp.get("success"), f"Expected success=True for {enh}")
            self.assertIn("enhanced_frame", resp)
            print(f"  [PASS] Existing enhancer '{enh}' passed (HTTP 200)")


def main():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEnhancerAPI)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        sys.exit(1)
    print("\n" + "=" * 60)
    print("  All Face Enhancer API Verification Checks Passed Successfully!")
    print("=" * 60 + "\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
