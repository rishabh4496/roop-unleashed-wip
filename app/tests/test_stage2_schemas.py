"""Unit tests for Stage 2: Backend API & Schema Synchronization.

Tests:
1. Schema Validation for EnhancerType, EnhancerSettings, ProcessVideoRequest, and PreviewRequest.
2. Verification that enhancer_type includes "gpen_realistic" and "ultramax".
3. Verification that enhancer_blend validates the range [0.0, 1.0] and defaults to 0.85.
4. Testing payload parsing in api.py without throwing 422 Unprocessable Entity errors.
5. FastAPI TestClient requests simulating React client payloads.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api_schemas import (
    CANONICAL_ENHANCER_MAP,
    EnhancerSettings,
    EnhancerType,
    PreviewRequest,
    ProcessVideoRequest,
    normalize_enhancer_type,
    parse_enhancer_from_payload,
)
from api import app
import roop.globals as roop_globals


def test_enhancer_type_enum_and_literals():
    """Verify that EnhancerType includes gpen_realistic and ultramax alongside standard options."""
    assert EnhancerType.GPEN_REALISTIC.value == "gpen_realistic"
    assert EnhancerType.ULTRAMAX.value == "ultramax"
    assert EnhancerType.GFPGAN.value == "gfpgan"
    assert EnhancerType.CODEFORMER.value == "codeformer"

    # Verify canonical normalization mapping
    assert normalize_enhancer_type("gpen_realistic") == "GPEN Realistic"
    assert normalize_enhancer_type("GPEN Realistic") == "GPEN Realistic"
    assert normalize_enhancer_type("ultramax") == "UltraMax"
    assert normalize_enhancer_type("UltraMax") == "UltraMax"
    assert normalize_enhancer_type("codeformer") == "Codeformer"
    assert normalize_enhancer_type("gfpgan") == "GFPGAN"
    assert normalize_enhancer_type("none") == "None"


def test_enhancer_settings_schema():
    """Verify EnhancerSettings schema validation, default blend 0.85, and range [0.0, 1.0]."""
    # Default construction
    s_default = EnhancerSettings()
    assert s_default.enhancer_type == "gpen_realistic"
    assert s_default.display_name == "GPEN Realistic"
    assert s_default.enhancer_blend == 0.85

    # Custom valid values
    s_custom = EnhancerSettings(enhancer_type="ultramax", enhancer_blend=0.75)
    assert s_custom.enhancer_type == "ultramax"
    assert s_custom.display_name == "UltraMax"
    assert s_custom.enhancer_blend == 0.75

    # Out of bounds clamping
    s_high = EnhancerSettings(enhancer_type="gpen_realistic", enhancer_blend=1.5)
    assert s_high.enhancer_blend == 1.0

    s_low = EnhancerSettings(enhancer_type="gpen_realistic", enhancer_blend=-0.5)
    assert s_low.enhancer_blend == 0.0

    # Non-numeric fallback
    s_invalid = EnhancerSettings(enhancer_type="gpen_realistic", enhancer_blend="invalid")
    assert s_invalid.enhancer_blend == 0.85


def test_process_video_request_schema():
    """Verify ProcessVideoRequest accepts both new and legacy fields without 422 errors."""
    # 1. New schema format
    req1 = ProcessVideoRequest(enhancer_type="gpen_realistic", enhancer_blend=0.85)
    assert req1.resolve_enhancer() == "GPEN Realistic"
    assert req1.resolve_blend() == 0.85

    # 2. Legacy schema format from existing React UI
    req2 = ProcessVideoRequest(enhancer="UltraMax", blend_ratio=0.70)
    assert req2.resolve_enhancer() == "UltraMax"
    assert req2.resolve_blend() == 0.70

    # 3. Payload with arbitrary extra frontend fields (must not raise ValidationError)
    req3 = ProcessVideoRequest(
        enhancer_type="ultramax",
        enhancer_blend=0.9,
        react_client_version="2.4.0",
        arbitrary_unseen_param="hello_world",
    )
    assert req3.resolve_enhancer() == "UltraMax"
    assert req3.resolve_blend() == 0.9
    assert req3.model_extra.get("arbitrary_unseen_param") == "hello_world"


def test_parse_enhancer_from_payload_utility():
    """Test priority and resolution in parse_enhancer_from_payload."""
    # Priority 1: enhancer_type / enhancer_blend overrides legacy
    p1 = {"enhancer_type": "gpen_realistic", "enhancer": "None", "enhancer_blend": 0.95, "blend_ratio": 0.5}
    name, blend = parse_enhancer_from_payload(p1)
    assert name == "GPEN Realistic"
    assert blend == 0.95

    # Priority 2: legacy keys when enhancer_type is absent
    p2 = {"enhancer": "UltraMax", "blend_ratio": 0.65}
    name, blend = parse_enhancer_from_payload(p2)
    assert name == "UltraMax"
    assert blend == 0.65

    # Empty payload uses fallbacks
    p3 = {}
    name, blend = parse_enhancer_from_payload(p3, fallback_enhancer="GPEN", fallback_blend=0.85)
    assert name == "GPEN"
    assert blend == 0.85


def test_fastapi_meta_endpoint():
    """Verify /api/meta endpoint returns new enhancer names, enhancer_types, and default_enhancer_blend."""
    client = TestClient(app)
    response = client.get("/api/meta")
    assert response.status_code == 200
    data = response.json()
    assert "GPEN Realistic" in data["enhancers"]
    assert "UltraMax" in data["enhancers"]
    assert "gpen_realistic" in data["enhancer_types"]
    assert "ultramax" in data["enhancer_types"]
    assert data["default_enhancer_blend"] == 0.85


def test_fastapi_save_settings_with_new_fields():
    """Verify /api/settings handles enhancer_type and enhancer_blend without 422 errors."""
    client = TestClient(app)
    payload = {
        "enhancer_type": "gpen_realistic",
        "enhancer_blend": 0.85,
    }
    response = client.post("/api/settings", json=payload)
    # 200 OK or 409 if another test kept render state; must NOT be 422
    assert response.status_code in (200, 409)
    assert response.status_code != 422


def test_fastapi_swap_request_payload_no_422():
    """Verify /api/swap handles full payload with new fields without 422 Unprocessable Entity."""
    client = TestClient(app)
    payload = {
        "enhancer_type": "ultramax",
        "enhancer_blend": 0.85,
        "detection": "All faces",
        "swap_model": "inswapper",
        "blend_ratio": 0.85,
    }
    response = client.post("/api/swap", json=payload)
    # The request should be parsed cleanly without 422 error
    assert response.status_code != 422
