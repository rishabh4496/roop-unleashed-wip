"""Pydantic schemas and validation for backend API synchronization.

Defines schemas for:
- EnhancerType (Enum / Literal)
- EnhancerSettings
- ProcessVideoRequest
- PreviewRequest

Ensures incoming requests from React or third-party API clients parse
enhancer_type and enhancer_blend safely without throwing 422 Unprocessable Entity
errors or breaking fallback defaults.
"""

from __future__ import annotations

import enum
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator


class EnhancerType(str, enum.Enum):
    NONE = "none"
    CODEFORMER = "codeformer"
    CODEFORMER_FP16 = "codeformer_fp16"
    DMDNET = "dmdnet"
    GFPGAN = "gfpgan"
    GPEN_256 = "gpen_256"
    GPEN = "gpen"
    GPEN_1024 = "gpen_1024"
    GPEN_2048 = "gpen_2048"
    GPEN_ULTIMATE = "gpen_ultimate"
    GPEN_REALISTIC = "gpen_realistic"
    ULTRAMAX = "ultramax"
    RESTOREFORMER_PLUS_PLUS = "restoreformer++"
    RESTORE_ULTRA = "restore_ultra"
    KEEP = "keep"


EnhancerTypeLiteral = Literal[
    "none",
    "None",
    "codeformer",
    "Codeformer",
    "codeformer_fp16",
    "Codeformer (fp16)",
    "dmdnet",
    "DMDNet",
    "gfpgan",
    "GFPGAN",
    "gpen_256",
    "GPEN 256",
    "gpen",
    "GPEN",
    "gpen_1024",
    "GPEN 1024",
    "gpen_2048",
    "GPEN 2048",
    "gpen_ultimate",
    "GPEN Ultimate",
    "gpen_realistic",
    "GPEN Realistic",
    "ultramax",
    "UltraMax",
    "restoreformer++",
    "Restoreformer++",
    "restore_ultra",
    "Restore Ultra",
    "keep",
    "KEEP (sidecar)",
]

# Canonical lookup mapping all variations (lowercase, display name, snake_case)
# to the standard backend roop.globals.selected_enhancer string name.
CANONICAL_ENHANCER_MAP: Dict[str, str] = {
    "none": "None",
    "codeformer": "Codeformer",
    "codeformer_fp16": "Codeformer (fp16)",
    "codeformer (fp16)": "Codeformer (fp16)",
    "dmdnet": "DMDNet",
    "gfpgan": "GFPGAN",
    "gpen": "GPEN",
    "gpen_256": "GPEN 256",
    "gpen 256": "GPEN 256",
    "gpen_512": "GPEN",
    "gpen 512": "GPEN",
    "gpen_1024": "GPEN 1024",
    "gpen 1024": "GPEN 1024",
    "gpen_2048": "GPEN 2048",
    "gpen 2048": "GPEN 2048",
    "gpen_ultimate": "GPEN Ultimate",
    "gpen ultimate": "GPEN Ultimate",
    "gpen_realistic": "GPEN Realistic",
    "gpen realistic": "GPEN Realistic",
    "ultramax": "UltraMax",
    "ultra_max": "UltraMax",
    "restoreformer++": "Restoreformer++",
    "restoreformer_plus_plus": "Restoreformer++",
    "restore_ultra": "Restore Ultra",
    "restore ultra": "Restore Ultra",
    "keep": "KEEP (sidecar)",
    "keep (sidecar)": "KEEP (sidecar)",
}


def normalize_enhancer_type(name: Optional[str], default: str = "None") -> str:
    """Normalize any incoming enhancer string into the canonical backend name."""
    if not name:
        return default
    key = str(name).strip().lower()
    return CANONICAL_ENHANCER_MAP.get(key, name)


class EnhancerSettings(BaseModel):
    """Enhancer parameters schema with validation and lenient fallbacks."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    enhancer_type: Optional[str] = Field(
        default="gpen_realistic",
        description="Enhancer model type (e.g. gpen_realistic, ultramax, gfpgan, codeformer)",
    )
    enhancer_blend: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Blend ratio for enhancement strength [0.0, 1.0]",
    )
    codeformer_fidelity: Optional[float] = Field(default=0.5, ge=0.0, le=1.0)
    enhancer_align: Optional[bool] = Field(default=False)
    color_match_after_enhance: Optional[bool] = Field(default=False)

    @field_validator("enhancer_blend", mode="before")
    @classmethod
    def validate_blend(cls, v: Any) -> float:
        if v is None:
            return 0.85
        try:
            val = float(v)
            return max(0.0, min(1.0, val))
        except (TypeError, ValueError):
            return 0.85

    @field_validator("enhancer_type", mode="before")
    @classmethod
    def validate_type(cls, v: Any) -> str:
        if v is None:
            return "gpen_realistic"
        val = str(v).strip().lower().replace(" ", "_").replace("-", "_")
        if val in ("gpen_realistic", "gpenrealistic"):
            return "gpen_realistic"
        if val in ("ultramax", "ultra_max"):
            return "ultramax"
        return str(v)

    @property
    def display_name(self) -> str:
        return normalize_enhancer_type(self.enhancer_type, default="GPEN Realistic")


class ProcessVideoRequest(BaseModel):
    """Full swap request payload schema.

    Permits extra fields so client versions with differing parameter sets
    never produce 422 Unprocessable Entity errors.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    # Enhancer fields (both new explicit schema and existing legacy keys supported)
    enhancer_type: Optional[str] = Field(default=None)
    enhancer_blend: Optional[float] = Field(default=None)
    enhancer: Optional[str] = Field(default=None)
    selected_enhancer: Optional[str] = Field(default=None)
    blend_ratio: Optional[float] = Field(default=None)

    # Core pipeline options
    detection: Optional[str] = Field(default="All faces")
    output_method: Optional[str] = Field(default=None)
    video_method: Optional[str] = Field(default=None)
    upscale: Optional[str] = Field(default="256px")
    mask_engine: Optional[str] = Field(default="None")
    mask_engine_2: Optional[str] = Field(default="None")
    clip_text: Optional[str] = Field(default="")
    face_distance: Optional[float] = Field(default=0.75)
    num_swap_steps: Optional[int] = Field(default=1)
    swap_model: Optional[str] = Field(default="inswapper")

    def resolve_enhancer(self, fallback: str = "GPEN") -> str:
        """Resolve effective enhancer name prioritizing enhancer_type then enhancer."""
        val = self.enhancer_type or self.enhancer or self.selected_enhancer or fallback
        return normalize_enhancer_type(val, default=fallback)

    def resolve_blend(self, fallback: float = 0.85) -> float:
        """Resolve effective blend ratio prioritizing enhancer_blend then blend_ratio."""
        val = self.enhancer_blend if self.enhancer_blend is not None else self.blend_ratio
        if val is None:
            return fallback
        try:
            return max(0.0, min(1.0, float(val)))
        except (TypeError, ValueError):
            return fallback


class PreviewRequest(BaseModel):
    """Preview request payload schema with safe fallbacks."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    enhancer_type: Optional[str] = Field(default=None)
    enhancer_blend: Optional[float] = Field(default=None)
    enhancer: Optional[str] = Field(default=None)
    blend_ratio: Optional[float] = Field(default=None)
    detection: Optional[str] = Field(default="All faces")
    face_distance: Optional[float] = Field(default=0.75)
    swap_model: Optional[str] = Field(default="inswapper")
    mask_engine: Optional[str] = Field(default="None")
    clip_text: Optional[str] = Field(default="")

    def resolve_enhancer(self, fallback: str = "None") -> str:
        val = self.enhancer_type or self.enhancer or fallback
        return normalize_enhancer_type(val, default=fallback)

    def resolve_blend(self, fallback: float = 0.85) -> float:
        val = self.enhancer_blend if self.enhancer_blend is not None else self.blend_ratio
        if val is None:
            return fallback
        try:
            return max(0.0, min(1.0, float(val)))
        except (TypeError, ValueError):
            return fallback


def parse_enhancer_from_payload(
    payload: Dict[str, Any],
    fallback_enhancer: str = "None",
    fallback_blend: float = 0.85,
) -> Tuple[str, float]:
    """Parse enhancer name and blend ratio from any payload dict.

    Priority:
    1. enhancer_type / enhancer_blend
    2. enhancer / blend_ratio
    3. selected_enhancer
    4. fallback values
    """
    raw_enhancer = (
        payload.get("enhancer_type")
        or payload.get("enhancer")
        or payload.get("selected_enhancer")
        or fallback_enhancer
    )
    enhancer_name = normalize_enhancer_type(raw_enhancer, default=fallback_enhancer)

    raw_blend = payload.get("enhancer_blend")
    if raw_blend is None:
        raw_blend = payload.get("blend_ratio")
    if raw_blend is None:
        blend_val = fallback_blend
    else:
        try:
            blend_val = max(0.0, min(1.0, float(raw_blend)))
        except (TypeError, ValueError):
            blend_val = fallback_blend

    return enhancer_name, blend_val
