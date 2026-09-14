"""Model registry and download verification with SHA-256 integrity checks.

Registers URLs, local filenames, model templates, and cryptographic hashes
for face enhancers including GPEN Realistic and the UltraMax pipeline models.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import ssl
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional

_LOGGER = logging.getLogger(__name__)

# Socket timeout for downloads in seconds
DOWNLOAD_TIMEOUT = 30
DOWNLOAD_CHUNK_SIZE = 1024 * 1024  # 1MB


@dataclass(frozen=True)
class ModelSpec:
    key: str
    filename: str
    url: str
    sha256: str
    size_bytes: int
    template: str
    description: str


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "gpen_bfr_512": ModelSpec(
        key="gpen_bfr_512",
        filename="GPEN-BFR-512.onnx",
        url="https://huggingface.co/countfloyd/deepfake/resolve/main/GPEN-BFR-512.onnx",
        sha256="0960f836488735444d508b588e44fb5dfd19c68fde9163ad7878aa24d1d5115e",
        size_bytes=284449835,
        template="ffhq_512",
        description="GPEN-BFR-512 ONNX face restoration model (primary GPEN Realistic engine)",
    ),
    "gpen_bfr_256": ModelSpec(
        key="gpen_bfr_256",
        filename="gpen_bfr_256.onnx",
        url="https://huggingface.co/facefusion/models-3.0.0/resolve/main/gpen_bfr_256.onnx",
        sha256="bad8bf0426873828df2dbf4e3b3d9ababba9da7965b8b72426569486f7ae5c25",
        size_bytes=71844390,
        template="ffhq_512",
        description="GPEN-BFR-256 ONNX model for fast/soft tier",
    ),
    "codeformer_fp16": ModelSpec(
        key="codeformer_fp16",
        filename="CodeFormer/codeformer.fp16.onnx",
        url="https://huggingface.co/countfloyd/deepfake/resolve/main/CodeFormer/codeformer.fp16.onnx",
        sha256="5414b2c989eb93feb7b5bccd22a1a9f3842a127411932663d03884af140a48b6",
        size_bytes=188373398,
        template="ffhq_512",
        description="CodeFormer FP16 ONNX model (UltraMax structural base)",
    ),
    "codeformer_fp32": ModelSpec(
        key="codeformer_fp32",
        filename="CodeFormer/CodeFormerv0.1.onnx",
        url="https://huggingface.co/countfloyd/deepfake/resolve/main/CodeFormer/CodeFormerv0.1.onnx",
        sha256="9aa48fc4b21224d85784c9a58885201284ec8e590b988126db2c07495b421d36",
        size_bytes=376483489,
        template="ffhq_512",
        description="CodeFormer FP32 reference model",
    ),
}


def compute_file_sha256(file_path: str, chunk_size: int = 1024 * 1024) -> str:
    """Compute hex SHA-256 digest of a local file."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Cannot hash non-existent file: {file_path}")
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest().lower()


def verify_file_integrity(file_path: str, expected_sha256: str) -> bool:
    """Verify that file exists, is non-empty, and matches expected SHA-256."""
    if not os.path.isfile(file_path):
        return False
    if os.path.getsize(file_path) == 0:
        return False
    actual_hash = compute_file_sha256(file_path)
    return actual_hash.lower() == expected_sha256.lower().strip()


def resolve_model_cache_path(spec: ModelSpec, base_models_dir: Optional[str] = None) -> str:
    """Resolve absolute destination path for a registered model."""
    if base_models_dir is None:
        from roop.utilities import resolve_relative_path
        base_models_dir = resolve_relative_path("../models")
    return os.path.abspath(os.path.join(base_models_dir, spec.filename))


def ensure_model_downloaded(
    spec_or_key: ModelSpec | str,
    base_models_dir: Optional[str] = None,
    force_download: bool = False,
) -> str:
    """Ensure a registered model exists locally with verified SHA-256 integrity.

    Downloads model if missing or if hash validation fails.
    Returns the absolute path to verified local model file.
    """
    if isinstance(spec_or_key, str):
        if spec_or_key not in MODEL_REGISTRY:
            raise KeyError(f"Unknown model key '{spec_or_key}'. Available: {list(MODEL_REGISTRY.keys())}")
        spec = MODEL_REGISTRY[spec_or_key]
    else:
        spec = spec_or_key

    dest_path = resolve_model_cache_path(spec, base_models_dir)
    dest_dir = os.path.dirname(dest_path)
    os.makedirs(dest_dir, exist_ok=True)

    if not force_download and os.path.isfile(dest_path):
        if verify_file_integrity(dest_path, spec.sha256):
            return dest_path
        _LOGGER.warning(
            "[ModelRegistry] Checksum mismatch for %s (%s). Re-downloading...",
            spec.key,
            dest_path,
        )
        try:
            os.remove(dest_path)
        except OSError:
            pass

    # Download to temporary .part file
    part_path = dest_path + ".part"
    if os.path.exists(part_path):
        try:
            os.remove(part_path)
        except OSError:
            pass

    # Offline check
    from roop.utilities import network_downloads_allowed
    if not network_downloads_allowed():
        raise RuntimeError(
            f"Cannot download model '{spec.key}' because offline mode is enabled and local file is invalid/missing."
        )

    ctx = ssl.create_default_context()
    if hasattr(ssl, "_create_unverified_context"):
        ctx = ssl._create_unverified_context()

    _LOGGER.info("[ModelRegistry] Downloading %s from %s...", spec.filename, spec.url)
    req = urllib.request.Request(spec.url, headers={"User-Agent": "roop-unleashed/1.0"})
    with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT, context=ctx) as response:
        with open(part_path, "wb") as out_file:
            shutil.copyfileobj(response, out_file, length=DOWNLOAD_CHUNK_SIZE)

    # Validate integrity before finalizing file
    if not verify_file_integrity(part_path, spec.sha256):
        actual = compute_file_sha256(part_path)
        try:
            os.remove(part_path)
        except OSError:
            pass
        raise ValueError(
            f"Downloaded file for '{spec.key}' failed integrity check! Expected SHA256: {spec.sha256}, got: {actual}"
        )

    os.replace(part_path, dest_path)
    _LOGGER.info("[ModelRegistry] Verified and saved %s", dest_path)
    return dest_path
