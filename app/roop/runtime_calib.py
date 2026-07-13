"""Learned runtime estimation for video swaps.

The Face Swap tab shows an "EST. RUNTIME" before a run starts. A hardcoded
heuristic is a poor predictor because real speed depends heavily on the settings
combination (swap model, enhancer + size, detector engine/resolution, pixel
boost, thread count, GPU, TRT precision…). This module records the ACTUAL
wall-clock ms/frame after every completed video, keyed by a settings signature,
and serves that measurement back for the pre-run estimate — so the estimate gets
more accurate over time, per settings combo.

Design notes:
- One JSON store (app/runtime_calibration.json). Never fatal: every op is
  wrapped so a corrupt/locked file can't break processing or the estimate.
- The signature is built from the SAME payload the frontend sends to /api/swap
  and to /api/runtime_estimate, so the record-time and estimate-time keys match.
  The reader is tolerant of the couple of fields whose key differs between the
  swap payload ("enhancer") and the raw settings object ("selected_enhancer").
- ms/frame is stored as an exponential moving average (recent runs weighted
  more, so a new GPU/driver shifts the estimate) plus a sample count.
"""
import os
import json
import threading
import time

from roop.utilities import resolve_relative_path

_LOCK = threading.Lock()
_ALPHA = 0.35          # EMA weight for the newest run
_VERSION = 1

# Perf-relevant settings. Each tuple is (canonical_name, [payload keys to try]).
# The enhancer name already encodes GPEN size ("GPEN 1024"), so no size field is
# needed. Order is irrelevant — the signature sorts by canonical name.
_SIG_FIELDS = [
    ("swap_model",        ["swap_model"]),
    ("enhancer",          ["enhancer", "selected_enhancer"]),
    ("detection",         ["detection", "face_detection_mode"]),
    ("det_size",          ["face_detector_size"]),
    ("detector",          ["detector_engine"]),
    ("swap_steps",        ["num_swap_steps"]),
    ("upscale",           ["upscale", "subsample_upscale"]),
    ("track",             ["track_identities"]),
    ("temporal",          ["temporal_detection"]),
    ("mask",              ["mask_engine"]),
    ("stab_face",         ["stabilize_face"]),
    ("stab_enh",          ["stabilize_enhancer"]),
]


def _path():
    return resolve_relative_path('../runtime_calibration.json')


def _load():
    try:
        with open(_path(), 'r', encoding='utf-8') as fh:
            data = json.load(fh)
        if isinstance(data, dict) and "entries" in data:
            return data
    except Exception:
        pass
    return {"version": _VERSION, "entries": {}, "global_ms_per_frame": None,
            "global_samples": 0}


def _save(data):
    try:
        tmp = _path() + ".tmp"
        with open(tmp, 'w', encoding='utf-8') as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp, _path())
    except Exception:
        pass


def _norm(v):
    if isinstance(v, bool):
        return "1" if v else "0"
    return str(v)


def signature_from_payload(payload, gpu="", threads="", precision=""):
    """Build a stable signature string from a swap/estimate payload plus the
    server-side environment (GPU, thread count, TRT precision) that the payload
    doesn't carry."""
    parts = []
    for name, keys in _SIG_FIELDS:
        val = None
        for k in keys:
            if k in payload and payload[k] is not None:
                val = payload[k]
                break
        parts.append(f"{name}={_norm(val)}")
    parts.append(f"threads={_norm(threads)}")
    parts.append(f"precision={_norm(precision)}")
    parts.append(f"gpu={_norm(gpu)}")
    return "|".join(parts)


def record(signature, frames, elapsed_ms):
    """Fold one completed run into the store. Ignores tiny/degenerate runs whose
    ms/frame would be noise."""
    try:
        frames = int(frames)
        elapsed_ms = float(elapsed_ms)
        if not signature or frames < 24 or elapsed_ms < 1000.0:
            return
        mpf = elapsed_ms / frames
        with _LOCK:
            data = _load()
            e = data["entries"].get(signature)
            if e and e.get("samples", 0) > 0:
                e["ms_per_frame"] = (1 - _ALPHA) * e["ms_per_frame"] + _ALPHA * mpf
                e["samples"] = e.get("samples", 0) + 1
            else:
                e = {"ms_per_frame": mpf, "samples": 1}
            e["last_ms_per_frame"] = mpf
            e["updated"] = int(time.time())
            data["entries"][signature] = e
            # Global fallback for never-seen signatures.
            g = data.get("global_ms_per_frame")
            data["global_ms_per_frame"] = mpf if g is None else (1 - _ALPHA) * g + _ALPHA * mpf
            data["global_samples"] = data.get("global_samples", 0) + 1
            _save(data)
    except Exception:
        pass


def predict(signature):
    """Return {ms_per_frame, samples, source} for a signature, or None when there
    is no data at all. source is 'measured' (exact signature) or 'global'."""
    try:
        with _LOCK:
            data = _load()
        e = data["entries"].get(signature)
        if e and e.get("samples", 0) > 0:
            return {"ms_per_frame": e["ms_per_frame"], "samples": e["samples"],
                    "source": "measured"}
        g = data.get("global_ms_per_frame")
        if g is not None:
            return {"ms_per_frame": g, "samples": 0, "source": "global"}
    except Exception:
        pass
    return None
