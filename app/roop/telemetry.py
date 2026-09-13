"""Per-frame diagnostic telemetry logger for video face swapping and stability tracking.

Instruments the frame processing loop with exact status, pose estimation,
match similarity, and failure diagnostics.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from collections import deque
import threading

# Actions
ACTION_SWAPPED = "SWAPPED"
ACTION_ROI_RETRY = "ROI_RETRY"
ACTION_LAST_SWAP_HOLD = "LAST_SWAP_HOLD"
ACTION_RAW_FALLBACK = "RAW_FALLBACK"

# Failure Flags
FAIL_SCRFD = "FAIL_SCRFD"
FAIL_SIMILARITY = "FAIL_SIMILARITY"
FAIL_LANDMARKS = "FAIL_LANDMARKS"

_TELEMETRY_LOG_ENABLED = os.environ.get("ROOP_TELEMETRY_LOG", "1") != "0"
_TELEMETRY_HISTORY: deque = deque(maxlen=1000)
_TELEMETRY_LOCK = threading.Lock()


def format_telemetry(
    frame_idx: Optional[int],
    detected: bool,
    det_score: float,
    yaw_pitch_roll: Optional[Sequence[float]],
    match_sim: float,
    action: str,
    failure_flag: Optional[str] = None,
) -> str:
    """Format the diagnostic telemetry string:

    Frame [N] | Detected: [T/F] | Det Score: [0.XX] | Yaw/Pitch/Roll: [Y, P, R] | Match Sim: [0.XX] | Action: [SWAPPED | ROI_RETRY | LAST_SWAP_HOLD | RAW_FALLBACK]
    """
    f_idx = frame_idx if frame_idx is not None else 0
    det_char = "T" if detected else "F"
    det_s = float(det_score) if det_score is not None else 0.0

    if yaw_pitch_roll is not None and len(yaw_pitch_roll) >= 3:
        y = f"{float(yaw_pitch_roll[0]):+.1f}"
        p = f"{float(yaw_pitch_roll[1]):+.1f}"
        r = f"{float(yaw_pitch_roll[2]):+.1f}"
    else:
        y, p, r = "0.0", "0.0", "0.0"

    sim_val = float(match_sim) if match_sim is not None else 0.0

    msg = (
        f"Frame [{f_idx}] | Detected: [{det_char}] | Det Score: [{det_s:.2f}] | "
        f"Yaw/Pitch/Roll: [{y}, {p}, {r}] | Match Sim: [{sim_val:.2f}] | Action: [{action}]"
    )
    if failure_flag:
        msg += f" | Failure: [{failure_flag}]"

    return msg


def log_frame_telemetry(
    frame_idx: Optional[int],
    detected: bool,
    det_score: float,
    yaw_pitch_roll: Optional[Sequence[float]],
    match_sim: float,
    action: str,
    failure_flag: Optional[str] = None,
) -> str:
    """Log telemetry to progress output and keep ring-buffer history."""
    msg = format_telemetry(
        frame_idx, detected, det_score, yaw_pitch_roll, match_sim, action, failure_flag
    )

    if _TELEMETRY_LOG_ENABLED:
        try:
            from roop.procmgr_runtime import bar_write
            bar_write(msg)
        except Exception:
            print(msg)

    record = {
        "frame_idx": frame_idx,
        "detected": detected,
        "det_score": det_score,
        "yaw_pitch_roll": tuple(yaw_pitch_roll) if yaw_pitch_roll is not None else (0.0, 0.0, 0.0),
        "match_sim": match_sim,
        "action": action,
        "failure_flag": failure_flag,
        "msg": msg,
    }

    with _TELEMETRY_LOCK:
        _TELEMETRY_HISTORY.append(record)

    return msg


def get_telemetry_history() -> List[Dict[str, Any]]:
    """Return in-memory telemetry records."""
    with _TELEMETRY_LOCK:
        return list(_TELEMETRY_HISTORY)


def clear_telemetry_history() -> None:
    """Clear in-memory telemetry records."""
    with _TELEMETRY_LOCK:
        _TELEMETRY_HISTORY.clear()
