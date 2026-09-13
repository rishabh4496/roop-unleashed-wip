"""Temporal Stabilization & Anti-Flickering Module for Video Face Swapping.

Applies Exponential Moving Average (EMA) smoothing across consecutive frames to:
1. 5-point and 68-point facial landmark coordinates.
2. ArcFace 512-d identity embedding vectors (with L2 re-normalization).
3. Mask boundary blending alpha values (anti-edge buzzing).

Configurable via `temporal_smooth_strength` (0.0 to 1.0, default 0.3).
"""

from __future__ import annotations

import logging
import math
import threading
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

_LOGGER = logging.getLogger(__name__)


class TemporalTrack:
    """Stateful temporal track for a single persistent face across consecutive frames."""

    def __init__(self, track_id: int, centroid: np.ndarray, size: float, frame_idx: int):
        self.track_id = track_id
        self.centroid = np.asarray(centroid, dtype=np.float64)
        self.size = float(size)
        self.last_frame_idx = frame_idx

        # Smoothing states
        self.prev_kps_5: Optional[np.ndarray] = None
        self.prev_lm_68: Optional[np.ndarray] = None
        self.prev_embedding: Optional[np.ndarray] = None
        self.prev_mask_crop: Optional[np.ndarray] = None
        self.prev_mask_shape: Optional[Tuple[int, int]] = None


class TemporalStabilizer:
    """Manages temporal smoothing across consecutive video frames with multi-face tracking."""

    def __init__(
        self,
        strength: float = 0.3,
        max_missing_frames: int = 8,
        match_scale_factor: float = 0.6,
        max_displacement_ratio: float = 1.2,
    ):
        """Args:

        strength: Temporal smoothing strength in [0.0, 1.0].
                  0.0 = no smoothing (raw passthrough).
                  1.0 = heavy smoothing (high inertia).
                  Default 0.3 balances jitter suppression and motion responsiveness.
        max_missing_frames: Maximum frames a face track can vanish before dropping.
        match_scale_factor: Association radius relative to face bounding size.
        max_displacement_ratio: Motion threshold above which tracking resets to avoid ghosting.
        """
        self.strength = float(min(1.0, max(0.0, strength)))
        self.max_missing = int(max_missing_frames)
        self.match_scale = float(match_scale_factor)
        self.max_displacement = float(max_displacement_ratio)

        self._tracks: List[TemporalTrack] = []
        self._next_track_id = 0
        self._lock = threading.RLock()

    def set_strength(self, strength: float) -> None:
        """Update smoothing strength dynamically."""
        self.strength = float(min(1.0, max(0.0, strength)))

    def reset(self) -> None:
        """Reset all temporal history and tracks."""
        with self._lock:
            self._tracks.clear()
            self._next_track_id = 0

    # ── Track Matching ───────────────────────────────────────────────────────

    def _find_or_create_track(
        self, centroid: np.ndarray, size: float, frame_idx: int
    ) -> TemporalTrack:
        """Associate observation with nearest existing face track or create a new one."""
        centroid = np.asarray(centroid, dtype=np.float64)
        size = max(1.0, float(size))

        best_track = None
        best_dist = float("inf")

        # Prune stale tracks
        self._tracks = [
            tr for tr in self._tracks
            if (frame_idx - tr.last_frame_idx) <= self.max_missing
        ]

        for tr in self._tracks:
            dt = frame_idx - tr.last_frame_idx
            if dt <= 0:
                continue
            dist = float(np.linalg.norm(tr.centroid - centroid))
            # Match radius scales with face bounding size
            max_dist = self.match_scale * max(size, tr.size)
            if dist < best_dist and dist <= max_dist:
                best_dist = dist
                best_track = tr

        if best_track is not None:
            # Check for shot cut / sudden teleportation
            if best_dist > self.max_displacement * size:
                # Sudden motion jump: reset track state to avoid ghosting
                best_track.prev_kps_5 = None
                best_track.prev_lm_68 = None
                best_track.prev_embedding = None
                best_track.prev_mask_crop = None

            best_track.centroid = centroid
            best_track.size = size
            best_track.last_frame_idx = frame_idx
            return best_track

        # New track
        new_track = TemporalTrack(self._next_track_id, centroid, size, frame_idx)
        self._next_track_id += 1
        self._tracks.append(new_track)
        return new_track

    # ── 1. Facial Landmark Coordinates Smoothing ─────────────────────────────

    def smooth_landmarks_5pt(
        self,
        kps: Optional[np.ndarray],
        centroid: Optional[np.ndarray] = None,
        size: Optional[float] = None,
        frame_idx: int = 0,
    ) -> Optional[np.ndarray]:
        """Apply EMA smoothing to 5-point facial keypoints (shape (5, 2))."""
        if kps is None or self.strength <= 0.0:
            return kps

        kps_arr = np.asarray(kps, dtype=np.float32)
        if kps_arr.ndim != 2 or kps_arr.shape[0] != 5 or kps_arr.shape[1] != 2:
            return kps

        with self._lock:
            c = centroid if centroid is not None else kps_arr.mean(axis=0)
            s = size if size is not None else max(float(np.ptp(kps_arr[:, 0])), float(np.ptp(kps_arr[:, 1])), 1.0)
            track = self._find_or_create_track(c, s, frame_idx)

            if track.prev_kps_5 is None or track.prev_kps_5.shape != kps_arr.shape:
                track.prev_kps_5 = kps_arr.copy()
                return kps_arr

            # EMA: alpha * current + (1 - alpha) * previous
            # where alpha = 1.0 - strength
            alpha = max(0.05, 1.0 - self.strength)
            smoothed = alpha * kps_arr + (1.0 - alpha) * track.prev_kps_5
            track.prev_kps_5 = smoothed.copy()
            return smoothed.astype(np.float32)

    def smooth_landmarks_68pt(
        self,
        lm68: Optional[np.ndarray],
        centroid: Optional[np.ndarray] = None,
        size: Optional[float] = None,
        frame_idx: int = 0,
    ) -> Optional[np.ndarray]:
        """Apply EMA smoothing to 68-point facial landmarks (shape (68, 2) or (68, 3))."""
        if lm68 is None or self.strength <= 0.0:
            return lm68

        lm_arr = np.asarray(lm68, dtype=np.float32)
        if lm_arr.ndim != 2 or lm_arr.shape[0] != 68 or lm_arr.shape[1] not in (2, 3):
            return lm68

        with self._lock:
            c = centroid if centroid is not None else lm_arr[:, :2].mean(axis=0)
            s = size if size is not None else max(float(np.ptp(lm_arr[:, 0])), float(np.ptp(lm_arr[:, 1])), 1.0)
            track = self._find_or_create_track(c, s, frame_idx)

            if track.prev_lm_68 is None or track.prev_lm_68.shape != lm_arr.shape:
                track.prev_lm_68 = lm_arr.copy()
                return lm_arr

            alpha = max(0.05, 1.0 - self.strength)
            smoothed = alpha * lm_arr + (1.0 - alpha) * track.prev_lm_68
            track.prev_lm_68 = smoothed.copy()
            return smoothed.astype(np.float32)

    # ── 2. ArcFace 512-d Identity Embedding Vector Smoothing ──────────────────

    def smooth_embedding(
        self,
        embedding: Optional[np.ndarray],
        centroid: Optional[np.ndarray] = None,
        size: Optional[float] = None,
        frame_idx: int = 0,
    ) -> Optional[np.ndarray]:
        """Apply EMA smoothing to ArcFace 512-d identity embedding followed by L2 re-normalization."""
        if embedding is None or self.strength <= 0.0:
            return embedding

        emb = np.asarray(embedding, dtype=np.float32)
        orig_shape = emb.shape
        flat_emb = emb.ravel()
        if flat_emb.size != 512:
            return embedding

        norm = float(np.linalg.norm(flat_emb))
        if norm > 1e-6:
            flat_emb = flat_emb / norm

        with self._lock:
            c = centroid if centroid is not None else np.array([0.0, 0.0], dtype=np.float64)
            s = size if size is not None else 100.0
            track = self._find_or_create_track(c, s, frame_idx)

            if track.prev_embedding is None or track.prev_embedding.size != 512:
                track.prev_embedding = flat_emb.copy()
                return flat_emb.reshape(orig_shape).astype(np.float32)

            # EMA on unit sphere
            alpha = max(0.05, 1.0 - self.strength)
            smoothed = alpha * flat_emb + (1.0 - alpha) * track.prev_embedding

            # Re-normalize to unit hypersphere
            s_norm = float(np.linalg.norm(smoothed))
            if s_norm > 1e-6:
                smoothed = smoothed / s_norm

            track.prev_embedding = smoothed.copy()
            return smoothed.reshape(orig_shape).astype(np.float32)

    # ── 3. Mask Boundary Blending Alpha Values Smoothing ──────────────────────

    def smooth_mask(
        self,
        mask: np.ndarray,
        centroid: Optional[np.ndarray] = None,
        size: Optional[float] = None,
        frame_idx: int = 0,
        bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> np.ndarray:
        """Apply EMA smoothing to mask blending alpha values to suppress boundary buzzing.

        Works on both full-frame masks and cropped matte buffers.
        """
        if mask is None or self.strength <= 0.0:
            return mask

        mask_f = np.asarray(mask, dtype=np.float32)
        h, w = mask_f.shape[:2]

        with self._lock:
            c = centroid if centroid is not None else np.array([w * 0.5, h * 0.5], dtype=np.float64)
            s = size if size is not None else float(max(w, h))
            track = self._find_or_create_track(c, s, frame_idx)

            # If working with a localized bounding box on full frame:
            if bbox is not None and len(bbox) == 4:
                bx, by, bw, bh = bbox
                bx = max(0, min(w - 1, int(bx)))
                by = max(0, min(h - 1, int(by)))
                bw = max(1, min(w - bx, int(bw)))
                bh = max(1, min(h - by, int(bh)))

                crop = mask_f[by : by + bh, bx : bx + bw]
                if track.prev_mask_crop is not None and track.prev_mask_crop.shape == crop.shape:
                    alpha = max(0.05, 1.0 - self.strength)
                    smoothed_crop = alpha * crop + (1.0 - alpha) * track.prev_mask_crop
                    track.prev_mask_crop = smoothed_crop.copy()
                    mask_f[by : by + bh, bx : bx + bw] = smoothed_crop
                else:
                    track.prev_mask_crop = crop.copy()
                return mask_f.astype(mask.dtype)

            # Direct mask smoothing (e.g. crop-space mask)
            if track.prev_mask_crop is not None and track.prev_mask_crop.shape == mask_f.shape:
                alpha = max(0.05, 1.0 - self.strength)
                smoothed = alpha * mask_f + (1.0 - alpha) * track.prev_mask_crop
                track.prev_mask_crop = smoothed.copy()
                return smoothed.astype(mask.dtype)

            track.prev_mask_crop = mask_f.copy()
            return mask


# Global singleton stabilizer instance
global_temporal_stabilizer = TemporalStabilizer(strength=0.3)
