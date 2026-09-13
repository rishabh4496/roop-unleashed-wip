"""Temporal Swap Hold & Optical Flow Motion Persistence Buffer.

Holds and warps the last swapped face across brief detection dropouts (1-3 frames)
using Pyramidal Lucas-Kanade optical flow (cv2.calcOpticalFlowPyrLK) and affine
velocity projection, preventing sudden frame-by-frame flicker or binary reversion.
"""

from __future__ import annotations

import threading
from typing import Optional, Tuple

import cv2
import numpy as np


class TemporalSwapHoldBuffer:
    """Manages optical-flow warping of the previous valid swap across dropped frames."""

    def __init__(self, max_holds: int = 3):
        self.max_holds = int(max_holds)
        self.consecutive_holds = 0
        self.prev_frame_gray: Optional[np.ndarray] = None
        self.prev_composite: Optional[np.ndarray] = None
        self.prev_mask: Optional[np.ndarray] = None  # 2D float32 [0.0, 1.0]
        self.prev_bbox: Optional[np.ndarray] = None
        self.prev_velocity: Optional[np.ndarray] = None  # 2x3 affine matrix
        self.lock = threading.RLock()

    def record_swap(
        self,
        frame: np.ndarray,
        composite: np.ndarray,
        mask: Optional[np.ndarray] = None,
        bbox: Optional[np.ndarray] = None,
    ) -> None:
        """Record a successful swap to maintain the persistence cache."""
        with self.lock:
            self.consecutive_holds = 0
            self.prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.prev_composite = composite.copy()

            if mask is not None:
                m = mask.squeeze()
                if m.dtype != np.float32:
                    self.prev_mask = (m.astype(np.float32) / 255.0).clip(0.0, 1.0)
                else:
                    self.prev_mask = m.copy().clip(0.0, 1.0)
            elif bbox is not None:
                self.prev_mask = self._generate_bbox_mask(frame.shape[:2], bbox)
            else:
                self.prev_mask = None

            self.prev_bbox = np.asarray(bbox, dtype=np.float32).copy() if bbox is not None else None
            self.prev_velocity = None

    def can_hold(self) -> bool:
        """Check if buffer has a valid cached swap within the hold limit."""
        with self.lock:
            return (
                self.prev_composite is not None
                and self.prev_frame_gray is not None
                and self.consecutive_holds < self.max_holds
            )

    def hold_and_warp(self, curr_frame: np.ndarray) -> Optional[np.ndarray]:
        """Track motion from previous frame to curr_frame and composite warped swap.

        Returns warped composited frame, or None if buffer is exhausted.
        """
        with self.lock:
            if not self.can_hold():
                return None

            curr_frame_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            h, w = curr_frame.shape[:2]

            # Scene-cut / camera transition check (zero-overhead 32x32 thumbnail MAD)
            prev_small = cv2.resize(self.prev_frame_gray, (32, 32), interpolation=cv2.INTER_AREA)
            curr_small = cv2.resize(curr_frame_gray, (32, 32), interpolation=cv2.INTER_AREA)
            frame_diff = float(np.mean(np.abs(curr_small.astype(np.float32) - prev_small.astype(np.float32))))

            # Hard scene cut threshold: if frames differ radically, reset and abort hold
            if frame_diff > 45.0:
                self.reset()
                return None

            # 1. Feature tracking inside previous face ROI
            M = None
            valid_count = 0
            if self.prev_bbox is not None:
                bw = float(self.prev_bbox[2] - self.prev_bbox[0])
                bh = float(self.prev_bbox[3] - self.prev_bbox[1])
                x1 = max(0, int(round(self.prev_bbox[0] - bw * 0.15)))
                y1 = max(0, int(round(self.prev_bbox[1] - bh * 0.15)))
                x2 = min(w, int(round(self.prev_bbox[2] + bw * 0.15)))
                y2 = min(h, int(round(self.prev_bbox[3] + bh * 0.15)))

                if x2 > x1 and y2 > y1:
                    roi_mask = np.zeros_like(self.prev_frame_gray)
                    roi_mask[y1:y2, x1:x2] = 255

                    p0 = cv2.goodFeaturesToTrack(
                        self.prev_frame_gray,
                        maxCorners=120,
                        qualityLevel=0.01,
                        minDistance=5,
                        mask=roi_mask,
                    )

                    if p0 is not None and len(p0) >= 4:
                        p1, st, _ = cv2.calcOpticalFlowPyrLK(
                            self.prev_frame_gray,
                            curr_frame_gray,
                            p0,
                            None,
                            winSize=(21, 21),
                            maxLevel=3,
                            criteria=(
                                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                30,
                                0.01,
                            ),
                        )
                        valid_p0 = p0[st.ravel() == 1]
                        valid_p1 = p1[st.ravel() == 1]
                        valid_count = len(valid_p0)

                        if valid_count >= 4:
                            M, _ = cv2.estimateAffinePartial2D(valid_p0, valid_p1)

            # Check if tracking completely failed on a significantly changed scene
            if valid_count < 4 and frame_diff > 30.0:
                self.reset()
                return None

            # Validate affine transformation plausibility (scale & translation bounds)
            is_valid_M = False
            if M is not None and np.isfinite(M).all():
                det = float(M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0])
                tx = abs(float(M[0, 2]))
                ty = abs(float(M[1, 2]))
                # Determinant must be positive and within reasonable scale range [0.25, 4.0]
                # Translation must be within 40% of frame dimensions
                if 0.25 <= det <= 4.0 and tx <= 0.40 * w and ty <= 0.40 * h:
                    is_valid_M = True

            if is_valid_M:
                self.prev_velocity = M.copy()
            else:
                # If M is degenerate or tracking failed, fallback to prev_velocity only if frame_diff < 30.0
                if self.prev_velocity is not None and frame_diff < 30.0:
                    M = self.prev_velocity.copy()
                elif frame_diff < 25.0:
                    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
                else:
                    self.reset()
                    return None

            # 2. Warp previous swap composite & mask
            warped_composite = cv2.warpAffine(
                self.prev_composite,
                M,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )

            if self.prev_mask is not None:
                warped_mask = cv2.warpAffine(
                    self.prev_mask,
                    M,
                    (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
            else:
                warped_mask = self._generate_bbox_mask(
                    (h, w),
                    self.prev_bbox if self.prev_bbox is not None else np.array([0, 0, w, h]),
                )

            # 3. Composite onto current frame
            if warped_mask.ndim == 2:
                alpha = np.repeat(warped_mask[:, :, np.newaxis], 3, axis=2)
            else:
                alpha = warped_mask

            alpha = np.clip(alpha, 0.0, 1.0)
            held_composite = (
                warped_composite.astype(np.float32) * alpha
                + curr_frame.astype(np.float32) * (1.0 - alpha)
            ).clip(0, 255).astype(np.uint8)

            # 4. Update state for chained multi-frame holds
            self.consecutive_holds += 1
            self.prev_frame_gray = curr_frame_gray
            self.prev_composite = held_composite
            self.prev_mask = warped_mask

            if self.prev_bbox is not None:
                bx1, by1, bx2, by2 = self.prev_bbox
                box_pts = np.array(
                    [[bx1, by1], [bx2, by1], [bx2, by2], [bx1, by2]], dtype=np.float32
                )
                warped_pts = cv2.transform(box_pts.reshape(1, -1, 2), M).reshape(-1, 2)
                self.prev_bbox = np.array(
                    [
                        max(0.0, float(warped_pts[:, 0].min())),
                        max(0.0, float(warped_pts[:, 1].min())),
                        min(float(w), float(warped_pts[:, 0].max())),
                        min(float(h), float(warped_pts[:, 1].max())),
                    ],
                    dtype=np.float32,
                )

            return held_composite

    def reset(self) -> None:
        """Clear all cached frames and counters."""
        with self.lock:
            self.consecutive_holds = 0
            self.prev_frame_gray = None
            self.prev_composite = None
            self.prev_mask = None
            self.prev_bbox = None
            self.prev_velocity = None

    @staticmethod
    def _generate_bbox_mask(shape: Tuple[int, int], bbox: np.ndarray) -> np.ndarray:
        """Create a feathered elliptical mask matching bbox bounds."""
        h, w = shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        if bbox is None or len(bbox) < 4:
            return mask
        x1 = max(0, min(w, int(round(float(bbox[0])))))
        y1 = max(0, min(h, int(round(float(bbox[1])))))
        x2 = max(0, min(w, int(round(float(bbox[2])))))
        y2 = max(0, min(h, int(round(float(bbox[3])))))
        if x2 <= x1 or y2 <= y1:
            return mask
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        ax, ay = max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2)
        cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 1.0, -1)
        ksize = max(5, int(round(min(ax, ay) * 0.3)) | 1)
        return cv2.GaussianBlur(mask, (ksize, ksize), 0)
