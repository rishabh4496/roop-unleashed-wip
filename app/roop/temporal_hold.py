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
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Smooth velocity projection across successive swaps
            if bbox is not None and self.prev_bbox is not None and self.prev_frame_gray is not None:
                dx = float((bbox[0] + bbox[2]) - (self.prev_bbox[0] + self.prev_bbox[2])) / 2.0
                dy = float((bbox[1] + bbox[3]) - (self.prev_bbox[1] + self.prev_bbox[3])) / 2.0
                h, w = frame.shape[:2]
                if abs(dx) <= 0.25 * w and abs(dy) <= 0.25 * h:
                    self.prev_velocity = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)

            self.prev_frame_gray = curr_gray
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

    def can_hold(self) -> bool:
        """Check if buffer has a valid cached swap within the hold limit."""
        with self.lock:
            if not (
                self.prev_composite is not None
                and self.prev_frame_gray is not None
                and self.consecutive_holds < self.max_holds
            ):
                return False

            # Check if previous face bounding box has already moved out of the frame
            if self.prev_bbox is not None and self.prev_frame_gray is not None:
                h, w = self.prev_frame_gray.shape[:2]
                bx1, by1, bx2, by2 = self.prev_bbox
                if bx2 <= 5.0 or bx1 >= (w - 5.0) or by2 <= 5.0 or by1 >= (h - 5.0):
                    return False

            return True

    def estimate_motion(self, curr_frame_gray: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """Estimate inter-frame motion matrix M and return (M, latency_ms)."""
        import time
        t0 = time.perf_counter_ns()
        h, w = curr_frame_gray.shape[:2]

        # Scene-cut / camera transition check (zero-overhead 32x32 thumbnail MAD)
        prev_small = cv2.resize(self.prev_frame_gray, (32, 32), interpolation=cv2.INTER_AREA)
        curr_small = cv2.resize(curr_frame_gray, (32, 32), interpolation=cv2.INTER_AREA)
        frame_diff = float(np.mean(np.abs(curr_small.astype(np.float32) - prev_small.astype(np.float32))))

        if frame_diff > 45.0:
            self.reset()
            t1 = time.perf_counter_ns()
            return None, (t1 - t0) / 1_000_000.0

        M = None
        # Fast path: smooth inter-frame velocity extrapolation (<0.3 ms)
        if self.prev_velocity is not None and frame_diff < 30.0:
            M = self.prev_velocity.copy()
        elif self.prev_bbox is not None:
            # Fallback: cropped ROI optical flow tracking
            bw = float(self.prev_bbox[2] - self.prev_bbox[0])
            bh = float(self.prev_bbox[3] - self.prev_bbox[1])
            x1 = max(0, int(round(self.prev_bbox[0] - bw * 0.15)))
            y1 = max(0, int(round(self.prev_bbox[1] - bh * 0.15)))
            x2 = min(w, int(round(self.prev_bbox[2] + bw * 0.15)))
            y2 = min(h, int(round(self.prev_bbox[3] + bh * 0.15)))

            if x2 > x1 and y2 > y1:
                roi_prev = self.prev_frame_gray[y1:y2, x1:x2]
                roi_curr = curr_frame_gray[y1:y2, x1:x2]
                p0 = cv2.goodFeaturesToTrack(roi_prev, maxCorners=30, qualityLevel=0.02, minDistance=6)
                if p0 is not None and len(p0) >= 4:
                    p1, st, _ = cv2.calcOpticalFlowPyrLK(
                        roi_prev, roi_curr, p0, None,
                        winSize=(15, 15), maxLevel=1,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
                    )
                    v_p0 = p0[st.ravel() == 1]
                    v_p1 = p1[st.ravel() == 1]
                    if len(v_p0) >= 4:
                        M_roi, _ = cv2.estimateAffinePartial2D(v_p0, v_p1)
                        if M_roi is not None and np.isfinite(M_roi).all():
                            M = M_roi.copy()

        # Validate affine transformation plausibility (scale & translation bounds)
        is_valid_M = False
        if M is not None and np.isfinite(M).all():
            det = float(M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0])
            tx = abs(float(M[0, 2]))
            ty = abs(float(M[1, 2]))
            if 0.25 <= det <= 4.0 and tx <= 0.40 * w and ty <= 0.40 * h:
                is_valid_M = True

        if is_valid_M:
            self.prev_velocity = M.copy()
        else:
            if self.prev_velocity is not None and frame_diff < 30.0:
                M = self.prev_velocity.copy()
            elif frame_diff < 25.0:
                M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
            else:
                self.reset()
                M = None

        t1 = time.perf_counter_ns()
        latency_ms = (t1 - t0) / 1_000_000.0
        return M, latency_ms

    def hold_and_warp(self, curr_frame: np.ndarray) -> Optional[np.ndarray]:
        """Track motion from previous frame to curr_frame and composite warped swap.

        Returns warped composited frame, or None if buffer is exhausted.
        """
        with self.lock:
            if not self.can_hold():
                return None

            curr_frame_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            h, w = curr_frame.shape[:2]

            M, tracking_ms = self.estimate_motion(curr_frame_gray)
            self.last_tracking_latency_ms = tracking_ms
            if M is None:
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

            # If the warped mask has negligible coverage, the face has exited the frame
            if warped_mask is not None and float(warped_mask.max()) < 0.02:
                self.reset()
                return None

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
            if self.prev_bbox is not None:
                bx1, by1, bx2, by2 = self.prev_bbox
                box_pts = np.array(
                    [[bx1, by1], [bx2, by1], [bx2, by2], [bx1, by2]], dtype=np.float32
                )
                warped_pts = cv2.transform(box_pts.reshape(1, -1, 2), M).reshape(-1, 2)
                min_x = float(warped_pts[:, 0].min())
                max_x = float(warped_pts[:, 0].max())
                min_y = float(warped_pts[:, 1].min())
                max_y = float(warped_pts[:, 1].max())

                # If face bounding box has completely left the frame, terminate hold and reset
                if max_x <= 5.0 or min_x >= float(w - 5.0) or max_y <= 5.0 or min_y >= float(h - 5.0):
                    self.reset()
                    return None

                clamped_x1 = max(0.0, min(float(w), min_x))
                clamped_y1 = max(0.0, min(float(h), min_y))
                clamped_x2 = max(0.0, min(float(w), max_x))
                clamped_y2 = max(0.0, min(float(h), max_y))

                if (clamped_x2 - clamped_x1) < 5.0 or (clamped_y2 - clamped_y1) < 5.0:
                    self.reset()
                    return None

                self.prev_bbox = np.array(
                    [clamped_x1, clamped_y1, clamped_x2, clamped_y2],
                    dtype=np.float32,
                )

            self.consecutive_holds += 1
            self.prev_frame_gray = curr_frame_gray
            self.prev_composite = held_composite
            self.prev_mask = warped_mask

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
