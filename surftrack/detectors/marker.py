"""Two-color fluorescent tape detector (deferred channel — not in default stack).

Detects a nose-color blob and a tail-color blob *inside a provided ROI only*
(the predicted board region), so full-frame color false positives cannot occur.
Ports the nose->tail vector math from the retired marker_simulator.py.

Enable later by adding to the config stack once tape is on the board and the
orientation phase begins.
"""

from __future__ import annotations

from math import atan2, sqrt
from typing import Optional

import cv2
import numpy as np

from ..detections import Detection
from .base import BaseDetector

# HSV ranges for fluorescent tape under a daylight-locked white balance.
# Re-calibrate against real footage before relying on these.
TAPE_HSV = {
    "orange": (np.array([5, 120, 120]), np.array([22, 255, 255])),
    "pink": (np.array([140, 90, 120]), np.array([170, 255, 255])),
    "green": (np.array([40, 100, 100]), np.array([80, 255, 255])),
    "yellow": (np.array([22, 100, 120]), np.array([35, 255, 255])),
}


def board_vector(nose: tuple[float, float], tail: tuple[float, float]) -> Optional[dict]:
    """Board center/angle/length from nose and tail points (from marker_simulator)."""
    nx, ny = nose
    tx, ty = tail
    dx, dy = nx - tx, ny - ty
    length = sqrt(dx * dx + dy * dy)
    if length < 1e-6:
        return None
    angle = atan2(dy, dx)
    return {
        "center": ((nx + tx) / 2.0, (ny + ty) / 2.0),
        "angle_deg": float(np.degrees(angle)),
        "length": length,
        "nose": (nx, ny),
        "tail": (tx, ty),
    }


class MarkerDetector(BaseDetector):
    """Detect nose/tail tape blobs inside an ROI; yields a 'board_marker' detection."""

    name = "marker"

    def __init__(self, nose_color: str = "orange", tail_color: str = "pink", min_blob_px: int = 12):
        self.nose_color = nose_color
        self.tail_color = tail_color
        self.min_blob_px = min_blob_px
        self.roi: Optional[tuple[int, int, int, int]] = None  # set per-frame by fusion

    def set_roi(self, roi: Optional[tuple[int, int, int, int]]) -> None:
        """Fusion sets this each frame to the predicted board bbox (x1,y1,x2,y2)."""
        self.roi = roi

    def _find_blob(self, hsv_roi: np.ndarray, color: str) -> Optional[tuple[float, float]]:
        lower, upper = TAPE_HSV[color]
        mask = cv2.inRange(hsv_roi, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = max(contours, key=cv2.contourArea, default=None)
        if best is None or cv2.contourArea(best) < self.min_blob_px:
            return None
        m = cv2.moments(best)
        if m["m00"] == 0:
            return None
        return (m["m10"] / m["m00"], m["m01"] / m["m00"])

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if self.roi is None:
            return []
        x1, y1, x2, y2 = [int(v) for v in self.roi]
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 4 or y2 - y1 < 4:
            return []

        hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
        nose = self._find_blob(hsv, self.nose_color)
        tail = self._find_blob(hsv, self.tail_color)
        if nose is None or tail is None:
            return []

        nose = (nose[0] + x1, nose[1] + y1)
        tail = (tail[0] + x1, tail[1] + y1)
        info = board_vector(nose, tail)
        if info is None:
            return []

        xs = [nose[0], tail[0]]
        ys = [nose[1], tail[1]]
        det = Detection(
            cls_name="board_marker",
            bbox=(min(xs), min(ys), max(xs), max(ys)),
            conf=1.0,
            source=self.name,
        )
        det.board_info = info  # angle/length metadata for downstream use
        return [det]
