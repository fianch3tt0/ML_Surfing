"""Common detection types shared by all detector plugins."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# COCO keypoint indices used by YOLO-pose models
KP_LEFT_ANKLE = 15
KP_RIGHT_ANKLE = 16


@dataclass
class Detection:
    """One detected object in full-frame pixel coordinates.

    bbox is (x1, y1, x2, y2). keypoints, when present, is an (N, 3) array of
    (x, y, confidence) rows in COCO keypoint order (pose detectors only).
    """

    cls_name: str
    bbox: tuple[float, float, float, float]
    conf: float
    source: str  # name of the detector plugin that produced this
    keypoints: Optional[np.ndarray] = None
    track_id: Optional[int] = None

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def ankle_midpoint(self, min_kp_conf: float = 0.3) -> Optional[tuple[float, float]]:
        """Midpoint of visible ankles — the anchor point for the board prior."""
        if self.keypoints is None or len(self.keypoints) <= KP_RIGHT_ANKLE:
            return None
        ankles = [
            self.keypoints[i]
            for i in (KP_LEFT_ANKLE, KP_RIGHT_ANKLE)
            if self.keypoints[i][2] >= min_kp_conf
        ]
        if not ankles:
            return None
        xs = [a[0] for a in ankles]
        ys = [a[1] for a in ankles]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    def to_dict(self) -> dict:
        d = {
            "cls": self.cls_name,
            "bbox": [round(float(v), 1) for v in self.bbox],
            "conf": round(float(self.conf), 3),
            "source": self.source,
        }
        if self.track_id is not None:
            d["track_id"] = int(self.track_id)
        if self.keypoints is not None:
            d["keypoints"] = [[round(float(v), 1) for v in kp] for kp in self.keypoints]
        return d
