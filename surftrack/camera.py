"""Camera profiles: per-camera preprocessing so GoPro and phone footage are
interchangeable inputs.

A profile carries the optics metadata (HFOV — used for pixel<->meter estimates
in analysis) and any preprocessing: optional downscale for inference speed and
optional undistortion (only needed if footage was NOT recorded in a linear/
distortion-free mode; fill in fisheye coefficients after calibration).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class CameraProfile:
    name: str
    hfov_deg: float = 70.0
    proc_width: int = 0            # downscale frames to this width before inference; 0 = off
    undistort: bool = False
    # OpenCV fisheye/pinhole distortion (set after calibrating the actual camera)
    k: Optional[list] = None       # 3x3 intrinsic matrix (row-major, at native resolution)
    d: Optional[list] = None       # distortion coefficients

    _maps: Optional[tuple] = None

    @classmethod
    def from_config(cls, cameras_cfg: dict, name: str) -> "CameraProfile":
        profiles = cameras_cfg.get("profiles", {})
        if name not in profiles:
            raise ValueError(f"Unknown camera profile '{name}'. Available: {list(profiles)}")
        return cls(name=name, **profiles[name])

    def process(self, frame: np.ndarray) -> np.ndarray:
        if self.undistort and self.k is not None and self.d is not None:
            if self._maps is None:
                h, w = frame.shape[:2]
                k = np.array(self.k, dtype=np.float64)
                d = np.array(self.d, dtype=np.float64)
                self._maps = cv2.initUndistortRectifyMap(
                    k, d, None, k, (w, h), cv2.CV_16SC2
                )
            frame = cv2.remap(frame, self._maps[0], self._maps[1], cv2.INTER_LINEAR)
        if self.proc_width and frame.shape[1] > self.proc_width:
            scale = self.proc_width / frame.shape[1]
            frame = cv2.resize(frame, (self.proc_width, int(frame.shape[0] * scale)))
        return frame
