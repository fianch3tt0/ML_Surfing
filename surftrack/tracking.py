"""Cross-frame smoothing with dropout coasting.

A constant-velocity smoother per tracked object: measurements pull the state
toward the detection, and during spray dropouts the box coasts along its
velocity (decaying) instead of vanishing, so downstream consumers see a
continuous trajectory with an explicit freshness signal.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class SmoothedBox:
    """EMA + velocity smoother over (cx, cy, w, h) with coasting."""

    def __init__(self, pos_alpha: float = 0.4, size_alpha: float = 0.25, vel_alpha: float = 0.3,
                 coast_decay: float = 0.9):
        self.pos_alpha = pos_alpha
        self.size_alpha = size_alpha
        self.vel_alpha = vel_alpha
        self.coast_decay = coast_decay
        self.state: Optional[np.ndarray] = None  # [cx, cy, w, h]
        self.vel = np.zeros(2)
        self.frames_since_update = 0
        self.last_conf = 0.0

    def update(self, bbox: tuple[float, float, float, float], conf: float) -> None:
        x1, y1, x2, y2 = bbox
        meas = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1])
        if self.state is None:
            self.state = meas
        else:
            new_vel = meas[:2] - self.state[:2]
            self.vel = (1 - self.vel_alpha) * self.vel + self.vel_alpha * new_vel
            self.state[:2] += self.pos_alpha * (meas[:2] - self.state[:2])
            self.state[2:] += self.size_alpha * (meas[2:] - self.state[2:])
        self.frames_since_update = 0
        self.last_conf = conf

    def coast(self) -> None:
        """No measurement this frame: advance along decayed velocity."""
        if self.state is not None:
            self.state[:2] += self.vel
            self.vel *= self.coast_decay
        self.frames_since_update += 1

    def reset(self) -> None:
        self.state = None
        self.vel = np.zeros(2)
        self.frames_since_update = 0
        self.last_conf = 0.0

    @property
    def bbox(self) -> Optional[tuple[float, float, float, float]]:
        if self.state is None:
            return None
        cx, cy, w, h = self.state
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    @property
    def center(self) -> Optional[tuple[float, float]]:
        if self.state is None:
            return None
        return (float(self.state[0]), float(self.state[1]))

    def is_fresh(self, max_coast_frames: int) -> bool:
        return self.state is not None and self.frames_since_update <= max_coast_frames
