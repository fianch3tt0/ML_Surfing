"""Base interface every detector plugin implements."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..detections import Detection


class BaseDetector(ABC):
    """A detector takes a BGR frame and returns detections in full-frame pixels.

    Implementations must be stateless across frames except for internal
    tracking state (e.g. ultralytics persist=True); the fusion layer owns all
    cross-frame reasoning beyond track IDs.
    """

    name: str = "base"

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]:
        ...

    def close(self) -> None:
        """Release any resources. Default: nothing."""
