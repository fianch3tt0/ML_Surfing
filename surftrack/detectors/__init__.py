"""Detector plugins. All implement BaseDetector.detect(frame) -> list[Detection]."""

from .base import BaseDetector
from .registry import build_detectors

__all__ = ["BaseDetector", "build_detectors"]
