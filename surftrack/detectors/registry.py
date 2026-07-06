"""Build detector plugins from config — swapping models is a YAML edit."""

from __future__ import annotations

from .base import BaseDetector


def build_detectors(config: dict) -> list[BaseDetector]:
    """Instantiate the detector stack listed under config['detectors'].

    Each entry: {type: yolo|yolo_pose|rtdetr|marker, ...kwargs passed through}.
    Import lazily so a missing optional dependency only breaks the plugin that
    needs it.
    """
    detectors: list[BaseDetector] = []
    shared = {
        "device": config.get("device", "auto"),
        "imgsz": config.get("imgsz", 960),
    }
    for entry in config.get("detectors", []):
        entry = dict(entry)
        dtype = entry.pop("type")
        if dtype == "yolo":
            from .yolo import YoloDetector

            detectors.append(YoloDetector(**{**shared, **entry}))
        elif dtype == "yolo_pose":
            from .yolo import YoloPoseDetector

            detectors.append(YoloPoseDetector(**{**shared, **entry}))
        elif dtype == "rtdetr":
            from .rtdetr import RTDETRDetector

            detectors.append(RTDETRDetector(**{**shared, **entry}))
        elif dtype == "marker":
            from .marker import MarkerDetector

            detectors.append(MarkerDetector(**entry))
        else:
            raise ValueError(f"Unknown detector type in config: {dtype}")
    return detectors
