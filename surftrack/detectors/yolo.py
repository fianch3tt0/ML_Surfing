"""Ultralytics-based detectors (YOLO detect, YOLO pose, shared result parsing)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..detections import Detection
from .base import BaseDetector

# COCO class ids in pretrained models
COCO_PERSON = 0
COCO_SURFBOARD = 37


def resolve_device(device: str = "auto") -> str:
    """auto -> cuda (dev PC) -> mps (MacBook on the boat) -> cpu."""
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class _UltralyticsDetector(BaseDetector):
    """Shared wrapper around any ultralytics model (YOLO, RTDETR, pose)."""

    name = "ultralytics"

    def __init__(
        self,
        model_path: str,
        classes: Optional[list[int]] = None,
        conf: float = 0.25,
        imgsz: int = 960,
        device: str = "auto",
        track: bool = True,
        tracker: str = "bytetrack.yaml",
    ):
        self.model = self._load_model(model_path)
        self.classes = classes
        self.conf = conf
        self.imgsz = imgsz
        self.device = resolve_device(device)
        self.track = track
        self.tracker = tracker

    def _load_model(self, model_path: str):
        from ultralytics import YOLO

        return YOLO(model_path)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        kwargs = dict(
            source=frame,
            classes=self.classes,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        if self.track:
            results = self.model.track(persist=True, tracker=self.tracker, **kwargs)
        else:
            results = self.model.predict(**kwargs)
        return self._parse(results[0])

    def _parse(self, result) -> list[Detection]:
        detections: list[Detection] = []
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return detections
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None
        kpts = None
        if getattr(result, "keypoints", None) is not None and result.keypoints.data.numel() > 0:
            kpts = result.keypoints.data.cpu().numpy()  # (n, 17, 3)

        for i in range(len(xyxy)):
            detections.append(
                Detection(
                    cls_name=result.names[clss[i]],
                    bbox=tuple(float(v) for v in xyxy[i]),
                    conf=float(confs[i]),
                    source=self.name,
                    keypoints=kpts[i] if kpts is not None else None,
                    track_id=int(ids[i]) if ids is not None else None,
                )
            )
        return detections


class YoloDetector(_UltralyticsDetector):
    """Plain YOLO object detection (default: person + surfboard from COCO)."""

    name = "yolo"

    def __init__(self, model_path: str = "yolo11s.pt", classes=None, **kwargs):
        if classes is None:
            classes = [COCO_PERSON, COCO_SURFBOARD]
        super().__init__(model_path, classes=classes, **kwargs)


class YoloPoseDetector(_UltralyticsDetector):
    """YOLO pose: person boxes + COCO keypoints (ankles feed the board prior)."""

    name = "yolo_pose"

    def __init__(self, model_path: str = "yolo11s-pose.pt", **kwargs):
        # Pose models are person-only; no class filter needed.
        kwargs.pop("classes", None)
        super().__init__(model_path, classes=None, **kwargs)
