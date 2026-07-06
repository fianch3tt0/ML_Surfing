"""RT-DETR plugin — benchmark alternative to YOLO (CUDA-oriented)."""

from __future__ import annotations

from .yolo import _UltralyticsDetector, COCO_PERSON, COCO_SURFBOARD


class RTDETRDetector(_UltralyticsDetector):
    """Real-Time DETR via ultralytics. Same interface, swap via config."""

    name = "rtdetr"

    def __init__(self, model_path: str = "rtdetr-l.pt", classes=None, **kwargs):
        if classes is None:
            classes = [COCO_PERSON, COCO_SURFBOARD]
        super().__init__(model_path, classes=classes, **kwargs)

    def _load_model(self, model_path: str):
        from ultralytics import RTDETR

        return RTDETR(model_path)
