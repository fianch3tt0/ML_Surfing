"""Frame sources: live camera, video file, or a directory of images.

The whole pipeline consumes FrameSource so recorded lake footage and the live
boat camera go through the identical code path.
"""

from __future__ import annotations

import glob
import os
import time
from typing import Iterator, Optional

import cv2
import numpy as np

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
VIDEO_EXTS = (".mp4", ".mov", ".m4v", ".avi", ".mkv")


class FrameSource:
    """Iterates (frame_idx, timestamp_sec, frame_bgr) from any supported source."""

    def __init__(self, spec: str, warmup_reads: int = 5):
        self.spec = spec
        self.is_live = False
        self._cap: Optional[cv2.VideoCapture] = None
        self._image_paths: list[str] = []

        if spec.isdigit():
            # Live camera. Warm-up reads: some Windows webcams drop the first frames.
            self.is_live = True
            self._cap = cv2.VideoCapture(int(spec))
            if not self._cap.isOpened():
                raise RuntimeError(
                    f"Cannot open camera index {spec}. Try another index (e.g. --source 1)."
                )
            for _ in range(warmup_reads):
                self._cap.read()
                time.sleep(0.05)
        elif os.path.isdir(spec):
            for ext in IMAGE_EXTS:
                self._image_paths.extend(glob.glob(os.path.join(spec, f"*{ext}")))
            self._image_paths.sort()
            if not self._image_paths:
                raise RuntimeError(f"No images found in directory: {spec}")
        elif os.path.isfile(spec):
            if not spec.lower().endswith(VIDEO_EXTS):
                raise RuntimeError(f"Unsupported file type: {spec}")
            self._cap = cv2.VideoCapture(spec)
            if not self._cap.isOpened():
                raise RuntimeError(f"Cannot open video file: {spec}")
        else:
            raise RuntimeError(f"Source not found: {spec}")

    @property
    def fps(self) -> float:
        """Nominal FPS of the source (0 if unknown, e.g. image dirs)."""
        if self._cap is not None:
            return self._cap.get(cv2.CAP_PROP_FPS) or 0.0
        return 0.0

    @property
    def frame_size(self) -> Optional[tuple[int, int]]:
        """(width, height) if known before reading."""
        if self._cap is not None:
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if w > 0 and h > 0:
                return (w, h)
        return None

    def __iter__(self) -> Iterator[tuple[int, float, np.ndarray]]:
        idx = 0
        t_start = time.time()
        if self._cap is not None:
            nominal = self.fps
            while True:
                ok, frame = self._cap.read()
                if not ok:
                    break
                if self.is_live:
                    ts = time.time() - t_start
                elif nominal > 0:
                    ts = idx / nominal
                else:
                    ts = 0.0
                yield idx, ts, frame
                idx += 1
        else:
            for path in self._image_paths:
                frame = cv2.imread(path)
                if frame is None:
                    continue
                yield idx, float(idx), frame
                idx += 1

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
