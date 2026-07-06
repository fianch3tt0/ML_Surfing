"""Overlay rendering for live view and review videos."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .fusion import FrameState, DETECTED, COASTING, ESTIMATED, SURFING, DOWN, GONE

RIDER_COLOR = (80, 220, 40)      # green
BOARD_COLOR = (0, 160, 255)      # orange
EST_COLOR = (0, 210, 255)        # yellow-ish for estimated/coasting
ANCHOR_COLOR = (255, 120, 0)
STATE_COLORS = {SURFING: (80, 220, 40), DOWN: (0, 100, 255), GONE: (100, 100, 100)}

# COCO skeleton edges for pose keypoints
SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
]


def _dashed_rect(img, p1, p2, color, thickness=2, dash=12):
    x1, y1 = p1
    x2, y2 = p2
    for (ax, ay, bx, by) in ((x1, y1, x2, y1), (x2, y1, x2, y2), (x2, y2, x1, y2), (x1, y2, x1, y1)):
        length = int(np.hypot(bx - ax, by - ay))
        if length == 0:
            continue
        for start in range(0, length, dash * 2):
            end = min(start + dash, length)
            sx = int(ax + (bx - ax) * start / length)
            sy = int(ay + (by - ay) * start / length)
            ex = int(ax + (bx - ax) * end / length)
            ey = int(ay + (by - ay) * end / length)
            cv2.line(img, (sx, sy), (ex, ey), color, thickness)


def _label(img, text, x, y, color):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)


def draw_overlay(frame: np.ndarray, fs: FrameState, fps: Optional[float] = None,
                 show_raw: bool = False) -> np.ndarray:
    vis = frame.copy()
    H, W = vis.shape[:2]

    if show_raw:
        for d in fs.raw_detections:
            x1, y1, x2, y2 = [int(v) for v in d.bbox]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (160, 160, 160), 1)

    if fs.rider is not None:
        x1, y1, x2, y2 = [int(v) for v in fs.rider.bbox]
        color = RIDER_COLOR if fs.rider.mode == DETECTED else EST_COLOR
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        tag = f"rider {fs.rider.conf:.2f}"
        if fs.rider.track_id is not None:
            tag += f" #{fs.rider.track_id}"
        if fs.rider.mode == COASTING:
            tag += " (coast)"
        _label(vis, tag, x1, max(16, y1 - 6), color)
        if fs.rider.keypoints:
            kps = fs.rider.keypoints
            for a, b in SKELETON:
                if kps[a][2] > 0.3 and kps[b][2] > 0.3:
                    cv2.line(vis, (int(kps[a][0]), int(kps[a][1])),
                             (int(kps[b][0]), int(kps[b][1])), (200, 200, 60), 1)

    if fs.board is not None:
        x1, y1, x2, y2 = [int(v) for v in fs.board.bbox]
        if fs.board.mode == DETECTED:
            cv2.rectangle(vis, (x1, y1), (x2, y2), BOARD_COLOR, 2)
            _label(vis, f"board {fs.board.conf:.2f}", x1, min(H - 6, y2 + 18), BOARD_COLOR)
        else:
            _dashed_rect(vis, (x1, y1), (x2, y2), EST_COLOR, 2)
            _label(vis, f"board ({fs.board.mode})", x1, min(H - 6, y2 + 18), EST_COLOR)

    if fs.ankle_anchor is not None:
        cv2.circle(vis, (int(fs.ankle_anchor[0]), int(fs.ankle_anchor[1])), 5, ANCHOR_COLOR, -1)

    # state banner
    color = STATE_COLORS.get(fs.state, (200, 200, 200))
    cv2.rectangle(vis, (0, 0), (170, 34), (20, 20, 20), -1)
    cv2.putText(vis, fs.state, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    if fps is not None:
        _label(vis, f"{fps:.1f} FPS", W - 110, 24, (230, 230, 230))

    return vis
