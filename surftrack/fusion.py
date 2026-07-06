"""Fusion layer: merge per-frame detections into one rider + one board state.

Responsibilities:
- pick the rider (highest-confidence person, sticky to the previous track id)
- pick the board (surfboard detection gated to the region below the rider)
- coast through spray dropouts (tracking.SmoothedBox)
- when no board is detected for a while, *estimate* it from the rider's
  ankle midpoint (the valid version of the old feet-ROI idea: a prior on top
  of a real detector, not the detector itself)
- run the SURFING / DOWN / GONE state machine
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot
from typing import Optional

from .detections import Detection
from .tracking import SmoothedBox

SURFING = "SURFING"
DOWN = "DOWN"
GONE = "GONE"

# board modes
DETECTED = "detected"
COASTING = "coasting"
ESTIMATED = "estimated"


@dataclass
class ObjectState:
    bbox: tuple[float, float, float, float]
    conf: float
    mode: str  # detected | coasting | estimated
    track_id: Optional[int] = None
    keypoints: Optional[list] = None

    def to_dict(self) -> dict:
        d = {
            "bbox": [round(float(v), 1) for v in self.bbox],
            "conf": round(float(self.conf), 3),
            "mode": self.mode,
        }
        if self.track_id is not None:
            d["track_id"] = int(self.track_id)
        return d


@dataclass
class FrameState:
    state: str = GONE
    rider: Optional[ObjectState] = None
    board: Optional[ObjectState] = None
    ankle_anchor: Optional[tuple[float, float]] = None
    raw_detections: list[Detection] = field(default_factory=list)


class FusionEngine:
    def __init__(
        self,
        rider_coast_frames: int = 15,
        board_coast_frames: int = 20,
        gone_after_frames: int = 60,
        down_aspect: float = 0.9,          # rider bbox h/w below this -> in the water
        board_gate_factor: float = 1.6,    # board must lie within this * rider height of the anchor
        est_board_width_factor: float = 1.0,   # estimated board width as fraction of rider height
        est_board_aspect: float = 0.25,        # estimated board height / width
    ):
        self.rider_coast_frames = rider_coast_frames
        self.board_coast_frames = board_coast_frames
        self.gone_after_frames = gone_after_frames
        self.down_aspect = down_aspect
        self.board_gate_factor = board_gate_factor
        self.est_board_width_factor = est_board_width_factor
        self.est_board_aspect = est_board_aspect

        self.rider_box = SmoothedBox()
        self.board_box = SmoothedBox()
        self.rider_track_id: Optional[int] = None
        self.frames_without_rider = 0

    # ---- selection ----

    def _pick_rider(self, detections: list[Detection]) -> Optional[Detection]:
        people = [d for d in detections if d.cls_name == "person"]
        if not people:
            return None
        if self.rider_track_id is not None:
            same = [d for d in people if d.track_id == self.rider_track_id]
            if same:
                return max(same, key=lambda d: d.conf)
        return max(people, key=lambda d: d.conf)

    def _anchor(self, rider: Optional[Detection]) -> Optional[tuple[float, float]]:
        """Board anchor: ankle midpoint if pose available, else rider bbox bottom-center."""
        if rider is None:
            if self.rider_box.bbox is not None:
                x1, y1, x2, y2 = self.rider_box.bbox
                return ((x1 + x2) / 2.0, y2)
            return None
        mid = rider.ankle_midpoint()
        if mid is not None:
            return mid
        x1, y1, x2, y2 = rider.bbox
        return ((x1 + x2) / 2.0, y2)

    def _pick_board(self, detections: list[Detection],
                    anchor: Optional[tuple[float, float]],
                    rider_height: Optional[float]) -> Optional[Detection]:
        boards = [d for d in detections if d.cls_name == "surfboard"]
        if not boards:
            return None
        if anchor is None or rider_height is None or rider_height <= 0:
            return max(boards, key=lambda d: d.conf)
        # Gate: board center must be near the rider's feet (rejects distractor
        # boards on other boats / racks on shore).
        gate = self.board_gate_factor * rider_height
        near = [d for d in boards if hypot(d.center[0] - anchor[0], d.center[1] - anchor[1]) <= gate]
        if not near:
            return None
        return min(near, key=lambda d: hypot(d.center[0] - anchor[0], d.center[1] - anchor[1]))

    # ---- main update ----

    def update(self, detections: list[Detection]) -> FrameState:
        fs = FrameState(raw_detections=detections)

        rider_det = self._pick_rider(detections)
        if rider_det is not None:
            self.rider_box.update(rider_det.bbox, rider_det.conf)
            if rider_det.track_id is not None:
                self.rider_track_id = rider_det.track_id
            self.frames_without_rider = 0
        else:
            self.rider_box.coast()
            self.frames_without_rider += 1

        anchor = self._anchor(rider_det)
        fs.ankle_anchor = anchor

        rider_bbox = self.rider_box.bbox
        rider_height = (rider_bbox[3] - rider_bbox[1]) if rider_bbox else None

        board_det = self._pick_board(detections, anchor, rider_height)
        if board_det is not None:
            self.board_box.update(board_det.bbox, board_det.conf)
        else:
            self.board_box.coast()

        # ---- assemble rider state ----
        rider_visible = self.rider_box.is_fresh(self.rider_coast_frames)
        if rider_visible and rider_bbox is not None:
            fs.rider = ObjectState(
                bbox=rider_bbox,
                conf=self.rider_box.last_conf,
                mode=DETECTED if rider_det is not None else COASTING,
                track_id=self.rider_track_id,
                keypoints=(rider_det.keypoints.tolist()
                           if rider_det is not None and rider_det.keypoints is not None else None),
            )

        # ---- assemble board state ----
        if self.board_box.is_fresh(self.board_coast_frames) and self.board_box.bbox is not None:
            fs.board = ObjectState(
                bbox=self.board_box.bbox,
                conf=self.board_box.last_conf,
                mode=DETECTED if board_det is not None else COASTING,
            )
        elif rider_visible and anchor is not None and rider_height:
            # Estimation fallback: board sits under the ankles, sized off the rider.
            w = self.est_board_width_factor * rider_height
            h = max(8.0, self.est_board_aspect * w)
            cx, cy = anchor
            cy += 0.15 * h  # nose rides slightly below the ankle line
            fs.board = ObjectState(
                bbox=(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2),
                conf=0.0,
                mode=ESTIMATED,
            )

        # ---- state machine ----
        if self.frames_without_rider > self.gone_after_frames:
            fs.state = GONE
            self.rider_box.reset()
            self.board_box.reset()
            self.rider_track_id = None
        elif rider_visible and rider_bbox is not None:
            w = rider_bbox[2] - rider_bbox[0]
            h = rider_bbox[3] - rider_bbox[1]
            aspect = h / (w + 1e-6)
            fs.state = DOWN if aspect < self.down_aspect else SURFING
        else:
            fs.state = GONE if self.frames_without_rider > self.gone_after_frames else DOWN

        return fs
