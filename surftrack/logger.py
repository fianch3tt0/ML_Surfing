"""Per-frame JSONL output — the pipeline's data product.

One JSON object per line per frame:
{"frame": 12, "t": 0.4, "state": "SURFING",
 "rider": {"bbox": [...], "conf": 0.91, "mode": "detected", "track_id": 3},
 "board": {"bbox": [...], "conf": 0.55, "mode": "detected"},
 "anchor": [x, y], "fps": 42.1}
"""

from __future__ import annotations

import json
from typing import Optional

from .fusion import FrameState


class JsonlLogger:
    def __init__(self, path: str):
        self.path = path
        self._fh = open(path, "w", encoding="utf-8")

    def write(self, frame_idx: int, ts: float, fs: FrameState, fps: Optional[float] = None) -> None:
        rec = {
            "frame": frame_idx,
            "t": round(ts, 3),
            "state": fs.state,
            "rider": fs.rider.to_dict() if fs.rider else None,
            "board": fs.board.to_dict() if fs.board else None,
            "anchor": [round(v, 1) for v in fs.ankle_anchor] if fs.ankle_anchor else None,
        }
        if fps is not None:
            rec["fps"] = round(fps, 1)
        self._fh.write(json.dumps(rec) + "\n")

    def close(self) -> None:
        self._fh.close()
