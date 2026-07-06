"""surftrack CLI — wakesurf rider + board tracking.

Examples:
    python app.py --source lake_footage.mp4 --camera gopro --save out.mp4
    python app.py --source 0 --camera webcam          # live camera smoke test
    python app.py --source frames_dir --no-show --log run.jsonl

Keys while the window is open: q/ESC quit, r toggle raw detections.
"""

from __future__ import annotations

import argparse
import os
import time

import cv2
import yaml

from surftrack.camera import CameraProfile
from surftrack.detectors import build_detectors
from surftrack.fusion import FusionEngine
from surftrack.logger import JsonlLogger
from surftrack.sources import FrameSource
from surftrack.viz import draw_overlay


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Wakesurf rider + board tracker")
    p.add_argument("--source", required=True,
                   help="camera index ('0'), video file path, or image directory")
    p.add_argument("--config", default="configs/default.yaml", help="pipeline config YAML")
    p.add_argument("--cameras", default="configs/cameras.yaml", help="camera profiles YAML")
    p.add_argument("--camera", default=None,
                   help="camera profile name (gopro|phone|webcam); default: raw frames")
    p.add_argument("--save", default=None, help="write annotated video to this path")
    p.add_argument("--log", default=None,
                   help="JSONL output path (default: runs/<timestamp>.jsonl)")
    p.add_argument("--no-show", action="store_true", help="headless: no display window")
    p.add_argument("--show-raw", action="store_true", help="draw raw detections too")
    p.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = all)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    profile = None
    if args.camera:
        with open(args.cameras, "r", encoding="utf-8") as fh:
            profile = CameraProfile.from_config(yaml.safe_load(fh), args.camera)

    detectors = build_detectors(config)
    fusion = FusionEngine(**config.get("fusion", {}))

    log_path = args.log
    if log_path is None:
        os.makedirs("runs", exist_ok=True)
        log_path = os.path.join("runs", time.strftime("%Y%m%d_%H%M%S") + ".jsonl")
    logger = JsonlLogger(log_path)

    source = FrameSource(args.source)
    writer = None
    show = not args.no_show
    show_raw = args.show_raw
    if show:
        cv2.namedWindow("surftrack", cv2.WINDOW_NORMAL)

    fps_ema = None
    last_board_bbox = None
    n_frames = 0
    try:
        for frame_idx, ts, frame in source:
            t0 = time.perf_counter()
            if profile is not None:
                frame = profile.process(frame)

            detections = []
            for det in detectors:
                # marker detector needs the predicted board region as its ROI
                if hasattr(det, "set_roi"):
                    det.set_roi(last_board_bbox)
                detections.extend(det.detect(frame))

            fs = fusion.update(detections)
            last_board_bbox = fs.board.bbox if fs.board is not None else None

            dt = time.perf_counter() - t0
            inst_fps = 1.0 / max(dt, 1e-6)
            fps_ema = inst_fps if fps_ema is None else 0.9 * fps_ema + 0.1 * inst_fps

            logger.write(frame_idx, ts, fs, fps_ema)
            vis = draw_overlay(frame, fs, fps_ema, show_raw=show_raw)

            if args.save:
                if writer is None:
                    h, w = vis.shape[:2]
                    out_fps = source.fps if source.fps > 0 else 30.0
                    writer = cv2.VideoWriter(
                        args.save, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w, h)
                    )
                writer.write(vis)

            if show:
                cv2.imshow("surftrack", vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("r"):
                    show_raw = not show_raw

            n_frames += 1
            if args.max_frames and n_frames >= args.max_frames:
                break
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        for det in detectors:
            det.close()
        source.release()
        if writer is not None:
            writer.release()
        logger.close()
        cv2.destroyAllWindows()
        print(f"Processed {n_frames} frames. Log: {log_path}"
              + (f"  Video: {args.save}" if args.save else ""))


if __name__ == "__main__":
    main()
