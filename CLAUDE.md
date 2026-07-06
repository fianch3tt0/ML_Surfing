# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Wakesurf rider + surfboard tracking from a boat-mounted camera. Modular
pipeline: pluggable detectors (YOLO11-pose for the rider, YOLO11 detect for the
board, RT-DETR and tape-marker plugins available), a fusion layer with
rider-anchored board estimation and dropout coasting, JSONL data output, and a
quantitative eval harness. Development runs against recorded lake footage, not
live cameras.

## Commands

Windows (PowerShell) — the primary dev environment (CUDA PC; runtime also
targets macOS/MPS on a MacBook, so keep the runtime path free of CUDA-only deps):

```powershell
.\venv\Scripts\Activate.ps1
python app.py --source videos\clip.mp4 --camera gopro --save out.mp4  # main entry
python app.py --source 0 --camera webcam                              # live camera
python app.py --source clip.mp4 --no-show --max-frames 100            # headless smoke test
python tools\autolabel_sam2.py --video clip.mp4 --out eval\clip_gt.json  # SAM2 GT labeling
python evaluate.py --pred runs\run.jsonl --gt eval\clip_gt.json          # metrics
python tools\extract_frames.py                                           # sample frames from videos/
```

No linter or test suite. Verification = run `app.py` headless on a clip and
check the JSONL output, or run `evaluate.py` against SAM2-generated ground truth.

## Architecture

Data flow (all per-frame, orchestrated in `app.py`):

`FrameSource` (`surftrack/sources.py`: camera index | video file | image dir)
→ `CameraProfile.process` (`surftrack/camera.py`: per-camera undistort/downscale,
profiles in `configs/cameras.yaml`)
→ detector plugins (`surftrack/detectors/`, built from `configs/default.yaml`
by `registry.build_detectors`; all return `list[Detection]` in full-frame pixels)
→ `FusionEngine.update` (`surftrack/fusion.py`)
→ `draw_overlay` (`surftrack/viz.py`) + `JsonlLogger` (`surftrack/logger.py`).

Key invariants and design points:

- **Detectors are stateless plugins** implementing
  `BaseDetector.detect(frame) -> list[Detection]` (`surftrack/detectors/base.py`).
  Add a new model by writing a plugin and registering it in
  `detectors/registry.py`; select/configure it in `configs/default.yaml`.
- **Fusion owns all cross-frame logic**: rider selection (sticky track id),
  board gating (must be near the rider's ankle anchor — rejects distractor
  boards), coasting through spray dropouts (`surftrack/tracking.py`
  `SmoothedBox`), the ESTIMATED-board fallback sized off rider height, and the
  SURFING/DOWN/GONE state machine. Board `mode` is always one of
  `detected | coasting | estimated` — downstream consumers rely on it.
- **`evaluate.py` contract:** COCO ground-truth `image_id` == frame index.
  `tools/autolabel_sam2.py` produces GT in that format via SAM2 video
  propagation (one click per object per clip — never hand-label frames).
- The marker plugin (`detectors/marker.py`) is a deferred orientation channel:
  it only searches inside an ROI set by fusion (`set_roi`), never full-frame.
- Device selection is `auto` → cuda → mps → cpu (`detectors/yolo.py
  resolve_device`); model weights (`*.pt`) auto-download and are gitignored.

## Conventions & gotchas

- OpenCV frames are **BGR**; bboxes are `(x1, y1, x2, y2)` floats in full-frame
  pixels everywhere in the pipeline.
- Camera sources get warm-up reads (some Windows webcams drop first frames) —
  keep that in `FrameSource`.
- `videos/`, `frames/`, `runs/`, `*.pt` are gitignored data/artifacts.
- `venv/` is a checked-in-locally (gitignored) Python 3.10 env; don't read or
  edit files under it.
- Tuning lives in `configs/*.yaml`, not in code constants.
