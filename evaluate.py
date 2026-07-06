"""Evaluate a surftrack run (JSONL from app.py) against COCO-format ground truth.

Ground truth comes from tools/autolabel_sam2.py (SAM2 video propagation) or any
COCO JSON whose image ids equal frame indices.

Metrics per class (person, surfboard):
  - detection rate: fraction of GT frames where the tracked object matched (IoU >= thr)
  - precision / recall / AP@0.5 over 'detected'-mode frames, conf-ranked
  - center jitter: std of frame-to-frame change in (pred - GT) center residual
  - dropout run-length histogram: consecutive GT frames missed
Plus:
  - FP/min: 'detected'-mode predictions on frames with no GT of that class
            (run this on rider-free water clips to measure spray/reflection FPs)
  - FPS: mean of the fps field logged by app.py

Usage:
  python evaluate.py --pred runs/20260705_1.jsonl --gt eval/clip1_gt.json \
      --out eval/clip1_metrics.json --tags sunny,chop_light
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

CLASSES = ("person", "surfboard")
PRED_KEY = {"person": "rider", "surfboard": "board"}


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def load_gt(path: str) -> dict[str, dict[int, list]]:
    """COCO json -> {class_name: {frame_idx: [bbox_xyxy, ...]}}."""
    with open(path, "r", encoding="utf-8") as fh:
        coco = json.load(fh)
    cat_names = {c["id"]: c["name"] for c in coco["categories"]}
    gt: dict[str, dict[int, list]] = {c: defaultdict(list) for c in CLASSES}
    for ann in coco["annotations"]:
        name = cat_names.get(ann["category_id"])
        if name not in gt:
            continue
        x, y, w, h = ann["bbox"]
        gt[name][int(ann["image_id"])].append((x, y, x + w, y + h))
    return gt


def load_pred(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def average_precision(conf_tp: list[tuple[float, bool]], n_gt: int) -> float:
    """AP@0.5 from (confidence, is_tp) pairs, all-point interpolation."""
    if n_gt == 0 or not conf_tp:
        return 0.0
    conf_tp.sort(key=lambda x: -x[0])
    tp_cum = fp_cum = 0
    precisions, recalls = [], []
    for _, is_tp in conf_tp:
        tp_cum += int(is_tp)
        fp_cum += int(not is_tp)
        precisions.append(tp_cum / (tp_cum + fp_cum))
        recalls.append(tp_cum / n_gt)
    ap = 0.0
    prev_r = 0.0
    for i in range(len(recalls)):
        p_at = max(precisions[i:])
        ap += (recalls[i] - prev_r) * p_at
        prev_r = recalls[i]
    return ap


def run_lengths(missed_flags: list[bool]) -> list[int]:
    runs, cur = [], 0
    for m in missed_flags:
        if m:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    return runs


def evaluate_class(cls: str, preds: list[dict], gt_frames: dict[int, list],
                   iou_thr: float, clip_minutes: float) -> dict:
    key = PRED_KEY[cls]
    n_gt_frames = len(gt_frames)
    tp = fp = 0
    conf_tp: list[tuple[float, bool]] = []
    missed_flags: list[bool] = []
    residuals: list[tuple[float, float]] = []
    fp_on_empty = 0

    matched_frames = 0
    for rec in preds:
        f = rec["frame"]
        p = rec.get(key)
        gts = gt_frames.get(f, [])
        pred_detected = p is not None and p.get("mode") == "detected"

        if gts:
            best = max((iou(p["bbox"], g) for g in gts), default=0.0) if p else 0.0
            hit = p is not None and best >= iou_thr
            missed_flags.append(not hit)
            if hit:
                matched_frames += 1
                g = max(gts, key=lambda g: iou(p["bbox"], g))
                pc = ((p["bbox"][0] + p["bbox"][2]) / 2, (p["bbox"][1] + p["bbox"][3]) / 2)
                gc = ((g[0] + g[2]) / 2, (g[1] + g[3]) / 2)
                residuals.append((pc[0] - gc[0], pc[1] - gc[1]))
            if pred_detected:
                is_tp = best >= iou_thr
                tp += int(is_tp)
                fp += int(not is_tp)
                conf_tp.append((p.get("conf", 0.0), is_tp))
        elif pred_detected:
            fp += 1
            fp_on_empty += 1
            conf_tp.append((p.get("conf", 0.0), False))

    jitter = None
    if len(residuals) > 2:
        res = np.array(residuals)
        deltas = np.diff(res, axis=0)
        jitter = float(np.sqrt((deltas ** 2).sum(axis=1).mean()))

    dropouts = run_lengths(missed_flags)
    return {
        "gt_frames": n_gt_frames,
        "detection_rate": round(matched_frames / n_gt_frames, 4) if n_gt_frames else None,
        "precision": round(tp / (tp + fp), 4) if (tp + fp) else None,
        "recall": round(tp / n_gt_frames, 4) if n_gt_frames else None,
        "ap50": round(average_precision(conf_tp, n_gt_frames), 4) if n_gt_frames else None,
        "center_jitter_px": round(jitter, 2) if jitter is not None else None,
        "fp_per_min_on_empty": round(fp_on_empty / clip_minutes, 2) if clip_minutes else None,
        "dropout_runs": {
            "count": len(dropouts),
            "max": max(dropouts) if dropouts else 0,
            "mean": round(float(np.mean(dropouts)), 1) if dropouts else 0,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate surftrack JSONL vs COCO GT")
    ap.add_argument("--pred", required=True, help="JSONL from app.py")
    ap.add_argument("--gt", required=True, help="COCO json (image_id == frame index)")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--video-fps", type=float, default=30.0,
                    help="clip FPS for FP/min normalization")
    ap.add_argument("--tags", default="", help="condition tags recorded into the report")
    ap.add_argument("--out", default=None, help="write metrics JSON here")
    args = ap.parse_args()

    preds = load_pred(args.pred)
    gt = load_gt(args.gt)
    clip_minutes = len(preds) / args.video_fps / 60.0 if preds else 0.0

    fps_vals = [r["fps"] for r in preds if r.get("fps")]
    report = {
        "pred": args.pred,
        "gt": args.gt,
        "tags": [t for t in args.tags.split(",") if t],
        "frames": len(preds),
        "pipeline_fps_mean": round(float(np.mean(fps_vals)), 1) if fps_vals else None,
        "classes": {},
    }
    for cls in CLASSES:
        report["classes"][cls] = evaluate_class(cls, preds, gt[cls], args.iou, clip_minutes)

    print(json.dumps(report, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
