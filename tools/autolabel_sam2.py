"""Auto-label a clip with SAM2 video propagation -> COCO JSON.

Click each object ONCE on the first frame; SAM2 tracks it through the whole
clip. Minutes of effort per clip instead of hand-labeling every frame. The
output doubles as evaluation ground truth (evaluate.py) and as training labels
for fine-tuning the board model.

Interactive:
    python tools/autolabel_sam2.py --video clip.mp4 --out eval/clip_gt.json
    (left-click rider -> press 'p'; left-click board -> press 's'; ENTER to run)

Headless:
    python tools/autolabel_sam2.py --video clip.mp4 --out gt.json \
        --points "960,400=person;940,620=surfboard"

Contract: image ids in the COCO json == frame indices (evaluate.py relies on it).
Requires: pip install ultralytics (sam2.1_b.pt downloads automatically).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2

CATEGORIES = {"person": 1, "surfboard": 2}


def pick_points_interactive(video: str) -> list[tuple[int, int, str]]:
    cap = cv2.VideoCapture(video)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read first frame of {video}")

    picked: list[tuple[int, int, str]] = []
    pending: list[tuple[int, int]] = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pending.clear()
            pending.append((x, y))

    win = "autolabel: click object, then 'p'=person 's'=surfboard, ENTER=run, u=undo"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        vis = frame.copy()
        for (x, y, lbl) in picked:
            cv2.circle(vis, (x, y), 6, (0, 220, 60), -1)
            cv2.putText(vis, lbl, (x + 8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 60), 2)
        for (x, y) in pending:
            cv2.circle(vis, (x, y), 6, (0, 160, 255), 2)
        cv2.imshow(win, vis)
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 10):  # ENTER
            break
        if key == 27:
            sys.exit("Cancelled.")
        if key == ord("u") and picked:
            picked.pop()
        if key == ord("p") and pending:
            picked.append((*pending.pop(), "person"))
        if key == ord("s") and pending:
            picked.append((*pending.pop(), "surfboard"))
    cv2.destroyWindow(win)
    if not picked:
        sys.exit("No points picked.")
    return picked


def parse_points_arg(spec: str) -> list[tuple[int, int, str]]:
    out = []
    for part in spec.split(";"):
        xy, label = part.split("=")
        x, y = xy.split(",")
        label = label.strip()
        if label not in CATEGORIES:
            raise ValueError(f"Unknown label '{label}' (use person|surfboard)")
        out.append((int(x), int(y), label))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="SAM2 video propagation -> COCO GT")
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True, help="output COCO json path")
    ap.add_argument("--points", default=None,
                    help='headless prompts: "x,y=person;x,y=surfboard"')
    ap.add_argument("--model", default="sam2.1_b.pt")
    ap.add_argument("--imgsz", type=int, default=1024)
    args = ap.parse_args()

    prompts = parse_points_arg(args.points) if args.points else pick_points_interactive(args.video)
    points = [[x, y] for x, y, _ in prompts]
    labels_in = [1] * len(points)
    obj_classes = [lbl for _, _, lbl in prompts]
    print(f"Prompts: {prompts}")

    from ultralytics.models.sam import SAM2VideoPredictor

    predictor = SAM2VideoPredictor(
        overrides=dict(task="segment", mode="predict", model=args.model,
                       imgsz=args.imgsz, conf=0.25, save=False, verbose=False)
    )
    results = predictor(source=args.video, points=points, labels=labels_in, stream=True)

    images, annotations = [], []
    ann_id = 1
    for frame_idx, res in enumerate(results):
        h, w = res.orig_shape
        images.append({"id": frame_idx, "file_name": f"frame_{frame_idx:06d}",
                       "width": w, "height": h})
        if res.masks is None:
            continue
        for obj_i, poly_group in enumerate(res.masks.xy):
            if obj_i >= len(obj_classes) or len(poly_group) == 0:
                continue
            xs, ys = poly_group[:, 0], poly_group[:, 1]
            x1, y1, x2, y2 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
            if (x2 - x1) < 2 or (y2 - y1) < 2:
                continue
            annotations.append({
                "id": ann_id,
                "image_id": frame_idx,
                "category_id": CATEGORIES[obj_classes[obj_i]],
                "bbox": [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)],
                "area": round((x2 - x1) * (y2 - y1), 1),
                "iscrowd": 0,
                "segmentation": [poly_group.flatten().round(1).tolist()],
            })
            ann_id += 1
        if frame_idx % 100 == 0:
            print(f"  frame {frame_idx}: {ann_id - 1} annotations so far")

    coco = {
        "info": {"source_video": os.path.abspath(args.video), "tool": "autolabel_sam2"},
        "images": images,
        "annotations": annotations,
        "categories": [{"id": cid, "name": name} for name, cid in CATEGORIES.items()],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(coco, fh)
    print(f"Wrote {len(images)} frames / {len(annotations)} annotations -> {args.out}")


if __name__ == "__main__":
    main()
