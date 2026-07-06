# ML_Surfing — wakesurf rider + board tracking

Real-time detection and tracking of a wakesurfer and their board from a
boat-mounted camera. YOLO-based modular pipeline with rider-anchored board
estimation, dropout coasting, and a quantitative evaluation harness.

## Quick start

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt   # plus CUDA torch on the dev PC, see requirements.txt

# Run on recorded footage (the normal dev loop)
python app.py --source videos\clip.mp4 --camera gopro --save out.mp4

# Live camera
python app.py --source 0 --camera webcam
```

Output per run: an annotated window/video plus a JSONL log
(`runs/<timestamp>.jsonl`) with per-frame rider/board boxes, confidences,
tracking mode (`detected|coasting|estimated`) and the SURFING/DOWN/GONE state.

## Workflow

1. **Collect footage** — follow [DATA_COLLECTION.md](DATA_COLLECTION.md).
2. **Baseline** — run `app.py` over the clips with the pretrained models.
3. **Ground truth** — `python tools/autolabel_sam2.py --video clip.mp4 --out eval/clip_gt.json`
   (click rider + board once; SAM2 propagates through the clip).
4. **Measure** — `python evaluate.py --pred runs/run.jsonl --gt eval/clip_gt.json`
   → detection rate, precision/recall/AP@0.5, jitter, dropout histogram,
   spray-FP rate, FPS.
5. **Iterate** — swap models/thresholds in `configs/default.yaml`; re-measure.

## Layout

- `surftrack/` — pipeline package (sources, detector plugins, tracking, fusion, viz, logging)
- `configs/` — pipeline config + per-camera profiles (GoPro / phone)
- `app.py` — CLI entry point
- `evaluate.py` — metrics vs COCO ground truth
- `tools/` — SAM2 auto-labeler, frame extractor
