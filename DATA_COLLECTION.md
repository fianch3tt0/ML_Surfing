# First Boat Outing — Data Collection Protocol

Everything downstream (evaluation, fine-tuning, threshold tuning) is built on
this footage. Follow the checklist; the log sheet becomes `eval/manifest.yaml`.

## Mounting (both cameras)

- **Position:** wake tower, **2.0–2.6 m above the waterline**, centered, looking aft.
- **Pitch:** ~15° downward — horizon in the top ~15% of the frame.
- **Rigidity matters:** engine vibration causes rolling-shutter jello. Use a hard
  mount (no suction cups on flexing surfaces); add a rubber pad if available.
- If time allows, record 10 minutes from a **transom mount** too, tagged
  separately, so tower-vs-transom can be compared in eval.

## Camera settings

| Setting | GoPro | Phone |
|---|---|---|
| Resolution / rate | **4K60** | 4K60 (or max available at 60fps) |
| Lens / FOV | **Linear** (never SuperView/Wide) | Main lens (1×) — do NOT use ultrawide |
| Shutter | Lock ≥ 1/960 (Protune: Shutter 1/960 or 1/1920) | Use an app with manual control if possible; else default |
| White balance | **Lock to 5500K / Daylight** (not Auto) | Lock via manual camera app if possible |
| Exposure / EV | **−0.5 EV bias** (protect highlights on water) | Tap-hold to lock AE/AF on the wake region, drag exposure down slightly |
| ISO | Auto, max capped (GoPro: ISO max 400 in daylight) | Auto |
| HDR / stabilization | **Off** (HDR ghosts moving objects; stabilization crops and warps) | HDR video off if possible |
| Bitrate | Highest available | Highest available |

Locked WB + exposure is not optional — auto-everything drifts as the boat turns
relative to the sun and becomes a source of model domain shift.

## Recording checklist (~45–60 min total)

Per rider (≥2 riders, different builds and clothing colors if possible;
≥2 board colors if available):

- [ ] Deep-water start → getting up (×2–3)
- [ ] Normal surfing, both wake sides, ≥3 min each
- [ ] Aggressive carving / pumping, ≥2 min
- [ ] Rider drifting back off the wave to max distance, then recovering
- [ ] At least one fall left in the recording (don't cut it)

Environment passes (once per session):

- [ ] One pass driving **toward the sun** and one **away** (worst/best glare)
- [ ] **5 min empty wake at surf speed, no rider** ← the false-positive eval set
- [ ] **3 min idle water** (drifting, engine off)
- [ ] If another boat passes, keep recording (distractor data — people + boards)

Repeat what you can on **both** cameras; identical segments on GoPro and phone
let `evaluate.py` compare camera profiles directly.

## Log sheet (fill per clip, transfer to eval/manifest.yaml)

| Field | Example |
|---|---|
| Clip file | GX010042.mp4 |
| Camera + mode | gopro, 4K60 Linear |
| Mount | tower 2.3 m, ~15° down |
| Time / cloud cover | 14:30, scattered clouds |
| Sun relative to camera | behind_camera / into_sun / overcast |
| Wind / chop | light |
| Rider / clothing | Cam, black wetsuit |
| Board / color | blue 4'8" |
| Distance band | near (<8 m) |
| Rider-free clip? | no |

## Back home

1. Copy clips into `videos/` (gitignored).
2. Fill `eval/manifest.yaml`.
3. Baseline run: `python app.py --source videos/<clip> --camera gopro --save runs/<clip>_annotated.mp4 --no-show`
4. Pick 3–5 representative clips, generate GT: `python tools/autolabel_sam2.py --video videos/<clip> --out eval/<clip>_gt.json`
5. Metrics: `python evaluate.py --pred runs/<run>.jsonl --gt eval/<clip>_gt.json --tags sunny,chop_light --out eval/<clip>_metrics.json`
6. Decide from numbers (not eyeballs) whether Phase 3 fine-tuning is needed.
