# Marker Simulation Mode - ML_Surfing

## Overview

This document explains the **synthetic neon marker detection system** for testing surfboard detection without needing physical tape.

### The Problem
- OpenCV color/edge detection alone is **unreliable** for detecting surfboards in varying water/lighting conditions
- Roboflow keypoint detection requires large labeled datasets
- Physical neon tape requires equipment

### The Solution
**Synthetic marker simulation**: Add virtual bright neon markers to video frames in software, then detect them. This tests the full pipeline without hardware.

---

## How It Works

### 1. Marker Placement
Three synthetic bright markers are placed on each frame:
- **Nose marker** (top of board)
- **Tail marker** (bottom of board)  
- **Center marker** (middle of board)

They're positioned in the region of interest (ROI) below the surfer's feet using MediaPipe pose landmarks.

### 2. Marker Detection
- Markers are drawn in bright neon colors (Green, Pink, Yellow, or Orange)
- HSV color range detection extracts marker centroids
- Board orientation is calculated from the nose-to-tail vector
- Angle and length are computed and displayed

### 3. Board Drawing
- A bounding box is drawn around the detected board
- An arrow shows board orientation
- Real-time stats show angle and length

---

## Usage

### Enable Marker Simulation
Set `USE_MARKER_MODE = True` at the top of `app.py`:

```python
USE_MARKER_MODE = True       # Enable synthetic neon tape detection
MARKER_COLOR = 'green'       # Options: 'green', 'pink', 'yellow', 'orange'
```

### Runtime Controls in app.py

| Key | Action |
|-----|--------|
| `m` | Toggle marker simulation on/off |
| `d` | Toggle debug overlays |
| `t` | Toggle optional tracker |
| `q` or `ESC` | Quit |

### Command Line Example

```bash
# Activate venv
.\venv\Scripts\Activate.ps1

# Run app with marker detection enabled
python app.py
```

Press `m` during runtime to toggle between:
- **Marker simulation mode**: Shows synthetic markers + board overlay
- **OpenCV mode**: Uses traditional color/edge detection

---

## Demo Script (test_markers.py)  

A standalone demo that shows just the marker detection pipeline:

```bash
python test_markers.py
```

**Controls in demo:**
- `s` - Toggle marker simulation
- `c` - Cycle through marker colors
- `d` - Toggle detection overlay
- `q` or `ESC` - Quit

---

## Marker Colors

The simulator supports four bright neon colors:

| Color | HSV Range | Use Case |
|-------|-----------|----------|
| **Green** | H: 35-85, S/V: 100-255 | Best in most conditions |
| **Pink** | H: 125-165, S/V: 100-255 | Good contrast with water |
| **Yellow** | H: 15-35, S/V: 100-255 | Very bright |
| **Orange** | H: 5-25, S/V: 100-255 | Warm conditions |

Change color in code:
```python
MARKER_COLOR = 'pink'  # or 'green', 'yellow', 'orange'
```

---

## Next Steps: Real Tape

Once marker simulation works reliably:

1. **Buy neon tape** (Gaff tape, spike tape, or athletic tape in neon colors)
2. **Place on board**: Nose, tail, and center near rails
3. **Set `USE_MARKER_MODE = False`** in app.py to use real markers
4. **Adjust HSV ranges** in `marker_simulator.py` if needed for your tape

---

## Architecture

### Files

- **`marker_simulator.py`** - MarkerSimulator class for adding/detecting synthetic markers
- **`app.py`** - Main app with integrated marker mode toggle (`USE_MARKER_MODE`)
- **`test_markers.py`** - Standalone demo for testing markers

### MarkerSimulator Class

```python
from marker_simulator import MarkerSimulator

sim = MarkerSimulator(marker_color='green', marker_size=15)

# Add markers to frame
marked_frame, positions = sim.add_markers_to_frame(frame, board_roi)

# Detect markers by color  
centroids, confidence = sim.detect_markers_in_frame(marked_frame)

# Calculate board vectors
board_info = sim.calculate_board_vectors(centroids)

# Draw overlays
vis = sim.draw_board_overlay(marked_frame, board_info)
```

---

## Tips & Troubleshooting

### Markers Not Detecting?
- **Lighting**: Ensure markers are bright enough relative to background
- **HSV range**: Try adjusting HSV bounds in `marker_simulator.py`
- **Color conflicts**: Water might have same HSV as marker color — try a different color
  
### False Positives?
- Increase `MIN_DETECT_AREA_FULL` to require larger blobs
- Tighten the ROI window around feet using pose landmarks

### Slow Performance?
- Set `DETECT_EVERY = 5` to detect less frequently
- Reduce frame resolution
- Disable `SHOW_DEBUG_OVERLAYS`

---

## Future Improvements

- [ ] Train YOLOv8 on frames with synthetic markers
- [ ] Add real neon tape detection with auto-calibration
- [ ] Implement board stability metrics over time
- [ ] Add confidence scoring for detection robustness
- [ ] Create annotated dataset from synthetic marker frames

---

## References

**See also:**
- [marker_simulator.py](marker_simulator.py) - Full implementation
- [app.py](app.py) - Integration example (search for `USE_MARKER_MODE`)
- [test_markers.py](test_markers.py) - Demo/test script

---

**Author Notes:**
Synthetic marker simulation is a proven technique used in:
- Motion capture systems (e.g., Vicon, OptiTrack)
- Sports analytics (golf, baseball, tennis)
- Robotics vision pipelines

This approach significantly increases reliability while keeping complexity low.
