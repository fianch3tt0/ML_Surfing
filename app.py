# app.py
# Stable surfboard detector + MediaPipe pose
# Replace your current app.py with this file.
#
# Requirements: mediapipe, opencv-python, numpy
# Usage: source venv/bin/activate && python3 app.py

import cv2
import mediapipe as mp
import numpy as np
import time
from math import sqrt

# =========================
# Tunables - tweak these
# =========================
USE_TRACKER = False          # Disabled by default — trackers often drift to distant objects.
DETECT_EVERY = 3            # run detection every N frames (1 = every frame)
DETECT_RESIZE_W = 640       # width to scale ROI to for detection (speed/accuracy tradeoff)
EMA_ALPHA = 0.30            # smoothing factor for bounding box (0.0-1.0)
PAD_FACTOR = 0.12           # expand detected box by this fraction (helps include full board)
MIN_DETECT_AREA_FULL = 1200 # absolute minimum hull area in full-frame pixels (tune up if false positives)
CONF_AREA_REF = 14000.0     # map area -> confidence (increase to require larger area)
POLY_CONF_DRAW = 0.45       # polygon will be drawn only if confidence >= this
BOX_CONF_DRAW = 0.20        # box will be drawn if confidence >= this
BLUE_LOWER = np.array([85, 55, 45])   # HSV lower bound for board color (tweak)
BLUE_UPPER = np.array([140, 255, 255])# HSV upper bound for board color (tweak)
SHOW_DEBUG_OVERLAYS = False  # Toggle: draw ROI and mask for debugging
MAX_POLY_POINTS = 8         # approximate polygon points for drawing (keeps shape stable)
SHOULDER_CLOSE_PIX = 200    # if shoulders wider than this, person likely close to camera -> hide board boxes
# =========================

# Helpers
def clamp(v, a, b): return max(a, min(b, v))
def ema(prev, new, alpha):
    if prev is None:
        return np.array(new, dtype=float)
    return (1 - alpha) * np.array(prev, dtype=float) + alpha * np.array(new, dtype=float)

def expand_rect(x, y, w, h, pad, frame_w, frame_h):
    cx = x + w/2.0; cy = y + h/2.0
    w2 = w * (1.0 + pad); h2 = h * (1.0 + pad)
    x2 = int(clamp(cx - w2/2.0, 0, frame_w-1))
    y2 = int(clamp(cy - h2/2.0, 0, frame_h-1))
    w2 = int(clamp(w2, 1, frame_w - x2))
    h2 = int(clamp(h2, 1, frame_h - y2))
    return x2, y2, w2, h2

# Camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open camera")
    raise SystemExit

cv2.namedWindow("Surf Pose + Board Tracker", cv2.WINDOW_NORMAL)

# Mediapipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# State
frame_idx = 0
smoothed_box = None   # [cx, cy, w, h] in full-frame pixel coords
last_poly = None      # polygon points (Nx2) in full-frame coords
last_conf = 0.0
last_poly_age = 999

# Optional tracker (not used by default)
tracker = None

# FPS helper
t0 = time.time(); frames = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        H, W = frame.shape[:2]
        vis = frame.copy()

        # ---- pose (always run) ----
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        feet_center = None
        shoulder_dist = 0
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            try:
                la = lm[mp_pose.PoseLandmark.LEFT_ANKLE.value]
                ra = lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
                ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                lx, ly = clamp(int(la.x * W), 0, W-1), clamp(int(la.y * H), 0, H-1)
                rx, ry = clamp(int(ra.x * W), 0, W-1), clamp(int(ra.y * H), 0, H-1)
                feet_center = ((lx + rx)//2, (ly + ry)//2)
                s_lx, s_ly = clamp(int(ls.x * W), 0, W-1), clamp(int(ls.y * H), 0, H-1)
                s_rx, s_ry = clamp(int(rs.x * W), 0, W-1), clamp(int(rs.y * H), 0, H-1)
                shoulder_dist = sqrt((s_lx - s_rx)**2 + (s_ly - s_ry)**2)
            except Exception:
                feet_center = None

        # If person is very close (shoulders wide), hide board overlays -> helps facial closeups
        person_too_close = shoulder_dist >= SHOULDER_CLOSE_PIX
        if person_too_close:
            smoothed_box = None
            last_poly = None
            last_conf = 0.0
            last_poly_age = 999
            tracker = None

        # Use cheap tracker? (we're not using it by default because it caused drift)
        if USE_TRACKER and tracker is None and smoothed_box is not None:
            # initialize tracker on current smoothed box
            try:
                tracker = cv2.TrackerCSRT_create() if not hasattr(cv2, "legacy") else cv2.legacy.TrackerCSRT_create()
                x = int(smoothed_box[0] - smoothed_box[2]/2)
                y = int(smoothed_box[1] - smoothed_box[3]/2)
                w = int(smoothed_box[2]); h = int(smoothed_box[3])
                tracker.init(frame, (x,y,w,h))
            except Exception:
                tracker = None

        # Try tracker first (fast), only if enabled
        tracked_box = None
        if USE_TRACKER and tracker is not None and not person_too_close:
            ok, tb = tracker.update(frame)
            if ok:
                tx, ty, tw, th = [int(round(v)) for v in tb]
                if tw > 4 and th > 4:
                    tracked_box = (tx, ty, tw, th)

        # Run detection periodically (or if no tracker box)
        if (frame_idx % DETECT_EVERY == 0) and not person_too_close:
            # Build ROI around feet if we have it (prefer downwards area)
            if feet_center:
                fx, fy = feet_center
                # ROI parameters relative to frame
                roi_w = int(W * 0.9)
                roi_h = int(H * 0.7)
                rx1 = clamp(int(fx - roi_w*0.5), 0, W-1)
                ry1 = clamp(int(fy - roi_h*0.25), 0, H-1)
                rx2 = clamp(rx1 + roi_w, 0, W)
                ry2 = clamp(ry1 + roi_h, 0, H)
            else:
                rx1, ry1, rx2, ry2 = 0, 0, W, H

            roi = frame[ry1:ry2, rx1:rx2]
            if roi is None or roi.size == 0:
                roi = frame.copy(); rx1, ry1 = 0, 0

            # Scale ROI for detection speed
            scale = 1.0
            if roi.shape[1] > DETECT_RESIZE_W:
                scale = DETECT_RESIZE_W / float(roi.shape[1])
            small = cv2.resize(roi, (int(roi.shape[1]*scale), int(roi.shape[0]*scale)))
            sH, sW = small.shape[:2]

            # color mask (blue-ish)
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            mask_color = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)
            mask_color = cv2.morphologyEx(mask_color, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
            mask_color = cv2.morphologyEx(mask_color, cv2.MORPH_DILATE, np.ones((3,3), np.uint8))

            # edges
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (7,7), 0)
            edges = cv2.Canny(blur, 40, 110)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))

            # combine
            combined = cv2.bitwise_or(edges, mask_color)

            # find contours on combined map
            contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            all_pts = []
            for c in contours:
                a = cv2.contourArea(c)
                if a < 150 * (scale**2):  # ignore tiny blobs (in small scale)
                    continue
                # collect points
                pts = c.reshape(-1,2)
                all_pts.append(pts)

            detected_box = None
            detected_poly = None
            detected_conf = 0.0

            if len(all_pts) > 0:
                # merge all points into single array
                merged = np.vstack(all_pts)
                # convex hull in small coords
                hull = cv2.convexHull(merged)
                hull_area = cv2.contourArea(hull)
                hull_area_full = hull_area / (scale**2) if scale > 0 else hull_area
                if hull_area_full >= MIN_DETECT_AREA_FULL:
                    # min area rect in small coords, then scale up
                    rect = cv2.minAreaRect(hull)
                    (cx_s, cy_s), (rw_s, rh_s), ang = rect
                    # map center back to full-frame via ROI offset and scale
                    cx_full = cx_s / scale + rx1
                    cy_full = cy_s / scale + ry1
                    w_full = rw_s / scale
                    h_full = rh_s / scale
                    # aspect check (elognated board)
                    aspect = max(w_full, h_full) / (min(w_full, h_full) + 1e-6)
                    if 1.2 <= aspect <= 12.0:
                        bx = int(cx_full - w_full/2)
                        by = int(cy_full - h_full/2)
                        bw = int(round(w_full))
                        bh = int(round(h_full))
                        bx, by, bw, bh = expand_rect(bx, by, bw, bh, PAD_FACTOR, W, H)
                        detected_box = (bx, by, bw, bh)
                        # create polygon in full coords from hull scaled up
                        hull_full = (hull.astype(float) / scale) + np.array([[rx1, ry1]])
                        # approx poly to reduce noisy points
                        eps = 0.01 * cv2.arcLength(hull_full.astype(np.int32), True)
                        try:
                            approx = cv2.approxPolyDP(hull_full.astype(np.int32), eps, True)
                            if len(approx) > MAX_POLY_POINTS:
                                idxs = np.round(np.linspace(0, len(approx)-1, MAX_POLY_POINTS)).astype(int)
                                approx = approx[idxs]
                            detected_poly = approx.reshape(-1,2)
                        except Exception:
                            detected_poly = hull_full.reshape(-1,2).astype(int)
                        detected_conf = min(1.0, hull_area_full / CONF_AREA_REF)

            # color-fallback (very large color blob)
            if detected_box is None:
                contours_c, _ = cv2.findContours(mask_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in contours_c:
                    a = cv2.contourArea(c)
                    if a / (scale**2) > 3000:
                        x0, y0, w0, h0 = cv2.boundingRect(c)
                        bx = int(x0 / scale) + rx1
                        by = int(y0 / scale) + ry1
                        bw = int(w0 / scale); bh = int(h0 / scale)
                        bx, by, bw, bh = expand_rect(bx, by, bw, bh, PAD_FACTOR, W, H)
                        detected_box = (bx, by, bw, bh)
                        detected_conf = min(1.0, (a / (scale**2)) / CONF_AREA_REF)
                        detected_poly = None
                        break

            # adopt detection
            if detected_box is not None:
                last_conf = detected_conf
                last_poly = detected_poly
                last_poly_age = 0
                # set smoothed box immediately (so label jumps less)
                cx = detected_box[0] + detected_box[2]/2.0
                cy = detected_box[1] + detected_box[3]/2.0
                smoothed_box = ema(smoothed_box, [cx, cy, detected_box[2], detected_box[3]], EMA_ALPHA)
            else:
                # no detection -> age out polygon/confidence
                last_poly_age += 1
                last_conf = max(0.0, last_conf * 0.95)

            # if debug overlay requested show ROI & mask (scaled back)
            if SHOW_DEBUG_OVERLAYS:
                # draw ROI rect on vis
                cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (200, 200, 0), 1)
                mask_up = cv2.resize(mask_color, (rx2 - rx1, ry2 - ry1))
                # show mask in top-left small window
                mh, mw = mask_up.shape
                vis[0:mh, 0:mw, 2] = cv2.add(vis[0:mh, 0:mw, 2], mask_up)

        # If tracker was used but we decided to not use trackers by default, we ignore that path.

        # Decide final drawing: prefer smoothed_box, fallback to tracked_box
        draw_box = None
        if smoothed_box is not None and last_conf >= BOX_CONF_DRAW and not person_too_close:
            draw_box = smoothed_box
        elif tracked_box is not None and not person_too_close:
            # convert tracked box to center format
            tx, ty, tw, th = tracked_box
            draw_box = np.array([tx + tw/2.0, ty + th/2.0, tw, th], dtype=float)

        # Draw polygon if confident
        if last_poly is not None and last_conf >= POLY_CONF_DRAW and last_poly_age < 10 and not person_too_close:
            try:
                pts = last_poly.astype(int)
                cv2.polylines(vis, [pts], True, (0, 220, 80), 2)
            except Exception:
                pass

        # Draw box and label (with smoothing)
        if draw_box is not None:
            cx, cy, bw, bh = draw_box.astype(float)
            x = int(round(cx - bw/2.0)); y = int(round(cy - bh/2.0))
            w_box = int(round(bw)); h_box = int(round(bh))
            x = clamp(x, 0, W-1); y = clamp(y, 0, H-1)
            w_box = clamp(w_box, 1, W - x); h_box = clamp(h_box, 1, H - y)
            # draw rectangle
            cv2.rectangle(vis, (x, y), (x + w_box, y + h_box), (16, 200, 20), 3)
            # label inside box with clipping so it remains on board
            label = "Board"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            lx = int(clamp(cx - tw/2.0, x + 4, x + w_box - tw - 4))
            ly = int(clamp(cy + th/2.0, y + th + 4, y + h_box - 4))
            cv2.putText(vis, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10,10,10), 3)
            cv2.putText(vis, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230,255,230), 1)

        # Draw feet marker & pose (always)
        if feet_center:
            cv2.circle(vis, feet_center, 5, (0,180,255), -1)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(vis, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # status
        frames += 1
        if frames % 10 == 0:
            t1 = time.time()
            fps = frames / (t1 - t0 + 1e-6)
            t0 = t1; frames = 0
        else:
            fps = None
        info = f"Conf={last_conf:.2f}  Shoulders={int(shoulder_dist)}"
        if fps:
            info += f"  FPS~{fps:.1f}"
        cv2.putText(vis, info, (8, H-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1)

        cv2.imshow("Surf Pose + Board Tracker", vis)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        if key == ord('d'):
            # toggle debug overlays
            SHOW_DEBUG_OVERLAYS = not SHOW_DEBUG_OVERLAYS
        if key == ord('t'):
            # toggle tracker usage
            USE_TRACKER = not USE_TRACKER
            if not USE_TRACKER:
                tracker = None

        frame_idx += 1

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    try:
        pose.close()
    except Exception:
        pass
    cap.release()
    cv2.destroyAllWindows()
