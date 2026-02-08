"""
marker_simulator.py

Simulates neon tape markers on the surfboard for testing the detection pipeline.
Adds synthetic bright markers without needing physical tape.

HSV color ranges for bright neon markers:
- Neon Green: H ~80-85, S ~255, V ~255
- Neon Pink/Magenta: H ~140-165, S ~255, V ~255
- Neon Yellow: H ~20-30, S ~255, V ~255
- Neon Orange: H ~5-20, S ~255, V ~255
"""

import cv2
import numpy as np
from math import atan2, sqrt

class MarkerSimulator:
    def __init__(self, marker_color='green', marker_size=15):
        """
        Initialize marker simulator.
        
        Args:
            marker_color: 'green', 'pink', 'yellow', or 'orange'
            marker_size: radius of marker circles (pixels)
        """
        self.marker_size = marker_size
        self.marker_color = marker_color
        
        # BGR colors for neon markers (OpenCV uses BGR, not RGB)
        self.color_map = {
            'green': (0, 255, 0),      # Neon green
            'pink': (255, 0, 255),     # Neon pink/magenta
            'yellow': (0, 255, 255),   # Neon yellow
            'orange': (0, 165, 255),   # Neon orange
        }
        
        self.marker_positions = {}  # Store marker positions for this frame
    
    def add_markers_to_frame(self, frame, board_roi=None):
        """
        Add synthetic neon tape markers to frame.
        
        If no ROI provided, places markers in lower half of frame (typical surfboard area).
        Otherwise places them within/around the provided ROI.
        
        Args:
            frame: input frame (H, W, 3) BGR
            board_roi: Optional tuple (x1, y1, x2, y2) defining board region
            
        Returns:
            marked_frame: frame with markers added
            marker_positions: dict of marker names -> (x, y) positions
        """
        marked_frame = frame.copy()
        H, W = frame.shape[:2]
        color = self.color_map.get(self.marker_color, (0, 255, 0))
        
        # Determine marker placement region
        if board_roi is not None:
            x1, y1, x2, y2 = board_roi
            # Place markers within ROI
            roi_w = x2 - x1
            roi_h = y2 - y1
            
            # Nose marker (top of ROI)
            nose_x = int(x1 + roi_w * 0.5)
            nose_y = int(y1 + roi_h * 0.2)
            
            # Tail marker (bottom of ROI)
            tail_x = int(x1 + roi_w * 0.5)
            tail_y = int(y1 + roi_h * 0.8)
            
            # Center marker (middle of ROI)
            center_x = int(x1 + roi_w * 0.5)
            center_y = int(y1 + roi_h * 0.5)
            
        else:
            # Default: lower half of frame, centered horizontally
            board_y_start = int(H * 0.4)
            board_y_end = int(H * 0.95)
            board_x_center = W // 2
            
            # Nose (top of board region)
            nose_x = board_x_center
            nose_y = board_y_start
            
            # Tail (bottom of board region)
            tail_x = board_x_center
            tail_y = board_y_end
            
            # Center (middle of board region)
            center_x = board_x_center
            center_y = (board_y_start + board_y_end) // 2
        
        # Draw markers as circles with a small cross
        self.marker_positions = {
            'nose': (nose_x, nose_y),
            'tail': (tail_x, tail_y),
            'center': (center_x, center_y),
        }
        
        for name, (mx, my) in self.marker_positions.items():
            # Clamp to frame bounds
            mx = max(0, min(W - 1, mx))
            my = max(0, min(H - 1, my))
            self.marker_positions[name] = (mx, my)
            
            # Draw filled circle
            cv2.circle(marked_frame, (mx, my), self.marker_size, color, -1)
            
            # Draw cross through center
            cross_len = self.marker_size // 2
            cv2.line(marked_frame, (mx - cross_len, my), (mx + cross_len, my), (0, 0, 0), 1)
            cv2.line(marked_frame, (mx, my - cross_len), (mx, my + cross_len), (0, 0, 0), 1)
        
        return marked_frame, self.marker_positions
    
    def detect_markers_in_frame(self, frame, hsv_range=None):
        """
        Detect the synthetic markers in a frame by color.
        
        Args:
            frame: input frame (BGR)
            hsv_range: Optional (lower, upper) HSV ranges. If None, auto-detects for marker color.
            
        Returns:
            marker_centroids: dict of marker names -> (x, y) if detected, else None
            confidence: how many pixels matched the color (for confidence scoring)
        """
        if hsv_range is None:
            # Auto HSV ranges for marker colors
            hsv_ranges = {
                'green': (np.array([35, 100, 100]), np.array([85, 255, 255])),
                'pink': (np.array([125, 100, 100]), np.array([165, 255, 255])),
                'yellow': (np.array([15, 100, 100]), np.array([35, 255, 255])),
                'orange': (np.array([5, 100, 100]), np.array([25, 255, 255])),
            }
            hsv_range = hsv_ranges.get(self.marker_color, hsv_ranges['green'])
        
        lower, upper = hsv_range
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        centroids = {}
        confidences = {}
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50:  # Ignore tiny blobs
                continue
            
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                centroids[f'marker_{len(centroids)}'] = (cx, cy)
                confidences[f'marker_{len(centroids)}'] = area
        
        return centroids, confidences
    
    def calculate_board_vectors(self, centroids):
        """
        Calculate board orientation from detected marker centroids.
        
        Assumes 'nose' and 'tail' markers are detected.
        
        Args:
            centroids: dict of marker names -> (x, y)
            
        Returns:
            board_info: dict with keys:
                - 'center': (x, y) midpoint between nose and tail
                - 'angle': angle in radians (-pi to pi)
                - 'angle_deg': angle in degrees
                - 'length': distance from nose to tail
                - 'vector': (dx, dy) unit vector pointing from tail to nose
        """
        if 'nose' not in centroids or 'tail' not in centroids:
            return None
        
        nx, ny = centroids['nose']
        tx, ty = centroids['tail']
        
        # Center is midpoint
        cx = (nx + tx) / 2.0
        cy = (ny + ty) / 2.0
        
        # Vector from tail to nose
        dx = nx - tx
        dy = ny - ty
        length = sqrt(dx**2 + dy**2)
        
        if length < 1e-6:
            return None
        
        # Unit vector
        ux = dx / length
        uy = dy / length
        
        # Angle: atan2(y_delta, x_delta) gives angle from tail pointing toward nose
        angle = atan2(dy, dx)
        angle_deg = np.degrees(angle)
        
        return {
            'center': (cx, cy),
            'angle': angle,
            'angle_deg': angle_deg,
            'length': length,
            'vector': (ux, uy),
            'nose': (nx, ny),
            'tail': (tx, ty),
        }
    
    def draw_board_overlay(self, frame, board_info, color=(0, 255, 0), thickness=3):
        """
        Draw board orientation overlay on frame based on detected markers.
        
        Args:
            frame: input frame
            board_info: output from calculate_board_vectors()
            color: BGR color for overlay
            thickness: line thickness
            
        Returns:
            frame with overlay drawn
        """
        if board_info is None:
            return frame
        
        overlay = frame.copy()
        cx, cy = board_info['center']
        nx, ny = board_info['nose']
        tx, ty = board_info['tail']
        ux, uy = board_info['vector']
        
        # Draw line from tail to nose
        cv2.line(overlay, (int(tx), int(ty)), (int(nx), int(ny)), color, thickness)
        
        # Draw circle at center
        cv2.circle(overlay, (int(cx), int(cy)), 8, color, 2)
        
        # Draw direction arrow
        arrow_len = 50
        arrow_end_x = int(cx + arrow_len * ux)
        arrow_end_y = int(cy + arrow_len * uy)
        cv2.arrowedLine(overlay, (int(cx), int(cy)), (arrow_end_x, arrow_end_y), 
                        color, thickness, tipLength=0.3)
        
        # Add info text
        text = f"Angle: {board_info['angle_deg']:.1f}°  Len: {board_info['length']:.0f}px"
        cv2.putText(overlay, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, color, 2)
        
        return overlay
