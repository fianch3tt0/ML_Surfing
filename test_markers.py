#!/usr/bin/env python3
"""
test_markers.py

Real-time demo of marker simulator and detection.
Tests the synthetic neon marker pipeline without needing physical tape.

Keys:
  's' - toggle marker simulation on/off
  'c' - cycle through marker colors (green, pink, yellow, orange)
  'd' - toggle detection overlay
  'q' or ESC - quit
"""

import cv2
import sys
from marker_simulator import MarkerSimulator

def main():
    # Initialize capture (webcam or video file)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera. Trying sample video...")
        # Try to find a video file
        try:
            cap = cv2.VideoCapture("videos/sample.mp4")
            if not cap.isOpened():
                print("ERROR: No video file found either.")
                sys.exit(1)
        except:
            sys.exit(1)
    
    cv2.namedWindow("Marker Simulator Demo", cv2.WINDOW_NORMAL)
    
    marker_sim = MarkerSimulator(marker_color='green', marker_size=15)
    
    simulate_mode = True
    show_detection = True
    color_cycle = ['green', 'pink', 'yellow', 'orange']
    color_idx = 0
    
    frame_count = 0
    
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║     Marker Simulator Demo - Synthetic Neon Tape Test       ║
    ╠════════════════════════════════════════════════════════════╣
    ║ CONTROLS:                                                  ║
    ║   's' - Toggle marker simulation on/off                    ║
    ║   'c' - Cycle through marker colors                        ║
    ║   'd' - Toggle detection overlay                           ║
    ║   'q' or ESC - Quit                                        ║
    ╠════════════════════════════════════════════════════════════╣
    ║ This demo tests the full pipeline:                         ║
    ║   1. Add synthetic neon markers to frames                  ║
    ║   2. Detect markers by HSV color                           ║
    ║   3. Calculate board orientation from markers              ║
    ║   4. Display board vectors and angle                       ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video or camera disconnected")
                break
            
            H, W = frame.shape[:2]
            vis = frame.copy()
            
            # Add synthetic markers if enabled
            if simulate_mode:
                # Simulate markers in lower half of frame
                board_roi = (int(W*0.2), int(H*0.3), int(W*0.8), int(H*0.9))
                vis, marker_pos = marker_sim.add_markers_to_frame(vis, board_roi)
                
                if show_detection:
                    # Try to detect the markers we just added
                    centroids, confidences = marker_sim.detect_markers_in_frame(vis)
                    
                    # For this test, we'll map detected centroids to nose/tail manually
                    # In real use, we'd need clustering logic
                    if len(centroids) >= 2:
                        # Sort by y-coordinate: smallest y = nose, largest y = tail
                        sorted_markers = sorted(centroids.items(), 
                                              key=lambda item: item[1][1])
                        detected_nose = sorted_markers[0][1]
                        detected_tail = sorted_markers[-1][1]
                        
                        detected_centroids = {
                            'nose': detected_nose,
                            'tail': detected_tail,
                        }
                        
                        board_info = marker_sim.calculate_board_vectors(detected_centroids)
                        vis = marker_sim.draw_board_overlay(vis, board_info, 
                                                           color=(0, 255, 0), thickness=2)
                    
                    # Draw detected marker centroids
                    for name, (mx, my) in centroids.items():
                        cv2.circle(vis, (mx, my), 5, (255, 0, 0), -1)
            
            # Draw info
            mode_text = f"Mode: {'SIMULATE' if simulate_mode else 'DETECT ONLY'} | Color: {marker_sim.marker_color} | Overlay: {'ON' if show_detection else 'OFF'}"
            cv2.putText(vis, mode_text, (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 1)
            
            cv2.imshow("Marker Simulator Demo", vis)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q or ESC
                break
            elif key == ord('s'):
                simulate_mode = not simulate_mode
                print(f"Simulation: {'ON' if simulate_mode else 'OFF'}")
            elif key == ord('c'):
                color_idx = (color_idx + 1) % len(color_cycle)
                marker_sim.marker_color = color_cycle[color_idx]
                print(f"Marker color: {marker_sim.marker_color}")
            elif key == ord('d'):
                show_detection = not show_detection
                print(f"Detection overlay: {'ON' if show_detection else 'OFF'}")
            
            frame_count += 1
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nProcessed {frame_count} frames")
        print("Demo complete!")

if __name__ == '__main__':
    main()
